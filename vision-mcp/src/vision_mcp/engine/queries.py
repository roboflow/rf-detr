# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Historical reads over the aggregated SQLite schema."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from vision_mcp.analytics.metrics import percentile, safe_rate
from vision_mcp.api_contract import (
    Bucket,
    ClassCounts,
    ConfidenceDistribution,
    DetectionRate,
    DwellResult,
    DwellStats,
    EntryExitCounts,
    EntryExitResult,
    ErrorList,
    ErrorRecord,
    EventList,
    EventRecord,
    FrameDropRate,
    LatencyStats,
    LineCrossing,
    LineCrossingResult,
    ThroughputStats,
    TimeWindow,
    UniqueObjectCount,
)
from vision_mcp.clock import utc_iso
from vision_mcp.config import EngineConfig
from vision_mcp.query import TimeQuery, build_time_query
from vision_mcp.security import redact, redact_data
from vision_mcp.storage.database import Database

_MIN_UNIQUE_TRACK_FRAMES = 2
"""A one-frame association is detection noise, not a confirmed tracking instance."""


class HistoricalQueries:
    """Typed query methods; aggregated rows remain the only historical frame data."""

    def __init__(self, database: Database, config: EngineConfig) -> None:
        self._db = database
        self._config = config

    def resolve(self, window: str, interval: str) -> tuple[TimeQuery, TimeWindow]:
        """Validate and resolve the shared query grammar."""
        query = build_time_query(window, interval)
        return query, TimeWindow(
            window=window,
            interval=interval,
            start=utc_iso(query.start),
            end=utc_iso(query.end),
            bucket_count=query.bucket_count,
        )

    async def counts_by_class(self, stream_id: str, window: str, interval: str) -> ClassCounts:
        """Sum frame detections, grouped by class."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT class_name, SUM(detections) AS n FROM detection_summaries "
            "WHERE stream_id = ? AND bucket_start >= ? AND bucket_start < ? GROUP BY class_name",
            (stream_id, query.start, query.end),
        )
        counts = {str(row["class_name"]): int(row["n"]) for row in rows}
        return ClassCounts(
            stream_id=stream_id,
            window=view,
            frame_detections=sum(counts.values()),
            counts_by_class=counts,
        )

    async def unique_objects(self, stream_id: str, window: str, interval: str) -> UniqueObjectCount:
        """Count distinct confirmed track instances, never frame detections."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT class_name, COUNT(*) AS n FROM tracks WHERE stream_id = ? "
            "AND last_seen >= ? AND first_seen < ? AND frames >= ? GROUP BY class_name",
            (stream_id, query.start, query.end, _MIN_UNIQUE_TRACK_FRAMES),
        )
        counts = {str(row["class_name"]): int(row["n"]) for row in rows}
        return UniqueObjectCount(
            stream_id=stream_id, window=view, unique_objects=sum(counts.values()), by_class=counts
        )

    async def detection_rate(self, stream_id: str, window: str, interval: str) -> DetectionRate:
        """Return frame detections per second in requested buckets."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT bucket_start, frame_detections FROM processing_metrics "
            "WHERE stream_id = ? AND bucket_start >= ? AND bucket_start < ?",
            (stream_id, query.start, query.end),
        )
        values = _bucket_sums(query, rows, "frame_detections")
        buckets = [
            Bucket(bucket_start=utc_iso(start), value=round(value / query.interval_seconds, 4))
            for start, value in values
        ]
        total = sum(value for _, value in values)
        return DetectionRate(
            stream_id=stream_id,
            window=view,
            buckets=buckets,
            mean_per_second=round(safe_rate(total, query.window_seconds), 4),
        )

    async def confidence(self, stream_id: str, window: str, interval: str) -> ConfidenceDistribution:
        """Combine stored confidence bins and weighted per-class means."""
        query, view = self.resolve(window, interval)
        histogram = await self._db.fetch_all(
            "SELECT bin_index, SUM(count) AS n FROM confidence_histogram WHERE stream_id = ? "
            "AND bucket_start >= ? AND bucket_start < ? GROUP BY bin_index",
            (stream_id, query.start, query.end),
        )
        summary = await self._db.fetch_one(
            "SELECT SUM(detections) AS n, SUM(detections * mean_confidence) AS weighted "
            "FROM detection_summaries WHERE stream_id = ? AND bucket_start >= ? AND bucket_start < ?",
            (stream_id, query.start, query.end),
        )
        counts = [0] * 10
        for row in histogram:
            counts[int(row["bin_index"])] = int(row["n"])
        samples = int(summary["n"] or 0) if summary else 0
        weighted = float(summary["weighted"] or 0.0) if summary else 0.0
        return ConfidenceDistribution(
            stream_id=stream_id,
            window=view,
            bin_edges=[index / 10 for index in range(11)],
            counts=counts,
            samples=samples,
            mean=None if samples == 0 else round(weighted / samples, 4),
        )

    async def entry_exit(self, stream_id: str, window: str, interval: str) -> EntryExitResult:
        """Count zone entries and exits separately."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT zone, SUM(CASE WHEN entered_at >= ? THEN 1 ELSE 0 END) AS entries, "
            "SUM(CASE WHEN exited_at >= ? AND exited_at < ? THEN 1 ELSE 0 END) AS exits "
            "FROM zone_transitions WHERE stream_id = ? AND (entered_at < ? AND COALESCE(exited_at, ?) >= ?) "
            "GROUP BY zone",
            (query.start, query.start, query.end, stream_id, query.end, query.end, query.start),
        )
        by_zone = {str(row["zone"]): (int(row["entries"]), int(row["exits"])) for row in rows}
        zones = []
        for name in self._config.streams[stream_id].zones:
            entries, exits = by_zone.get(name, (0, 0))
            zones.append(EntryExitCounts(zone=name, entries=entries, exits=exits, net=entries - exits))
        return EntryExitResult(stream_id=stream_id, window=view, zones=zones)

    async def dwell(self, stream_id: str, window: str, interval: str) -> DwellResult:
        """Calculate exact dwell percentiles from completed zone visits."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT zone, dwell_seconds FROM zone_transitions WHERE stream_id = ? "
            "AND exited_at >= ? AND exited_at < ? AND dwell_seconds IS NOT NULL ORDER BY dwell_seconds",
            (stream_id, query.start, query.end),
        )
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["zone"])].append(float(row["dwell_seconds"]))
        stats = [_dwell(name, grouped[name]) for name in self._config.streams[stream_id].zones]
        return DwellResult(stream_id=stream_id, window=view, zones=stats)

    async def crossings(self, stream_id: str, window: str, interval: str) -> LineCrossingResult:
        """Return directional line-crossing events newest first."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT line, track_id, class_name, at, direction FROM line_crossings "
            "WHERE stream_id = ? AND at >= ? AND at < ? ORDER BY at DESC",
            (stream_id, query.start, query.end),
        )
        totals: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        crossings = []
        for row in rows:
            direction = str(row["direction"])
            totals[str(row["line"])][direction] += 1
            crossings.append(
                LineCrossing(
                    line=str(row["line"]),
                    track_id=int(row["track_id"]),
                    class_name=str(row["class_name"]),
                    direction=direction,  # type: ignore[arg-type]
                    at=utc_iso(float(row["at"])),
                )
            )
        return LineCrossingResult(stream_id=stream_id, window=view, crossings=crossings, totals=dict(totals))

    async def latency(self, stream_id: str, window: str, interval: str) -> LatencyStats:
        """Combine aggregate latency samples without pretending frame rows exist."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT latency_samples, latency_mean_ms, latency_p50_ms, latency_p95_ms, "
            "latency_p99_ms, latency_max_ms FROM processing_metrics WHERE stream_id = ? "
            "AND bucket_start >= ? AND bucket_start < ?",
            (stream_id, query.start, query.end),
        )
        samples = sum(int(row["latency_samples"]) for row in rows)
        weighted = sum(int(row["latency_samples"]) * float(row["latency_mean_ms"] or 0.0) for row in rows)
        return LatencyStats(
            scope=stream_id,
            window=view,
            samples=samples,
            mean_ms=None if samples == 0 else round(weighted / samples, 2),
            p50_ms=_weighted_stat(rows, "latency_p50_ms"),
            p95_ms=_weighted_stat(rows, "latency_p95_ms"),
            p99_ms=_weighted_stat(rows, "latency_p99_ms"),
            max_ms=max(
                (float(row["latency_max_ms"]) for row in rows if row["latency_max_ms"] is not None),
                default=None,
            ),
        )

    async def throughput(self, stream_id: str, window: str, interval: str) -> ThroughputStats:
        """Return processed-frame throughput and bucket series."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT bucket_start, bucket_seconds, processed_frames FROM processing_metrics WHERE stream_id = ? "
            "AND bucket_start >= ? AND bucket_start < ?",
            (stream_id, query.start, query.end),
        )
        values = _bucket_rates(query, rows, "processed_frames")
        total = int(sum(value for _, value, _ in values))
        observed_seconds = sum(seconds for _, _, seconds in values)
        return ThroughputStats(
            stream_id=stream_id,
            window=view,
            processed_frames=total,
            processed_fps=round(safe_rate(total, observed_seconds), 4),
            target_fps=self._config.streams[stream_id].processing_fps,
            buckets=[
                Bucket(bucket_start=utc_iso(start), value=round(safe_rate(value, seconds), 4))
                for start, value, seconds in values
            ],
        )

    async def drop_rate(self, stream_id: str, window: str, interval: str) -> FrameDropRate:
        """Return a safe zero when no frames were captured."""
        query, view = self.resolve(window, interval)
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(captured_frames), 0) AS captured, "
            "COALESCE(SUM(processed_frames), 0) AS processed, "
            "COALESCE(SUM(dropped_frames), 0) AS dropped FROM processing_metrics "
            "WHERE stream_id = ? AND bucket_start >= ? AND bucket_start < ?",
            (stream_id, query.start, query.end),
        )
        captured, processed, dropped = (
            (int(row["captured"]), int(row["processed"]), int(row["dropped"])) if row else (0, 0, 0)
        )
        return FrameDropRate(
            stream_id=stream_id,
            window=view,
            captured_frames=captured,
            processed_frames=processed,
            dropped_frames=dropped,
            drop_rate=round(safe_rate(dropped, captured), 4),
        )

    async def events(self, stream_id: str | None, window: str, interval: str, limit: int) -> EventList:
        """Return recent persisted business events."""
        query, view = self.resolve(window, interval)
        where = "at >= ? AND at < ?"
        params: list[Any] = [query.start, query.end]
        if stream_id is not None:
            where += " AND stream_id = ?"
            params.append(stream_id)
        params.append(limit)
        rows = await self._db.fetch_all(
            f"SELECT * FROM events WHERE {where} ORDER BY at DESC LIMIT ?", params
        )
        events = [
            EventRecord(
                event_id=str(row["event_id"]),
                stream_id=str(row["stream_id"]),
                event_type=str(row["event_type"]),  # type: ignore[arg-type]
                at=utc_iso(float(row["at"])),
                severity=str(row["severity"]),  # type: ignore[arg-type]
                details=redact_data(json.loads(str(row["details"]))),
                artifact_id=None if row["artifact_id"] is None else str(row["artifact_id"]),
            )
            for row in rows
        ]
        return EventList(stream_id=stream_id, window=view, events=events)

    async def errors(self, window: str, interval: str, limit: int) -> ErrorList:
        """Return redacted recent errors."""
        query, view = self.resolve(window, interval)
        rows = await self._db.fetch_all(
            "SELECT at, source, code, message FROM errors WHERE at >= ? AND at < ? ORDER BY at DESC LIMIT ?",
            (query.start, query.end, limit),
        )
        return ErrorList(
            window=view,
            errors=[
                ErrorRecord(
                    at=utc_iso(float(row["at"])),
                    source=redact(row["source"]),
                    code=str(row["code"]),
                    message=redact(row["message"]),
                )
                for row in rows
            ],
        )


def _bucket_sums(query: TimeQuery, rows: list[Any], column: str) -> list[tuple[float, float]]:
    """Fold stored aggregation buckets into requested response buckets."""
    totals = [0.0] * query.bucket_count
    for row in rows:
        index = int((float(row["bucket_start"]) - query.start) // query.interval_seconds)
        if 0 <= index < len(totals):
            totals[index] += float(row[column])
    return list(zip(query.bucket_starts, totals, strict=True))


def _bucket_rates(query: TimeQuery, rows: list[Any], column: str) -> list[tuple[float, float, float]]:
    """Fold values and their observed durations into requested response buckets."""
    totals = [0.0] * query.bucket_count
    seconds = [0.0] * query.bucket_count
    for row in rows:
        index = int((float(row["bucket_start"]) - query.start) // query.interval_seconds)
        if 0 <= index < len(totals):
            totals[index] += float(row[column])
            seconds[index] += float(row["bucket_seconds"])
    return list(zip(query.bucket_starts, totals, seconds, strict=True))


def _weighted_stat(rows: list[Any], column: str) -> float | None:
    """Weight a stored per-bucket percentile by its sample count."""
    values = [
        (float(row[column]), int(row["latency_samples"]))
        for row in rows
        if row[column] is not None and int(row["latency_samples"]) > 0
    ]
    count = sum(weight for _, weight in values)
    return None if count == 0 else round(sum(value * weight for value, weight in values) / count, 2)


def _dwell(zone: str, samples: list[float]) -> DwellStats:
    """Create one exact dwell distribution."""
    ordered = sorted(samples)
    return DwellStats(
        zone=zone,
        samples=len(ordered),
        mean_seconds=None if not ordered else round(sum(ordered) / len(ordered), 2),
        p50_seconds=percentile(ordered, 50),
        p95_seconds=percentile(ordered, 95),
        max_seconds=None if not ordered else round(ordered[-1], 2),
    )
