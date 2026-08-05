# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The analytics observer: everything one stream computes per processed frame.

`StreamRuntime` knows how to capture frames and run a model. It knows nothing about tracks, zones or events. This
observer is the piece that plugs into it, and it is the only place where per-frame work becomes durable state.

Ordering matters and is fixed here: associate tracks, resolve spatial membership against the tracked subset, retire
tracks that have aged out (closing their open zone visits), then persist. Live counts returned to the runtime come from
the same pass, so the preview, `get_stream_status` and `get_zone_occupancy` can never disagree about a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_mcp.analytics.events import EventSink
from vision_mcp.analytics.metrics import MetricsCollector
from vision_mcp.analytics.spatial import Crossing, SpatialIndex, SpatialUpdate, Violation, ZoneVisit
from vision_mcp.analytics.tracking import Tracker, TrackRecord
from vision_mcp.api_contract import ActiveTracks, TrackInfo, ZoneOccupancy, ZoneOccupancyResult
from vision_mcp.clock import utc_iso
from vision_mcp.config import EngineConfig, StreamEntry
from vision_mcp.inference.detector import InferenceOutput
from vision_mcp.logging_setup import get_logger
from vision_mcp.storage.database import Database, Statement
from vision_mcp.streams.frames import Frame
from vision_mcp.streams.runtime import ObservedState

logger = get_logger("vision-mcp.analytics")


@dataclass(slots=True)
class _Live:
    """The latest frame's derived state, served to live tools without touching SQLite."""

    zone_counts: dict[str, int]
    zone_by_class: dict[str, dict[str, int]]
    line_counts: dict[str, tuple[int, int]]


class StreamAnalytics:
    """Tracking, zones, lines, events and metrics for one stream."""

    def __init__(
        self,
        stream_id: str,
        entry: StreamEntry,
        config: EngineConfig,
        database: Database,
        events: EventSink,
    ) -> None:
        self.stream_id = stream_id
        self.entry = entry
        self._database = database
        self._events = events
        self._tracker = Tracker(config.tracking, entry.processing_fps)
        self._spatial = SpatialIndex(entry)
        self.metrics = MetricsCollector(stream_id, config.metrics.latency_samples)
        self._live = _Live({}, {}, {})

    @property
    def spatial(self) -> SpatialIndex:
        """Resolved zones and lines, for the preview overlay."""
        return self._spatial

    async def observe(self, stream_id: str, frame: Frame, output: InferenceOutput) -> ObservedState:
        """Fold one inference result into tracks, zones, lines, events and metrics."""
        at = frame.captured_at
        self._spatial.configure(frame.size.width, frame.size.height)
        tracked = self._tracker.update(output.raw, output.detections, at)
        classes = {record.track_id: record.class_name for record in self._tracker.records.values()}
        update = self._spatial.update(tracked.tracked, classes, at)
        closed = self._close_visits(tracked.expired)
        self._live = _Live(update.zone_counts, update.zone_by_class, update.line_counts)
        self.metrics.observe(tracked.detections, output.inference_ms)
        await self._persist(tracked.expired, update.entries, [*update.exits, *closed], update.crossings)
        await self._emit(update, [*update.exits, *closed], at, frame.array)
        return ObservedState(
            detections=tracked.detections,
            raw=tracked.raw,
            active_tracks=self._tracker.active_count,
            zone_counts=update.zone_counts,
            line_counts=update.line_counts,
        )

    async def close(self) -> None:
        """Persist whatever survived the last frame so a stop does not lose history."""
        remaining = self._tracker.flush()
        visits = self._close_visits(remaining)
        await self._persist(remaining, [], visits, [])

    def active_tracks(self) -> ActiveTracks:
        """Live answer for `get_active_tracks`, including zone membership."""
        zones: dict[int, list[str]] = {}
        for visit in self._spatial.open_visits:
            zones.setdefault(visit.track_id, []).append(visit.zone)
        tracks = [
            _track_info(record, sorted(zones.get(record.track_id, [])))
            for record in sorted(self._tracker.active_records, key=lambda item: item.track_id)
        ]
        return ActiveTracks(stream_id=self.stream_id, active_tracks=self._tracker.active_count, tracks=tracks)

    def zone_occupancy(self) -> ZoneOccupancyResult:
        """Live answer for `get_zone_occupancy`."""
        zones = [
            ZoneOccupancy(
                zone=name,
                occupancy=self._live.zone_counts.get(name, 0),
                by_class=self._live.zone_by_class.get(name, {}),
                limit=self.entry.occupancy_limits.get(name),
                over_limit=_over(self._live.zone_counts.get(name, 0), self.entry.occupancy_limits.get(name)),
            )
            for name in self.entry.zones
        ]
        return ZoneOccupancyResult(stream_id=self.stream_id, zones=zones)

    def line_counts(self) -> dict[str, tuple[int, int]]:
        """Cumulative in/out totals per line since the stream started."""
        return dict(self._live.line_counts)

    def track_statements(self) -> list[Statement]:
        """Upsert current track lifecycles at aggregation cadence, not per frame."""
        return _track_rows(self.stream_id, list(self._tracker.records.values()))

    async def _persist(
        self,
        expired: list[TrackRecord],
        entries: list[ZoneVisit],
        exits: list[ZoneVisit],
        crossings: list[Crossing],
    ) -> None:
        """Queue this frame's durable rows.

        Never blocks the inference loop.
        """
        statements: list[Statement] = []
        statements.extend(_track_rows(self.stream_id, expired))
        statements.extend(_entry_rows(self.stream_id, entries))
        statements.extend(_exit_rows(self.stream_id, exits))
        statements.extend(_crossing_rows(self.stream_id, crossings))
        if statements:
            self._database.submit(statements)

    def _close_visits(self, records: list[TrackRecord]) -> list[ZoneVisit]:
        """Close each track's open zone visits at that track's own last sighting."""
        closed: list[ZoneVisit] = []
        for record in records:
            closed.extend(self._spatial.close_tracks((record.track_id,), record.last_seen))
        return closed

    async def _emit(
        self,
        update: SpatialUpdate,
        exits: list[ZoneVisit],
        at: float,
        frame: np.ndarray[Any, Any],
    ) -> None:
        """Turn this frame's spatial changes into events."""
        for visit in update.entries:
            await self._events.emit(self.stream_id, "zone_entry", at, _visit_details(visit))
        for visit in exits:
            await self._events.emit(self.stream_id, "zone_exit", visit.exited_at or at, _visit_details(visit))
        for crossing in update.crossings:
            await self._events.emit(self.stream_id, "line_crossing", crossing.at, _crossing_details(crossing))
        for violation in update.violations:
            await self._events.emit(
                self.stream_id,
                "occupancy_violation",
                at,
                _violation_details(violation),
                severity="warning",
                frame=frame,
            )


def _track_info(record: TrackRecord, zones: list[str]) -> TrackInfo:
    """Contract view of one track record."""
    return TrackInfo(
        track_id=record.track_id,
        class_name=record.class_name,
        confidence=round(record.confidence, 4),
        box=record.box,
        first_seen=utc_iso(record.first_seen),
        last_seen=utc_iso(record.last_seen),
        age_seconds=round(record.duration_seconds, 2),
        zones=zones,
    )


def _over(occupancy: int, limit: int | None) -> bool:
    """Whether a zone is above its configured limit."""
    return limit is not None and occupancy > limit


def _visit_details(visit: ZoneVisit) -> dict[str, str | int | float]:
    """Event payload for a zone entry or exit."""
    details: dict[str, str | int | float] = {
        "zone": visit.zone,
        "track_id": visit.track_id,
        "class_name": visit.class_name,
    }
    if visit.dwell_seconds is not None:
        details["dwell_seconds"] = round(visit.dwell_seconds, 2)
    return details


def _crossing_details(crossing: Crossing) -> dict[str, str | int | float]:
    """Event payload for a line crossing."""
    return {
        "line": crossing.line,
        "track_id": crossing.track_id,
        "class_name": crossing.class_name,
        "direction": crossing.direction,
    }


def _violation_details(violation: Violation) -> dict[str, str | int | float]:
    """Event payload for an occupancy violation."""
    return {"zone": violation.zone, "occupancy": violation.occupancy, "limit": violation.limit}


def _track_rows(stream_id: str, records: list[TrackRecord]) -> list[Statement]:
    """Lifecycle rows for retired tracks; live tracks are never written."""
    return [
        (
            "INSERT OR REPLACE INTO tracks (stream_id, track_id, class_name, first_seen, last_seen,"
            " frames, mean_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                stream_id,
                record.track_id,
                record.class_name,
                record.first_seen,
                record.last_seen,
                record.frames,
                round(record.mean_confidence, 4),
            ),
        )
        for record in records
    ]


def _entry_rows(stream_id: str, visits: list[ZoneVisit]) -> list[Statement]:
    """Open one zone_transitions row per entry."""
    return [
        (
            "INSERT OR IGNORE INTO zone_transitions (stream_id, zone, track_id, class_name, entered_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (stream_id, visit.zone, visit.track_id, visit.class_name, visit.entered_at),
        )
        for visit in visits
    ]


def _exit_rows(stream_id: str, visits: list[ZoneVisit]) -> list[Statement]:
    """Close the matching row when a visit ends."""
    return [
        (
            "UPDATE zone_transitions SET exited_at = ?, dwell_seconds = ?"
            " WHERE stream_id = ? AND zone = ? AND track_id = ? AND entered_at = ?",
            (
                visit.exited_at,
                round(visit.dwell_seconds or 0.0, 3),
                stream_id,
                visit.zone,
                visit.track_id,
                visit.entered_at,
            ),
        )
        for visit in visits
    ]


def _crossing_rows(stream_id: str, crossings: list[Crossing]) -> list[Statement]:
    """One row per line crossing."""
    return [
        (
            "INSERT INTO line_crossings (stream_id, line, track_id, class_name, at, direction)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                stream_id,
                crossing.line,
                crossing.track_id,
                crossing.class_name,
                crossing.at,
                crossing.direction,
            ),
        )
        for crossing in crossings
    ]
