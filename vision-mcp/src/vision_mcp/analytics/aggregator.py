# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The background task that turns live counters into SQLite rows.

One task serves every stream. It wakes on a fixed interval, closes the current bucket for each stream and hands the
statements to the database writer thread. Nothing here awaits a disk write: a slow disk must never slow down capture or
inference.

Frame counters on the capture side are cumulative for the life of a stream, so the aggregator keeps the previous reading
and stores the delta. Counters reset when a stream restarts, which shows up as a smaller reading and is treated as a
fresh baseline rather than a negative delta.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from vision_mcp.analytics.events import EventSink
from vision_mcp.analytics.metrics import BucketCounts, HealthSample, MetricsCollector
from vision_mcp.analytics.observer import StreamAnalytics
from vision_mcp.clock import epoch
from vision_mcp.logging_setup import get_logger
from vision_mcp.storage.database import Database, Statement
from vision_mcp.streams.manager import StreamManager
from vision_mcp.streams.runtime import StreamRuntime

logger = get_logger("vision-mcp.aggregator")


class MetricsAggregator:
    """Flushes every stream's metric bucket on a fixed cadence."""

    def __init__(
        self,
        database: Database,
        manager: StreamManager,
        analytics: Mapping[str, StreamAnalytics],
        interval_seconds: float,
        events: EventSink | None = None,
    ) -> None:
        self._database = database
        self._manager = manager
        self._analytics = analytics
        self._interval = interval_seconds
        self._events = events
        self._task: asyncio.Task[None] | None = None
        self._bucket_start = 0.0
        self._previous: dict[str, tuple[int, int]] = {}
        self._previous_states: dict[str, str] = {}

    @property
    def running(self) -> bool:
        """Whether the flush task is alive."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Begin flushing.

        Idempotent.
        """
        if self.running:
            return
        self._bucket_start = _now()
        self._previous_states = {runtime.stream_id: runtime.state for runtime in self._manager}
        self._task = asyncio.create_task(self._loop(), name="metrics-aggregator")

    async def stop(self) -> None:
        """Flush one last bucket and stop."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.flush()

    def flush(self) -> None:
        """Close the current bucket for every stream and queue the rows."""
        now = _now()
        elapsed = max(now - self._bucket_start, 0.001)
        statements: list[Statement] = []
        for runtime in self._manager:
            statements.extend(self._statements(runtime, now, elapsed))
        self._bucket_start = now
        if statements:
            self._database.submit(statements)

    def _statements(self, runtime: StreamRuntime, at: float, elapsed: float) -> list[Statement]:
        """Metric and health rows for one stream.

        Health is sampled at ``at``; metrics cover the bucket that ends there.
        """
        analytics = self._analytics.get(runtime.stream_id)
        status = runtime.status()
        sample = HealthSample(
            stream_id=runtime.stream_id,
            at=at,
            state=status.state,
            health=status.health,
            processed_fps=status.processed_fps,
            queue_depth=status.queue_depth,
            dropped_frames=status.dropped_frames,
            last_error=status.last_error,
        )
        rows = [sample.statement()]
        if analytics is not None:
            counts = self._counts(
                runtime.stream_id, status.captured_frames, status.dropped_frames, analytics.metrics
            )
            rows.extend(analytics.metrics.flush(self._bucket_start, elapsed, counts))
            rows.extend(analytics.track_statements())
        return rows

    def _counts(
        self, stream_id: str, captured: int, dropped: int, collector: MetricsCollector
    ) -> BucketCounts:
        """Bucket-local frame counters, derived from cumulative capture-side totals."""
        previous_captured, previous_dropped = self._previous.get(stream_id, (0, 0))
        if captured < previous_captured or dropped < previous_dropped:
            previous_captured, previous_dropped = 0, 0
        self._previous[stream_id] = (captured, dropped)
        return BucketCounts(
            captured=max(0, captured - previous_captured),
            processed=collector.processed_frames,
            dropped=max(0, dropped - previous_dropped),
        )

    async def _loop(self) -> None:
        """Flush on the configured interval until cancelled."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                self.flush()
                await self._emit_state_changes()
            except asyncio.CancelledError:
                raise
            except Exception:  # a bad bucket must not kill aggregation
                logger.exception("metrics flush failed")

    async def _emit_state_changes(self) -> None:
        """Persist stream failure/recovery events only when capture state changes."""
        if self._events is None:
            return
        for runtime in self._manager:
            previous = self._previous_states.get(runtime.stream_id)
            current = runtime.state
            self._previous_states[runtime.stream_id] = current
            if current == previous:
                continue
            if current in {"disconnected", "reconnecting"}:
                await self._events.emit(
                    runtime.stream_id,
                    "stream_failure",
                    epoch(),
                    {"state": current},
                    severity="error",
                )
            elif current == "connected" and previous in {"disconnected", "reconnecting"}:
                await self._events.emit(
                    runtime.stream_id,
                    "stream_recovered",
                    epoch(),
                    {"state": current},
                )


def _now() -> float:
    """Bucket boundaries are wall-clock, because every read filters by wall-clock windows."""
    return epoch()
