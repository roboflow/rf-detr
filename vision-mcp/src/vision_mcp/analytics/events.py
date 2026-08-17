# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Business-level events: persistence, optional snapshots and change notifications.

Events are the fifth metric family in the README's list, and the only one that is not a count of something.
They are rare by construction — an entry, an exit, a crossing, a limit breach, a stream failing or recovering
— so each one is worth a durable row.

Snapshots are deliberately stingy. Only violations and failures are worth an image, and even then the frame is
stored once, as an artifact reference, never as bytes in a payload.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import numpy as np

from vision_mcp.api_contract import EventRecord, EventType
from vision_mcp.clock import utc_iso
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import redact
from vision_mcp.storage.artifacts import ArtifactStore
from vision_mcp.storage.database import Database, Statement

logger = get_logger("vision-mcp.events")

Severity = Literal["info", "warning", "error"]
Notifier = Callable[[EventRecord], Awaitable[None]]

_SNAPSHOT_EVENTS = frozenset({"occupancy_violation"})
"""Event types worth an artifact.

Everything else references the stream's live frames.
"""

_RECENT_LIMIT = 256
"""In-memory ring size, used for cheap resource reads; SQLite remains the source of truth."""


class EventSink:
    """Writes events to SQLite, attaches snapshots and fans out notifications."""

    def __init__(
        self,
        database: Database,
        artifacts: ArtifactStore,
        snapshot_quality: int = 80,
    ) -> None:
        self._database = database
        self._artifacts = artifacts
        self._quality = snapshot_quality
        self._recent: deque[EventRecord] = deque(maxlen=_RECENT_LIMIT)
        self._notifiers: list[Notifier] = []

    def subscribe(self, notifier: Notifier) -> None:
        """Register a callback invoked once per event, after persistence is queued."""
        self._notifiers.append(notifier)

    def recent(self, stream_id: str | None = None, limit: int = 50) -> list[EventRecord]:
        """Newest events still held in memory, optionally filtered by stream."""
        chosen = [item for item in reversed(self._recent) if stream_id in (None, item.stream_id)]
        return chosen[:limit]

    async def emit(
        self,
        stream_id: str,
        event_type: EventType,
        at: float,
        details: dict[str, str | int | float],
        severity: Severity = "info",
        frame: np.ndarray[Any, Any] | None = None,
    ) -> EventRecord:
        """Persist one event and notify subscribers.

        Args:
            stream_id: Stream the event belongs to.
            event_type: One of the contract's event types.
            at: Wall-clock time of the occurrence, in epoch seconds.
            details: Small scalar payload describing the occurrence.
            severity: Severity band used by `get_recent_errors` and health explanations.
            frame: Frame to snapshot, used only for event types that warrant one.
        """
        artifact_id = await self._snapshot(stream_id, event_type, frame)
        record = EventRecord(
            event_id=uuid.uuid4().hex,
            stream_id=stream_id,
            event_type=event_type,
            at=utc_iso(at),
            severity=severity,
            details=details,
            artifact_id=artifact_id,
        )
        self._database.submit([_insert(record, at)])
        self._recent.append(record)
        await self._notify(record)
        return record

    async def record_error(self, source: str, code: str, message: str, at: float) -> None:
        """Log a failure for `get_recent_errors`.

        Credentials are redacted first.
        """
        await self._database.write(
            [
                (
                    "INSERT INTO errors (at, source, code, message) VALUES (?, ?, ?, ?)",
                    (at, redact(source), code, redact(message)),
                )
            ]
        )

    async def _snapshot(
        self, stream_id: str, event_type: EventType, frame: np.ndarray[Any, Any] | None
    ) -> str | None:
        """Store a JPEG for event types that warrant one; never fail the event over it."""
        if frame is None or event_type not in _SNAPSHOT_EVENTS:
            return None
        from vision_mcp.streams.preview import encode_jpeg  # local: drawing is not a core dependency

        try:
            ref = await self._artifacts.save(
                kind="event_snapshot", data=encode_jpeg(frame, self._quality), stream_id=stream_id
            )
        except Exception:
            logger.exception("event snapshot failed", extra={"stream_id": stream_id})
            return None
        return ref.artifact_id

    async def _notify(self, record: EventRecord) -> None:
        """Run subscribers, isolating each from the others and from the caller."""
        for notifier in self._notifiers:
            try:
                await notifier(record)
            except Exception:
                logger.exception("event notification failed", extra={"stream_id": record.stream_id})


def _insert(record: EventRecord, at: float) -> Statement:
    """The INSERT for one event row, with details stored as compact JSON."""
    return (
        "INSERT INTO events (event_id, stream_id, event_type, at, severity, details, artifact_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record.event_id,
            record.stream_id,
            record.event_type,
            at,
            record.severity,
            json.dumps(record.details, separators=(",", ":")),
            record.artifact_id,
        ),
    )
