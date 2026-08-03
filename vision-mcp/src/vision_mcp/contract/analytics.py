"""Tracking, zone, line, event and error payloads."""

from __future__ import annotations

from typing import Literal

from vision_mcp.contract.base import BoundingBox, Envelope, Payload, TimeWindow


class TrackInfo(Payload):
    """A live tracked object. IDs are motion-consistency handles, not identities."""

    track_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    first_seen: str
    last_seen: str
    age_seconds: float
    zones: list[str]


class ActiveTracks(Envelope):
    """Tracks alive in the latest processed frame."""

    stream_id: str
    active_tracks: int
    tracks: list[TrackInfo]


class ZoneOccupancy(Payload):
    """Current occupancy of one zone."""

    zone: str
    occupancy: int
    by_class: dict[str, int]
    limit: int | None
    over_limit: bool


class ZoneOccupancyResult(Envelope):
    """Occupancy for every configured zone on a stream."""

    stream_id: str
    zones: list[ZoneOccupancy]


class EntryExitCounts(Payload):
    """Entries and exits for one zone over a window."""

    zone: str
    entries: int
    exits: int
    net: int


class EntryExitResult(Envelope):
    """Zone entry/exit counts over a window."""

    stream_id: str
    window: TimeWindow
    zones: list[EntryExitCounts]


class DwellStats(Payload):
    """Dwell-time distribution for one zone, in seconds."""

    zone: str
    samples: int
    mean_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    max_seconds: float | None


class DwellResult(Envelope):
    """Dwell statistics over a window."""

    stream_id: str
    window: TimeWindow
    zones: list[DwellStats]


class LineCrossing(Payload):
    """One directional crossing event."""

    line: str
    track_id: int
    class_name: str
    direction: Literal["in", "out"]
    at: str


class LineCrossingResult(Envelope):
    """Crossings over a window, newest first."""

    stream_id: str
    window: TimeWindow
    crossings: list[LineCrossing]
    totals: dict[str, dict[str, int]]


EventType = Literal[
    "zone_entry", "zone_exit", "line_crossing", "occupancy_violation", "stream_failure", "stream_recovered"
]


class EventRecord(Payload):
    """A business-level occurrence."""

    event_id: str
    stream_id: str
    event_type: EventType
    at: str
    severity: Literal["info", "warning", "error"]
    details: dict[str, str | int | float]
    artifact_id: str | None = None


class EventList(Envelope):
    """Recent events, newest first."""

    stream_id: str | None
    window: TimeWindow
    events: list[EventRecord]


class ErrorRecord(Payload):
    """A logged failure, already redacted."""

    at: str
    source: str
    code: str
    message: str


class ErrorList(Envelope):
    """Recent errors, newest first."""

    window: TimeWindow
    errors: list[ErrorRecord]
