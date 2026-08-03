"""Stream, worker and system status payloads."""

from __future__ import annotations

from typing import Literal

from vision_mcp.contract.base import Envelope, Health, ImageSize, Payload, StreamState


class StreamSummary(Payload):
    """Compact per-stream state for listings."""

    stream_id: str
    state: StreamState
    health: Health
    model: str
    source: str
    processed_frames: int
    current_objects: int
    active_tracks: int


class StreamList(Envelope):
    """All configured streams."""

    streams: list[StreamSummary]


class StreamStatus(Envelope):
    """Full state of one stream. `source` is redacted; RTSP credentials never appear."""

    stream_id: str
    state: StreamState
    health: Health
    health_reasons: list[str]
    model: str
    source: str
    frame_size: ImageSize | None
    target_fps: float
    processed_fps: float
    captured_frames: int
    processed_frames: int
    dropped_frames: int
    queue_depth: int
    queue_capacity: int
    active_tracks: int
    current_objects: int
    last_frame_at: str | None
    started_at: str | None
    reconnect_attempts: int
    last_error: str | None


class LiveSnapshot(Envelope):
    """Cheap snapshot for the browser viewer."""

    stream_id: str
    state: StreamState
    health: Health
    counts_by_class: dict[str, int]
    current_objects: int
    active_tracks: int
    processed_fps: float
    queue_depth: int
    queue_capacity: int
    dropped_frames: int
    inference_ms_p50: float | None
    last_frame_at: str | None


class WorkerInfo(Payload):
    """One long-lived thread or task owned by the engine."""

    name: str
    kind: Literal["capture", "inference", "db_writer", "cleanup", "stream_loop", "aggregator"]
    alive: bool
    detail: str | None = None


class WorkerList(Envelope):
    """Every worker the engine is running."""

    workers: list[WorkerInfo]
    thread_count: int


class SystemStatus(Envelope):
    """Engine-wide status."""

    version: str
    started_at: str
    uptime_seconds: float
    device: str
    mps_fallback_enabled: bool
    streams_configured: int
    streams_running: int
    models_loaded: int
    database_ok: bool
    artifacts_bytes: int
    health: Health
    health_reasons: list[str]
