"""Shared primitives every contract module builds on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Bumped when a response shape changes incompatibly; present on every top-level response.
SCHEMA_VERSION = "1.0"

StreamState = Literal["starting", "connected", "reconnecting", "disconnected", "stopped"]
Health = Literal["healthy", "degraded", "unhealthy", "unknown"]


class Payload(BaseModel):
    """Base for nested payloads."""

    model_config = ConfigDict(extra="forbid")


class Envelope(Payload):
    """Base for every top-level response."""

    schema_version: str = SCHEMA_VERSION


class BoundingBox(Payload):
    """Pixel coordinates in the processed frame."""

    x1: float
    y1: float
    x2: float
    y2: float


class ImageSize(Payload):
    """Size of the image inference actually ran on."""

    width: int
    height: int


class Bucket(Payload):
    """One time bucket in an aggregated series."""

    bucket_start: str
    value: float


class TimeWindow(Payload):
    """Echo of the resolved query window."""

    window: str
    interval: str
    start: str
    end: str
    bucket_count: int


WindowValue = Literal["5m", "15m", "1h", "6h", "24h", "7d"]
IntervalValue = Literal["1s", "10s", "1m", "5m", "1h", "1d"]


class HistoricalQuery(Payload):
    """The one accepted time-window and interval grammar for historical tools."""

    time_window: WindowValue = "15m"
    interval: IntervalValue = "1m"

    @model_validator(mode="after")
    def _valid_bucket_count(self) -> HistoricalQuery:
        from vision_mcp.query import build_time_query

        build_time_query(self.time_window, self.interval)
        return self


class ArtifactRef(Payload):
    """A stored file, referenced by generated ID. Never a filesystem path."""

    artifact_id: str
    uri: str
    kind: Literal["frame", "event_snapshot", "crop"]
    media_type: str
    bytes: int
    created_at: str
    stream_id: str | None = None


class ArtifactResult(Envelope):
    """A single artifact reference."""

    artifact: ArtifactRef


class ErrorResponse(Envelope):
    """Engine failure body; mirrors VisionError."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
