# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Engine configuration: YAML in, validated Pydantic models out.

Config is the only place the engine accepts operator input, so every field is checked here rather than at the point of
use. Nothing in this module imports torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vision_mcp.errors import ErrorCode, VisionError

Task = Literal["detection", "segmentation", "keypoints"]

#: Architectures this engine will instantiate, grouped by the task they can serve.
ARCHITECTURES: dict[str, Task] = {
    "RFDETRNano": "detection",
    "RFDETRSmall": "detection",
    "RFDETRMedium": "detection",
    "RFDETRLarge": "detection",
    "RFDETRSegNano": "segmentation",
    "RFDETRSegSmall": "segmentation",
    "RFDETRSegMedium": "segmentation",
    "RFDETRSegLarge": "segmentation",
    "RFDETRKeypointPreview": "keypoints",
}

DEFAULT_RESOLUTIONS: dict[str, int] = {
    "RFDETRNano": 384,
    "RFDETRSmall": 512,
    "RFDETRMedium": 576,
    "RFDETRLarge": 704,
    "RFDETRSegNano": 312,
    "RFDETRSegSmall": 384,
    "RFDETRSegMedium": 432,
    "RFDETRSegLarge": 504,
    "RFDETRKeypointPreview": 576,
}

Normalised = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Normalised, Normalised]


class StrictModel(BaseModel):
    """Base for config sections; unknown keys are operator typos and must fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ZoneConfig(StrictModel):
    """A polygon in normalised frame coordinates, resolved to pixels at stream start."""

    polygon: list[Point] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def _distinct_vertices(cls, polygon: list[Point]) -> list[Point]:
        if len({tuple(point) for point in polygon}) < 3:
            raise ValueError("zone polygon needs at least 3 distinct vertices")
        return polygon


class LineConfig(StrictModel):
    """A counting line in normalised frame coordinates."""

    start: Point
    end: Point

    @model_validator(mode="after")
    def _not_degenerate(self) -> LineConfig:
        if self.start == self.end:
            raise ValueError("line start and end must differ")
        return self


class ModelEntry(StrictModel):
    """One loadable model.

    `checkpoint` selects custom weights; otherwise pretrained COCO weights.
    """

    architecture: str
    checkpoint: str | None = None
    task: Task = "detection"
    device: str = "auto"
    confidence: float = Field(default=0.4, gt=0.0, lt=1.0)
    resolution: int | None = Field(default=None, ge=224, le=1568)

    @property
    def effective_resolution(self) -> int:
        """Configured resolution or the RF-DETR architecture's native default."""
        return self.resolution or DEFAULT_RESOLUTIONS[self.architecture]

    @field_validator("architecture")
    @classmethod
    def _known_architecture(cls, architecture: str) -> str:
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"unknown architecture {architecture!r}; expected one of {sorted(ARCHITECTURES)}"
            )
        return architecture

    @field_validator("device")
    @classmethod
    def _known_device(cls, device: str) -> str:
        if device in {"auto", "cpu", "mps", "cuda"} or device.startswith("cuda:"):
            return device
        raise ValueError(f"unknown device {device!r}; expected auto, cpu, mps, cuda or cuda:N")

    @model_validator(mode="after")
    def _task_matches_architecture(self) -> ModelEntry:
        expected = ARCHITECTURES[self.architecture]
        if expected != self.task:
            raise ValueError(f"{self.architecture} serves task {expected!r}, not {self.task!r}")
        return self


class ReconnectConfig(StrictModel):
    """Exponential backoff bounds for a dropped source."""

    initial_seconds: float = Field(default=1.0, gt=0)
    max_seconds: float = Field(default=30.0, gt=0)
    multiplier: float = Field(default=2.0, gt=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> ReconnectConfig:
        if self.max_seconds < self.initial_seconds:
            raise ValueError("reconnect.max_seconds must be >= initial_seconds")
        return self


class StreamEntry(StrictModel):
    """One capture source and the analytics attached to it."""

    source: int | str
    model: str
    processing_fps: float = Field(default=3.0, gt=0, le=60)
    queue_size: int = Field(default=2, ge=1, le=16)
    confidence: float | None = Field(default=None, gt=0.0, lt=1.0)
    classes: list[str] | None = None
    zones: dict[str, ZoneConfig] = Field(default_factory=dict)
    lines: dict[str, LineConfig] = Field(default_factory=dict)
    reconnect: ReconnectConfig = ReconnectConfig()
    occupancy_limits: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _limits_name_known_zones(self) -> StreamEntry:
        unknown = set(self.occupancy_limits) - set(self.zones)
        if unknown:
            raise ValueError(f"occupancy_limits reference unknown zones: {sorted(unknown)}")
        return self


class TrackingConfig(StrictModel):
    """ByteTrack parameters and dwell bookkeeping."""

    lost_track_seconds: float = Field(default=1.0, gt=0, le=60)
    minimum_matching_threshold: float = Field(default=0.8, gt=0, lt=1)
    track_activation_threshold: float = Field(default=0.25, gt=0, lt=1)


class StorageConfig(StrictModel):
    """Database, artifact directory and retention."""

    database: Path = Path("./data/vision.db")
    artifacts: Path = Path("./data/artifacts")
    retention_days: int = Field(default=7, ge=1, le=365)
    max_artifact_bytes: int = Field(default=8 * 1024 * 1024, ge=64 * 1024)
    cleanup_interval_seconds: float = Field(default=3600.0, gt=0)


class SecurityConfig(StrictModel):
    """Filesystem roots and outbound fetch policy."""

    filesystem_roots: list[Path] = Field(default_factory=list)
    allowed_url_hosts: list[str] = Field(default_factory=list)
    allow_private_network: bool = False
    max_download_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    download_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_pixels: int = Field(default=40_000_000, ge=10_000)


class MetricsConfig(StrictModel):
    """Live-window sizes and how often live metrics are flushed to SQLite."""

    aggregation_interval_seconds: float = Field(default=30.0, ge=5, le=3600)
    latency_samples: int = Field(default=512, ge=32, le=8192)


class EngineHttpConfig(StrictModel):
    """Local HTTP bind.

    Loopback only; there is no authentication layer.
    """

    host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class DebugConfig(StrictModel):
    """Browser preview.

    Off unless explicitly enabled.
    """

    preview: bool = False
    preview_fps: float = Field(default=5.0, gt=0, le=30)
    jpeg_quality: int = Field(default=70, ge=20, le=95)


class EngineConfig(StrictModel):
    """The whole validated configuration."""

    engine: EngineHttpConfig = EngineHttpConfig()
    models: dict[str, ModelEntry]
    streams: dict[str, StreamEntry] = Field(default_factory=dict)
    tracking: TrackingConfig = TrackingConfig()
    storage: StorageConfig = StorageConfig()
    security: SecurityConfig = SecurityConfig()
    metrics: MetricsConfig = MetricsConfig()
    debug: DebugConfig = DebugConfig()

    @model_validator(mode="after")
    def _streams_reference_known_models(self) -> EngineConfig:
        for stream_id, stream in self.streams.items():
            if stream.model not in self.models:
                raise ValueError(f"stream {stream_id!r} references unknown model {stream.model!r}")
            entry = self.models[stream.model]
            if stream.classes:
                _validate_class_filter(stream_id, stream.classes, entry)
        return self

    def model_for_stream(self, stream_id: str) -> ModelEntry:
        """The model entry backing a stream."""
        return self.models[self.streams[stream_id].model]


def _validate_class_filter(stream_id: str, classes: list[str], entry: ModelEntry) -> None:
    """Class filters are checked against COCO names only for pretrained models."""
    if entry.checkpoint is not None:
        return  # custom checkpoints define their own label set, unknown until load
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES  # heavy import, only when filtering

    unknown = sorted(set(classes) - set(COCO_CLASS_NAMES))
    if unknown:
        raise ValueError(f"stream {stream_id!r} filters unknown COCO classes: {unknown}")


def load_config(path: str | Path) -> EngineConfig:
    """Read and validate a YAML config file.

    Raises:
        VisionError: INVALID_ARGUMENT when the file is missing, unparsable or fails validation.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise VisionError(ErrorCode.INVALID_ARGUMENT, "Config file not found.", {"path": str(config_path)})
    try:
        raw: Any = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Config is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise VisionError(ErrorCode.INVALID_ARGUMENT, "Config root must be a mapping.")
    try:
        return EngineConfig.model_validate(raw)
    except ValueError as exc:
        raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Config validation failed: {exc}") from exc
