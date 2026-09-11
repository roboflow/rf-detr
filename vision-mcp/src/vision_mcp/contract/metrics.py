# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection-count and performance metric payloads.

The five metric definitions are kept in separate models on purpose: frame detections and unique objects must
never be interchangeable.
"""

from __future__ import annotations

from vision_mcp.contract.base import Bucket, Envelope, Payload, TimeWindow


class CurrentCounts(Envelope):
    """Detections in the latest processed frame only."""

    stream_id: str
    current_objects: int
    counts_by_class: dict[str, int]
    frame_at: str | None


class ClassCounts(Envelope):
    """Frame detections summed over a window, per class.

    Not unique objects.
    """

    stream_id: str
    window: TimeWindow
    frame_detections: int
    counts_by_class: dict[str, int]


class UniqueObjectCount(Envelope):
    """Distinct confirmed tracking IDs seen in a window.

    Always <= frame_detections.
    """

    stream_id: str
    window: TimeWindow
    unique_objects: int
    by_class: dict[str, int]


class DetectionRate(Envelope):
    """Detections per second per bucket."""

    stream_id: str
    window: TimeWindow
    buckets: list[Bucket]
    mean_per_second: float


class ConfidenceDistribution(Envelope):
    """Histogram of detection confidence over a window."""

    stream_id: str
    window: TimeWindow
    bin_edges: list[float]
    counts: list[int]
    samples: int
    mean: float | None


class LatencyStats(Envelope):
    """Inference latency percentiles in milliseconds."""

    scope: str
    window: TimeWindow
    samples: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None


class ThroughputStats(Envelope):
    """Processed frames per second against the configured target."""

    stream_id: str
    window: TimeWindow
    processed_frames: int
    processed_fps: float
    target_fps: float
    buckets: list[Bucket]


class FrameDropRate(Envelope):
    """Fraction of captured frames dropped by the bounded queue."""

    stream_id: str
    window: TimeWindow
    captured_frames: int
    processed_frames: int
    dropped_frames: int
    drop_rate: float


class QueueMetrics(Envelope):
    """Bounded-queue state per stream."""

    stream_id: str
    depth: int
    capacity: int
    high_water: int
    dropped_frames: int
    inference_queue_depth: int


class GpuDevice(Payload):
    """One NVIDIA device, when pynvml is present."""

    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilisation_percent: float
    temperature_c: float | None


class GpuMetrics(Envelope):
    """GPU metrics, or a structured explanation of why they are unavailable."""

    available: bool
    reason: str | None
    device: str
    devices: list[GpuDevice]
