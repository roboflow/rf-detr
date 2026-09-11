# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The single definition of every payload crossing the engine HTTP boundary.

The engine returns these models and the MCP server parses the same classes; neither side defines its own copy
of a shape. Definitions live in `vision_mcp.contract.*` only because one module would exceed the project's
size limit — this module is the import surface.
"""

from __future__ import annotations

from vision_mcp.contract.analytics import (
    ActiveTracks,
    DwellResult,
    DwellStats,
    EntryExitCounts,
    EntryExitResult,
    ErrorList,
    ErrorRecord,
    EventList,
    EventRecord,
    EventType,
    LineCrossing,
    LineCrossingResult,
    TrackInfo,
    ZoneOccupancy,
    ZoneOccupancyResult,
)
from vision_mcp.contract.base import (
    SCHEMA_VERSION,
    ArtifactRef,
    ArtifactResult,
    BoundingBox,
    Bucket,
    Envelope,
    ErrorResponse,
    Health,
    HistoricalQuery,
    ImageSize,
    Payload,
    StreamState,
    TimeWindow,
)
from vision_mcp.contract.inference import (
    CompareResult,
    CountsByClass,
    CropResult,
    Detection,
    DetectionResult,
    ModelInfo,
    ModelList,
    ModelStatus,
    ModelSummary,
)
from vision_mcp.contract.metrics import (
    ClassCounts,
    ConfidenceDistribution,
    CurrentCounts,
    DetectionRate,
    FrameDropRate,
    GpuDevice,
    GpuMetrics,
    LatencyStats,
    QueueMetrics,
    ThroughputStats,
    UniqueObjectCount,
)
from vision_mcp.contract.status import (
    LiveSnapshot,
    StreamList,
    StreamStatus,
    StreamSummary,
    SystemStatus,
    WorkerInfo,
    WorkerList,
)

__all__ = [
    "SCHEMA_VERSION",
    "ActiveTracks",
    "ArtifactRef",
    "ArtifactResult",
    "BoundingBox",
    "Bucket",
    "ClassCounts",
    "CompareResult",
    "ConfidenceDistribution",
    "CountsByClass",
    "CropResult",
    "CurrentCounts",
    "Detection",
    "DetectionRate",
    "DetectionResult",
    "DwellResult",
    "DwellStats",
    "EntryExitCounts",
    "EntryExitResult",
    "Envelope",
    "ErrorList",
    "ErrorRecord",
    "ErrorResponse",
    "EventList",
    "EventRecord",
    "EventType",
    "FrameDropRate",
    "GpuDevice",
    "GpuMetrics",
    "Health",
    "HistoricalQuery",
    "ImageSize",
    "LatencyStats",
    "LineCrossing",
    "LineCrossingResult",
    "LiveSnapshot",
    "ModelInfo",
    "ModelList",
    "ModelStatus",
    "ModelSummary",
    "Payload",
    "QueueMetrics",
    "StreamList",
    "StreamState",
    "StreamStatus",
    "StreamSummary",
    "SystemStatus",
    "ThroughputStats",
    "TimeWindow",
    "TrackInfo",
    "UniqueObjectCount",
    "WorkerInfo",
    "WorkerList",
    "ZoneOccupancy",
    "ZoneOccupancyResult",
]
