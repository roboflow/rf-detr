# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""M1-M6 MCP tool and resource declarations."""

from __future__ import annotations

from typing import Any

from mcp import types

_STRING = {"type": "string"}
_STREAM = {"stream_id": _STRING}
_MODEL = {"model": _STRING}
_PERIOD = {
    "time_window": {"type": "string", "enum": ["5m", "15m", "1h", "6h", "24h", "7d"], "default": "15m"},
    "interval": {"type": "string", "enum": ["1s", "10s", "1m", "5m", "1h", "1d"], "default": "1m"},
}
_IMAGE = {
    **_MODEL,
    "source": _STRING,
    "confidence": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
    "classes": {"type": "array", "items": _STRING},
    "annotate": {"type": "boolean", "default": False},
}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Build a strict JSON object schema."""
    result: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        result["required"] = required
    return result


def tool_definitions() -> list[types.Tool]:
    """Every Phase 1 tool, excluding directory jobs and advanced health explanation."""
    definitions: list[tuple[str, str, dict[str, Any], list[str] | None]] = [
        ("list_models", "List configured RF-DETR models.", {}, None),
        ("get_model_info", "Get a configured model's architecture, task, and labels.", _MODEL, ["model"]),
        (
            "detect_objects",
            "Detect objects in one image. current_objects means this image only.",
            _IMAGE,
            ["model", "source"],
        ),
        (
            "segment_instances",
            "Segment instances in one image with a segmentation model.",
            _IMAGE,
            ["model", "source"],
        ),
        (
            "detect_keypoints",
            "Detect keypoints in one image with a keypoint model.",
            _IMAGE,
            ["model", "source"],
        ),
        (
            "count_objects",
            "Count detections by class in one image; these are current_objects.",
            _IMAGE,
            ["model", "source"],
        ),
        ("find_objects", "Find selected classes in one image.", _IMAGE, ["model", "source"]),
        (
            "crop_detections",
            "Store generated crop artifacts for detections in one image.",
            _IMAGE,
            ["model", "source"],
        ),
        (
            "compare_detections",
            "Compare per-class detections between two images.",
            {**_MODEL, "left": _STRING, "right": _STRING},
            ["model", "left", "right"],
        ),
        ("get_system_status", "Get engine, database, model, and stream health.", {}, None),
        ("get_stream_status", "Get detailed live stream state.", _STREAM, ["stream_id"]),
        ("list_active_streams", "List configured streams and whether they are running.", {}, None),
        ("get_model_status", "Get model load timestamp, device, and inference counters.", _MODEL, ["model"]),
        ("get_worker_status", "Get capture, inference, database, and maintenance worker state.", {}, None),
        (
            "get_current_counts",
            "Get current_objects: detections in the latest processed frame only.",
            _STREAM,
            ["stream_id"],
        ),
        (
            "get_counts_by_class",
            "Get frame_detections summed across frames, never unique objects.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_unique_object_count",
            "Get unique_objects: distinct multi-frame tracking instances in a period.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_detection_rate",
            "Get frame detections per second in time buckets.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_confidence_distribution",
            "Get the detection confidence histogram for a period.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_active_tracks",
            "Get active_tracks visible in the latest processed frame only.",
            _STREAM,
            ["stream_id"],
        ),
        ("get_zone_occupancy", "Get current tracked occupancy for configured zones.", _STREAM, ["stream_id"]),
        (
            "get_entry_exit_counts",
            "Get business events for zone entries and exits.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_dwell_times",
            "Get dwell-time samples and percentiles for completed zone visits.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_line_crossing_events",
            "Get directional line-crossing business events.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_inference_latency",
            "Get inference latency statistics from aggregated samples.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_processing_throughput",
            "Get processed frames per second, not object counts.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        (
            "get_frame_drop_rate",
            "Get bounded-queue frame drop rate with safe zero handling.",
            {**_STREAM, **_PERIOD},
            ["stream_id"],
        ),
        ("get_gpu_metrics", "Get NVIDIA GPU metrics or a structured unsupported result.", {}, None),
        (
            "get_queue_metrics",
            "Get queue depth, capacity, high-water mark, and dropped frames.",
            _STREAM,
            ["stream_id"],
        ),
        (
            "get_latest_annotated_frame",
            "Create an annotated artifact from the latest in-memory frame.",
            _STREAM,
            ["stream_id"],
        ),
        (
            "get_event_snapshot",
            "Look up the snapshot artifact attached to an event.",
            {"event_id": _STRING},
            ["event_id"],
        ),
        (
            "get_recent_detection_events",
            "Get recent entries, exits, crossings, violations, and stream events.",
            {"stream_id": _STRING, **_PERIOD, "limit": {"type": "integer", "minimum": 1, "maximum": 500}},
            None,
        ),
        (
            "get_recent_errors",
            "Get recent recursively redacted engine errors.",
            {**_PERIOD, "limit": {"type": "integer", "minimum": 1, "maximum": 500}},
            None,
        ),
    ]
    return [
        types.Tool(name=name, description=description, input_schema=_schema(properties, required))
        for name, description, properties, required in definitions
    ]


def resources() -> list[types.Resource]:
    """Fixed non-job resources."""
    return [
        types.Resource(name="system-status", uri="vision://system/status", mime_type="application/json"),
        types.Resource(name="streams", uri="vision://streams", mime_type="application/json"),
        types.Resource(name="models", uri="vision://models", mime_type="application/json"),
    ]


def resource_templates() -> list[types.ResourceTemplate]:
    """Parameterized non-job resources."""
    return [
        types.ResourceTemplate(
            name="stream-status",
            uri_template="vision://streams/{stream_id}/status",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            name="stream-counts",
            uri_template="vision://streams/{stream_id}/counts",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            name="stream-events",
            uri_template="vision://streams/{stream_id}/events",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            name="model-status",
            uri_template="vision://models/{model_name}/status",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            name="artifact",
            uri_template="vision://artifacts/{artifact_id}",
            mime_type="application/octet-stream",
        ),
    ]
