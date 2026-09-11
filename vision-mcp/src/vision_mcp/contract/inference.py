# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Single-image inference and model description payloads."""

from __future__ import annotations

from vision_mcp.contract.base import ArtifactRef, BoundingBox, Envelope, ImageSize, Payload


class Detection(Payload):
    """One detected object.

    `track_id` is present only for tracked stream detections.
    """

    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    track_id: int | None = None
    mask_area_px: int | None = None
    keypoints: list[list[float]] | None = None


class DetectionResult(Envelope):
    """Result of a single-image inference call."""

    model: str
    task: str
    source: str
    image: ImageSize
    detections: list[Detection]
    current_objects: int
    inference_ms: float
    artifact: ArtifactRef | None = None


class CountsByClass(Envelope):
    """Detections grouped by class for one image."""

    model: str
    source: str
    counts: dict[str, int]
    total: int


class CompareResult(Envelope):
    """Per-class delta between two images scored by the same model."""

    model: str
    left: CountsByClass
    right: CountsByClass
    delta: dict[str, int]


class CropResult(Envelope):
    """Artifact references for crops cut from one image."""

    model: str
    source: str
    crops: list[ArtifactRef]


class ModelSummary(Payload):
    """Configured model, whether or not it is loaded."""

    name: str
    architecture: str
    task: str
    device: str
    confidence: float
    checkpoint: str | None
    loaded: bool


class ModelList(Envelope):
    """All configured models."""

    models: list[ModelSummary]


class ModelInfo(Envelope):
    """Static description of one model, including its label set when loaded."""

    model: ModelSummary
    resolution: int | None
    class_names: list[str] | None
    class_count: int | None


class ModelStatus(Envelope):
    """Runtime state of one model.

    `loaded_at` proves weights were not reloaded.
    """

    name: str
    loaded: bool
    device: str | None
    loaded_at: str | None
    load_seconds: float | None
    inference_count: int
    last_inference_at: str | None
    mean_latency_ms: float | None
    queue_depth: int
