# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Heuristic camera-viewpoint (overhead vs eye-level) estimation for FlowCount.

Distinguishes overhead/high-mounted cameras from eye-level cameras using two
signals derived from a sample of person bounding boxes, per the heuristic
outlined in ``FlowCount/README.md``:

- **box aspect ratio**: bird's-eye views show near-square person silhouettes;
  eye-level views show tall/narrow silhouettes.
- **size vs. vertical-position correlation**: in eye-level (perspective) shots,
  people lower in the frame (closer to the camera) have larger boxes; in
  overhead shots, box size is roughly independent of frame position.

The resulting estimate maps to one of the FlowCount presets so a caller can
auto-select ``overhead`` or ``eye-level`` instead of guessing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Sequence

CameraViewpoint = Literal["overhead", "eye_level"]

_DEFAULT_ASPECT_RATIO_THRESHOLD = 1.4
_DEFAULT_CORRELATION_THRESHOLD = 0.35
_MIN_SAMPLE_COUNT = 2


@dataclass(frozen=True)
class ViewpointEstimate:
    """Result of :func:`estimate_camera_viewpoint`.

    Attributes:
        viewpoint: Estimated camera viewpoint.
        confidence: Agreement between the aspect-ratio and correlation signals,
            in ``{0.0, 0.5, 1.0}`` (0.0 when there were too few boxes to judge).
        median_aspect_ratio: Median height/width ratio of the sampled boxes.
        size_position_correlation: Pearson correlation between box area and
            normalized vertical center (0.0 when it could not be computed).
        sample_count: Number of boxes used (after discarding degenerate ones).
    """

    viewpoint: CameraViewpoint
    confidence: float
    median_aspect_ratio: float
    size_position_correlation: float
    sample_count: int


def estimate_camera_viewpoint(
    boxes: Sequence[tuple[float, float, float, float]],
    frame_height: float,
    *,
    aspect_ratio_threshold: float = _DEFAULT_ASPECT_RATIO_THRESHOLD,
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> ViewpointEstimate:
    """Estimate whether person boxes come from an overhead or eye-level camera.

    Args:
        boxes: Person bounding boxes as ``(x1, y1, x2, y2)`` tuples, ideally
            sampled across several frames of a clip.
        frame_height: Height of the video frame in pixels, used to normalize
            box vertical position.
        aspect_ratio_threshold: Median height/width ratio above which boxes
            are considered "tall" (an eye-level signal).
        correlation_threshold: Size-vs-position correlation above which boxes
            are considered "perspective-scaled" (an eye-level signal).

    Returns:
        A :class:`ViewpointEstimate` with the classified viewpoint and the
        underlying signal values.
    """
    aspect_ratios: list[float] = []
    positions: list[float] = []
    areas: list[float] = []

    for x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        aspect_ratios.append(height / width)
        areas.append(width * height)
        positions.append(((y1 + y2) / 2) / frame_height if frame_height > 0 else 0.0)

    sample_count = len(aspect_ratios)
    if sample_count < _MIN_SAMPLE_COUNT:
        return ViewpointEstimate(
            viewpoint="overhead",
            confidence=0.0,
            median_aspect_ratio=aspect_ratios[0] if aspect_ratios else 0.0,
            size_position_correlation=0.0,
            sample_count=sample_count,
        )

    median_aspect_ratio = statistics.median(aspect_ratios)
    try:
        correlation = statistics.correlation(positions, areas)
    except statistics.StatisticsError:
        correlation = 0.0

    eye_level_votes = 0
    if median_aspect_ratio > aspect_ratio_threshold:
        eye_level_votes += 1
    if correlation > correlation_threshold:
        eye_level_votes += 1

    eye_level_score = eye_level_votes / 2
    viewpoint: CameraViewpoint = "eye_level" if eye_level_score >= 0.5 else "overhead"
    confidence = eye_level_score if viewpoint == "eye_level" else 1.0 - eye_level_score

    return ViewpointEstimate(
        viewpoint=viewpoint,
        confidence=confidence,
        median_aspect_ratio=median_aspect_ratio,
        size_position_correlation=correlation,
        sample_count=sample_count,
    )


def preset_for_viewpoint(estimate: ViewpointEstimate) -> Literal["overhead", "eye-level"]:
    """Map a :class:`ViewpointEstimate` to a FlowCount preset name.

    Args:
        estimate: Result of :func:`estimate_camera_viewpoint`.

    Returns:
        ``"overhead"`` or ``"eye-level"``, matching the preset names documented
        in ``FlowCount/README.md``.
    """
    return "overhead" if estimate.viewpoint == "overhead" else "eye-level"
