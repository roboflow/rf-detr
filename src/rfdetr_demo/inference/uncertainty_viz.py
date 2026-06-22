# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-joint heatmap rendering for RF-DETR keypoint uncertainty ellipses.

Backward-compatible facade — prefer ``rfdetr_demo.inference.uncertainty`` for new code.
"""

from __future__ import annotations

from rfdetr_demo.inference.uncertainty.constants import (
    COCO17_KEYPOINT_NAMES,
    DEFAULT_HEATMAP_DECAY,
    DEFAULT_HEATMAP_OPACITY,
    DEFAULT_UNCERTAINTY_MAX_AXIS_PX,
    DEFAULT_UNCERTAINTY_SIGMA,
)
from rfdetr_demo.inference.uncertainty.geometry import (
    clamp_ellipse_axes,
    covariance_trace,
    decompose_covariance,
    joint_index_to_bgr,
    resolve_max_ellipse_axis,
)
from rfdetr_demo.inference.uncertainty.heatmap import (
    KeypointJointHeatmapAnnotator,
    KeypointMagnitudeHeatmapAnnotator,
    trace_to_bgr,
)
from rfdetr_demo.inference.uncertainty.styles import (
    KeypointCrossAnnotator,
    KeypointFilledEllipseAnnotator,
    KeypointOutlineAnnotator,
    annotate_joint_colored_vertices,
)

__all__ = [
    "COCO17_KEYPOINT_NAMES",
    "DEFAULT_HEATMAP_DECAY",
    "DEFAULT_HEATMAP_OPACITY",
    "DEFAULT_UNCERTAINTY_MAX_AXIS_PX",
    "DEFAULT_UNCERTAINTY_SIGMA",
    "KeypointCrossAnnotator",
    "KeypointFilledEllipseAnnotator",
    "KeypointJointHeatmapAnnotator",
    "KeypointMagnitudeHeatmapAnnotator",
    "KeypointOutlineAnnotator",
    "annotate_joint_colored_vertices",
    "clamp_ellipse_axes",
    "covariance_trace",
    "decompose_covariance",
    "joint_index_to_bgr",
    "resolve_max_ellipse_axis",
    "trace_to_bgr",
]
