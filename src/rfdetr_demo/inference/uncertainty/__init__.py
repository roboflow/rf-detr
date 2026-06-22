# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint uncertainty visualization subpackage."""

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
