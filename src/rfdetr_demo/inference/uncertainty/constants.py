# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Default constants for keypoint uncertainty visualization."""

from __future__ import annotations

DEFAULT_UNCERTAINTY_MAX_AXIS_PX: float = 36.0
DEFAULT_UNCERTAINTY_SIGMA: float = 1.5
DEFAULT_HEATMAP_OPACITY: float = 0.38
DEFAULT_HEATMAP_DECAY: float = 3.0

COCO17_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
