# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared types and constants for video demo inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

TaskName = Literal["detect", "segment", "keypoint"]
ModelSize = Literal["nano", "small", "medium", "large"]
KeypointUncertaintyStyle = Literal[
    "none",
    "ellipse",
    "halo",
    "heatmap",
    "magnitude",
    "outline",
    "cross",
    "filled",
]

UNCERTAINTY_STYLE_LABELS: dict[str, str] = {
    "heatmap": "ヒートマップ（関節色）",
    "magnitude": "ヒートマップ（不確実性）",
    "halo": "ハロー楕円",
    "ellipse": "楕円塗り",
    "outline": "楕円輪郭",
    "cross": "十字線",
    "filled": "単色塗り",
}

COCO_PERSON_CLASS_ID = 1

ProgressCallback = Callable[[int, int, dict[str, int]], None]
PreviewCallback = Callable[[np.ndarray, int, int], None]


class VideoProcessingCancelledError(Exception):
    """Raised when the user cancels an in-progress video export."""
