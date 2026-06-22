# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Downscale BGR frames for live preview (no GUI dependency)."""

from __future__ import annotations

import cv2
import numpy as np


def resize_bgr_for_preview(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a BGR frame for lightweight GUI preview."""
    if max_width <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    if width <= max_width:
        return frame_bgr
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def fit_bgr_for_preview(
    frame_bgr: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Scale a BGR frame to fit inside ``max_width`` × ``max_height`` (aspect preserved)."""
    if max_width <= 0 or max_height <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return frame_bgr
    scale = min(max_width / float(width), max_height / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
