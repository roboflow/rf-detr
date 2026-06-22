# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Overlay rendering for video demo."""

from rfdetr_demo.inference.overlays.detection import (
    render_detection_overlay,
    render_segment_overlay,
    render_tune_cache_entry,
    render_tune_cache_sequence,
)
from rfdetr_demo.inference.overlays.keypoint import (
    KeypointOverlaySettings,
    build_keypoint_uncertainty_annotator,
    render_keypoint_overlay,
)

__all__ = [
    "KeypointOverlaySettings",
    "build_keypoint_uncertainty_annotator",
    "render_detection_overlay",
    "render_keypoint_overlay",
    "render_segment_overlay",
    "render_tune_cache_entry",
    "render_tune_cache_sequence",
]
