# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video inference pipeline."""

from rfdetr_demo.cli.run_video import main, parse_args
from rfdetr_demo.inference.overlays.detection import render_tune_cache_sequence
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.runner import run_demo
from rfdetr_demo.inference.types import (
    KeypointUncertaintyStyle,
    ModelSize,
    TaskName,
    VideoProcessingCancelledError,
)
from rfdetr_demo.paths import default_output_path, resolve_default_source

__all__ = [
    "KeypointOverlaySettings",
    "KeypointUncertaintyStyle",
    "ModelSize",
    "TaskName",
    "VideoProcessingCancelledError",
    "default_output_path",
    "main",
    "parse_args",
    "render_keypoint_overlay",
    "render_tune_cache_sequence",
    "resolve_default_source",
    "run_demo",
]
