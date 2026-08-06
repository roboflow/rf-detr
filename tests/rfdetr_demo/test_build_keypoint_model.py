# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the keypoint model builder resolution passthrough."""

from __future__ import annotations

from unittest.mock import patch

from rfdetr_demo.inference.models import build_keypoint_model


def test_build_keypoint_model_forwards_resolution() -> None:
    with patch("rfdetr.RFDETRKeypointPreview") as ctor:
        build_keypoint_model(resolution=768)
        ctor.assert_called_once_with(resolution=768)


def test_build_keypoint_model_default_has_no_resolution() -> None:
    with patch("rfdetr.RFDETRKeypointPreview") as ctor:
        build_keypoint_model()
        ctor.assert_called_once_with()
