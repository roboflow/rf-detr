# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Focused predict() contract tests for keypoint and non-keypoint outputs."""

import numpy as np
import PIL.Image
import supervision as sv

from .helpers import _DummyModel, _DummyRFDETR


def test_predict_returns_supervision_keypoints() -> None:
    """Keypoint model predictions return ``sv.KeyPoints`` with detection details."""
    image = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
    model = _DummyRFDETR()
    model.model = _DummyModel(labels=[0, 1], include_keypoints=True)

    key_points = model.predict(image)

    assert isinstance(key_points, sv.KeyPoints)
    assert key_points.xy.shape == (2, 17, 2)
    assert key_points.keypoint_confidence.shape == (2, 17)
    assert key_points.data["xyxy"].shape == (2, 4)
    assert key_points.detection_confidence.shape == (2,)
    assert np.isfinite(key_points.xy).all()
    assert np.isfinite(key_points.keypoint_confidence).all()
    assert "keypoint_precision_cholesky" in key_points.data
    keypoint_precision = key_points.data["keypoint_precision_cholesky"]
    assert isinstance(keypoint_precision, np.ndarray)
    assert keypoint_precision.shape == (2, 17, 3)
    assert np.isfinite(keypoint_precision).all()
    assert "source_image" in key_points.data
    assert len(key_points.data["source_image"]) == 2


def test_predict_default_detection_without_keypoints_unchanged() -> None:
    """Default detection prediction keeps legacy output structure."""
    image = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
    model = _DummyRFDETR()

    detections = model.predict(image)

    assert "keypoints" not in detections.data
    assert not hasattr(detections, "keypoints")
    assert "class_name" in detections.data
    assert "source_shape" in detections.data
    assert detections.data["source_shape"].shape[1] == 2
