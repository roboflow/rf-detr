# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Focused predict() contract tests for keypoint and non-keypoint outputs."""

import numpy as np
import PIL.Image

from tests.models.test_predict import _DummyModel, _DummyRFDETR


def test_predict_returns_keypoints_in_detections_data() -> None:
    """Keypoint model predictions expose keypoints in ``detections.data``."""
    image = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
    model = _DummyRFDETR()
    model.model = _DummyModel(labels=[0, 1], include_keypoints=True)

    detections = model.predict(image)

    assert "keypoints" in detections.data
    keypoints = detections.data["keypoints"]
    assert isinstance(keypoints, np.ndarray)
    assert keypoints.shape == (2, 17, 3)
    assert np.isfinite(keypoints).all()
    assert "keypoint_precision_cholesky" in detections.data
    keypoint_precision = detections.data["keypoint_precision_cholesky"]
    assert isinstance(keypoint_precision, np.ndarray)
    assert keypoint_precision.shape == (2, 17, 3)
    assert np.isfinite(keypoint_precision).all()


def test_predict_default_detection_without_keypoints_unchanged() -> None:
    """Default detection prediction keeps legacy output structure."""
    image = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
    model = _DummyRFDETR()

    detections = model.predict(image)

    assert "keypoints" not in detections.data
    assert "class_name" in detections.data
    assert "source_shape" in detections.data
    assert detections.data["source_shape"].shape[1] == 2
