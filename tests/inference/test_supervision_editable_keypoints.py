# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke test for RF-DETR keypoints carried by Supervision KeyPoints."""

import numpy as np
import supervision as sv


def test_rfdetr_keypoints_include_detection_details() -> None:
    """RF-DETR-style keypoints preserve detection boxes and scores."""
    key_points = sv.KeyPoints(
        xy=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        keypoint_confidence=np.array([[0.9, 0.8]], dtype=np.float32),
        detection_confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([1], dtype=int),
        data={"xyxy": np.array([[0, 0, 10, 10]], dtype=np.float32)},
    )

    assert key_points.xy.shape == (1, 2, 2)
    np.testing.assert_array_equal(key_points.data["xyxy"], np.array([[0, 0, 10, 10]], dtype=np.float32))
    np.testing.assert_array_equal(key_points.detection_confidence, np.array([0.95], dtype=np.float32))
