# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke test for editable supervision keypoint API visibility."""

import numpy as np
import supervision as sv


def test_editable_supervision_keypoint_api_visible_to_rfdetr() -> None:
    """RF-DETR runtime sees `KeyPoints.from_rfdetr` from editable supervision install."""
    assert hasattr(sv.KeyPoints, "from_rfdetr")

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        data={"keypoints": np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]], dtype=np.float32)},
    )
    key_points = sv.KeyPoints.from_rfdetr(detections)
    assert key_points.xy.shape == (1, 2, 2)
