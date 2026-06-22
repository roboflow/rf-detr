# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for IoU person associator."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.person_associator import PersonAssociator


def _key_points_at(x_offset: float) -> sv.KeyPoints:
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    xy[0, 0] = (100.0 + x_offset, 200.0)
    xy[0, 11] = (90.0 + x_offset, 250.0)
    xy[0, 12] = (110.0 + x_offset, 250.0)
    return sv.KeyPoints(
        xy=xy,
        visible=np.ones((1, 17), dtype=bool),
        keypoint_confidence=np.full((1, 17), 0.9, dtype=np.float32),
        detection_confidence=np.array([0.9], dtype=np.float32),
    )


def test_associator_keeps_track_on_small_motion() -> None:
    associator = PersonAssociator()
    first = associator.assign(_key_points_at(0.0))
    second = associator.assign(_key_points_at(5.0))
    assert first[0] == 0
    assert second[0] == 0


def test_associator_creates_new_track_for_distant_detection() -> None:
    associator = PersonAssociator()
    associator.assign(_key_points_at(0.0))
    two_person = sv.KeyPoints(
        xy=np.stack([_key_points_at(0.0).xy[0], _key_points_at(400.0).xy[0]]),
        visible=np.ones((2, 17), dtype=bool),
        keypoint_confidence=np.full((2, 17), 0.9, dtype=np.float32),
        detection_confidence=np.array([0.9, 0.9], dtype=np.float32),
    )
    ids = associator.assign(two_person)
    assert ids[0] == 0
    assert ids[1] == 1
