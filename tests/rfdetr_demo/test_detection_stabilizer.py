# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for detection stabilizer (NMS + track hold)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.detection_stabilizer import (
    DetectionStabilizer,
    DetectionStabilizerSettings,
    merge_key_points,
    nms_detection_indices,
    subset_key_points,
)


def _box_key_points(
    *,
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
) -> sv.KeyPoints:
    num = len(boxes)
    xy = np.zeros((num, 17, 2), dtype=np.float32)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        xy[index, 0] = ((x1 + x2) / 2, (y1 + y2) / 2)
        xy[index, 11] = (x1, y2)
        xy[index, 12] = (x2, y2)
    xyxy = np.asarray(boxes, dtype=np.float32)
    return sv.KeyPoints(
        xy=xy,
        visible=np.ones((num, 17), dtype=bool),
        keypoint_confidence=np.full((num, 17), 0.9, dtype=np.float32),
        detection_confidence=np.asarray(confidences, dtype=np.float32),
        data={"xyxy": xyxy},
    )


def test_nms_suppresses_overlapping_duplicate() -> None:
    key_points = _box_key_points(
        boxes=[
            (100.0, 100.0, 200.0, 300.0),
            (105.0, 105.0, 205.0, 305.0),
            (400.0, 100.0, 500.0, 300.0),
        ],
        confidences=[0.62, 0.88, 0.80],
    )
    keep = nms_detection_indices(key_points, iou_threshold=0.50)
    filtered = subset_key_points(key_points, keep)
    assert len(filtered) == 2
    assert 1 in keep


def test_stabilizer_holds_missing_detection() -> None:
    stabilizer = DetectionStabilizer(
        settings=DetectionStabilizerSettings(
            nms_iou_threshold=0.50,
            max_missed=2,
            match_iou_threshold=0.15,
        ),
    )
    frame0 = _box_key_points(
        boxes=[(100.0, 100.0, 200.0, 300.0), (400.0, 100.0, 500.0, 300.0)],
        confidences=[0.90, 0.90],
    )
    frame1 = _box_key_points(
        boxes=[(400.0, 100.0, 500.0, 300.0)],
        confidences=[0.90],
    )

    result0 = stabilizer.apply(frame0, 0)
    result1 = stabilizer.apply(frame1, 1)

    assert result0.stats.active_track_count == 2
    assert result1.stats.raw_count == 1
    assert result1.stats.active_track_count == 2
    assert result1.stats.ghost_count == 1
    flags = result1.key_points.data.get("track_is_ghost")
    assert flags is not None
    assert flags.tolist().count(True) == 1


def test_merge_key_points_concatenates_rows() -> None:
    left = _box_key_points(boxes=[(0.0, 0.0, 10.0, 10.0)], confidences=[0.9])
    right = _box_key_points(boxes=[(20.0, 0.0, 30.0, 10.0)], confidences=[0.8])
    merged = merge_key_points([left, right])
    assert len(merged) == 2
    assert merged.data["xyxy"].shape == (2, 4)
