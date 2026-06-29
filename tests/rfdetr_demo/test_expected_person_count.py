# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for expected person count cap/fill (P3 dance-demo stabilization)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings

# Five dancers across the frame (mn1-2 style layout).
_BASE_BOXES: list[tuple[float, float, float, float]] = [
    (80.0, 100.0, 180.0, 300.0),
    (230.0, 100.0, 330.0, 300.0),
    (380.0, 100.0, 480.0, 300.0),
    (530.0, 100.0, 630.0, 300.0),
    (680.0, 100.0, 780.0, 300.0),
]
_DUPLICATE_CENTER = (385.0, 105.0, 475.0, 295.0)

# Raw detection count pattern mimicking mn1-2 flicker (4–6 people).
_MN1_RAW_COUNTS: list[int] = [5, 6, 5, 4, 5, 6, 5, 4, 5, 5, 6, 5, 4, 5, 5, 6, 5, 4, 5, 5]


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


def _boxes_for_raw_count(raw_count: int) -> list[tuple[float, float, float, float]]:
    if raw_count == 6:
        return [* _BASE_BOXES, _DUPLICATE_CENTER]
    if raw_count == 4:
        return [_BASE_BOXES[0], _BASE_BOXES[1], _BASE_BOXES[3], _BASE_BOXES[4]]
    return list(_BASE_BOXES)


def _confidences_for_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    low_center: bool = False,
) -> list[float]:
    values: list[float] = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2.0
        if low_center and 350.0 <= cx <= 520.0:
            values.append(0.58)
        elif box == _DUPLICATE_CENTER:
            values.append(0.62)
        else:
            values.append(0.88)
    return values


def test_expected_person_count_caps_sixth_detection() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            expected_person_count=5,
            hysteresis_enabled=True,
            new_track_min_confidence=0.65,
        ),
        frame_width=1280,
    )
    boxes = _boxes_for_raw_count(6)
    result = pipeline.apply(
        _box_key_points(boxes=boxes, confidences=_confidences_for_boxes(boxes)),
        0,
    )
    assert result.stats.active_track_count == 5
    assert result.stats.raw_count == 6


def test_expected_person_count_fills_missing_center() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            max_missed=2,
            expected_person_count=5,
            fill_below_expected=True,
            fill_extra_missed=3,
            hysteresis_enabled=False,
        ),
        frame_width=1280,
    )
    pipeline.apply(
        _box_key_points(
            boxes=_boxes_for_raw_count(5),
            confidences=_confidences_for_boxes(_boxes_for_raw_count(5)),
        ),
        0,
    )
    result = pipeline.apply(
        _box_key_points(
            boxes=_boxes_for_raw_count(4),
            confidences=_confidences_for_boxes(_boxes_for_raw_count(4)),
        ),
        1,
    )
    assert result.stats.active_track_count == 5
    assert result.stats.ghost_count >= 1


def test_mn1_pattern_meets_acceptance_with_expected_count() -> None:
    """Acceptance: 5±0 on ≥80% of frames when expected_person_count=5 is set."""
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            max_missed=2,
            expected_person_count=5,
            fill_below_expected=True,
            fill_extra_missed=3,
            hysteresis_enabled=True,
            new_track_min_confidence=0.65,
        ),
        frame_width=1280,
    )
    stabilized_counts: list[int] = []
    for frame_index, raw_count in enumerate(_MN1_RAW_COUNTS):
        boxes = _boxes_for_raw_count(raw_count)
        low_center = raw_count == 4
        key_points = _box_key_points(
            boxes=boxes,
            confidences=_confidences_for_boxes(boxes, low_center=low_center),
        )
        result = pipeline.apply(key_points, frame_index)
        stabilized_counts.append(result.stats.active_track_count)

    at_five = sum(1 for count in stabilized_counts if count == 5)
    assert at_five / len(stabilized_counts) >= 0.80
    assert max(stabilized_counts) <= 5
    assert sum(1 for count in stabilized_counts if count == 6) == 0
