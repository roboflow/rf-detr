# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for detection-based tracking (Stage 1 counter)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.inference.callbacks import make_detection_track_callback
from rfdetr_demo.tracking.keypoints_ops import detections_to_key_points, track_ids_from_key_points
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings


def _person_detections(boxes: list[tuple[float, float, float, float]]) -> sv.Detections:
    xyxy = np.asarray(boxes, dtype=np.float32).reshape(len(boxes), 4)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full((len(boxes),), 0.9, dtype=np.float32),
        class_id=np.ones((len(boxes),), dtype=np.int64),  # COCO_PERSON_CLASS_ID = 1
    )


def test_detections_to_key_points_carries_boxes() -> None:
    detections = _person_detections([(10.0, 20.0, 30.0, 60.0), (100.0, 100.0, 140.0, 200.0)])
    key_points = detections_to_key_points(detections)

    assert len(key_points) == 2
    assert key_points.data["xyxy"].shape == (2, 4)
    np.testing.assert_allclose(key_points.data["xyxy"][1], [100.0, 100.0, 140.0, 200.0])
    # Joints are absent so appearance ReID falls back to the box.
    assert key_points.visible is not None
    assert not key_points.visible.any()


class _FakeDetectionModel:
    def __init__(self, per_frame: list[sv.Detections]) -> None:
        self._per_frame = per_frame
        self._index = 0

    def predict(self, _frame_rgb: object, **_kwargs: object) -> sv.Detections:
        detections = self._per_frame[self._index]
        self._index += 1
        return detections


def test_detection_track_callback_assigns_stable_ids() -> None:
    frame0 = _person_detections([(100.0, 100.0, 160.0, 300.0), (400.0, 100.0, 460.0, 300.0)])
    frame1 = _person_detections([(110.0, 100.0, 170.0, 300.0), (410.0, 100.0, 470.0, 300.0)])
    model = _FakeDetectionModel([frame0, frame1])
    pipeline = PersonTrackPipeline(settings=PersonTrackSettings(enabled=True), frame_width=640)
    stats: dict[str, int] = {"processed_frames": 0, "total_detections": 0}

    captured: list[sv.KeyPoints] = []
    original = PersonTrackPipeline.apply

    def _spy(self: PersonTrackPipeline, key_points: sv.KeyPoints, frame_index: int, frame: object = None):
        result = original(self, key_points, frame_index, frame)
        captured.append(result.key_points)
        return result

    PersonTrackPipeline.apply = _spy  # type: ignore[method-assign]
    try:
        callback = make_detection_track_callback(model, 0.5, True, stats, pipeline)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        callback(blank, 0)
        callback(blank, 1)
    finally:
        PersonTrackPipeline.apply = original  # type: ignore[method-assign]

    ids_frame0 = sorted(t for t in track_ids_from_key_points(captured[0]) if t is not None)
    ids_frame1 = sorted(t for t in track_ids_from_key_points(captured[1]) if t is not None)
    assert ids_frame0 == [0, 1]
    # The two people barely moved, so ids persist across frames.
    assert ids_frame1 == [0, 1]
    assert stats["frame_active_tracks"] == 2
