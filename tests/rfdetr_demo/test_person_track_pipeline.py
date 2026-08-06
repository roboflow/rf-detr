# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for unified person track pipeline."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.appearance import (
    appearance_histogram,
    appearance_roi,
    histogram_similarity,
)
from rfdetr_demo.tracking.keypoints_ops import track_ids_from_key_points
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.track_store import _match_tracks_to_detections
from rfdetr_demo.tracking.types import TRACK_IS_GHOST_KEY, PersonTrackSettings


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


def test_pipeline_attaches_track_ids() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(max_missed=2),
        frame_width=1280,
    )
    frame0 = _box_key_points(
        boxes=[(100.0, 100.0, 200.0, 300.0), (400.0, 100.0, 500.0, 300.0)],
        confidences=[0.90, 0.90],
    )
    result = pipeline.apply(frame0, 0)
    track_ids = track_ids_from_key_points(result.key_points)
    assert len(track_ids) == 2
    assert all(track_id is not None for track_id in track_ids)


def test_pipeline_hold_sets_ghost_flag() -> None:
    pipeline = PersonTrackPipeline(settings=PersonTrackSettings(max_missed=2), frame_width=1280)
    frame0 = _box_key_points(
        boxes=[(100.0, 100.0, 200.0, 300.0), (400.0, 100.0, 500.0, 300.0)],
        confidences=[0.90, 0.90],
    )
    frame1 = _box_key_points(
        boxes=[(400.0, 100.0, 500.0, 300.0)],
        confidences=[0.90],
    )
    pipeline.apply(frame0, 0)
    result1 = pipeline.apply(frame1, 1)
    flags = result1.key_points.data.get(TRACK_IS_GHOST_KEY)
    assert flags is not None
    assert flags.tolist().count(True) == 1


def test_sticky_center_extends_hold() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            max_missed=1,
            sticky_center_track=True,
            sticky_max_missed=3,
            center_x_fraction=(0.28, 0.48),
        ),
        frame_width=1280,
    )
    center_box = (300.0, 100.0, 420.0, 300.0)
    side_box = (800.0, 100.0, 900.0, 300.0)
    pipeline.apply(_box_key_points(boxes=[center_box, side_box], confidences=[0.9, 0.9]), 0)
    result1 = pipeline.apply(_box_key_points(boxes=[side_box], confidences=[0.9]), 1)
    assert result1.stats.ghost_count == 1
    result2 = pipeline.apply(_box_key_points(boxes=[side_box], confidences=[0.9]), 2)
    assert result2.stats.ghost_count == 1


def _center_x(key_points: sv.KeyPoints, detection_index: int) -> float:
    return float(key_points.xy[detection_index, 0, 0])


def test_motion_retains_track_across_fast_move() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(motion_enabled=True, motion_smoothing=0.5),
        frame_width=1280,
    )
    first = pipeline.apply(_box_key_points(boxes=[(100.0, 100.0, 200.0, 300.0)], confidences=[0.9]), 0)
    pipeline.apply(_box_key_points(boxes=[(150.0, 100.0, 250.0, 300.0)], confidences=[0.9]), 1)
    pipeline.apply(_box_key_points(boxes=[(200.0, 100.0, 300.0, 300.0)], confidences=[0.9]), 2)
    # Raw IoU with the previous box is below match_iou_threshold; only the
    # motion-predicted box overlaps the new detection.
    fast = pipeline.apply(_box_key_points(boxes=[(280.0, 100.0, 380.0, 300.0)], confidences=[0.9]), 3)

    first_id = track_ids_from_key_points(first.key_points)[0]
    fast_id = track_ids_from_key_points(fast.key_points)[0]
    assert fast_id == first_id


def test_motion_disabled_switches_track_on_fast_move() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(motion_enabled=False),
        frame_width=1280,
    )
    first = pipeline.apply(_box_key_points(boxes=[(100.0, 100.0, 200.0, 300.0)], confidences=[0.9]), 0)
    pipeline.apply(_box_key_points(boxes=[(150.0, 100.0, 250.0, 300.0)], confidences=[0.9]), 1)
    pipeline.apply(_box_key_points(boxes=[(200.0, 100.0, 300.0, 300.0)], confidences=[0.9]), 2)
    fast = pipeline.apply(_box_key_points(boxes=[(280.0, 100.0, 380.0, 300.0)], confidences=[0.9]), 3)

    first_id = track_ids_from_key_points(first.key_points)[0]
    fast_id = track_ids_from_key_points(fast.key_points)[0]
    assert fast_id != first_id


def test_motion_advances_ghost_keypoints() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(motion_enabled=True, motion_smoothing=0.5, max_missed=3),
        frame_width=1280,
    )
    pipeline.apply(_box_key_points(boxes=[(100.0, 100.0, 200.0, 300.0)], confidences=[0.9]), 0)
    pipeline.apply(_box_key_points(boxes=[(150.0, 100.0, 250.0, 300.0)], confidences=[0.9]), 1)
    last = pipeline.apply(_box_key_points(boxes=[(200.0, 100.0, 300.0, 300.0)], confidences=[0.9]), 2)
    ghost = pipeline.apply(_box_key_points(boxes=[], confidences=[]), 3)

    assert ghost.stats.ghost_count == 1
    # The held ghost should predict forward, not freeze at the last position.
    assert _center_x(ghost.key_points, 0) > _center_x(last.key_points, 0)


def _box_width(key_points: sv.KeyPoints, detection_index: int) -> float:
    xyxy = key_points.data["xyxy"]
    box = xyxy[detection_index]
    return float(box[2] - box[0])


def test_motion_grows_ghost_box_for_approaching_person() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(motion_enabled=True, motion_smoothing=0.5, max_missed=3),
        frame_width=1280,
    )
    # A centered person whose box grows each frame (walking toward the camera).
    pipeline.apply(_box_key_points(boxes=[(100.0, 100.0, 200.0, 300.0)], confidences=[0.9]), 0)
    pipeline.apply(_box_key_points(boxes=[(90.0, 90.0, 210.0, 310.0)], confidences=[0.9]), 1)
    last = pipeline.apply(_box_key_points(boxes=[(80.0, 80.0, 220.0, 320.0)], confidences=[0.9]), 2)
    ghost = pipeline.apply(_box_key_points(boxes=[], confidences=[]), 3)

    assert ghost.stats.ghost_count == 1
    # Scale prediction should keep growing the held box, not freeze its size.
    assert _box_width(ghost.key_points, 0) > _box_width(last.key_points, 0)


def test_motion_gate_rejects_implausible_long_range_match() -> None:
    track_boxes = [np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)]
    detection_boxes = [np.array([60.0, 0.0, 160.0, 100.0], dtype=np.float64)]

    ungated_matched, _, _ = _match_tracks_to_detections(
        track_boxes,
        detection_boxes,
        match_iou_threshold=0.15,
        gate_distances=None,
    )
    gated_matched, unmatched_tracks, _ = _match_tracks_to_detections(
        track_boxes,
        detection_boxes,
        match_iou_threshold=0.15,
        gate_distances=[50.0],
    )
    # Centers are 60px apart: the overlapping boxes match without a gate but the
    # 50px gate disqualifies the pair.
    assert ungated_matched == [(0, 0)]
    assert gated_matched == []
    assert unmatched_tracks == {0}


def _person_key_points(
    *,
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
) -> sv.KeyPoints:
    """Build keypoints with visible torso joints (5, 6, 11, 12) inside each box."""
    num = len(boxes)
    xy = np.zeros((num, 17, 2), dtype=np.float32)
    visible = np.zeros((num, 17), dtype=bool)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        width = x2 - x1
        height = y2 - y1
        xy[index, 0] = ((x1 + x2) / 2, (y1 + y2) / 2)
        xy[index, 5] = (x1 + 0.3 * width, y1 + 0.3 * height)
        xy[index, 6] = (x1 + 0.7 * width, y1 + 0.3 * height)
        xy[index, 11] = (x1 + 0.3 * width, y1 + 0.6 * height)
        xy[index, 12] = (x1 + 0.7 * width, y1 + 0.6 * height)
        for joint in (0, 5, 6, 11, 12):
            visible[index, joint] = True
    return sv.KeyPoints(
        xy=xy,
        visible=visible,
        keypoint_confidence=np.full((num, 17), 0.9, dtype=np.float32),
        detection_confidence=np.asarray(confidences, dtype=np.float32),
        data={"xyxy": np.asarray(boxes, dtype=np.float32)},
    )


def _frame_with_person(
    box: tuple[float, float, float, float],
    color_bgr: tuple[int, int, int],
    *,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = (int(value) for value in box)
    frame[y1:y2, x1:x2] = color_bgr
    return frame


def test_appearance_similarity_separates_colors() -> None:
    box = (100.0, 100.0, 200.0, 300.0)
    key_points = _person_key_points(boxes=[box], confidences=[0.9])
    roi = appearance_roi(key_points, 0, np.asarray(box, dtype=np.float64))
    assert roi is not None

    red = appearance_histogram(_frame_with_person(box, (0, 0, 255)), roi)
    red_again = appearance_histogram(_frame_with_person(box, (0, 0, 255)), roi)
    green = appearance_histogram(_frame_with_person(box, (0, 255, 0)), roi)

    assert histogram_similarity(red, red_again) > 0.95
    assert histogram_similarity(red, green) < 0.05


def test_build_encoder_selects_backend() -> None:
    from rfdetr_demo.tracking.appearance import EmbeddingEncoder, HistogramEncoder, build_appearance_encoder

    assert isinstance(build_appearance_encoder(backend="histogram", model_path=None), HistogramEncoder)
    assert isinstance(build_appearance_encoder(backend="embedding", model_path="m.onnx"), EmbeddingEncoder)


def test_cosine_similarity_of_l2_vectors() -> None:
    from rfdetr_demo.tracking.appearance import cosine_similarity

    a = np.array([1.0, 0.0])
    orthogonal = np.array([0.0, 1.0])
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, orthogonal) == 0.0


def test_embedding_encoder_normalizes_and_scores(monkeypatch: object) -> None:
    from rfdetr_demo.tracking import appearance

    class _FakeSession:
        def __init__(self, vector: np.ndarray) -> None:
            self._vector = vector

        def run(self, _outputs: object, _inputs: dict[str, object]) -> list[np.ndarray]:
            return [self._vector[np.newaxis, :]]

    encoder = appearance.EmbeddingEncoder("unused.onnx")
    # Bypass the ONNX session with a deterministic fake returning an unnormalized vector.
    encoder._session = _FakeSession(np.array([3.0, 4.0, 0.0], dtype=np.float32))
    encoder._input_name = "input"

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    roi = np.array([5.0, 5.0, 35.0, 35.0], dtype=np.float64)
    descriptor = encoder.encode(frame, roi)
    assert descriptor is not None
    assert abs(float(np.linalg.norm(descriptor)) - 1.0) < 1e-9
    assert encoder.similarity(descriptor, descriptor) == 1.0


def test_reid_cost_blend_prefers_matching_appearance() -> None:
    track_boxes = [np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)]
    # D0 overlaps more (higher IoU) but has a different appearance; D1 overlaps
    # less but matches the track appearance exactly.
    detection_boxes = [
        np.array([10.0, 0.0, 110.0, 100.0], dtype=np.float64),
        np.array([40.0, 0.0, 140.0, 100.0], dtype=np.float64),
    ]
    track_descriptors = [np.array([1.0, 0.0, 0.0, 0.0])]
    det_descriptors = [np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])]

    geometry_only, _, _ = _match_tracks_to_detections(
        track_boxes,
        detection_boxes,
        match_iou_threshold=0.15,
        reid_weight=0.0,
    )
    with_appearance, _, _ = _match_tracks_to_detections(
        track_boxes,
        detection_boxes,
        match_iou_threshold=0.15,
        track_descriptors=track_descriptors,
        det_descriptors=det_descriptors,
        reid_weight=0.7,
    )

    assert geometry_only == [(0, 0)]
    assert with_appearance == [(0, 1)]


def test_reid_revives_track_id_after_long_occlusion() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            max_missed=1,
            reid_enabled=True,
            reid_similarity_threshold=0.5,
            reid_max_gallery_frames=30,
        ),
        frame_width=640,
    )
    box = (100.0, 100.0, 200.0, 300.0)
    red = (0, 0, 255)
    kp = _person_key_points(boxes=[box], confidences=[0.9])

    first = pipeline.apply(kp, 0, _frame_with_person(box, red))
    original_id = track_ids_from_key_points(first.key_points)[0]

    # Occlude for several frames so the track is dropped into the gallery.
    empty = _person_key_points(boxes=[], confidences=[])
    for index in range(1, 4):
        pipeline.apply(empty, index, np.zeros((480, 640, 3), dtype=np.uint8))

    reappear = pipeline.apply(kp, 4, _frame_with_person(box, red))
    revived_id = track_ids_from_key_points(reappear.key_points)[0]

    assert revived_id == original_id


def test_reid_disabled_assigns_new_id_after_long_occlusion() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(max_missed=1, reid_enabled=False),
        frame_width=640,
    )
    box = (100.0, 100.0, 200.0, 300.0)
    red = (0, 0, 255)
    kp = _person_key_points(boxes=[box], confidences=[0.9])

    first = pipeline.apply(kp, 0, _frame_with_person(box, red))
    original_id = track_ids_from_key_points(first.key_points)[0]

    empty = _person_key_points(boxes=[], confidences=[])
    for index in range(1, 4):
        pipeline.apply(empty, index)

    reappear = pipeline.apply(kp, 4, _frame_with_person(box, red))
    new_id = track_ids_from_key_points(reappear.key_points)[0]

    assert new_id != original_id


def test_hysteresis_blocks_low_confidence_new_track() -> None:
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(
            max_missed=2,
            hysteresis_enabled=True,
            new_track_min_confidence=0.65,
        ),
        frame_width=1280,
    )
    pipeline.apply(
        _box_key_points(boxes=[(100.0, 100.0, 200.0, 300.0)], confidences=[0.90]),
        0,
    )
    result = pipeline.apply(
        _box_key_points(
            boxes=[(100.0, 100.0, 200.0, 300.0), (400.0, 100.0, 500.0, 300.0)],
            confidences=[0.90, 0.58],
        ),
        1,
    )
    assert result.stats.active_track_count == 1
