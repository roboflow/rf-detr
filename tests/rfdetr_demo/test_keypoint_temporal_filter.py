# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for keypoint temporal filter."""

from __future__ import annotations

from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter, MotionPlausibilitySettings
from tests.rfdetr_demo.helpers import single_person_keypoints


def test_single_person_keeps_stable_track() -> None:
    settings = MotionPlausibilitySettings(enabled=True, max_speed_fraction_per_sec=0.5, ema_alpha=1.0)
    filt = KeypointTemporalFilter(settings, frame_width=640, frame_height=480, fps=30.0, frame_stride=1)
    first = filt.apply(single_person_keypoints(x=100.0, y=200.0), frame_index=0)
    second = filt.apply(single_person_keypoints(x=105.0, y=200.0), frame_index=1)
    assert first.xy[0, 0, 0] == 100.0
    assert second.xy[0, 0, 0] == 105.0


def test_impossible_speed_is_rejected_in_hold_mode() -> None:
    settings = MotionPlausibilitySettings(
        enabled=True,
        max_speed_fraction_per_sec=0.01,
        reject_mode="hold",
        max_consecutive_holds=8,
    )
    filt = KeypointTemporalFilter(settings, frame_width=640, frame_height=480, fps=30.0, frame_stride=1)
    filt.apply(single_person_keypoints(x=100.0, y=200.0), frame_index=0)
    corrected = filt.apply(single_person_keypoints(x=500.0, y=200.0), frame_index=1)
    assert corrected.xy[0, 0, 0] == 100.0
    assert filt.stats.speed_rejections >= 1


def test_hold_streak_breaks_after_max_consecutive_holds() -> None:
    settings = MotionPlausibilitySettings(
        enabled=True,
        max_speed_fraction_per_sec=0.01,
        reject_mode="hold",
        max_consecutive_holds=2,
        ema_alpha=1.0,
    )
    filt = KeypointTemporalFilter(settings, frame_width=640, frame_height=480, fps=30.0, frame_stride=1)
    filt.apply(single_person_keypoints(x=100.0, y=200.0), frame_index=0)
    filt.apply(single_person_keypoints(x=500.0, y=200.0), frame_index=1)
    released = filt.apply(single_person_keypoints(x=520.0, y=200.0), frame_index=2)
    assert released.xy[0, 0, 0] == 520.0


def test_temporal_filter_skips_ghost_tracks() -> None:
    import numpy as np
    import supervision as sv

    from rfdetr_demo.tracking.detection_stabilizer import TRACK_IS_GHOST_KEY

    settings = MotionPlausibilitySettings(
        enabled=True,
        max_speed_fraction_per_sec=0.01,
        reject_mode="hold",
    )
    filt = KeypointTemporalFilter(settings, frame_width=640, frame_height=480, fps=30.0, frame_stride=1)
    ghost = single_person_keypoints(x=100.0, y=200.0)
    ghost = sv.KeyPoints(
        xy=ghost.xy,
        visible=ghost.visible,
        keypoint_confidence=ghost.keypoint_confidence,
        detection_confidence=ghost.detection_confidence,
        data={TRACK_IS_GHOST_KEY: np.array([True], dtype=bool)},
    )
    out = filt.apply(ghost, frame_index=0)
    moved = sv.KeyPoints(
        xy=out.xy.copy(),
        visible=out.visible,
        keypoint_confidence=out.keypoint_confidence,
        detection_confidence=out.detection_confidence,
        data={TRACK_IS_GHOST_KEY: np.array([True], dtype=bool)},
    )
    moved.xy[0, 0, 0] = 500.0
    second = filt.apply(moved, frame_index=1)
    assert second.xy[0, 0, 0] == 500.0
    assert filt.stats.speed_rejections == 0
