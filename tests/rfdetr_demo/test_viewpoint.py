# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for camera-viewpoint estimation (overhead vs eye-level)."""

from __future__ import annotations

from rfdetr_demo.tracking.viewpoint import (
    ViewpointEstimate,
    estimate_camera_viewpoint,
    preset_for_viewpoint,
)


def _box(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


class TestEstimateCameraViewpoint:
    def test_overhead_boxes_are_squarish_with_uniform_size(self) -> None:
        # Bird's-eye view: near-square person silhouettes, size independent of y-position.
        boxes = [
            _box(cx=100.0, cy=50.0, width=40.0, height=44.0),
            _box(cx=300.0, cy=200.0, width=42.0, height=40.0),
            _box(cx=500.0, cy=350.0, width=38.0, height=42.0),
            _box(cx=700.0, cy=500.0, width=41.0, height=39.0),
            _box(cx=900.0, cy=650.0, width=39.0, height=43.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 1.0
        assert estimate.sample_count == 5

    def test_eye_level_boxes_are_tall_and_grow_with_y(self) -> None:
        # Ground-level view: tall/narrow silhouettes, boxes near the bottom (closer) are larger.
        boxes = [
            _box(cx=100.0, cy=100.0, width=20.0, height=40.0),
            _box(cx=200.0, cy=250.0, width=35.0, height=70.0),
            _box(cx=300.0, cy=400.0, width=55.0, height=110.0),
            _box(cx=400.0, cy=550.0, width=80.0, height=160.0),
            _box(cx=500.0, cy=680.0, width=110.0, height=220.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "eye_level"
        assert estimate.confidence == 1.0
        assert estimate.sample_count == 5

    def test_too_few_boxes_default_to_low_confidence_overhead(self) -> None:
        boxes = [_box(cx=100.0, cy=100.0, width=40.0, height=40.0)]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 0.0
        assert estimate.sample_count == 1

    def test_no_boxes_default_to_low_confidence_overhead(self) -> None:
        estimate = estimate_camera_viewpoint([], frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 0.0
        assert estimate.sample_count == 0

    def test_conflicting_signals_yield_moderate_confidence(self) -> None:
        # Tall boxes (eye-level signal) but size is symmetric around the frame
        # center rather than growing with y, so it does not correlate with
        # vertical position (overhead signal).
        boxes = [
            _box(cx=100.0, cy=50.0, width=30.0, height=70.0),
            _box(cx=300.0, cy=200.0, width=34.0, height=68.0),
            _box(cx=500.0, cy=350.0, width=28.0, height=72.0),
            _box(cx=700.0, cy=500.0, width=34.0, height=68.0),
            _box(cx=900.0, cy=650.0, width=30.0, height=70.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "eye_level"
        assert estimate.confidence == 0.5

    def test_degenerate_zero_width_boxes_are_skipped(self) -> None:
        boxes = [
            (100.0, 100.0, 100.0, 140.0),  # zero width, ignored
            *[_box(cx=100.0 + i * 50, cy=100.0 + i * 100, width=40.0, height=42.0) for i in range(4)],
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.sample_count == 4


class TestPresetForViewpoint:
    def test_overhead_maps_to_overhead_preset(self) -> None:
        estimate = ViewpointEstimate(
            viewpoint="overhead",
            confidence=1.0,
            median_aspect_ratio=1.0,
            size_position_correlation=0.0,
            sample_count=5,
        )

        assert preset_for_viewpoint(estimate) == "overhead"

    def test_eye_level_maps_to_eye_level_preset(self) -> None:
        estimate = ViewpointEstimate(
            viewpoint="eye_level",
            confidence=1.0,
            median_aspect_ratio=2.0,
            size_position_correlation=0.8,
            sample_count=5,
        )

        assert preset_for_viewpoint(estimate) == "eye-level"
