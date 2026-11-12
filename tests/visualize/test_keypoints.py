# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private RF-DETR keypoint visualization helpers."""

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from rfdetr.utilities.keypoints import precision_cholesky_to_pixel_covariance
from rfdetr.visualize.keypoints import _key_points_for_display, _keypoint_prediction_records


class TestPrecisionCholeskyToPixelCovariance:
    """Tests for `precision_cholesky_to_pixel_covariance`."""

    def test_identity_precision_scales_to_pixel_variance(self) -> None:
        """Identity normalized precision should scale to width/height pixel variance."""
        precision_cholesky = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        source_shape = np.array([[10.0, 20.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=precision_cholesky, source_shape=source_shape
        )

        expected = np.array([[[[400.0, 0.0], [0.0, 100.0]]]], dtype=np.float32)
        np.testing.assert_allclose(covariance, expected, rtol=1e-4, atol=1e-6)

    def test_does_not_clamp_log_cholesky(self) -> None:
        """Covariance display should use raw RF-DETR precision parameters."""
        precision_cholesky = np.array([[[25.0, 0.0, 0.0]]], dtype=np.float32)
        source_shape = np.array([[1.0, 1.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=precision_cholesky, source_shape=source_shape
        )

        np.testing.assert_allclose(covariance[0, 0, 0, 0], np.exp(-50.0), rtol=1e-4, atol=1e-28)

    @pytest.mark.parametrize(
        ("precision_shape", "source_shape", "message"),
        [
            ((1, 2, 4), (1, 2), "precision_cholesky must have shape"),
            ((2, 1, 3), (1, 2), "source_shape must have shape"),
        ],
    )
    def test_rejects_bad_shape(self, precision_shape: tuple, source_shape: tuple, message: str) -> None:
        """Invalid precision and source shapes should fail before annotation."""
        with pytest.raises(ValueError) as exc_info:
            precision_cholesky_to_pixel_covariance(
                precision_cholesky=np.zeros(precision_shape, dtype=np.float32),
                source_shape=np.zeros(source_shape, dtype=np.float32),
            )

        assert message in str(exc_info.value)


class TestKeyPointsForDisplay:
    """Tests for `_key_points_for_display`."""

    def test_builds_keypoints_with_covariance_and_masks_low_confidence(self) -> None:
        """RF-DETR keypoints become annotator-ready keypoints with covariance and a low-confidence visibility mask."""
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9, 0.1]], dtype=np.float32),
            detection_confidence=np.array([0.95], dtype=np.float32),
            class_id=np.array([3], dtype=int),
            data={
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
                "xyxy": np.array([[0, 0, 10, 10]], dtype=np.float32),
            },
        )

        key_points = _key_points_for_display(predictions, keypoint_threshold=0.2)

        np.testing.assert_allclose(
            key_points.xy, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            key_points.keypoint_confidence, np.array([[0.9, 0.1]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_array_equal(key_points.visible, np.array([[True, False]]))
        np.testing.assert_array_equal(key_points.class_id, np.array([3]))
        assert "covariance" in key_points.data
        assert key_points.data["covariance"].shape == (1, 2, 2, 2)

    def test_accepts_keypoints_directly(self) -> None:
        """RF-DETR KeyPoints should be annotator-ready without converting through Detections."""
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9, 0.1]], dtype=np.float32),
            detection_confidence=np.array([0.95], dtype=np.float32),
            data={
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
                "xyxy": np.array([[0, 0, 10, 10]], dtype=np.float32),
            },
        )

        key_points = _key_points_for_display(predictions, keypoint_threshold=0.2)

        np.testing.assert_allclose(
            key_points.xy, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_array_equal(key_points.visible, np.array([[True, False]]))
        np.testing.assert_array_equal(key_points.data["xyxy"], predictions.data["xyxy"])
        np.testing.assert_array_equal(key_points.detection_confidence, predictions.detection_confidence)
        assert "covariance" in key_points.data

    def test_preserves_existing_covariance(self) -> None:
        """Display preparation should not overwrite covariance emitted by prediction."""
        covariance = np.array([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=np.float32)
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9]], dtype=np.float32),
            data={
                "covariance": covariance,
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
            },
        )

        key_points = _key_points_for_display(predictions)

        np.testing.assert_array_equal(key_points.data["covariance"], covariance)

    def test_rejects_keypoints_without_confidence_channel(self) -> None:
        """RF-DETR display helper expects per-keypoint confidence."""
        key_points = sv.KeyPoints(xy=np.array([[[1.0, 2.0]]], dtype=np.float32))

        with pytest.raises(ValueError, match="Expected RF-DETR keypoints"):
            _key_points_for_display(key_points)

    def test_returns_empty_keypoints_unchanged(self) -> None:
        """Empty KeyPoints (zero detections) should be returned unchanged without raising."""
        empty_predictions = sv.KeyPoints.empty()

        result = _key_points_for_display(empty_predictions)

        assert len(result) == 0, f"Expected empty KeyPoints, got len={len(result)}"


def test_keypoint_prediction_records_flattens_visible_keypoints() -> None:
    """Prediction records should expose detection and keypoint confidence for visible non-zero points."""
    key_points = sv.KeyPoints(
        xy=np.array([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]], dtype=np.float32),
        keypoint_confidence=np.array([[0.9, 0.99, 0.1]], dtype=np.float32),
        detection_confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([2], dtype=int),
        visible=np.array([[True, True, False]]),
        data={"class_name": np.array(["dartboard"], dtype=object)},
    )

    records = _keypoint_prediction_records(key_points, image=Path("/tmp/sample.jpg"), keypoint_threshold=0.2)

    assert records == [
        {
            "image": "sample.jpg",
            "detection_index": 0,
            "class_id": 2,
            "class_name": "dartboard",
            "detection_confidence": pytest.approx(0.95),
            "keypoint_index": 0,
            "x": pytest.approx(1.0),
            "y": pytest.approx(2.0),
            "keypoint_confidence": pytest.approx(0.9),
        }
    ]
