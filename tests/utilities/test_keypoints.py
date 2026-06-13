# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for keypoint utility functions in rfdetr.utilities.keypoints."""

import numpy as np

from rfdetr.utilities.keypoints import precision_cholesky_to_pixel_covariance


class TestPrecisionCholeskyToPixelCovariance:
    """Group: precision_cholesky_to_pixel_covariance — non-finite input handling."""

    def test_nan_in_single_slot_produces_nan_only_in_that_slot(self) -> None:
        """NaN params in one detection slot should propagate NaN only to that slot's output."""
        # N=2, K=1: first slot valid, second slot has NaN in all three params.
        params = np.array(
            [[[0.0, 0.0, 0.0]], [[np.nan, 0.0, 0.0]]],
            dtype=np.float32,
        )
        source_shape = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=params,
            source_shape=source_shape,
        )

        # First slot (valid) should be all-finite.
        assert np.isfinite(covariance[0, 0]).all(), f"First slot expected all-finite, got {covariance[0, 0]}"
        # Second slot (NaN input) should be all-NaN.
        assert np.isnan(covariance[1, 0]).all(), f"Second slot expected all-NaN, got {covariance[1, 0]}"

    def test_all_inf_params_produce_all_nan_covariance(self) -> None:
        """Infinite precision params should produce all-NaN pixel covariances."""
        params = np.full((1, 1, 3), np.inf, dtype=np.float32)
        source_shape = np.array([[10.0, 20.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=params,
            source_shape=source_shape,
        )

        assert np.isnan(covariance).all(), f"Expected all-NaN output for all-inf inputs, got {covariance}"

    def test_mixed_valid_and_nan_rows_isolates_nan_to_bad_row(self) -> None:
        """First detection valid, second detection NaN — only second row should be NaN."""
        params = np.array(
            [[[0.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]]],
            dtype=np.float32,
        )
        source_shape = np.array([[10.0, 20.0], [5.0, 8.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=params,
            source_shape=source_shape,
        )

        # Row 0 — valid identity input, covariance should be finite.
        assert np.isfinite(covariance[0]).all(), f"Row 0 expected all-finite, got {covariance[0]}"
        # Row 1 — NaN input, covariance should be all-NaN.
        assert np.isnan(covariance[1]).all(), f"Row 1 expected all-NaN, got {covariance[1]}"
