# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for keypoint utility functions in rfdetr.utilities.keypoints."""

import numpy as np
import pytest

from rfdetr.utilities.keypoints import (
    _is_bg_first_schema,
    _to_active_first,
    _to_bg_first,
    precision_cholesky_to_pixel_covariance,
    schemas_semantically_equal,
)


class TestIsBgFirstSchema:
    """Group: is_bg_first_schema — schema classification."""

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            pytest.param([0, 17], True, id="bg-first-single-class"),
            pytest.param([0, 17, 4], True, id="bg-first-multi-class"),
            pytest.param([0], True, id="bg-only-slot"),
            pytest.param([17], False, id="active-first-single"),
            pytest.param([17, 4], False, id="active-first-multi"),
            pytest.param([], False, id="empty-schema"),
        ],
    )
    def test_classification(self, schema: list[int], expected: bool) -> None:
        """is_bg_first_schema returns expected bool for each schema form."""
        assert _is_bg_first_schema(schema) == expected


class TestToActiveFirst:
    """Group: to_active_first — strip leading background slot."""

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            pytest.param([0, 17], [17], id="bg-first-to-active"),
            pytest.param([0, 17, 4], [17, 4], id="bg-first-multi-class"),
            pytest.param([0], [], id="bg-only-to-empty"),
            pytest.param([17], [17], id="already-active-first"),
            pytest.param([17, 4], [17, 4], id="already-active-multi"),
            pytest.param([], [], id="empty-schema"),
            pytest.param([0, 0, 17], [0, 17], id="multi-leading-zero-strips-one"),
        ],
    )
    def test_conversion(self, schema: list[int], expected: list[int]) -> None:
        """to_active_first strips only the first leading zero slot."""
        assert _to_active_first(schema) == expected

    def test_returns_new_list(self) -> None:
        """to_active_first always returns a new list, never the input object."""
        schema = [17]
        result = _to_active_first(schema)
        assert result is not schema


class TestToBgFirst:
    """Group: to_bg_first — prepend background slot."""

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            pytest.param([17], [0, 17], id="active-first-to-bg"),
            pytest.param([17, 4], [0, 17, 4], id="active-first-multi"),
            pytest.param([0, 17], [0, 17], id="already-bg-first-no-op"),
            pytest.param([], [], id="empty-schema-no-op"),
        ],
    )
    def test_conversion(self, schema: list[int], expected: list[int]) -> None:
        """to_bg_first prepends 0 only when schema is active-first and non-empty."""
        assert _to_bg_first(schema) == expected

    def test_returns_new_list(self) -> None:
        """to_bg_first always returns a new list, never the input object."""
        schema = [0, 17]
        result = _to_bg_first(schema)
        assert result is not schema


class TestSchemasSemanticallyEqual:
    """Group: schemas_semantically_equal — cross-form equality."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            pytest.param([0, 17], [17], True, id="bg-first-eq-active-first"),
            pytest.param([17], [17], True, id="identical-active-first"),
            pytest.param([0, 17], [0, 17], True, id="identical-bg-first"),
            pytest.param([0, 17], [0, 33], False, id="different-keypoint-counts"),
            pytest.param([17], [33], False, id="active-first-mismatch"),
            pytest.param([0], [], True, id="bg-only-eq-empty"),
            pytest.param([], [], True, id="empty-eq-empty"),
            pytest.param([0, 17, 4], [17, 4], True, id="multi-class-cross-form"),
        ],
    )
    def test_equality(self, a: list[int], b: list[int], expected: bool) -> None:
        """schemas_semantically_equal returns expected result for each pair."""
        assert schemas_semantically_equal(a, b) == expected

    def test_symmetric(self) -> None:
        """schemas_semantically_equal(a, b) == schemas_semantically_equal(b, a)."""
        assert schemas_semantically_equal([0, 17], [17]) == schemas_semantically_equal([17], [0, 17])


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
