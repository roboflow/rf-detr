# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for :func:`rfdetr.detr._validate_shape_dims`.

Tests call the helper directly so each validation path has a single focused
test without the export/predict scaffolding overhead.
"""

import pytest

from rfdetr.detr import _validate_shape_dims


class TestValidateShapeDimsHappyPath:
    """_validate_shape_dims returns normalised (height, width) for valid inputs."""

    def test_exact_plain_ints(self) -> None:
        """Plain int dims divisible by block_size are returned unchanged."""
        assert _validate_shape_dims((56, 112), 14, 14, 1) == (56, 112)

    def test_returns_plain_int_tuple(self) -> None:
        """Return type is always a tuple of plain Python int."""
        h, w = _validate_shape_dims((56, 56), 14, 14, 1)
        assert type(h) is int and type(w) is int

    def test_numpy_int_accepted(self) -> None:
        """numpy.int64 dims are accepted via operator.index and normalised."""
        import numpy as np

        h, w = _validate_shape_dims((np.int64(56), np.int64(112)), 14, 14, 1)
        assert (h, w) == (56, 112)
        assert type(h) is int and type(w) is int

    def test_non_square_shape(self) -> None:
        """Non-square (H != W) shapes are returned correctly."""
        assert _validate_shape_dims((112, 224), 14, 14, 1) == (112, 224)

    def test_block_size_from_num_windows(self) -> None:
        """block_size = patch_size * num_windows; both dims divisible by it."""
        # patch_size=16, num_windows=2 → block_size=32
        assert _validate_shape_dims((64, 128), 32, 16, 2) == (64, 128)


class TestValidateShapeDimsArityErrors:
    """_validate_shape_dims raises ValueError for non-two-element shapes."""

    def test_one_element_raises(self) -> None:
        """Single-element tuple must raise ValueError."""
        with pytest.raises(ValueError, match="shape must be a sequence"):
            _validate_shape_dims((56,), 14, 14, 1)

    def test_three_element_raises(self) -> None:
        """Three-element tuple must raise ValueError."""
        with pytest.raises(ValueError, match="shape must be a sequence"):
            _validate_shape_dims((56, 56, 3), 14, 14, 1)

    def test_scalar_raises(self) -> None:
        """Bare scalar (not a sequence) must raise ValueError."""
        with pytest.raises(ValueError, match="shape must be a sequence"):
            _validate_shape_dims(56, 14, 14, 1)  # type: ignore[arg-type]


class TestValidateShapeDimsBoolRejection:
    """_validate_shape_dims rejects bool dims (isinstance(True, int) is True)."""

    def test_bool_height_raises(self) -> None:
        """Bool height must raise ValueError even though bool is an int subtype."""
        with pytest.raises(ValueError, match="height must be an integer"):
            _validate_shape_dims((True, 56), 14, 14, 1)  # type: ignore[arg-type]

    def test_bool_width_raises(self) -> None:
        """Bool width must raise ValueError."""
        with pytest.raises(ValueError, match="width must be an integer"):
            _validate_shape_dims((56, False), 14, 14, 1)  # type: ignore[arg-type]


class TestValidateShapeDimsFloatRejection:
    """_validate_shape_dims rejects float dims via operator.index."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((56.0, 56.0), id="both_floats"),
            pytest.param((56.0, 56), id="float_height"),
            pytest.param((56, 56.0), id="float_width"),
        ],
    )
    def test_float_dim_raises(self, shape: tuple) -> None:
        """Float dims must raise ValueError (operator.index rejects them)."""
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_shape_dims(shape, 14, 14, 1)


class TestValidateShapeDimsPositiveCheck:
    """_validate_shape_dims rejects zero and negative dimensions."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 56), id="zero_height"),
            pytest.param((56, 0), id="zero_width"),
            pytest.param((-14, 56), id="negative_height"),
            pytest.param((56, -14), id="negative_width"),
        ],
    )
    def test_non_positive_dim_raises(self, shape: tuple[int, int]) -> None:
        """Zero or negative dims must raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            _validate_shape_dims(shape, 14, 14, 1)


class TestValidateShapeDimsDivisibilityCheck:
    """_validate_shape_dims enforces divisibility by block_size."""

    @pytest.mark.parametrize(
        "shape, block_size",
        [
            pytest.param((55, 56), 14, id="height_not_divisible"),
            pytest.param((56, 55), 14, id="width_not_divisible"),
            pytest.param((48, 64), 32, id="height_not_divisible_large_block"),
        ],
    )
    def test_indivisible_shape_raises(self, shape: tuple[int, int], block_size: int) -> None:
        """Shapes not divisible by block_size must raise ValueError."""
        with pytest.raises(ValueError, match=f"divisible by {block_size}"):
            _validate_shape_dims(shape, block_size, 14, 1)

    def test_error_message_includes_patch_size_and_num_windows(self) -> None:
        """Error message must name patch_size and num_windows for debuggability."""
        with pytest.raises(ValueError, match="patch_size=16") as exc_info:
            _validate_shape_dims((48, 64), 32, 16, 2)
        assert "num_windows=2" in str(exc_info.value)
