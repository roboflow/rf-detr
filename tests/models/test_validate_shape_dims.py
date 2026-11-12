# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for :func:`rfdetr.detr._validate_shape_dims` and :func:`rfdetr.detr._resolve_patch_size`.

Tests call each helper directly so each validation path has a single focused test without the export/predict scaffolding
overhead.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from rfdetr.detr import _resolve_patch_size, _validate_shape_dims


class TestValidateShapeDimsHappyPath:
    """_validate_shape_dims returns normalised (height, width) for valid inputs."""

    @pytest.mark.parametrize(
        ("shape", "block_size", "patch_size", "num_windows"),
        [
            pytest.param((56, 112), 14, 14, 1, id="plain_ints"),
            pytest.param((112, 224), 14, 14, 1, id="non_square"),
            # block_size = patch_size * num_windows = 16 * 2
            pytest.param((64, 128), 32, 16, 2, id="block_size_from_num_windows"),
        ],
    )
    def test_returns_divisible_shape_unchanged(
        self, shape: tuple[int, int], block_size: int, patch_size: int, num_windows: int
    ) -> None:
        """Plain int dims divisible by block_size are returned unchanged."""
        assert _validate_shape_dims(shape, block_size, patch_size, num_windows) == shape

    def test_returns_plain_int_tuple(self) -> None:
        """Return type is always a tuple of plain Python int."""
        h, w = _validate_shape_dims((56, 56), 14, 14, 1)
        assert type(h) is int and type(w) is int

    def test_numpy_int_accepted(self) -> None:
        """numpy.int64 dims are accepted via operator.index and normalised."""
        h, w = _validate_shape_dims((np.int64(56), np.int64(112)), 14, 14, 1)
        assert (h, w) == (56, 112)
        assert type(h) is int and type(w) is int


class TestValidateShapeDimsArityErrors:
    """_validate_shape_dims raises ValueError for non-two-element shapes."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((56,), id="one_element"),
            pytest.param((56, 56, 3), id="three_elements"),
            pytest.param(56, id="scalar"),
        ],
    )
    def test_non_pair_raises(self, shape: object) -> None:
        """A one-element tuple, a three-element tuple, or a bare scalar must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims(shape, 14, 14, 1)  # type: ignore[arg-type]
        assert "shape must be a sequence" in str(exc_info.value)


class TestValidateShapeDimsInvalidDim:
    """_validate_shape_dims rejects bool, float, and non-positive dimension values."""

    @pytest.mark.parametrize(
        ("shape", "dim_name"),
        [
            pytest.param((True, 56), "height", id="bool_height"),
            pytest.param((56, False), "width", id="bool_width"),
        ],
    )
    def test_bool_dim_raises(self, shape: tuple, dim_name: str) -> None:
        """Bool dims must raise ValueError even though bool is an int subtype."""
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims(shape, 14, 14, 1)  # type: ignore[arg-type]
        assert f"{dim_name} must be an integer" in str(exc_info.value)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((56.0, 56.0), id="both_float"),
            pytest.param((56.0, 56), id="float_height"),
            pytest.param((56, 56.0), id="float_width"),
        ],
    )
    def test_float_dim_raises(self, shape: tuple) -> None:
        """Float dims must raise ValueError (operator.index rejects them)."""
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims(shape, 14, 14, 1)
        assert "must be an integer" in str(exc_info.value)

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
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims(shape, 14, 14, 1)
        assert "positive integers" in str(exc_info.value)


class TestValidateShapeDimsDivisibilityCheck:
    """_validate_shape_dims enforces divisibility by block_size."""

    @pytest.mark.parametrize(
        ("shape", "block_size"),
        [
            pytest.param((55, 56), 14, id="height_not_divisible"),
            pytest.param((56, 55), 14, id="width_not_divisible"),
            pytest.param((48, 64), 32, id="height_not_divisible_large_block"),
        ],
    )
    def test_indivisible_shape_raises(self, shape: tuple[int, int], block_size: int) -> None:
        """Shapes not divisible by block_size must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims(shape, block_size, 14, 1)
        assert f"divisible by {block_size}" in str(exc_info.value)

    def test_error_message_includes_patch_size_and_num_windows(self) -> None:
        """Error message must name patch_size and num_windows for debuggability."""
        with pytest.raises(ValueError) as exc_info:
            _validate_shape_dims((48, 64), 32, 16, 2)
        assert "patch_size=16" in str(exc_info.value)
        assert "num_windows=2" in str(exc_info.value)


class TestResolvePatchSize:
    """_resolve_patch_size resolves and validates patch_size for export()/predict()."""

    @pytest.mark.parametrize(
        ("patch_size", "model_config", "expected"),
        [
            pytest.param(None, SimpleNamespace(patch_size=16), 16, id="none_reads_model_config"),
            pytest.param(None, SimpleNamespace(), 14, id="none_falls_back_to_14_when_config_missing"),
            pytest.param(14, SimpleNamespace(patch_size=14), 14, id="explicit_matching_config"),
        ],
    )
    def test_resolves_patch_size(self, patch_size: int | None, model_config: SimpleNamespace, expected: int) -> None:
        """None resolves from model_config (default 14); an explicit value equal to the config is accepted."""
        assert _resolve_patch_size(patch_size, model_config, "export") == expected

    def test_explicit_mismatch_raises(self) -> None:
        """Providing patch_size != model_config.patch_size must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_patch_size(16, SimpleNamespace(patch_size=14), "export")
        assert "does not match" in str(exc_info.value)

    def test_mismatch_error_includes_caller_name(self) -> None:
        """Mismatch error message includes the caller name for context."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_patch_size(16, SimpleNamespace(patch_size=14), "predict")
        assert "predict" in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0, -1, True, False])
    def test_invalid_explicit_patch_size_raises(self, bad: int) -> None:
        """Non-positive-int patch_size must raise ValueError before the mismatch check."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_patch_size(bad, SimpleNamespace(patch_size=bad), "export")
        assert "patch_size must be a positive integer" in str(exc_info.value)

    def test_invalid_config_patch_size_raises(self) -> None:
        """Bad patch_size in model_config (when caller passes None) must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_patch_size(None, SimpleNamespace(patch_size=0), "export")
        assert "patch_size must be a positive integer" in str(exc_info.value)
