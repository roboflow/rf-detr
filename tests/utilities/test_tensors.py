# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for rfdetr.utilities.tensors.

Covers:
- ``_bilinear_grid_sample`` parity (manual gather path vs ``F.grid_sample``).
- ``nested_tensor_from_tensor_list`` with ``block_size`` (backbone-aware batch rounding).
- ``make_collate_fn`` factory.
"""

import pickle
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
import torch.testing

from rfdetr.utilities.tensors import (
    _bilinear_grid_sample,
    make_collate_fn,
    nested_tensor_from_tensor_list,
)


def _grid_sample_reference(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Ground-truth output from F.grid_sample for comparison."""
    return F.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def _call_manual_path(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Force the manual gather-based code path by mocking input.device.type.

    The function checks ``input.device.type != "mps"`` to decide which branch
    to take.  We patch ``torch.Tensor.device`` to return an object whose
    ``.type`` is ``"mps"`` so the manual path runs on a normal CPU tensor.
    """

    class _FakeMPSDevice:
        type = "mps"

        def __eq__(self, other):
            return False

        def __repr__(self):
            return "device(type='mps')"

    with patch.object(torch.Tensor, "device", new_callable=lambda: property(lambda self: _FakeMPSDevice())):
        return _bilinear_grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seed():
    """Fix random seed for reproducible grid/input generation."""
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Test scenarios as parametrize parameters
# ---------------------------------------------------------------------------

_PADDING_ALIGN_COMBOS = [
    pytest.param("zeros", False, id="zeros-no_align"),
    pytest.param("border", False, id="border-no_align"),
    pytest.param("zeros", True, id="zeros-align_corners"),
]

_LOW_PRECISION_DTYPES = [
    pytest.param(torch.float16, id="float16"),
    pytest.param(torch.bfloat16, id="bfloat16"),
]

_LOW_PRECISION_GRAD_TOLERANCES = {
    torch.float16: (1e-2, 2e-2),
    torch.bfloat16: (3e-2, 1e-1),
}


def _require_grid_sample_dtype_support(dtype: torch.dtype) -> None:
    """Skip test when current backend does not support grid_sample for dtype."""
    input = torch.randn(1, 1, 2, 2, dtype=dtype, requires_grad=True)
    grid = (torch.rand(1, 1, 1, 2, dtype=dtype) * 1.6 - 0.8).requires_grad_(True)
    try:
        out = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out.backward(torch.ones_like(out))
    except RuntimeError as error:
        pytest.skip(f"grid_sample dtype support missing for {dtype}: {error}")


class TestBilinearGridSampleParity:
    """Manual gather path must match F.grid_sample for all grid/padding combos."""

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_interior_grid_coordinates(self, seed, padding_mode, align_corners):
        """Grid values well inside [-1, 1] -- pure interpolation, no boundary effects."""
        input = torch.randn(1, 3, 8, 8)
        # Grid in [-0.8, 0.8] -- safely inside
        grid = torch.rand(1, 4, 4, 2) * 1.6 - 0.8

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_partially_outside_grid_coordinates(self, seed, padding_mode, align_corners):
        """Grid values spanning [-1.5, 1.5] -- some samples fall outside the image."""
        input = torch.randn(1, 3, 8, 8)
        grid = torch.rand(1, 6, 6, 2) * 3.0 - 1.5

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_exact_boundary_grid_values(self, seed, padding_mode, align_corners):
        """Grid values at exact boundaries: -1.0, 0.0, 1.0."""
        input = torch.randn(1, 2, 4, 4)
        # Manually craft grid with boundary values
        coords = torch.tensor([-1.0, 0.0, 1.0])
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, 3, 3, 2)

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_single_pixel_input(self, padding_mode, align_corners):
        """1x1 spatial input -- extreme edge case for index arithmetic."""
        input = torch.tensor([[[[3.14]]]])  # (1, 1, 1, 1)
        grid = torch.tensor([[[[0.0, 0.0]]]])  # (1, 1, 1, 2) -- center

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_batch_and_multichannel(self, seed, padding_mode, align_corners):
        """Batch size > 1 and multiple channels."""
        input = torch.randn(3, 5, 10, 12)
        grid = torch.rand(3, 7, 9, 2) * 2.0 - 1.0  # [-1, 1]

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_all_out_of_bounds(self, padding_mode, align_corners):
        """All grid coordinates far outside [-1, 1] -- tests OOB handling."""
        input = torch.randn(1, 2, 4, 4)
        # All coordinates at +5.0 -- far outside
        grid = torch.full((1, 3, 3, 2), 5.0)

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        _PADDING_ALIGN_COMBOS,
    )
    def test_negative_out_of_bounds(self, padding_mode, align_corners):
        """All grid coordinates at -5.0 -- far outside on the negative side."""
        input = torch.randn(1, 2, 4, 4)
        grid = torch.full((1, 3, 3, 2), -5.0)

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        [
            pytest.param("zeros", False, id="zeros-no_align"),
            pytest.param("border", False, id="border-no_align"),
        ],
    )
    def test_non_square_spatial_dimensions(self, seed, padding_mode, align_corners):
        """Non-square H != W input -- tests that x/y coordinate handling is correct."""
        input = torch.randn(1, 2, 5, 13)  # tall vs wide
        grid = torch.rand(1, 4, 6, 2) * 2.0 - 1.0

        expected = _grid_sample_reference(input, grid, padding_mode, align_corners)
        actual = _call_manual_path(input, grid, padding_mode, align_corners)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestBilinearGridSampleDelegation:
    """On non-MPS devices, the function delegates directly to F.grid_sample."""

    def test_cpu_delegates_to_grid_sample(self, seed):
        """On CPU, output should match F.grid_sample exactly (same code path)."""
        input = torch.randn(1, 3, 8, 8)
        grid = torch.rand(1, 4, 4, 2) * 2.0 - 1.0

        expected = _grid_sample_reference(input, grid, "zeros", False)
        actual = _bilinear_grid_sample(input, grid, padding_mode="zeros", align_corners=False)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_cpu_border_delegates_to_grid_sample(self, seed):
        """On CPU with border padding, output matches F.grid_sample exactly."""
        input = torch.randn(2, 4, 6, 6)
        grid = torch.rand(2, 3, 3, 2) * 3.0 - 1.5

        expected = _grid_sample_reference(input, grid, "border", False)
        actual = _bilinear_grid_sample(input, grid, padding_mode="border", align_corners=False)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestBilinearGridSampleOutputShape:
    """Output shape must be (N, C, Hg, Wg) for all inputs."""

    @pytest.mark.parametrize(
        "n, c, h, w, hg, wg",
        [
            pytest.param(1, 1, 1, 1, 1, 1, id="minimal"),
            pytest.param(1, 3, 8, 8, 4, 4, id="standard"),
            pytest.param(2, 5, 10, 12, 7, 9, id="batch_multichannel"),
            pytest.param(1, 1, 3, 7, 5, 5, id="non_square"),
        ],
    )
    def test_output_shape(self, n, c, h, w, hg, wg):
        """Manual path output shape is (N, C, Hg, Wg)."""
        input = torch.randn(n, c, h, w)
        grid = torch.rand(n, hg, wg, 2) * 2.0 - 1.0

        actual = _call_manual_path(input, grid)
        assert actual.shape == (n, c, hg, wg), f"Expected shape ({n}, {c}, {hg}, {wg}), got {actual.shape}"


class TestBilinearGridSampleGradient:
    """Gradient correctness for the manual gather path."""

    @pytest.mark.parametrize(
        "padding_mode, align_corners",
        [
            pytest.param("zeros", False, id="zeros-no_align"),
            pytest.param("border", False, id="border-no_align"),
            pytest.param("zeros", True, id="zeros-align_corners"),
        ],
    )
    def test_gradient_matches_grid_sample(self, seed, padding_mode, align_corners):
        """Gradients from manual path match those from F.grid_sample."""
        input_ref = torch.randn(1, 2, 6, 6, requires_grad=True)
        grid_ref = (torch.rand(1, 4, 4, 2) * 1.6 - 0.8).requires_grad_(True)

        # Clone for manual path
        input_man = input_ref.detach().clone().requires_grad_(True)
        grid_man = grid_ref.detach().clone().requires_grad_(True)

        # Forward
        out_ref = _grid_sample_reference(input_ref, grid_ref, padding_mode, align_corners)
        out_man = _call_manual_path(input_man, grid_man, padding_mode, align_corners)

        # Backward with same upstream gradient
        upstream = torch.randn_like(out_ref)
        out_ref.backward(upstream)
        out_man.backward(upstream)

        torch.testing.assert_close(
            input_man.grad,
            input_ref.grad,
            atol=1e-5,
            rtol=1e-5,
            msg="Input gradient mismatch between manual path and F.grid_sample",
        )
        torch.testing.assert_close(
            grid_man.grad,
            grid_ref.grad,
            atol=1e-5,
            rtol=1e-5,
            msg="Grid gradient mismatch between manual path and F.grid_sample",
        )

    def test_gradcheck_manual_path(self, seed):
        """torch.autograd.gradcheck passes on the manual path (double precision)."""
        input = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
        grid = (torch.rand(1, 3, 3, 2, dtype=torch.float64) * 1.6 - 0.8).requires_grad_(True)

        assert torch.autograd.gradcheck(
            lambda inp, grd: _call_manual_path(inp, grd, padding_mode="zeros", align_corners=False),
            (input, grid),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        ), "gradcheck failed for manual bilinear grid sample path"


class TestBilinearGridSampleLowPrecision:
    """Low-precision parity and gradients stay aligned with F.grid_sample."""

    @pytest.mark.parametrize("dtype", _LOW_PRECISION_DTYPES)
    def test_low_precision_parity(self, seed, dtype):
        """Manual path output matches F.grid_sample for low-precision inputs."""
        _require_grid_sample_dtype_support(dtype)

        input = torch.randn(2, 3, 6, 6, dtype=dtype)
        grid = torch.rand(2, 4, 4, 2, dtype=dtype) * 3.0 - 1.5

        expected = _grid_sample_reference(input, grid, padding_mode="zeros", align_corners=False)
        actual = _call_manual_path(input, grid, padding_mode="zeros", align_corners=False)

        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        assert actual.dtype == dtype

    @pytest.mark.parametrize("dtype", _LOW_PRECISION_DTYPES)
    def test_low_precision_gradient_parity(self, seed, dtype):
        """Manual path gradients match F.grid_sample gradients for low precision."""
        _require_grid_sample_dtype_support(dtype)
        atol, rtol = _LOW_PRECISION_GRAD_TOLERANCES[dtype]

        input_ref = torch.randn(1, 2, 6, 6, dtype=dtype, requires_grad=True)
        grid_ref = (torch.rand(1, 4, 4, 2, dtype=dtype) * 1.6 - 0.8).requires_grad_(True)

        input_man = input_ref.detach().clone().requires_grad_(True)
        grid_man = grid_ref.detach().clone().requires_grad_(True)

        out_ref = _grid_sample_reference(input_ref, grid_ref, padding_mode="zeros", align_corners=False)
        out_man = _call_manual_path(input_man, grid_man, padding_mode="zeros", align_corners=False)

        upstream = torch.randn_like(out_ref)
        out_ref.backward(upstream)
        out_man.backward(upstream)

        torch.testing.assert_close(input_man.grad, input_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(grid_man.grad, grid_ref.grad, atol=atol, rtol=rtol)
        assert input_man.grad is not None
        assert grid_man.grad is not None
        assert input_man.grad.dtype == dtype
        assert grid_man.grad.dtype == dtype


class TestBilinearGridSampleRealUseCases:
    """Parity tests matching the actual call sites in the codebase."""

    def test_ms_deform_attn_pattern(self, seed):
        """Matches ms_deform_attn_func: padding_mode='zeros', align_corners=False.

        The attention function passes (B*n_heads, head_dim, H, W) input and
        (B*n_heads, Len_q, P, 2) grid.
        """
        # Simulate B=2, n_heads=8, head_dim=32
        input = torch.randn(16, 32, 14, 14)
        grid = torch.rand(16, 100, 4, 2) * 2.0 - 1.0

        expected = _grid_sample_reference(input, grid, "zeros", False)
        actual = _call_manual_path(input, grid, "zeros", False)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_point_sample_pattern(self, seed):
        """Matches point_sample in segmentation: padding_mode='border', align_corners=False.

        point_sample transforms point_coords via ``2.0 * point_coords - 1.0`` to
        map [0, 1] -> [-1, 1].
        """
        input = torch.randn(4, 256, 28, 28)
        # Simulate point_coords in [0, 1], transformed to [-1, 1]
        point_coords_01 = torch.rand(4, 12544, 1, 2)
        grid = 2.0 * point_coords_01 - 1.0

        expected = _grid_sample_reference(input, grid, "border", False)
        actual = _call_manual_path(input, grid, "border", False)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestNestedTensorBlockSize:
    """``nested_tensor_from_tensor_list`` with block_size rounds batch max H/W up.

    This is the collator-level pad for backbone divisibility.  The rounded-up
    strip must be marked as padding in the mask so downstream attention skips
    it.  See https://github.com/roboflow/rf-detr/issues/983 for context.
    """

    @staticmethod
    def _image(c: int, h: int, w: int, fill: float = 1.0) -> torch.Tensor:
        """Return a ``(C, H, W)`` float32 tensor filled with the given value."""
        return torch.full((c, h, w), fill, dtype=torch.float32)

    def test_block_size_none_preserves_old_behavior(self) -> None:
        """Without block_size, the batch tensor is exactly batch-max H/W."""
        images = [self._image(3, 100, 200), self._image(3, 150, 180)]
        nested = nested_tensor_from_tensor_list(images)
        _, _, h, w = nested.tensors.shape
        assert (h, w) == (150, 200)
        # Mask reflects per-image sizes (no block rounding).
        assert nested.mask[0, :100, :200].any().item() is False
        assert nested.mask[0, 100:, :].all().item() is True
        assert nested.mask[1, :150, :180].any().item() is False
        assert nested.mask[1, :, 180:].all().item() is True

    def test_block_size_rounds_up(self) -> None:
        """batch-max is rounded up to the next multiple of block_size."""
        images = [self._image(3, 100, 200), self._image(3, 150, 180)]
        nested = nested_tensor_from_tensor_list(images, block_size=32)
        _, _, h, w = nested.tensors.shape
        # max_h=150 -> 160, max_w=200 -> 224
        assert (h, w) == (160, 224)

    def test_block_size_equal_to_max_is_noop(self) -> None:
        """When batch max already matches a multiple of block_size, no extra rounding."""
        images = [self._image(3, 128, 256)]
        nested = nested_tensor_from_tensor_list(images, block_size=32)
        _, _, h, w = nested.tensors.shape
        assert (h, w) == (128, 256)

    def test_divisor_pad_marked_in_mask(self) -> None:
        """All padded cells (both batch-level and divisor round-up) are marked True in the mask."""
        images = [self._image(3, 100, 200)]
        nested = nested_tensor_from_tensor_list(images, block_size=32)
        tensor = nested.tensors[0]
        mask = nested.mask[0]

        # Content region is the original 100x200; mask[:100, :200] must be False.
        assert mask[:100, :200].any().item() is False
        # The rounded-up strip (100:128 rows, 200:224 cols) must be True.
        assert mask[100:, :].all().item() is True
        assert mask[:, 200:].all().item() is True

        # Content region is the original fill; pad region is zero.
        assert torch.all(tensor[:, :100, :200] == 1.0)
        assert torch.all(tensor[:, 100:, :] == 0.0)
        assert torch.all(tensor[:, :, 200:] == 0.0)

    @pytest.mark.parametrize(
        "block_size,shape,expected",
        [
            pytest.param(32, (100, 100), (128, 128), id="both-rounded"),
            pytest.param(32, (128, 200), (128, 224), id="h-aligned-w-rounded"),
            pytest.param(32, (100, 256), (128, 256), id="h-rounded-w-aligned"),
            pytest.param(56, (100, 100), (112, 112), id="patch14-num-windows4"),
            pytest.param(64, (100, 100), (128, 128), id="block-size-64"),
        ],
    )
    def test_single_image_rounding_parametrized(self, block_size: int, shape: tuple, expected: tuple) -> None:
        """Single-image batch; round-up applied correctly for various block sizes."""
        images = [self._image(3, shape[0], shape[1])]
        nested = nested_tensor_from_tensor_list(images, block_size=block_size)
        _, _, h, w = nested.tensors.shape
        assert (h, w) == expected


class TestMakeCollateFn:
    """``make_collate_fn`` returns a picklable collate callable with block_size rounding baked in."""

    @staticmethod
    def _batch(*shapes: tuple[int, ...]) -> list[tuple[torch.Tensor, dict]]:
        """Build a list of ``(tensor, target_dict)`` pairs with given shapes.

        Args:
            *shapes: Variadic sequence of ``(C, H, W)`` shapes, one per image.

        Returns:
            List of ``(image_tensor, target_dict)`` pairs ready to pass to a
            collate callable.
        """
        batch = []
        for shape in shapes:
            img = torch.full(shape, 1.0, dtype=torch.float32)
            target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}
            batch.append((img, target))
        return batch

    def test_default_block_size_none_behaves_like_collate_fn(self) -> None:
        """With block_size=None, the factory returns a collate equivalent to the default."""
        collate = make_collate_fn()  # block_size=None
        samples, targets = collate(self._batch((3, 100, 200), (3, 150, 180)))
        _, _, h, w = samples.tensors.shape
        assert (h, w) == (150, 200)  # exact batch max
        assert len(targets) == 2

    def test_block_size_rounds_up_batch_max(self) -> None:
        """Factory with block_size=32 rounds batch-max up to 32-multiples."""
        collate = make_collate_fn(block_size=32)
        samples, _ = collate(self._batch((3, 100, 200), (3, 150, 180)))
        _, _, h, w = samples.tensors.shape
        assert (h, w) == (160, 224)

    def test_targets_passed_through(self) -> None:
        """Factory collator preserves the list-of-targets second element."""
        collate = make_collate_fn(block_size=32)
        samples, targets = collate(self._batch((3, 100, 200), (3, 150, 180)))
        assert isinstance(targets, tuple)
        assert len(targets) == 2
        for t in targets:
            assert set(t.keys()) == {"boxes", "labels"}

    def test_mixed_landscape_portrait_batch_masked_correctly(self) -> None:
        """Mixed-orientation batch: all pad (batch + divisor) correctly marked True in mask."""
        # landscape (H=100, W=200) and portrait (H=200, W=100).  block_size=32 rounds
        # batch max (200, 200) to (224, 224).
        collate = make_collate_fn(block_size=32)
        samples, _ = collate(self._batch((3, 100, 200), (3, 200, 100)))
        _, _, h, w = samples.tensors.shape
        assert (h, w) == (224, 224)

        # Each image's content region equals its original shape; everything else is pad.
        mask_a = samples.mask[0]
        mask_b = samples.mask[1]
        assert mask_a[:100, :200].any().item() is False
        assert mask_a[100:, :].all().item() is True
        assert mask_a[:, 200:].all().item() is True
        assert mask_b[:200, :100].any().item() is False
        assert mask_b[200:, :].all().item() is True
        assert mask_b[:, 100:].all().item() is True

    def test_make_collate_fn_is_picklable(self) -> None:
        """make_collate_fn returns a functools.partial picklable for num_workers > 0."""
        collate = make_collate_fn(block_size=32)
        assert pickle.dumps(collate) is not None
