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
- ``pack_targets``/``PackedTargets`` round-trip fidelity.
"""

import collections.abc
import pickle
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
import torch.testing
from torch.utils.data import DataLoader

from rfdetr.utilities.tensors import (
    PackedTargets,
    _bilinear_grid_sample,
    make_collate_fn,
    nested_tensor_from_tensor_list,
    pack_targets,
)


def _grid_sample_reference(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Ground-truth output from F.grid_sample for comparison.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> grid = torch.zeros(1, 1, 1, 2)
        >>> _grid_sample_reference(input, grid, align_corners=True)
        tensor([[[[1.5000]]]])
    """
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
    device_type: str = "mps",
) -> torch.Tensor:
    """Force the manual gather-based code path by mocking input.device.type.

    The function checks ``input.device.type not in ("mps", "xla")`` to decide which branch to take.  We patch
    ``torch.Tensor.device`` to return an object whose ``.type`` is *device_type* so the manual path runs on a normal CPU
    tensor without needing real MPS/XLA hardware.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> grid = torch.zeros(1, 1, 1, 2)
        >>> _call_manual_path(input, grid, align_corners=True)
        tensor([[[[1.5000]]]])
    """

    class _FakeDevice:
        type = device_type

        def __eq__(self, other):
            return False

        def __repr__(self):
            return f"device(type='{device_type}')"

    with patch.object(torch.Tensor, "device", new_callable=lambda: property(lambda self: _FakeDevice())):
        return _bilinear_grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seed():
    """Fix random seed for reproducible grid/input generation.

    Examples:
        Injected by pytest as a fixture argument -- calling it directly (bypassing pytest's fixture machinery)
        raises ``Failed: Fixture "seed" called directly``, so this is illustrative only, not run here. The
        underlying effect is a plain ``torch.manual_seed(42)`` call:

        >>> torch.manual_seed(42)  # doctest: +SKIP
    """
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
    """Skip test when current backend does not support grid_sample for dtype.

    Examples:
        Returns silently for a dtype the current backend supports:

        >>> _require_grid_sample_dtype_support(torch.float32)

        For a low-precision dtype the current backend may lack support for (e.g. float16/bfloat16 on some CPU
        builds), this calls ``pytest.skip`` instead of raising -- not run here since the outcome is
        backend-dependent.

        >>> _require_grid_sample_dtype_support(torch.float16)  # doctest: +SKIP
    """
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


class TestBilinearGridSampleDeviceRouting:
    """Both MPS and XLA route through the manual gather path (WP1 Task 1.1)."""

    @pytest.mark.parametrize(
        "device_type",
        [pytest.param("mps", id="mps"), pytest.param("xla", id="xla")],
    )
    def test_gather_path_taken_and_matches_reference(self, seed, device_type):
        """Manual gather path is taken for mps/xla -- F.grid_sample itself is never called -- and matches it."""
        input = torch.randn(1, 3, 8, 8)
        grid = torch.rand(1, 4, 4, 2) * 1.6 - 0.8

        expected = _grid_sample_reference(input, grid, "zeros", False)
        with patch.object(F, "grid_sample", wraps=F.grid_sample) as mock_grid_sample:
            actual = _call_manual_path(input, grid, "zeros", False, device_type=device_type)

        mock_grid_sample.assert_not_called()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestBilinearGridSampleXLAExecution:
    """Real torch_xla PJRT execution -- proves the gather path takes no aten:: CPU fallback (T1 lane, WP2)."""

    @pytest.mark.xla
    def test_gather_path_no_cpu_fallback_on_real_xla_device(self, seed):
        """On a real XLA device (any PJRT backend) the gather path runs with zero aten:: fallback ops."""
        pytest.importorskip("torch_xla")
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met

        device = xm.xla_device()
        input = torch.randn(1, 3, 8, 8, device=device)
        grid = torch.rand(1, 4, 4, 2, device=device) * 1.6 - 0.8

        met.clear_all()
        actual = _bilinear_grid_sample(input, grid, padding_mode="zeros", align_corners=False)
        xm.mark_step()

        report = met.metrics_report()
        aten_lines = [line for line in report.splitlines() if "aten::" in line.lower()]
        assert not aten_lines, "CPU fallback ops detected:\n" + "\n".join(aten_lines)

        expected = _grid_sample_reference(input.cpu(), grid.cpu(), "zeros", False)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


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

        The attention function passes (B*n_heads, head_dim, H, W) input and (B*n_heads, Len_q, P, 2) grid.
        """
        # Simulate B=2, n_heads=8, head_dim=32
        input = torch.randn(16, 32, 14, 14)
        grid = torch.rand(16, 100, 4, 2) * 2.0 - 1.0

        expected = _grid_sample_reference(input, grid, "zeros", False)
        actual = _call_manual_path(input, grid, "zeros", False)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_point_sample_pattern(self, seed):
        """Matches point_sample in segmentation: padding_mode='border', align_corners=False.

        point_sample transforms point_coords via ``2.0 * point_coords - 1.0`` to map [0, 1] -> [-1, 1].
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

    This is the collator-level pad for backbone divisibility.  The rounded-up strip must be marked as padding in the
    mask so downstream attention skips it.  See
    https://github.com/roboflow/rf-detr/issues/983
    for context.
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
        """Batch-max is rounded up to the next multiple of block_size."""
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
            List of ``(image_tensor, target_dict)`` pairs ready to pass to a collate callable.
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


class TestPackedTargets:
    """Packing a batch of target dicts must be lossless and behave as the unpacked batch does."""

    @staticmethod
    def _batch() -> list[dict[str, torch.Tensor]]:
        """Build a batch whose samples differ in instance count, including an empty one.

        Returns:
            Three target dicts holding 2, 0 and 1 instances respectively.

        Examples:
            >>> [int(target["labels"].numel()) for target in TestPackedTargets._batch()]
            [2, 0, 1]
        """
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                "labels": torch.tensor([3, 7]),
                "image_id": torch.tensor(11),
                "orig_size": torch.tensor([480, 640]),
            },
            {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor(12),
                "orig_size": torch.tensor([300, 400]),
            },
            {
                "boxes": torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
                "labels": torch.tensor([5]),
                "image_id": torch.tensor(13),
                "orig_size": torch.tensor([720, 1280]),
            },
        ]

    def test_round_trip_is_bit_identical(self) -> None:
        """Every field must come back with the same values, dtype and shape, not merely close ones."""
        batch = self._batch()
        restored = pack_targets(batch)
        assert len(restored) == len(batch)
        for original, rebuilt in zip(batch, restored):
            assert original.keys() == rebuilt.keys()
            for key, value in original.items():
                assert rebuilt[key].dtype == value.dtype
                assert rebuilt[key].shape == value.shape
                assert torch.equal(rebuilt[key], value)

    def test_a_sample_with_no_instances_survives_the_round_trip(self) -> None:
        """A zero-instance sample must not collapse into its neighbours' rows."""
        restored = pack_targets(self._batch())
        assert restored[1]["boxes"].shape == (0, 4)
        assert restored[1]["labels"].shape == (0,)
        assert torch.equal(restored[0]["boxes"], torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        assert torch.equal(restored[2]["boxes"], torch.tensor([[9.0, 10.0, 11.0, 12.0]]))

    def test_it_crosses_the_boundary_as_one_tensor_per_field(self) -> None:
        """The point of packing is the object count, so pin it: 4 fields, not 4 fields x 3 samples."""
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)
        assert sorted(packed.fields) == ["boxes", "image_id", "labels", "orig_size"]
        assert all(isinstance(tensor, torch.Tensor) for tensor in packed.fields.values())

    def test_it_is_not_a_sequence_abc_so_pin_memory_keeps_the_batch_packed(self) -> None:
        """PyTorch's pin-memory worker dispatches on Sequence before it looks for pin_memory().

        Subclassing the ABC would make it take the batch apart and pin 7 tensors per sample, rebuilding in the main
        process exactly what packing avoids.
        """
        assert not isinstance(pack_targets(self._batch()), collections.abc.Sequence)
        assert hasattr(PackedTargets, "pin_memory")

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pin_memory_actually_pins_every_field(self) -> None:
        """pin_memory() must pin each field's storage, not merely exist as a method."""
        packed = pack_targets(self._batch())
        pinned = packed.pin_memory()
        assert isinstance(pinned, PackedTargets)
        assert all(tensor.is_pinned() for tensor in pinned.fields.values())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pytorch_pin_memory_worker_routes_through_our_pin_memory(self) -> None:
        """The real DataLoader pin-memory dispatcher (torch/utils/data/_utils/pin_memory.py) must call
        PackedTargets.pin_memory() and keep the batch packed, not merely satisfy hasattr() in isolation."""
        from torch.utils.data._utils.pin_memory import pin_memory as torch_pin_memory

        packed = pack_targets(self._batch())
        result = torch_pin_memory(packed)
        assert isinstance(result, PackedTargets)
        assert all(tensor.is_pinned() for tensor in result.fields.values())

    def test_iterating_yields_each_sample_once(self) -> None:
        """Consumers that only iterate a batch, such as the dataset-grid saver, must keep working."""
        batch = self._batch()
        packed = pack_targets(batch)
        seen = [int(target["image_id"].item()) for target in packed]
        assert seen == [int(target["image_id"].item()) for target in batch]

    def test_indexing_matches_sequence_semantics(self) -> None:
        """Negative indices, slicing and out-of-range access must behave as they do on a plain sequence."""
        batch = self._batch()
        packed = pack_targets(batch)
        assert torch.equal(packed[-1]["labels"], batch[-1]["labels"])
        assert [t["image_id"].item() for t in packed[0:2]] == [11, 12]
        with pytest.raises(IndexError):
            packed[len(batch)]

    def test_a_field_with_mixed_dtypes_falls_back_instead_of_being_promoted(self) -> None:
        """torch.cat would promote int64 to float32 here rather than fail, silently changing the field.

        This is reachable on real COCO data: a sample with no annotations builds `iscrowd`/`area` from an
        empty list, which yields float32, while a populated sample yields int64.
        """
        batch = [
            {"iscrowd": torch.tensor([0, 1], dtype=torch.int64)},
            {"iscrowd": torch.zeros((0,), dtype=torch.float32)},
        ]
        result = pack_targets(batch)
        assert not isinstance(result, PackedTargets)
        assert result == tuple(batch)

    def test_a_grad_bearing_value_falls_back_by_identity(self) -> None:
        """Packing must not replace autograd-owned target tensors with a concatenated non-leaf tensor."""
        targets = (
            {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)},
            {"boxes": torch.tensor([[5.0, 6.0, 7.0, 8.0]], requires_grad=True)},
        )

        result = pack_targets(targets)

        assert result is targets
        assert all(target["boxes"].requires_grad for target in result)

    def test_a_field_with_mixed_devices_falls_back_by_identity(self) -> None:
        """Packing must reject a mixed-device field before ``torch.cat`` can raise."""
        targets = (
            {"boxes": torch.zeros(1, 4)},
            {"boxes": torch.empty(1, 4, device="meta")},
        )

        result = pack_targets(targets)

        assert result is targets

    def test_large_int64_values_are_never_routed_through_a_float_cat(self) -> None:
        """A value above float32's exact-integer range must come back exactly, or not be packed at all."""
        big = 2**54 + 1
        batch = [
            {"image_id": torch.tensor(big, dtype=torch.int64)},
            {"image_id": torch.tensor(7, dtype=torch.int64)},
        ]
        packed = pack_targets(batch)
        assert isinstance(packed, PackedTargets)
        assert packed[0]["image_id"].dtype == torch.int64
        assert int(packed[0]["image_id"].item()) == big

    def test_samples_with_different_keys_fall_back_instead_of_packing(self) -> None:
        """A custom dataset must keep working rather than raise; it just does not get the speed-up."""
        batch = [{"boxes": torch.zeros(1, 4)}, {"boxes": torch.zeros(1, 4), "extra": torch.zeros(1)}]
        result = pack_targets(batch)
        assert not isinstance(result, PackedTargets)
        assert result == tuple(batch)

    def test_a_non_tensor_value_falls_back_instead_of_packing(self) -> None:
        """A non-tensor field value must not reach torch.cat; the batch keeps working unpacked."""
        batch = [{"boxes": torch.zeros(1, 4), "path": "a.jpg"}, {"boxes": torch.zeros(1, 4), "path": "b.jpg"}]
        result = pack_targets(batch)  # type: ignore[arg-type]
        assert not isinstance(result, PackedTargets)

    def test_an_empty_batch_packs_to_nothing(self) -> None:
        """An empty batch must return an empty tuple rather than raise."""
        assert pack_targets([]) == ()

    def test_a_sparse_value_falls_back_instead_of_crashing(self) -> None:
        """torch.cat's reshape(-1) has no linear element order for a sparse tensor and raises RuntimeError; the gate
        must catch that before torch.cat, not let it propagate."""
        sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (4,)).coalesce()
        batch = [{"boxes": sparse}, {"boxes": sparse.clone()}]
        result = pack_targets(batch)
        assert not isinstance(result, PackedTargets)
        assert result == tuple(batch)

    def test_a_quantized_value_falls_back_instead_of_being_silently_requantized(self) -> None:
        """Two quantized tensors can share torch.dtype while using a different (scale, zero_point); torch.cat
        requantizes the second sample's values to the first sample's parameters instead of failing, so dtype equality
        alone does not catch this and the gate must check quantization explicitly."""
        first = torch.quantize_per_tensor(torch.tensor([0.03, 0.06]), scale=0.03, zero_point=0, dtype=torch.qint8)
        second = torch.quantize_per_tensor(torch.tensor([0.09, 0.12]), scale=0.1, zero_point=0, dtype=torch.qint8)
        assert first.dtype == second.dtype
        batch = [{"boxes": first}, {"boxes": second}]
        result = pack_targets(batch)
        assert not isinstance(result, PackedTargets)
        assert result == tuple(batch)

    def test_as_list_gives_dicts_that_can_be_mutated_independently(self) -> None:
        """`transfer_batch_to_device` hands these downstream, where they are mutated in place."""
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)
        materialised = packed.as_list()
        assert len({id(target) for target in materialised}) == len(materialised)
        materialised[0]["labels"] = torch.tensor([99, 99])
        assert torch.equal(materialised[2]["labels"], torch.tensor([5]))
        assert torch.equal(packed[0]["labels"], torch.tensor([3, 7]))

    def test_as_list_tensors_do_not_alias_the_packed_storage(self) -> None:
        """`as_list()` promises tensors that "behave exactly like the unpacked batch": on that path every sample already
        owns independent storage, so an in-place write on one sample's tensor must not be visible through another view
        of the same field -- not through `packed.fields`, and not through a second `as_list()` call.

        `__getitem__`/iteration stay lazy views by design (cheap read-only access, e.g. `DatasetGridSaver`); only the
        materialised copy needs this guarantee.
        """
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)
        materialised = packed.as_list()
        materialised[0]["labels"][0] = 999
        materialised[0]["boxes"][0, 0] = 999.0
        assert torch.equal(packed.fields["labels"], torch.tensor([3, 7, 5]))
        assert packed.fields["boxes"][0].item() == 1.0
        assert torch.equal(packed.as_list()[0]["labels"], torch.tensor([3, 7]))

    def test_to_list_matches_the_unpacked_batch_for_every_sample(self) -> None:
        """Direct materialisation preserves every sample's complete tensor contract.

        This catches a reconstruction that drops the empty sample, skips a field, reshapes scalar metadata, or silently
        changes a dtype while still passing the ownership-only regression below.
        """
        batch = self._batch()
        packed = pack_targets(batch)
        assert isinstance(packed, PackedTargets)

        materialised = packed.to_list(torch.device("cpu"))

        assert len(materialised) == len(batch)
        for original, rebuilt in zip(batch, materialised, strict=True):
            assert rebuilt.keys() == original.keys()
            for key, value in original.items():
                assert rebuilt[key].device.type == "cpu"
                assert rebuilt[key].dtype == value.dtype
                assert rebuilt[key].shape == value.shape
                assert torch.equal(rebuilt[key], value)

    def test_to_list_same_device_does_not_alias_packed_storage(self) -> None:
        """Direct materialisation must preserve the unpacked path's independent tensor ownership on CPU."""
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)

        materialised = packed.to_list(torch.device("cpu"))
        materialised[0]["labels"][0] = 999
        materialised[0]["boxes"][0, 0] = 999.0

        assert torch.equal(packed.fields["labels"], torch.tensor([3, 7, 5]))
        assert packed.fields["boxes"][0].item() == 1.0

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_list_transfers_pinned_views_directly_to_cuda(self) -> None:
        """Direct materialisation must preserve values and ownership across a non-blocking CUDA transfer."""
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)
        pinned = packed.pin_memory()

        materialised = pinned.to_list(torch.device("cuda"), non_blocking=True)
        torch.cuda.synchronize()

        assert materialised[0]["labels"].device.type == "cuda"
        assert torch.equal(materialised[0]["labels"].cpu(), torch.tensor([3, 7]))
        materialised[0]["labels"][0] = 999
        assert torch.equal(pinned.fields["labels"], torch.tensor([3, 7, 5]))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_list_bounds_the_transient_cuda_peak_for_a_mask_field(self) -> None:
        """``to_list``'s docstring claims the avoided duplicate "can be large for segmentation masks" -- pin that claim
        on the actual CUDA peak with a mask field, not just on values or ownership."""
        pinned = pack_targets(
            [
                {"labels": torch.tensor([3, 7]), "masks": torch.ones((2, 256, 256), dtype=torch.bool)},
                {"labels": torch.tensor([5]), "masks": torch.ones((1, 256, 256), dtype=torch.bool)},
            ]
        ).pin_memory()
        mask_bytes = pinned.fields["masks"].numel()

        def peak_extra(
            materialise: collections.abc.Callable[[], list[dict[str, torch.Tensor]]],
        ) -> int:
            """Return transient CUDA bytes beyond memory retained by the result.

            Examples:
                This helper requires CUDA and state from the enclosing test.

                >>> peak_extra(lambda: pinned.to_list("cuda"))  # doctest: +SKIP
                0
            """
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            out = materialise()
            torch.cuda.synchronize()
            extra = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
            del out
            return extra

        old_extra = peak_extra(lambda: pinned.to(torch.device("cuda"), non_blocking=True).as_list())
        new_extra = peak_extra(lambda: pinned.to_list(torch.device("cuda"), non_blocking=True))

        assert old_extra >= mask_bytes
        assert new_extra < mask_bytes // 2

    def test_to_returns_self_when_already_on_the_target_device(self) -> None:
        """``to()`` keeps the batch packed instead of materialising it, unlike ``to_list()``.

        Losing its only caller in ``transfer_batch_to_device`` (replaced by ``to_list()``) must not leave it untested: a
        no-op device request has to return the same instance, matching every no-op ``Tensor.to()`` call underneath it.
        """
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)

        same_device = packed.to(torch.device("cpu"))

        assert same_device is packed
        assert torch.equal(same_device.fields["labels"], torch.tensor([3, 7, 5]))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_moves_every_field_and_keeps_the_batch_packed_on_cuda(self) -> None:
        """A genuine device change must move every field and return a new packed batch, not a materialised list."""
        packed = pack_targets(self._batch())
        assert isinstance(packed, PackedTargets)

        moved = packed.to(torch.device("cuda"))

        assert isinstance(moved, PackedTargets)
        assert moved is not packed
        assert all(tensor.device.type == "cuda" for tensor in moved.fields.values())
        assert torch.equal(moved.fields["labels"].cpu(), torch.tensor([3, 7, 5]))

    def test_collate_packs_only_when_asked(self) -> None:
        """The collate contract only changes for callers that opt in."""
        images = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]
        batch = [(images[0], self._batch()[0]), (images[1], self._batch()[2])]
        _, plain = make_collate_fn(block_size=None)(batch)
        _, packed = make_collate_fn(block_size=None, pack=True)(batch)
        assert not isinstance(plain, PackedTargets)
        assert isinstance(packed, PackedTargets)
        for unpacked, rebuilt in zip(plain, packed):
            for key, value in unpacked.items():
                assert torch.equal(rebuilt[key], value)

    def test_worker_collation_packs_losslessly_and_falls_back_for_grad_targets(self) -> None:
        """A real worker must return packed plain targets and preserve grad-bearing targets unpacked."""
        targets = self._batch()[:2]
        dataset = [
            (torch.full((3, 8, 8), 1.0), targets[0]),
            (torch.full((3, 8, 8), 2.0), targets[1]),
        ]

        images, packed = next(
            iter(DataLoader(dataset, batch_size=2, collate_fn=make_collate_fn(pack=True), num_workers=1))
        )

        assert isinstance(packed, PackedTargets)
        assert torch.equal(images.tensors[0, :, :8, :8], dataset[0][0])
        assert torch.equal(images.tensors[1, :, :8, :8], dataset[1][0])
        for original, rebuilt in zip(targets, packed):
            assert original.keys() == rebuilt.keys()
            for key, value in original.items():
                assert rebuilt[key].dtype == value.dtype
                assert rebuilt[key].shape == value.shape
                assert torch.equal(rebuilt[key], value)

        grad_targets = (
            {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)},
            {"boxes": torch.tensor([[5.0, 6.0, 7.0, 8.0]], requires_grad=True)},
        )
        grad_dataset = [
            (torch.full((3, 8, 8), 3.0), grad_targets[0]),
            (torch.full((3, 8, 8), 4.0), grad_targets[1]),
        ]

        _, fallback = next(
            iter(DataLoader(grad_dataset, batch_size=2, collate_fn=make_collate_fn(pack=True), num_workers=1))
        )

        assert isinstance(fallback, tuple)
        assert not isinstance(fallback, PackedTargets)
        assert all(target["boxes"].requires_grad for target in fallback)
