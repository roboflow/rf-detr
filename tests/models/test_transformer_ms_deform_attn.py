# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the multi-scale deformable attention core, the ``MSDeformAttn`` module and their export compatibility."""

from collections.abc import Callable

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.utilities.tensors import _bilinear_grid_sample


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


_MSDeformInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]


def _keep_eager(module: MSDeformAttn) -> None:
    """Leave ``module`` in eager mode: the no-op counterpart of ``MSDeformAttn.export`` for parametrized tests.

    Examples:
        >>> module = MSDeformAttn(d_model=32, n_levels=2, n_heads=4, n_points=1)
        >>> _keep_eager(module)
        >>> module._export
        False
    """


_EAGER_OR_EXPORT = [pytest.param(_keep_eager, id="eager"), pytest.param(MSDeformAttn.export, id="export")]


_PASS_HW_SHAPES = [pytest.param(False, id="tensor-only"), pytest.param(True, id="hw-pairs")]


def _build_ms_deform_inputs(
    bsz: int = 1,
    n_heads: int = 2,
    head_dim: int = 4,
    len_q: int = 3,
    npts: int = 1,
    levels: list[tuple[int, int]] | None = None,
) -> _MSDeformInputs:
    """Build minimal valid inputs for ms_deform_attn_core_pytorch.

    Examples:
        >>> value, spatial_shapes, sampling_locations, attention_weights, levels = _build_ms_deform_inputs()
        >>> value.shape, spatial_shapes.shape, len(levels)
        (torch.Size([1, 2, 4, 20]), torch.Size([2, 2]), 2)


    Args:
        bsz: Batch size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        len_q: Number of query elements.
        npts: Number of sampling points per level.
        levels: List of (H, W) int pairs; defaults to [(4, 4), (2, 2)].

    Returns:
        Tuple of (value, spatial_shapes_tensor, sampling_locations,
                  attention_weights, spatial_shapes_hw).
    """
    if levels is None:
        levels = [(4, 4), (2, 2)]
    nlvl = len(levels)

    total_hw = sum(ht * wd for ht, wd in levels)
    spatial_shapes_tensor = torch.tensor(levels, dtype=torch.long)
    value = torch.randn(bsz, n_heads, head_dim, total_hw)
    # sampling_locations: (bsz, len_q, n_heads, nlvl, npts, 2) in [0, 1]
    sampling_locations = torch.rand(bsz, len_q, n_heads, nlvl, npts, 2)
    # attention_weights: (bsz, len_q, n_heads, nlvl * npts)
    attention_weights = torch.softmax(torch.randn(bsz, len_q, n_heads, nlvl * npts), dim=-1)

    return value, spatial_shapes_tensor, sampling_locations, attention_weights, levels


class TestMSDeformAttnCorePytorch:
    """Tests for ms_deform_attn_core_pytorch with Python int pair spatial shapes.

    Regression suite for torch.export.export compatibility: iterating over a spatial_shapes tensor yields FakeTensor
    scalars during FakeTensor tracing, which cannot be used as Python int split/view sizes.  The function now accepts an
    optional ``value_spatial_shapes_hw`` list of Python int pairs that bypasses tensor iteration.
    """

    @pytest.fixture
    def make_inputs(self) -> _MSDeformInputs:
        """Default two-level inputs: levels=[(4, 4), (2, 2)]."""
        return _build_ms_deform_inputs()

    @pytest.fixture
    def single_level_inputs(self) -> _MSDeformInputs:
        """Single-level inputs: levels=[(8, 8)]."""
        return _build_ms_deform_inputs(levels=[(8, 8)])

    @pytest.mark.parametrize("pass_hw", _PASS_HW_SHAPES)
    def test_output_shape(self, make_inputs: _MSDeformInputs, pass_hw: bool) -> None:
        """Output is ``(bsz, len_q, n_heads * head_dim)`` from the tensor-only and the Python int pair spatial shapes.

        The int pair path (``value_spatial_shapes_hw``) is the ``torch.export.export``-compatible one: tensor scalars
        from iterating a FakeTensor cannot be used as split/view sizes, so the caller passes explicit Python ints.
        """
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels if pass_hw else None,
        )

        bsz, n_heads, head_dim, _ = value.shape
        len_q = sampling_locations.shape[1]
        assert output.shape == (bsz, len_q, n_heads * head_dim)

    def test_tensor_and_hw_paths_produce_identical_outputs(self, make_inputs: _MSDeformInputs) -> None:
        """Python int pair path and tensor iteration path must produce the same result."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs

        out_tensor_path = ms_deform_attn_core_pytorch(
            value, spatial_shapes_tensor, sampling_locations, attention_weights
        )
        out_hw_path = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        torch.testing.assert_close(out_tensor_path, out_hw_path)

    def test_single_level(self, single_level_inputs: _MSDeformInputs) -> None:
        """Single-level case with Python int pair path must not crash."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = single_level_inputs

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        assert output.shape[0] == 1

    @pytest.mark.parametrize("pass_hw", _PASS_HW_SHAPES)
    def test_single_level_skips_sample_packing(
        self, mocker: MockerFixture, single_level_inputs: _MSDeformInputs, pass_hw: bool
    ) -> None:
        """Single-level attention reuses its sampled tensor without stacking it, with or without int pair shapes."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = single_level_inputs
        stack = mocker.patch.object(torch, "stack", wraps=torch.stack)

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels if pass_hw else None,
        )

        stack.assert_not_called()
        bsz, n_heads, head_dim, _ = value.shape
        len_q = sampling_locations.shape[1]
        assert output.shape == (bsz, len_q, n_heads * head_dim)

    @pytest.mark.parametrize(
        "sampling_layout",
        [pytest.param("rank-6", id="rank-6"), pytest.param("merged-rank-5", id="merged-rank-5")],
    )
    def test_single_level_two_points_matches_prechange_sample_packing(self, sampling_layout: str) -> None:
        """Single-level two-point output must match the former singleton stack-and-flatten packing.

        The direct singleton path avoids allocating a temporary rank-5 tensor, while the pre-change
        ``torch.stack([single_sample], dim=-2).flatten(-2)`` expression establishes the numerical contract for both
        eager rank-6 and export rank-5 sampling-location layouts.
        """
        value, spatial_shapes, sampling_locations, attention_weights, levels = _build_ms_deform_inputs(
            npts=2, levels=[(4, 4)]
        )
        if sampling_layout == "merged-rank-5":
            sampling_locations = sampling_locations.flatten(3, 4)

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        batch_size, n_heads, head_dim, _ = value.shape
        height, width = levels[0]
        grid_locations = sampling_locations[:, :, :, 0] if sampling_locations.ndim == 6 else sampling_locations
        sampling_grid = (2 * grid_locations - 1).transpose(1, 2).flatten(0, 1)
        single_sample = _bilinear_grid_sample(
            value.view(batch_size * n_heads, head_dim, height, width),
            sampling_grid,
            padding_mode="zeros",
            align_corners=False,
        )
        prechange_sampling_values = torch.stack([single_sample], dim=-2).flatten(-2)
        len_query = sampling_locations.shape[1]
        prechange_attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * n_heads, 1, len_query, 2)
        expected = (
            (prechange_sampling_values * prechange_attention_weights)
            .sum(-1)
            .view(batch_size, n_heads * head_dim, len_query)
        )

        torch.testing.assert_close(output, expected.transpose(1, 2).contiguous())

    def test_multiple_levels_keep_sample_packing(self, mocker: MockerFixture, make_inputs: _MSDeformInputs) -> None:
        """Multi-level attention should still stack one sampled tensor per feature level."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs
        stack = mocker.patch.object(torch, "stack", wraps=torch.stack)

        ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        stack.assert_called_once()
        assert len(stack.call_args.args[0]) == len(levels)
        assert stack.call_args.kwargs.get("dim") == -2


class TestMSDeformAttnModule:
    """Tests for MSDeformAttn.forward covering the export-compatibility changes.

    Validates the module-level parameter threading and export-mode assert guard introduced in the torch.export.export
    compatibility fix.
    """

    _d_model = 32
    _n_heads = 4
    _n_levels = 2
    _n_points = 1
    _hw_pairs: list[tuple[int, int]] = [(4, 4), (2, 2)]

    def _make_module_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """Build minimal valid inputs for MSDeformAttn.forward.

        Returns:
            Tuple of (query, reference_points, input_flatten,
                      input_spatial_shapes, input_level_start_index, hw_pairs).
        """
        hw_pairs = self._hw_pairs
        total_len = sum(ht * wd for ht, wd in hw_pairs)
        bsz, len_q = 1, 3

        query = torch.randn(bsz, len_q, self._d_model)
        reference_points = torch.rand(bsz, len_q, self._n_levels, 2)
        input_flatten = torch.randn(bsz, total_len, self._d_model)
        input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        # Cumulative start index per level: [0, H0*W0]
        starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(self._n_levels)]
        input_level_start_index = torch.tensor(starts, dtype=torch.long)

        return query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, hw_pairs

    def test_forward_without_hw_param_backward_compat(self) -> None:
        """MSDeformAttn.forward without hw param produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, _ = self._make_module_inputs()

        output = module(query, ref_pts, input_flatten, spatial_shapes, level_start_index)

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    @pytest.mark.parametrize("configure", _EAGER_OR_EXPORT)
    def test_forward_with_hw_param_keeps_query_shape(self, configure: Callable[[MSDeformAttn], None]) -> None:
        """``MSDeformAttn.forward`` with ``input_spatial_shapes_hw`` returns ``(bsz, len_q, d_model)`` in both modes."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        configure(module)
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_export_mode_forward_with_full_level_dim_and_last_dim_4(self) -> None:
        """Export mode forward with reference_points level dim == n_levels and last dim 4 must match eager output.

        Regression: test_forward_with_hw_param_keeps_query_shape[export] only exercises the n_ref_levels==n_levels
        skip-branch (ms_deform_attn.py:206-210) with last dim 2. This covers the sibling last-dim-4 branch combined
        with a level dim that is already n_levels (not the singleton-broadcast case).
        """
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        module.eval()
        query, _, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()
        ref_pts = torch.rand(query.shape[0], query.shape[1], self._n_levels, 4)

        with torch.no_grad():
            eager_out = module(
                query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
            )
            module.export()
            export_out = module(
                query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
            )

        bsz, len_q, _ = query.shape
        assert export_out.shape == (bsz, len_q, self._d_model)
        torch.testing.assert_close(export_out, eager_out, rtol=1e-5, atol=1e-5)

    def test_export_flag_set_after_export_call(self) -> None:
        """Calling .export() must set _export=True, enabling the torch._assert guard path."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        assert not module._export

        module.export()

        assert module._export

    @pytest.mark.parametrize(
        "last_dim,batch_size",
        [
            pytest.param(4, 1, id="last-dim-4-batch-1"),
            pytest.param(2, 1, id="last-dim-2-batch-1"),
            pytest.param(4, 2, id="last-dim-4-batch-2"),
        ],
    )
    def test_export_mode_broadcasts_singleton_level_dim(self, last_dim: int, batch_size: int) -> None:
        """Checks export mode accepts decoder-style ``(B, Q, 1, last_dim)`` refs when ``n_levels > 1``.

        Regression: the original case only covered last_dim=4 with batch_size=1. The singleton-broadcast
        ``.expand()`` (ms_deform_attn.py:194) feeds both the last-dim-2 (ms_deform_attn.py:199-205) and
        last-dim-4 (ms_deform_attn.py:206-210) sampling-location branches, and must also broadcast
        correctly when batch_size > 1 since the expand only touches the level axis, not batch. This also
        checks that gradients flow back through the ``.expand()`` view to the original singleton-shaped
        reference_points.
        """
        # Use n_points > 1 so a missing expand would yield length n_points vs n_levels*n_points.
        module = MSDeformAttn(d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=2)
        module.eval()
        hw_pairs = self._hw_pairs
        total_len = sum(ht * wd for ht, wd in hw_pairs)
        num_queries = 3
        query = torch.randn(batch_size, num_queries, self._d_model)
        # Decoder export shape: one shared box broadcast across feature levels.
        ref_pts = torch.rand(batch_size, num_queries, 1, last_dim, requires_grad=True)
        input_flatten = torch.randn(batch_size, total_len, self._d_model)
        spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(self._n_levels)]
        level_start_index = torch.tensor(starts, dtype=torch.long)
        assert ref_pts.shape[2] == 1
        assert self._n_levels > 1

        with torch.no_grad():
            eager_out = module(
                query,
                ref_pts,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )
        module.export()
        export_out = module(
            query,
            ref_pts,
            input_flatten,
            spatial_shapes,
            level_start_index,
            input_spatial_shapes_hw=hw_pairs,
        )

        torch.testing.assert_close(export_out.detach(), eager_out, rtol=1e-5, atol=1e-5)

        # Backward-pass check on the new `.expand()` view op (ms_deform_attn.py:194): gradients must
        # flow back to the original singleton-shaped reference_points, not just the expanded view.
        export_out.sum().backward()
        assert ref_pts.grad is not None
        assert ref_pts.grad.shape == ref_pts.shape

    @pytest.mark.parametrize(
        "last_dim",
        [pytest.param(2, id="last-dim-2"), pytest.param(4, id="last-dim-4")],
    )
    def test_export_mode_single_level_config_matches_eager(self, last_dim: int) -> None:
        """Export mode with n_levels=1 (degenerate no-op expand branch) must match eager output.

        Regression: TestMSDeformAttnModule hardcodes n_levels=2 everywhere else, so the
        ``n_ref_levels == 1`` no-op ``.expand(-1, -1, 1, -1)`` branch (ms_deform_attn.py:192-194) that
        fires specifically when self.n_levels == 1 was never exercised.
        """
        hw_pairs: list[tuple[int, int]] = [(4, 4)]
        d_model, n_heads, n_points, n_levels = 32, 4, 2, 1
        module = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        module.eval()
        total_len = sum(ht * wd for ht, wd in hw_pairs)
        batch_size, num_queries = 1, 3
        query = torch.randn(batch_size, num_queries, d_model)
        ref_pts = torch.rand(batch_size, num_queries, n_levels, last_dim)
        input_flatten = torch.randn(batch_size, total_len, d_model)
        spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            eager_out = module(
                query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
            )
            module.export()
            export_out = module(
                query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
            )

        assert export_out.shape == (batch_size, num_queries, d_model)
        torch.testing.assert_close(export_out, eager_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        ("configure", "last_dim", "n_points"),
        [
            pytest.param(MSDeformAttn.export, 4, 1, id="export-last-dim-4-points-1"),
            pytest.param(MSDeformAttn.export, 2, 1, id="export-last-dim-2-points-1"),
            pytest.param(MSDeformAttn.export, 4, 2, id="export-last-dim-4-points-2"),
            pytest.param(_keep_eager, 4, 1, id="eager-last-dim-4-points-1"),
        ],
    )
    def test_forward_rejects_invalid_reference_level_dim(
        self, configure: Callable[[MSDeformAttn], None], last_dim: int, n_points: int
    ) -> None:
        """Forward raises ``ValueError`` when the reference level dim is neither 1 nor ``n_levels``, in both modes.

        Regression: the level-dim guard (ms_deform_attn.py:192-198) is hoisted above the export/eager split, so both
        modes reject malformed input with the same message. It fires before the last-dim branch and before the
        n_points-dependent merged axis is built, so the export cases cover last_dim=2 and n_points>1 as well.
        """
        module = MSDeformAttn(d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=n_points)
        configure(module)
        query, _, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()
        bad_ref = torch.rand(query.shape[0], query.shape[1], self._n_levels + 1, last_dim)

        with pytest.raises(ValueError, match="level dim must be 1 or n_levels"):
            module(
                query,
                bad_ref,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )

    @pytest.mark.parametrize("configure", _EAGER_OR_EXPORT)
    @pytest.mark.parametrize("last_dim", [1, 3])
    def test_forward_rejects_invalid_reference_last_dim(
        self, configure: Callable[[MSDeformAttn], None], last_dim: int
    ) -> None:
        """Forward raises ``ValueError`` when reference_points last dim is neither 2 nor 4, in both modes.

        The ``Raises:`` entry of ``MSDeformAttn.forward`` names this contract (ms_deform_attn.py:211-216 for export
        mode, 233-238 for eager).
        """
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        configure(module)
        query, _, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()
        bad_ref = torch.rand(query.shape[0], query.shape[1], self._n_levels, last_dim)

        with pytest.raises(ValueError, match="Last dim of reference_points must be 2 or 4"):
            module(
                query,
                bad_ref,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )


class _ConcreteShapeDeformAttnCore(nn.Module):
    """Run ``ms_deform_attn_core_pytorch`` with concrete Python int ``(H, W)`` pairs, as ``torch.export`` needs.

    Examples:
        >>> _ConcreteShapeDeformAttnCore([(4, 4)]).hw
        [(4, 4)]
    """

    def __init__(self, hw: list[tuple[int, int]]) -> None:
        super().__init__()
        self.hw = hw

    def forward(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using concrete Python int pairs for export compatibility."""
        return ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights, value_spatial_shapes_hw=self.hw
        )


class _ExportedMSDeformAttn(nn.Module):
    """Export-mode ``MSDeformAttn`` with concrete spatial dimensions, one level per ``hw`` entry.

    Examples:
        >>> _ExportedMSDeformAttn([(4, 4)]).attn._export
        True
    """

    def __init__(self, hw: list[tuple[int, int]]) -> None:
        super().__init__()
        self.attn = MSDeformAttn(d_model=32, n_levels=len(hw), n_heads=4, n_points=2)
        self.attn.export()
        self.hw = hw

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using the module's Python int pairs for export compatibility."""
        return self.attn(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_spatial_shapes_hw=self.hw,
        )


@pytest.mark.parametrize(
    "levels", [pytest.param([(4, 4), (2, 2)], id="two-level"), pytest.param([(4, 4)], id="single-level")]
)
def test_ms_deform_attn_core_pytorch_export_compatible(levels: list[tuple[int, int]]) -> None:
    """``torch.export.export`` must succeed on ``ms_deform_attn_core_pytorch`` given ``value_spatial_shapes_hw``.

    Regression for the FakeTensor tracing failure: iterating ``spatial_shapes`` and using the scalar elements as
    split/view sizes fails because FakeTensor data is not allocated. Concrete Python ints bypass the tensor iteration.
    The single-level case traces the ``num_levels == 1`` branch, which replaces ``torch.stack(...).flatten(-2)`` with a
    direct index and needs its own trace to stay export-compatible.
    """
    value, spatial_shapes, sampling_locations, attention_weights, _ = _build_ms_deform_inputs(levels=levels)

    exported = torch.export.export(
        _ConcreteShapeDeformAttnCore(levels), args=(value, spatial_shapes, sampling_locations, attention_weights)
    )

    assert exported is not None


@pytest.mark.parametrize(
    "hw_pairs", [pytest.param([(4, 4), (2, 2)], id="two-level"), pytest.param([(4, 4)], id="single-level")]
)
def test_ms_deform_attn_module_export_compatible_with_singleton_level_dim(hw_pairs: list[tuple[int, int]]) -> None:
    """``torch.export.export`` must succeed on ``MSDeformAttn.forward`` with decoder-style singleton-level refs.

    Regression: running the module eagerly after ``module.export()`` never traces the ``reference_points.shape[2]``
    control-flow branch (ms_deform_attn.py:192-198) under a real FakeTensor-traced export, the regime ``export()`` is
    built for. The single-level case traces the rank-5 merged level-and-point sampling route the core consumes.
    """
    d_model, batch_size, num_queries = 32, 1, 3
    total_len = sum(ht * wd for ht, wd in hw_pairs)
    query = torch.randn(batch_size, num_queries, d_model)
    # Decoder export shape: one shared box broadcast across feature levels (n_ref_levels == 1 branch).
    reference_points = torch.rand(batch_size, num_queries, 1, 4)
    input_flatten = torch.randn(batch_size, total_len, d_model)
    input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
    starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(len(hw_pairs))]
    input_level_start_index = torch.tensor(starts, dtype=torch.long)

    exported = torch.export.export(
        _ExportedMSDeformAttn(hw_pairs),
        args=(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index),
    )

    assert exported is not None
