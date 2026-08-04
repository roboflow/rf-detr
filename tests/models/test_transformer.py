# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for transformer utilities, MS deformable attention core, and MSDeformAttn module."""

import io

import numpy as np
import pytest
import torch
from torch import nn

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import Transformer, gen_encoder_output_proposals, gen_sineembed_for_position


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


_MSDeformInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]


def _build_ms_deform_inputs(
    bsz: int = 1,
    n_heads: int = 2,
    head_dim: int = 4,
    len_q: int = 3,
    npts: int = 1,
    levels: list[tuple[int, int]] | None = None,
) -> _MSDeformInputs:
    """Build minimal valid inputs for ms_deform_attn_core_pytorch.

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


def test_gen_encoder_output_proposals_passes_ij_indexing_to_meshgrid(monkeypatch) -> None:
    """`gen_encoder_output_proposals` should call `torch.meshgrid` with explicit ij indexing."""
    original_meshgrid = torch.meshgrid
    call_count = 0

    def _meshgrid_with_indexing_assertion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("indexing") != "ij":
            raise AssertionError("torch.meshgrid must be called with indexing='ij'")
        return original_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_with_indexing_assertion)

    memory = torch.randn(1, 4, 8)
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        spatial_shapes=spatial_shapes,
    )

    assert call_count == 1


def test_gen_sineembed_for_position_keeps_box_dimensions_in_sin_cos_order() -> None:
    """4D box positional embeddings must use the pretrained sin/cos order for all dimensions."""
    pos_tensor = torch.tensor([[[0.125, 0.25, 0.5, 0.75]]], dtype=torch.float32)
    dim = 4
    scale = 2 * torch.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    expected_parts = []
    for coord_idx in (1, 0, 2, 3):
        coord = pos_tensor[:, :, coord_idx] * scale
        encoded = coord[:, :, None] / dim_t
        expected_parts.append(torch.stack((encoded[:, :, 0::2].sin(), encoded[:, :, 1::2].cos()), dim=3).flatten(2))
    expected = torch.cat(expected_parts, dim=2)

    actual = gen_sineembed_for_position(pos_tensor, dim=dim)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_gen_encoder_output_proposals_rejects_non_square_ij_indexing(monkeypatch) -> None:
    """Wrong meshgrid indexing (xy vs ij) produces different proposals for non-square spatial shapes."""
    original_meshgrid = torch.meshgrid

    def _meshgrid_wrong_indexing(*args, **kwargs):
        kwargs["indexing"] = "xy"
        return original_meshgrid(*args, **kwargs)

    # Use non-square spatial shapes so that ij vs xy indexing produces observably different outputs.
    memory = torch.randn(1, 8, 8)
    spatial_shapes = torch.tensor([[2, 4]], dtype=torch.long)

    correct_memory, correct_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_wrong_indexing)

    wrong_memory, wrong_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    assert not torch.allclose(correct_proposals, wrong_proposals), (
        "xy indexing must produce different proposals than ij indexing for non-square spatial shapes"
    )


def test_gen_encoder_output_proposals_accepts_int_tuple_spatial_shapes() -> None:
    """`gen_encoder_output_proposals` must accept `spatial_shapes` as a tensor of int pairs."""
    batch = 2
    ht, wd = 4, 4
    memory = torch.randn(batch, ht * wd, 8)
    spatial_shapes = torch.tensor([[ht, wd]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, ht * wd, 4)


def test_gen_encoder_output_proposals_accepts_python_int_pair_spatial_shapes() -> None:
    """`gen_encoder_output_proposals` must accept `spatial_shapes` as `list[tuple[int, int]]` with no padding mask.

    Regression: `Transformer.forward` passes Python int pairs derived from `src.shape`, so the
    export-driven call path uses `list[tuple[int, int]]` rather than a tensor.
    """
    batch, ht, wd, dim = 2, 4, 4, 8
    memory = torch.randn(batch, ht * wd, dim)
    spatial_shapes = [(ht, wd)]  # Python int pairs, as produced by Transformer.forward()

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
    )

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, ht * wd, 4)


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

    def test_with_tensor_spatial_shapes(self, make_inputs: _MSDeformInputs) -> None:
        """Baseline: passing only the tensor spatial_shapes still works."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, _ = make_inputs

        output = ms_deform_attn_core_pytorch(value, spatial_shapes_tensor, sampling_locations, attention_weights)

        bsz, n_heads, head_dim, _ = value.shape
        len_q = sampling_locations.shape[1]
        assert output.shape == (bsz, len_q, n_heads * head_dim)

    def test_with_python_int_pair_spatial_shapes(self, make_inputs: _MSDeformInputs) -> None:
        """Regression: value_spatial_shapes_hw list of Python int pairs must be accepted.

        This is the torch.export.export-compatible code path: tensor scalar values (from iterating over a FakeTensor)
        cannot be used as split/view sizes, so the caller passes explicit Python int pairs via value_spatial_shapes_hw
        instead.
        """
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
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

    def test_forward_with_hw_param_produces_correct_shape(self) -> None:
        """MSDeformAttn.forward with input_spatial_shapes_hw produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_export_mode_forward_with_hw_param(self) -> None:
        """MSDeformAttn.forward in export mode with hw param must not raise."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        module.export()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_export_mode_forward_with_full_level_dim_and_last_dim_4(self) -> None:
        """Export mode forward with reference_points level dim == n_levels and last dim 4 must match eager output.

        Regression: test_export_mode_forward_with_hw_param only exercises the n_ref_levels==n_levels skip-branch
        (ms_deform_attn.py:206-210) with last dim 2. This covers the sibling last-dim-4 branch combined with a
        level dim that is already n_levels (not the singleton-broadcast case).
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
        "last_dim,n_points",
        [
            pytest.param(4, 1, id="last-dim-4-points-1"),
            pytest.param(2, 1, id="last-dim-2-points-1"),
            pytest.param(4, 2, id="last-dim-4-points-2"),
        ],
    )
    def test_export_mode_rejects_invalid_reference_level_dim(self, last_dim: int, n_points: int) -> None:
        """Checks export mode raises when reference level dim is neither 1 nor ``n_levels``.

        Regression: the original case only covered last_dim=4 with n_points=1 (self._n_points). The
        level-dim guard (ms_deform_attn.py:195-198) fires before the last-dim branch and before the
        n_points-dependent merged axis is built, so it must also be verified with last_dim=2 and with
        n_points>1 (which changes the size of the merged n_levels*n_points sampling axis).
        """
        module = MSDeformAttn(d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=n_points)
        module.export()
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

    def test_eager_mode_rejects_invalid_reference_level_dim(self) -> None:
        """Eager mode forward must raise ValueError when reference level dim is neither 1 nor n_levels.

        Regression: the level-dim guard was hoisted above the export/eager split so both modes
        reject malformed input with the same message (ms_deform_attn.py:192-198), but only the
        export-mode path (test_export_mode_rejects_invalid_reference_level_dim) was covered.
        """
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        query, _, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()
        bad_ref = torch.rand(query.shape[0], query.shape[1], self._n_levels + 1, 4)

        with pytest.raises(ValueError, match="level dim must be 1 or n_levels"):
            module(
                query,
                bad_ref,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )

    @pytest.mark.parametrize(
        "last_dim",
        [pytest.param(1, id="last-dim-1"), pytest.param(3, id="last-dim-3")],
    )
    def test_eager_mode_rejects_invalid_reference_last_dim(self, last_dim: int) -> None:
        """Eager mode forward must raise ValueError when reference_points last dim is neither 2 nor 4.

        Regression: the ``Raises:`` docstring entry for MSDeformAttn.forward names this contract
        explicitly (ms_deform_attn.py:233-238), but no test previously exercised it.
        """
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
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

    @pytest.mark.parametrize(
        "last_dim",
        [pytest.param(1, id="last-dim-1"), pytest.param(3, id="last-dim-3")],
    )
    def test_export_mode_rejects_invalid_reference_last_dim(self, last_dim: int) -> None:
        """Export mode forward must raise ValueError when reference_points last dim is neither 2 nor 4.

        Regression: the ``Raises:`` docstring entry for MSDeformAttn.forward names this contract
        explicitly (ms_deform_attn.py:211-216), but no test previously exercised it.
        """
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        module.export()
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


class TestGenEncoderOutputProposalsDynamicBatch:
    """Regression tests for dynamic batch support in gen_encoder_output_proposals.

    Ensures that the ONNX-symbolic refactoring (PR #950 / issue #949) does not bake a fixed batch dimension into
    proposals and that output shapes are correct for varying batch sizes.
    """

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_output_shape_invariant_across_batch_sizes(self, batch_size: int) -> None:
        """Output shapes must scale correctly with batch size, with no baked constants.

        Args:
            batch_size: Number of images in the batch.
        """
        ht, wd, dim = 4, 4, 8
        memory = torch.randn(batch_size, ht * wd, dim)
        spatial_shapes = [(ht, wd)]

        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )

        assert output_memory.shape == (batch_size, ht * wd, dim)
        assert output_proposals.shape == (batch_size, ht * wd, 4)

    def test_proposals_semantically_equivalent_across_batch_sizes(self) -> None:
        """Proposals for batch=1 and batch=4 must be identical per image.

        Regression: if batch_size were baked as a constant, repeating the same image
        N times would produce different proposals for each copy.
        """
        ht, wd, dim = 4, 4, 8
        memory_single = torch.randn(1, ht * wd, dim)
        memory_multi = memory_single.expand(4, -1, -1).contiguous()
        spatial_shapes = [(ht, wd)]

        _, proposals_single = gen_encoder_output_proposals(
            memory_single, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )
        _, proposals_multi = gen_encoder_output_proposals(
            memory_multi, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )

        torch.testing.assert_close(proposals_single.expand(4, -1, -1), proposals_multi)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape_invariant_with_padding_mask(self, batch_size: int) -> None:
        """Output shapes must be correct when memory_padding_mask is provided with varying batch sizes.

        Regression for PR #950 / issue #949: the masked branch used .reshape(-1, h, w, 1) to infer the batch dimension
        dynamically; this test verifies the branch handles varying batch sizes without error.

        Args:
            batch_size: Number of images in the batch.
        """
        ht, wd, dim = 4, 4, 8
        total_hw = ht * wd
        memory = torch.randn(batch_size, total_hw, dim)
        # Mask shape: (batch, sum_hw) — True means padding (invalid position)
        memory_padding_mask = torch.zeros(batch_size, total_hw, dtype=torch.bool)
        spatial_shapes = [(ht, wd)]

        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, memory_padding_mask=memory_padding_mask, spatial_shapes=spatial_shapes
        )

        assert output_memory.shape == (batch_size, total_hw, dim)
        assert output_proposals.shape == (batch_size, total_hw, 4)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_onnx_export_with_dynamic_batch_axis(self, batch_size: int) -> None:
        """ONNX export with dynamic batch axis must run inference for batch sizes other than the trace batch.

        Regression for issue #949: exporting with a fixed trace batch baked `Reshape([8,...])` as a constant ONNX node,
        causing TRT engines to fail at inference for any batch != 8. Skipped when onnx or onnxruntime is not installed.
        """
        pytest.importorskip("onnx")
        onnxruntime = pytest.importorskip("onnxruntime")

        ht, wd, dim = 4, 4, 8
        spatial_shapes_list = [(ht, wd)]

        class _ProposalModule(torch.nn.Module):
            """Thin wrapper to export gen_encoder_output_proposals via torch.onnx."""

            def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Forward pass delegating to gen_encoder_output_proposals."""
                return gen_encoder_output_proposals(
                    memory, memory_padding_mask=None, spatial_shapes=spatial_shapes_list
                )

        module = _ProposalModule()
        trace_memory = torch.randn(2, ht * wd, dim)

        buf = io.BytesIO()
        torch.onnx.export(
            module,
            (trace_memory,),
            buf,
            input_names=["memory"],
            output_names=["output_memory", "output_proposals"],
            dynamic_axes={"memory": {0: "batch"}},
            opset_version=17,
        )
        buf.seek(0)
        onnx_bytes = buf.read()

        session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        memory_np = np.random.randn(batch_size, ht * wd, dim).astype(np.float32)
        out_memory, out_proposals = session.run(None, {"memory": memory_np})
        assert out_memory.shape == (batch_size, ht * wd, dim), f"wrong memory shape for batch={batch_size}"
        assert out_proposals.shape == (batch_size, ht * wd, 4), f"wrong proposals shape for batch={batch_size}"


def test_ms_deform_attn_core_pytorch_export_compatible() -> None:
    """torch.export.export must succeed on a module using ms_deform_attn_core_pytorch with hw param.

    Regression test for the FakeTensor tracing failure: iterating over spatial_shapes and using the scalar elements as
    split/view sizes fails during torch.export.export because FakeTensor data is not allocated. Passing
    value_spatial_shapes_hw (concrete Python ints from a module attribute) bypasses the tensor iteration entirely.
    """
    levels: list[tuple[int, int]] = [(4, 4), (2, 2)]
    bsz, n_heads, head_dim = 1, 2, 4
    total_hw = sum(ht * wd for ht, wd in levels)
    len_q, nlvl, npts = 3, len(levels), 1

    class _MinimalDeformAttn(torch.nn.Module):
        """Minimal wrapper to test torch.export.export on the hw-param code path."""

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
                value,
                spatial_shapes,
                sampling_locations,
                attention_weights,
                value_spatial_shapes_hw=self.hw,
            )

    value = torch.randn(bsz, n_heads, head_dim, total_hw)
    spatial_shapes = torch.tensor(levels, dtype=torch.long)
    sampling_locations = torch.rand(bsz, len_q, n_heads, nlvl, npts, 2)
    attention_weights = torch.softmax(torch.randn(bsz, len_q, n_heads, nlvl * npts), dim=-1)

    module = _MinimalDeformAttn(hw=levels)

    exported = torch.export.export(module, args=(value, spatial_shapes, sampling_locations, attention_weights))
    assert exported is not None


def test_ms_deform_attn_module_export_compatible_with_singleton_level_dim() -> None:
    """torch.export.export must succeed on MSDeformAttn.forward with decoder-style singleton-level refs.

    Regression test: TestMSDeformAttnModule.test_export_mode_broadcasts_singleton_level_dim only calls
    module.export() and then runs the module eagerly in Python — it never traces through
    torch.export.export itself, so the reference_points.shape[2] control-flow branch
    (ms_deform_attn.py:192-198) was never verified under a real FakeTensor-traced export, which is
    the actual regime the export() mode is designed for.
    """
    hw_pairs: list[tuple[int, int]] = [(4, 4), (2, 2)]
    d_model, n_heads, n_levels, n_points = 32, 4, 2, 2
    total_len = sum(ht * wd for ht, wd in hw_pairs)
    batch_size, num_queries = 1, 3

    class _MSDeformAttnExportWrapper(torch.nn.Module):
        """Thin wrapper exporting MSDeformAttn.forward via torch.export.export."""

        def __init__(self, hw: list[tuple[int, int]]) -> None:
            super().__init__()
            self.attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
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

    query = torch.randn(batch_size, num_queries, d_model)
    # Decoder export shape: one shared box broadcast across feature levels (n_ref_levels == 1 branch).
    reference_points = torch.rand(batch_size, num_queries, 1, 4)
    input_flatten = torch.randn(batch_size, total_len, d_model)
    input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
    starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(n_levels)]
    input_level_start_index = torch.tensor(starts, dtype=torch.long)

    module = _MSDeformAttnExportWrapper(hw=hw_pairs)

    exported = torch.export.export(
        module,
        args=(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index),
    )
    assert exported is not None


class _FixedTopkScores(nn.Module):
    """Returns pre-set per-position scores regardless of its input.

    Stubs ``enc_out_class_embed`` so the two-stage top-k selection in ``Transformer.forward`` picks known positions in a
    known, deliberately out-of-position-order rank.
    """

    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ignore ``x`` and return the fixed scores."""
        return self.scores


def test_two_stage_topk_gather_selects_correct_rows_out_of_position_order(monkeypatch) -> None:
    """The two-stage top-k gather must copy each selected proposal's exact memory row and box.

    ``torch.topk`` ranks proposals by score, not by position, so the selected indices are rarely in ascending position
    order. This pins that ``memory_ts``/``boxes_ts`` reproduce the source rows picked by an out-of-order, per-batch-row-
    distinct selection, and additionally asserts that ``Tensor.repeat`` is never called during the forward pass:
    ``gen_encoder_output_proposals`` and the decoder layers used here don't call it either, so the only way it could
    fire is the two-stage gather's index broadcast regressing from ``expand`` back to ``repeat``.
    """
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries = 2, 16, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(batch_size, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(batch_size, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(batch_size, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
    )

    # Deliberately out-of-position-order and batch-row-distinct top-3 picks.
    scores = torch.full((batch_size, total_hw, 1), -100.0)
    scores[0, 17, 0], scores[0, 2, 0], scores[0, 9, 0] = 30.0, 20.0, 10.0
    scores[1, 5, 0], scores[1, 19, 0], scores[1, 0, 0] = 25.0, 15.0, 5.0
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(scores)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    repeat_calls: list[tuple[object, ...]] = []
    original_repeat = torch.Tensor.repeat

    def _tracking_repeat(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        repeat_calls.append(args)
        return original_repeat(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "repeat", _tracking_repeat)

    _, _, memory_ts, boxes_ts = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert not repeat_calls, (
        "the two-stage top-k gather must broadcast its index via Tensor.expand, not Tensor.repeat "
        f"(transformer.py:381-390); got {len(repeat_calls)} call(s) to Tensor.repeat during forward: {repeat_calls}"
    )

    # Ground truth computed independently of the gather under test: the same flatten/proposal
    # machinery Transformer.forward uses internally, then plain (non-gather) row indexing.
    memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
    mask_flatten = torch.cat([m.flatten(1) for m in masks], 1)
    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, mask_flatten, spatial_shapes_hw, unsigmoid=True
    )
    output_memory_gidx = transformer.enc_output_norm[0](transformer.enc_output[0](output_memory))
    coord_unselected = transformer.enc_out_bbox_embed[0](output_memory_gidx) + output_proposals
    chosen_idx = scores.squeeze(-1).topk(num_queries, dim=1).indices  # mirrors forward()'s torch.topk call

    assert torch.equal(chosen_idx, torch.tensor([[17, 2, 9], [5, 19, 0]]))  # sanity: out of position order
    expected_memory = torch.stack([output_memory_gidx[b, chosen_idx[b]] for b in range(batch_size)])
    # forward() returns boxes_ts.sigmoid() when bbox_reparam=False (transformer.py:531).
    expected_coord = torch.stack([coord_unselected[b, chosen_idx[b]] for b in range(batch_size)]).sigmoid()
    assert torch.equal(memory_ts, expected_memory)
    assert torch.equal(boxes_ts, expected_coord)
