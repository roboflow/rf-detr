# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for transformer utilities, MS deformable attention core, and MSDeformAttn module."""

import copy
import io
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_model
from rfdetr.models.math import MLP
from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import (
    Transformer,
    TransformerDecoderLayer,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
)
from rfdetr.utilities.tensors import _bilinear_grid_sample


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


_MSDeformInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]


def test_decoder_grouping_reuses_query_tensor_as_key() -> None:
    """Grouped self-attention should preserve layout while reusing query as key."""

    class _RecordingSelfAttention(nn.Module):
        """Fake self-attention that records whether ``query`` and ``key`` are the same object."""

        def __init__(self) -> None:
            super().__init__()
            self.query_is_key = False
            self.query: torch.Tensor | None = None
            self.key: torch.Tensor | None = None
            self.value: torch.Tensor | None = None

        def forward(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
        ) -> tuple[torch.Tensor, None]:
            """Record attention inputs, then return zeros shaped like ``query``."""
            self.query_is_key = query is key
            self.query = query.detach().clone()
            self.key = key.detach().clone()
            self.value = value.detach().clone()
            return torch.zeros_like(query), None

    class _ZeroCrossAttention(nn.Module):
        """Fake cross-attention that ignores its inputs and returns zeros shaped like ``query``."""

        def forward(self, query: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Return a zero tensor shaped like ``query``."""
            return torch.zeros_like(query)

    layer = TransformerDecoderLayer(
        d_model=16,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=32,
        dropout=0,
        group_detr=3,
        num_feature_levels=2,
    )
    self_attn = _RecordingSelfAttention()
    layer.self_attn = self_attn
    layer.cross_attn = _ZeroCrossAttention()

    tgt = torch.arange(2 * 12 * 16, dtype=torch.float32).reshape(2, 12, 16)
    memory = torch.zeros(2, 20, 16)
    query_pos = torch.full_like(tgt, 0.5)

    layer.forward_post(tgt=tgt, memory=memory, query_pos=query_pos)

    assert self_attn.query_is_key
    assert self_attn.query is not None
    assert self_attn.key is not None
    assert self_attn.value is not None

    expected_query = torch.cat((tgt + query_pos).split(4, dim=1), dim=0)
    expected_value = torch.cat(tgt.split(4, dim=1), dim=0)
    assert self_attn.query.shape == (6, 4, 16)
    torch.testing.assert_close(self_attn.query, expected_query)
    torch.testing.assert_close(self_attn.key, expected_query)
    torch.testing.assert_close(self_attn.value, expected_value)


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


@pytest.mark.parametrize("position_layout", ["real", "strided"])
def test_transformer_packs_single_level_position_without_redundant_copy(position_layout: str) -> None:
    """Reuse real contiguous position storage while preserving contiguous output for custom strided input."""
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries, height, width = 1, 16, 3, 4, 4
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 2)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    if position_layout == "real":
        # PositionEmbeddingSine produces contiguous BHWC storage viewed as NCHW. Flattening spatial dimensions and
        # transposing back to B(HW)C is already contiguous and aliases the original storage.
        position_storage = torch.randn(batch_size, height, width, hidden_dim)
        position = position_storage.permute(0, 3, 1, 2)
    else:
        position = torch.randn(batch_size, hidden_dim, height, width)
    position.requires_grad_(True)
    flattened_position = position.flatten(2).transpose(1, 2)
    assert flattened_position.is_contiguous() is (position_layout == "real")
    seen_decoder_positions: list[torch.Tensor] = []

    handle = transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_decoder_positions.append(kwargs["pos"]), with_kwargs=True
    )
    try:
        transformer(
            [torch.randn(batch_size, hidden_dim, height, width)],
            [torch.zeros(batch_size, height, width, dtype=torch.bool)],
            [position],
            torch.rand(num_queries, 4),
            torch.randn(num_queries, hidden_dim),
        )
    finally:
        handle.remove()

    assert len(seen_decoder_positions) == 1
    assert torch.equal(seen_decoder_positions[0], flattened_position)
    assert seen_decoder_positions[0].is_contiguous()
    if position_layout == "real":
        assert seen_decoder_positions[0].data_ptr() == flattened_position.data_ptr()

    seen_decoder_positions[0].sum().backward()
    assert position.grad is not None


@pytest.mark.parametrize("mask_layout", ["real", "strided"])
def test_transformer_packs_single_level_mask_without_redundant_copy(mask_layout: str) -> None:
    """Reuse real padding-mask storage while preserving contiguous output for custom strided input."""
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries, height, width = 1, 16, 3, 4, 4
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 2)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    if mask_layout == "real":
        mask = torch.zeros(batch_size, height, width, dtype=torch.bool)
    else:
        mask_storage = torch.zeros(batch_size, height, width * 2, dtype=torch.bool)
        mask = mask_storage[:, :, ::2]
    flattened_mask = mask.flatten(1)
    assert flattened_mask.is_contiguous() is (mask_layout == "real")
    seen_decoder_masks: list[torch.Tensor] = []

    handle = transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_decoder_masks.append(kwargs["memory_key_padding_mask"]),
        with_kwargs=True,
    )
    try:
        transformer(
            [torch.randn(batch_size, hidden_dim, height, width)],
            [mask],
            [torch.randn(batch_size, hidden_dim, height, width)],
            torch.rand(num_queries, 4),
            torch.randn(num_queries, hidden_dim),
        )
    finally:
        handle.remove()

    assert len(seen_decoder_masks) == 1
    assert torch.equal(seen_decoder_masks[0], flattened_mask)
    assert seen_decoder_masks[0].is_contiguous()
    if mask_layout == "real":
        assert seen_decoder_masks[0].data_ptr() == flattened_mask.data_ptr()


@pytest.mark.parametrize("memory_layout", ["real", "strided"])
def test_transformer_packs_single_level_memory_without_redundant_copy(memory_layout: str) -> None:
    """Reuse real contiguous projector-feature storage while preserving contiguous output for custom strided input."""
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries, height, width = 1, 16, 3, 4, 4
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 2)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    if memory_layout == "real":
        # MultiScaleProjector's final stage norm is unconditionally the permute-based LayerNorm defined in
        # projector.py, which leaves contiguous BHWC storage viewed as NCHW. Flattening spatial dimensions and
        # transposing back to B(HW)C is already contiguous and aliases the original storage.
        src_storage = torch.randn(batch_size, height, width, hidden_dim)
        src = src_storage.permute(0, 3, 1, 2)
    else:
        src = torch.randn(batch_size, hidden_dim, height, width)
    src.requires_grad_(True)
    flattened_src = src.flatten(2).transpose(1, 2)
    assert flattened_src.is_contiguous() is (memory_layout == "real")
    seen_decoder_memories: list[torch.Tensor] = []

    handle = transformer.decoder.register_forward_pre_hook(
        lambda _module, args, _kwargs: seen_decoder_memories.append(args[1]), with_kwargs=True
    )
    try:
        transformer(
            [src],
            [torch.zeros(batch_size, height, width, dtype=torch.bool)],
            [torch.randn(batch_size, hidden_dim, height, width)],
            torch.rand(num_queries, 4),
            torch.randn(num_queries, hidden_dim),
        )
    finally:
        handle.remove()

    assert len(seen_decoder_memories) == 1
    assert torch.equal(seen_decoder_memories[0], flattened_src)
    assert seen_decoder_memories[0].is_contiguous()
    if memory_layout == "real":
        assert seen_decoder_memories[0].data_ptr() == flattened_src.data_ptr()

    seen_decoder_memories[0].sum().backward()
    assert src.grad is not None


@pytest.mark.parametrize("memory_layout", ["real", "strided"])
def test_transformer_packs_single_level_cross_attn_memory_without_redundant_copy(memory_layout: str) -> None:
    """Reuse real contiguous dual-projector storage while preserving contiguous output for custom strided input."""
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries, height, width = 1, 16, 3, 4, 4
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
        dual_projector_kp_only=True,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 2)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    if memory_layout == "real":
        cross_src_storage = torch.randn(batch_size, height, width, hidden_dim)
        cross_src = cross_src_storage.permute(0, 3, 1, 2)
    else:
        cross_src = torch.randn(batch_size, hidden_dim, height, width)
    cross_src.requires_grad_(True)
    flattened_cross_src = cross_src.flatten(2).transpose(1, 2)
    assert flattened_cross_src.is_contiguous() is (memory_layout == "real")
    seen_cross_attn_memories: list[torch.Tensor] = []

    handle = transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_cross_attn_memories.append(kwargs["kp_cross_attn_memory"]),
        with_kwargs=True,
    )
    try:
        transformer(
            [torch.randn(batch_size, hidden_dim, height, width)],
            [torch.zeros(batch_size, height, width, dtype=torch.bool)],
            [torch.randn(batch_size, hidden_dim, height, width)],
            torch.rand(num_queries, 4),
            torch.randn(num_queries, hidden_dim),
            cross_attn_srcs=[cross_src],
        )
    finally:
        handle.remove()

    assert len(seen_cross_attn_memories) == 1
    assert torch.equal(seen_cross_attn_memories[0], flattened_cross_src)
    assert seen_cross_attn_memories[0].is_contiguous()
    if memory_layout == "real":
        assert seen_cross_attn_memories[0].data_ptr() == flattened_cross_src.data_ptr()

    seen_cross_attn_memories[0].sum().backward()
    assert cross_src.grad is not None


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

    def test_single_level_skips_sample_packing(
        self, monkeypatch: pytest.MonkeyPatch, single_level_inputs: _MSDeformInputs
    ) -> None:
        """Single-level attention should reuse its sampled tensor without stacking it."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = single_level_inputs
        stack = Mock(wraps=torch.stack)
        monkeypatch.setattr(torch, "stack", stack)

        ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        stack.assert_not_called()

    def test_single_level_with_tensor_spatial_shapes_skips_sample_packing(
        self, monkeypatch: pytest.MonkeyPatch, single_level_inputs: _MSDeformInputs
    ) -> None:
        """Single-level packing skip must also apply on the tensor-only fallback (no value_spatial_shapes_hw)."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, _ = single_level_inputs
        stack = Mock(wraps=torch.stack)
        monkeypatch.setattr(torch, "stack", stack)

        output = ms_deform_attn_core_pytorch(value, spatial_shapes_tensor, sampling_locations, attention_weights)

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

    def test_multiple_levels_keep_sample_packing(
        self, monkeypatch: pytest.MonkeyPatch, make_inputs: _MSDeformInputs
    ) -> None:
        """Multi-level attention should still stack one sampled tensor per feature level."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs
        stack = Mock(wraps=torch.stack)
        monkeypatch.setattr(torch, "stack", stack)

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


def test_ms_deform_attn_core_pytorch_export_compatible_single_level() -> None:
    """torch.export.export must succeed with a single feature level (num_levels == 1 packing skip).

    Regression test for the singleton-packing change: the two-level case above already covers the general
    torch.export path, but the num_levels == 1 branch replaces torch.stack(...).flatten(-2) with a direct index and
    needs its own FakeTensor trace to confirm that substitution stays export-compatible.
    """
    levels: list[tuple[int, int]] = [(4, 4)]
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


def test_ms_deform_attn_module_export_compatible_single_level() -> None:
    """A one-level exported MSDeformAttn module must trace its rank-5 sampling route.

    Regression: the existing single-level core trace passes rank-6 sampling locations,
    while ``MSDeformAttn.export()`` creates the rank-5 merged level-and-point layout
    consumed by the core during an actual module export.
    """
    hw_pairs: list[tuple[int, int]] = [(4, 4)]
    d_model, n_heads, n_points = 32, 4, 2
    batch_size, num_queries = 1, 3

    class _SingleLevelMSDeformAttnExportWrapper(torch.nn.Module):
        """Export MSDeformAttn with concrete one-level spatial dimensions."""

        def __init__(self, hw: list[tuple[int, int]]) -> None:
            super().__init__()
            self.attn = MSDeformAttn(d_model=d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
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
            """Run the one-level export path with concrete spatial dimensions."""
            return self.attn(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_spatial_shapes_hw=self.hw,
            )

    query = torch.randn(batch_size, num_queries, d_model)
    reference_points = torch.rand(batch_size, num_queries, 1, 4)
    input_flatten = torch.randn(batch_size, 16, d_model)
    input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
    input_level_start_index = torch.tensor([0], dtype=torch.long)
    module = _SingleLevelMSDeformAttnExportWrapper(hw=hw_pairs)

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
    distinct selection, and additionally asserts that every ``torch.gather`` index used by the two-stage selection is a
    broadcast view produced by ``Tensor.expand`` (its broadcast dim keeps stride 0), not a materialised copy.
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

    gather_index_calls: list[torch.Tensor] = []
    original_gather = torch.gather

    def _tracking_gather(input: torch.Tensor, dim: int, index: torch.Tensor, **kwargs: object) -> torch.Tensor:
        gather_index_calls.append(index)
        return original_gather(input, dim, index, **kwargs)

    monkeypatch.setattr(torch, "gather", _tracking_gather)

    _, _, memory_ts, boxes_ts = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    # Every gather index used by the two-stage top-k selection (Transformer.forward two-stage top-k
    # gather) must still be a broadcast view produced by Tensor.expand: its broadcast dim keeps
    # stride 0. Tensor.repeat, Tensor.tile, Tensor.expand(...).contiguous(), and
    # Tensor.repeat_interleave all re-materialise that same allocation into a nonzero-stride tensor,
    # so checking the index's stride catches all four regression variants directly instead of only
    # detecting the literal absence of a Tensor.repeat call.
    assert gather_index_calls, "expected torch.gather to be called during the two-stage top-k selection"
    for index in gather_index_calls:
        assert index.stride(-1) == 0, (
            "the two-stage top-k gather index must broadcast its last dim via Tensor.expand "
            "(Transformer.forward two-stage top-k gather); got a materialised index with nonzero "
            f"last-dim stride {index.stride()} for shape {tuple(index.shape)}"
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
    # forward() returns boxes_ts.sigmoid() when bbox_reparam=False (Transformer.forward two-stage return).
    expected_coord = torch.stack([coord_unselected[b, chosen_idx[b]] for b in range(batch_size)]).sigmoid()
    assert torch.equal(memory_ts, expected_memory)
    assert torch.equal(boxes_ts, expected_coord)


def _make_out_of_order_scores(total_hw: int, picks: list[int]) -> torch.Tensor:
    """Build batch=1 per-position class scores with `picks` as the strictly descending top-k winners.

    Args:
        total_hw: Total number of flattened spatial positions across all feature levels.
        picks: Position indices to rank first, second, third, ... in descending score order.

    Returns:
        Score tensor of shape (1, total_hw, 1); every position not in `picks` scores -100.0.

    Examples:
        >>> _make_out_of_order_scores(5, [3, 1]).squeeze(-1).tolist()
        [[-100.0, 20.0, -100.0, 30.0, -100.0]]
    """
    scores = torch.full((1, total_hw, 1), -100.0)
    picks_tensor = torch.tensor(picks)
    scores[0, picks_tensor, 0] = 30.0 - 10.0 * torch.arange(len(picks), dtype=torch.float32)
    return scores


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_topk_gather_broadcasts_correctly_across_groups_in_training_mode(
    monkeypatch, bbox_reparam: bool
) -> None:
    """With group_detr>1 and the module left in its default training mode, every per-group gather index
    must still broadcast via Tensor.expand, and the concatenated memory_ts/boxes_ts must reproduce each
    group's exact top-k rows.

    Regression: test_two_stage_topk_gather_selects_correct_rows_out_of_position_order only exercises
    group_detr=1, where the ``group_detr = self.group_detr if self.training else 1`` guard in
    Transformer.forward degenerates to a single gather per stage. This pins the group_detr>1 branch,
    which loops the same two gathers twice more (once per extra group) and concatenates the results
    along the query dimension. The module intentionally never calls .eval(): group_detr>1 only takes
    effect while nn.Module.training is True (its default), and calling .eval() would silently fall back
    to the already-covered group_detr=1 path.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 3, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(1, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(1, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

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
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    assert transformer.training  # default nn.Module state; group_detr>1 only takes effect while training

    # Deliberately out-of-position-order, distinct picks per group.
    picks_per_group = [[17, 2, 9], [5, 19, 0], [11, 14, 3]]
    scores_per_group = [_make_out_of_order_scores(total_hw, picks) for picks in picks_per_group]
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(scores) for scores in scores_per_group])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )

    gather_index_calls: list[torch.Tensor] = []
    original_gather = torch.gather

    def _tracking_gather(input: torch.Tensor, dim: int, index: torch.Tensor, **kwargs: object) -> torch.Tensor:
        gather_index_calls.append(index)
        return original_gather(input, dim, index, **kwargs)

    monkeypatch.setattr(torch, "gather", _tracking_gather)

    _, _, memory_ts, boxes_ts = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    # Two gather calls (refpoint, memory) per group -- the only torch.gather call sites in
    # Transformer.forward's two-stage top-k selection.
    assert len(gather_index_calls) == 2 * group_detr
    for index in gather_index_calls:
        assert index.stride(-1) == 0, (
            "every per-group two-stage top-k gather index must broadcast its last dim via Tensor.expand "
            f"(Transformer.forward two-stage top-k gather); got a materialised index with nonzero "
            f"last-dim stride {index.stride()} for shape {tuple(index.shape)}"
        )

    # Ground truth computed independently of the gather under test, per group.
    memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
    mask_flatten = torch.cat([m.flatten(1) for m in masks], 1)
    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, mask_flatten, spatial_shapes_hw, unsigmoid=not bbox_reparam
    )
    picks_tensors = [torch.tensor(picks) for picks in picks_per_group]
    output_memory_per_group = [
        transformer.enc_output_norm[g](transformer.enc_output[g](output_memory)) for g in range(group_detr)
    ]
    if bbox_reparam:
        coord_unselected_per_group = []
        for g in range(group_detr):
            delta = transformer.enc_out_bbox_embed[g](output_memory_per_group[g])
            coord_unselected_per_group.append(
                torch.cat(
                    [
                        delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2],
                        delta[..., 2:].exp() * output_proposals[..., 2:],
                    ],
                    dim=-1,
                )
            )
    else:
        coord_unselected_per_group = [
            transformer.enc_out_bbox_embed[g](output_memory_per_group[g]) + output_proposals for g in range(group_detr)
        ]

    expected_memory = torch.cat(
        [output_memory_per_group[g][0, picks_tensors[g]] for g in range(group_detr)], dim=0
    ).unsqueeze(0)
    # forward() returns boxes_ts.sigmoid() only when bbox_reparam=False.
    expected_coord = torch.cat(
        [coord_unselected_per_group[g][0, picks_tensors[g]] for g in range(group_detr)], dim=0
    ).unsqueeze(0)
    if not bbox_reparam:
        expected_coord = expected_coord.sigmoid()
    torch.testing.assert_close(memory_ts, expected_memory, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(boxes_ts, expected_coord, atol=1e-5, rtol=1e-4)

    parameters = [parameter for module in transformer.enc_out_bbox_embed for parameter in module.parameters()]
    new_gradients = torch.autograd.grad(boxes_ts.sum(), [*srcs, *parameters], retain_graph=True)
    reference_gradients = torch.autograd.grad(expected_coord.sum(), [*srcs, *parameters])
    for new_gradient, reference_gradient in zip(new_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(new_gradient, reference_gradient, atol=1e-5, rtol=1e-4)


def _build_two_stage_transformer_with_production_shaped_heads(
    hidden_dim: int, num_queries: int, group_detr: int, num_classes: int, bbox_reparam: bool
) -> Transformer:
    """Build a two-stage ``Transformer`` whose group heads are the concrete types LWDETR constructs.

    ``Transformer.__init__`` already builds ``enc_output``/``enc_output_norm`` as real
    ``nn.Linear``/``nn.LayerNorm``. This additionally assigns real ``nn.Linear``/``MLP`` instances to
    ``enc_out_class_embed``/``enc_out_bbox_embed`` (which start ``None`` and are set externally by
    ``LWDETR`` in production), matching the layout ``Transformer._two_stage_batching_eligible`` requires
    for the batched fast path.

    Args:
        hidden_dim: Model width.
        num_queries: Queries selected per group.
        group_detr: Number of independent groups.
        num_classes: Class-head output width.
        bbox_reparam: Whether the transformer uses the reparameterised box-delta path.

    Returns:
        A two-stage ``Transformer`` left in its default training mode.

    Examples:
        >>> t = _build_two_stage_transformer_with_production_shaped_heads(16, 3, 2, 5, False)
        >>> t._two_stage_batching_eligible()
        True
    """
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )
    return transformer


def test_two_stage_batching_eligible_true_for_production_shaped_two_stage_modules() -> None:
    """The batched fast path's eligibility guard must accept the exact layout LWDETR constructs."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=4, num_classes=5, bbox_reparam=False
    )
    assert transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_true_for_real_build_model_rfdetr_nano() -> None:
    """The fast path activates on RFDETRNano assembled through the real production constructor.

    ``_build_two_stage_transformer_with_production_shaped_heads`` manually assigns the head types
    ``LWDETR.__init__`` constructs; this test instead calls ``build_model()`` -- the same function
    ``RFDETRNano`` uses -- so the eligibility guard is proven against the actual shipped wiring,
    not a hand-built stand-in.
    """
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    model = build_model(ns)
    assert model.group_detr == 13
    assert model.transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_false_for_linear_subclass_in_enc_output() -> None:
    """A subclassed nn.Linear must be rejected, not accepted via a permissive isinstance check.

    The fast path only reads `.weight`/`.bias` directly and never calls the module itself, so a subclass with its own
    overridden `forward` would have that override silently skipped if the guard let it through.
    """

    class _DoublingLinear(nn.Linear):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x) * 2

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    transformer.enc_output = nn.ModuleList([_DoublingLinear(16, 16) for _ in range(2)])
    assert not transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_false_for_bias_free_linear_in_enc_output() -> None:
    """A bias=False nn.Linear must be rejected, not crash inside torch.stack over a None bias.

    `_stack_linear_params` unconditionally stacks every module's `.bias`; without this guard a `None` bias would raise a
    `TypeError` instead of falling back to the generic loop, which calls the module directly and tolerates a missing
    bias.
    """
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    transformer.enc_output = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(2)])
    assert not transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_false_for_affine_free_layer_norm() -> None:
    """An elementwise_affine=False nn.LayerNorm must be rejected, not crash inside torch.stack over None.

    `_two_stage_group_selection` unconditionally stacks every `enc_output_norm` module's `.weight` and `.bias`; both are
    `None` when `elementwise_affine=False`, which would otherwise reach `torch.stack` instead of falling back to the
    generic loop.
    """
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    transformer.enc_output_norm = nn.ModuleList([nn.LayerNorm(16, elementwise_affine=False) for _ in range(2)])
    assert not transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_false_for_multi_axis_layer_norm() -> None:
    """A multi-axis LayerNorm must use the generic path even when every group has the same shape."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    transformer.enc_output_norm = nn.ModuleList([nn.LayerNorm((20, 16)) for _ in range(2)])
    assert not transformer._two_stage_batching_eligible()


@pytest.mark.parametrize(
    "mismatch",
    [
        "group-count",
        "layer-norm-eps",
        "layer-norm-dtype",
        "class-head-shape",
        "class-head-dtype",
        "bbox-depth",
        "bbox-layer-shape",
    ],
)
def test_two_stage_batching_eligible_false_for_heterogeneous_group_modules(mismatch: str) -> None:
    """Groups that cannot share one stacked operation must fall back to their individual module calls."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )

    if mismatch == "group-count":
        transformer.enc_output = nn.ModuleList([transformer.enc_output[0]])
    elif mismatch == "layer-norm-eps":
        transformer.enc_output_norm[1].eps = 0.1
    elif mismatch == "layer-norm-dtype":
        transformer.enc_output_norm[1].double()
    elif mismatch == "class-head-shape":
        transformer.enc_out_class_embed[1] = nn.Linear(16, 6)
    elif mismatch == "class-head-dtype":
        transformer.enc_out_class_embed[1].double()
    elif mismatch == "bbox-depth":
        transformer.enc_out_bbox_embed[1] = MLP(16, 16, 4, num_layers=2)
    else:
        transformer.enc_out_bbox_embed[1] = MLP(16, 12, 4, num_layers=3)

    assert not transformer._two_stage_batching_eligible()


@pytest.mark.parametrize(
    "case",
    [
        "encoder-hook",
        "norm-hook",
        "class-hook",
        "bbox-hook",
        "bbox-layer-hook",
        "instance-forward",
        "compiled-child",
        "global-hook",
    ],
)
def test_two_stage_batching_eligible_false_when_module_call_semantics_are_observable(case: str) -> None:
    """Hooks and per-instance call overrides must keep the generic module-calling path."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    handle = None
    if case == "encoder-hook":
        transformer.enc_output[0].register_forward_hook(Mock(return_value=None))
    elif case == "norm-hook":
        transformer.enc_output_norm[0].register_forward_pre_hook(Mock(return_value=None))
    elif case == "class-hook":
        transformer.enc_out_class_embed[0].register_full_backward_hook(Mock(return_value=None))
    elif case == "bbox-hook":
        transformer.enc_out_bbox_embed[0].register_full_backward_pre_hook(Mock(return_value=None))
    elif case == "bbox-layer-hook":
        transformer.enc_out_bbox_embed[0].layers[0].register_forward_hook(Mock(return_value=None))
    elif case == "instance-forward":
        transformer.enc_output[0].forward = Mock(side_effect=transformer.enc_output[0].forward)
    elif case == "compiled-child":
        transformer.enc_output[0]._compiled_call_impl = Mock()
    else:
        handle = nn.modules.module.register_module_forward_hook(Mock(return_value=None))

    try:
        assert not transformer._two_stage_batching_eligible()
    finally:
        if handle is not None:
            handle.remove()


def test_two_stage_forward_preserves_hooked_child_module_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public forward route must execute child hooks through the generic fallback."""
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    hook = Mock(return_value=None)
    transformer.enc_output[0].register_forward_hook(hook)
    selection_mock = Mock(side_effect=transformer._two_stage_group_selection)
    monkeypatch.setattr(transformer, "_two_stage_group_selection", selection_mock)

    spatial_shapes_hw = [(4, 4), (2, 2)]
    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert hook.call_count == 1
    assert selection_mock.call_count == 0


def test_two_stage_group_selection_dispatches_through_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Transformer.forward` must invoke the batched fast path exactly once when eligible."""
    hidden_dim, num_queries, group_detr = 16, 3, 4
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    assert transformer._two_stage_batching_eligible()

    selection_mock = Mock(side_effect=transformer._two_stage_group_selection)
    monkeypatch.setattr(transformer, "_two_stage_group_selection", selection_mock)

    spatial_shapes_hw = [(4, 4), (2, 2)]
    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert selection_mock.call_count == 1


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_group_selection_matches_generic_loop_forward_and_gradient(bbox_reparam: bool) -> None:
    """The batched fast path (group_detr>1) must reproduce the generic per-group loop's fp32 outputs and gradients.

    Runs the SAME transformer twice from identical inputs: once through ``Transformer.forward``'s normal
    routing (which takes the fast path because ``_two_stage_batching_eligible()`` is True here), and once
    with that guard monkeypatched to force the generic loop. In fp32 every op the fast path uses (batched
    matmul, one shared-then-affine LayerNorm, batched topk/gather) is a direct algebraic restatement of
    the loop's own ops, so forward and backward both match to float32 rounding noise from the different
    (batched vs. looped) reduction order, not a correctness gap. The test uses a tolerance rather than
    ``torch.equal`` because larger GEMMs may choose a different accumulation order; see
    ``Transformer._two_stage_group_selection`` for the mixed-precision limitation.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 5, 4
    spatial_shapes_hw = [(4, 4), (2, 2)]

    srcs = [torch.randn(2, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=bbox_reparam
    )
    assert transformer._two_stage_batching_eligible()

    _, _, memory_fast, boxes_fast = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
    fast_loss = memory_fast.sum() + boxes_fast.sum()
    # enc_out_class_embed is excluded: its output only ranks torch.topk's (non-differentiable)
    # selection indices, so it structurally never receives a gradient from memory_ts/boxes_ts,
    # in the loop and the batched path alike.
    fast_modules = [*transformer.enc_output, *transformer.enc_output_norm, *transformer.enc_out_bbox_embed]
    fast_parameters = [parameter for module in fast_modules for parameter in module.parameters()]
    fast_gradients = torch.autograd.grad(fast_loss, [*srcs, *fast_parameters])

    transformer_loop = copy.deepcopy(transformer)
    transformer_loop._two_stage_batching_eligible = lambda: False
    srcs_loop = [src.detach().clone().requires_grad_(True) for src in srcs]
    _, _, memory_loop, boxes_loop = transformer_loop(
        srcs_loop, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
    loop_loss = memory_loop.sum() + boxes_loop.sum()
    loop_modules = [
        *transformer_loop.enc_output,
        *transformer_loop.enc_output_norm,
        *transformer_loop.enc_out_bbox_embed,
    ]
    loop_parameters = [parameter for module in loop_modules for parameter in module.parameters()]
    loop_gradients = torch.autograd.grad(loop_loss, [*srcs_loop, *loop_parameters])

    # A batched GEMM can select a different accumulation order than separate calls, so compare
    # within float32 tolerance rather than requiring bit-for-bit equality.
    torch.testing.assert_close(memory_fast, memory_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(boxes_fast, boxes_loop, atol=1e-4, rtol=1e-4)
    for fast_gradient, loop_gradient in zip(fast_gradients, loop_gradients, strict=True):
        torch.testing.assert_close(fast_gradient, loop_gradient, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_stage_group_selection_bf16_produces_finite_valid_selection_with_gradients() -> None:
    """Under bf16 autocast the batched fast path must still produce finite outputs with gradients that reach every
    group's own parameters -- not necessarily bit-identical to the generic loop.

    ``Transformer._two_stage_group_selection`` documents that ``torch.baddbmm``'s batched-GEMM kernel can
    accumulate in a different order than the loop's separate GEMM calls, so a near-tied ``torch.topk``
    ranking can legitimately pick a different, equally valid query under bf16. A hard numeric-parity bound
    would be flaky across GPU architectures and problem sizes depending on whether that specific run
    happens to hit a tie, so this test instead pins the invariant that must ALWAYS hold regardless of
    which tie-break wins: finite values and gradients reaching every parameter.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 32, 20, 13
    spatial_shapes_hw = [(16, 16), (8, 8)]  # matches num_feature_levels=2 in the shared builder
    device = "cuda"

    srcs = [torch.randn(2, hidden_dim, ht, wd, device=device, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool, device=device) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd, device=device) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4, device=device)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim, device=device)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=90, bbox_reparam=True
    ).to(device)
    assert transformer._two_stage_batching_eligible()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, memory_ts, boxes_ts = transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
        )
        loss = memory_ts.float().sum() + boxes_ts.float().sum()

    assert torch.isfinite(memory_ts).all()
    assert torch.isfinite(boxes_ts).all()

    # memory_ts/boxes_ts are the two-stage encoder outputs only (the decoder's own parameters correctly
    # get no gradient from this loss), so only check the modules _two_stage_group_selection actually uses
    # for a DIFFERENTIABLE output. enc_out_class_embed is excluded: its output only ranks torch.topk's
    # (non-differentiable) selection indices, so it structurally never receives a gradient here.
    two_stage_modules = [
        *transformer.enc_output,
        *transformer.enc_output_norm,
        *transformer.enc_out_bbox_embed,
    ]
    named_parameters = [
        (name, parameter)
        for module in two_stage_modules
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    inputs = [*srcs, *[parameter for _, parameter in named_parameters]]
    names = [f"srcs[{i}]" for i in range(len(srcs))] + [name for name, _ in named_parameters]
    gradients = torch.autograd.grad(loss, inputs)
    for name, gradient in zip(names, gradients, strict=True):
        assert torch.isfinite(gradient).all(), f"{name} gradient has non-finite entries"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_stage_group_selection_compiles_with_finite_gradients() -> None:
    """The batched fast path must be compatible with torch.compile.

    Unlike ONNX/TorchScript export (which always forces `group_detr=1` and never reaches this code), `module_model.py`
    applies `torch.compile` directly to the training model, where `group_detr>1` and this new path are exactly what
    training exercises. This asserts compile-time and run-time compatibility (finite outputs and gradients through a
    compiled call, using the same `capture_scalar_outputs` config `module_model.py` itself sets before compiling) -- not
    a timing claim; a local single-GPU smoke run is not a substitute for this project's separately reported L4 step-time
    evidence.
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 5, 4
    spatial_shapes_hw = [(8, 8), (4, 4)]
    device = "cuda"

    srcs = [torch.randn(2, hidden_dim, ht, wd, device=device, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool, device=device) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd, device=device) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4, device=device)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim, device=device)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=True
    ).to(device)
    assert transformer._two_stage_batching_eligible()
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled_transformer = torch.compile(transformer, dynamic=True)

        _, _, memory_ts, boxes_ts = compiled_transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
        )
        loss = memory_ts.sum() + boxes_ts.sum()
        parameters = [
            parameter
            for module in [*transformer.enc_output, *transformer.enc_output_norm, *transformer.enc_out_bbox_embed]
            for parameter in module.parameters()
        ]
        gradients = torch.autograd.grad(loss, [*srcs, *parameters])

    assert torch.isfinite(memory_ts).all()
    assert torch.isfinite(boxes_ts).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def _build_two_stage_transformer_with_keypoints(
    hidden_dim: int, num_queries: int, group_detr: int, num_classes: int, num_keypoints: int
) -> Transformer:
    """Build a decoder-free, two-stage ``Transformer`` with GroupPose keypoint heads enabled.

    ``num_decoder_layers=0`` isolates the encoder-side computation this PR touches (
    :meth:`Transformer._two_stage_group_selection` and the untouched per-group
    ``enc_out_keypoint_embed`` loop that consumes its output) from the decoder's own keypoint
    cross-attention plumbing, which this change does not modify.

    Args:
        hidden_dim: Model width.
        num_queries: Queries selected per group.
        group_detr: Number of independent groups.
        num_classes: Class-head output width.
        num_keypoints: Keypoints per instance.

    Returns:
        A decoder-free, two-stage, GroupPose-keypoint-enabled ``Transformer`` left in its default
        training mode.

    Examples:
        >>> t = _build_two_stage_transformer_with_keypoints(16, 3, 2, 5, 4)
        >>> t._two_stage_batching_eligible()
        True
    """
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=0,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=group_detr,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[num_keypoints],
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )
    return transformer


def test_two_stage_group_selection_matches_generic_loop_for_grouppose_keypoint_config() -> None:
    """The batched fast path must also match the generic loop for a GroupPose keypoint config.

    ``_two_stage_batching_eligible`` only inspects ``enc_output``/``enc_output_norm``/
    ``enc_out_class_embed``/``enc_out_bbox_embed`` -- the same concrete types ``LWDETR.__init__``
    constructs regardless of ``use_grouppose_keypoints`` -- so the fast path also activates for
    keypoint/pose models (``RFDETRKeypointPreviewConfig`` defaults to ``group_detr=13``, same as every
    detection config). ``enc_kp_predictions`` is derived from the two-stage ``memory_ts``/``boxes_ts``
    via ``keypoint_query_initializer_enc`` and the untouched per-group ``enc_out_keypoint_embed`` loop,
    so a divergence in ``memory_ts``/``boxes_ts`` between the two paths could still surface here even
    though it does not for a plain detection model.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr, num_keypoints = 16, 5, 4, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]

    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = _build_two_stage_transformer_with_keypoints(
        hidden_dim, num_queries, group_detr, num_classes=7, num_keypoints=num_keypoints
    )
    assert transformer._two_stage_batching_eligible()

    outputs_fast = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    transformer_loop = copy.deepcopy(transformer)
    transformer_loop._two_stage_batching_eligible = lambda: False
    outputs_loop = transformer_loop(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    # return_values layout with num_decoder_layers=0: (hs=None, references=None, memory_ts, boxes_ts,
    # keypoint_hs=None, enc_kp_predictions, keypoint_memory_ts) -- see Transformer.forward's tail.
    memory_fast, boxes_fast, enc_kp_fast = outputs_fast[2], outputs_fast[3], outputs_fast[5]
    memory_loop, boxes_loop, enc_kp_loop = outputs_loop[2], outputs_loop[3], outputs_loop[5]

    assert enc_kp_fast is not None and enc_kp_loop is not None
    torch.testing.assert_close(memory_fast, memory_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(boxes_fast, boxes_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(enc_kp_fast, enc_kp_loop, atol=1e-4, rtol=1e-4)


def test_two_stage_topk_gather_selects_correct_rows_with_bbox_reparam(monkeypatch) -> None:
    """With bbox_reparam=True, the coordinate-delta reparameterisation path must still gather the exact selected rows
    via a stride-0 broadcast index, and boxes_ts must be returned un-sigmoided.

    Regression: test_two_stage_topk_gather_selects_correct_rows_out_of_position_order only exercises
    bbox_reparam=False (the ``enc_out_bbox_embed(...) + output_proposals`` branch of Transformer.forward).
    bbox_reparam=True instead builds the unselected coordinates from a cx/cy delta scaled by proposal
    size plus a log-space w/h delta, and Transformer.forward skips the final ``.sigmoid()`` on
    ``boxes_ts`` in that mode -- neither computation shares code with the bbox_reparam=False branch, so
    this covers the gather correctness independently for it.
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
        bbox_reparam=True,
        group_detr=1,
    )

    scores = torch.full((batch_size, total_hw, 1), -100.0)
    scores[0, 17, 0], scores[0, 2, 0], scores[0, 9, 0] = 30.0, 20.0, 10.0
    scores[1, 5, 0], scores[1, 19, 0], scores[1, 0, 0] = 25.0, 15.0, 5.0
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(scores)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    gather_index_calls: list[torch.Tensor] = []
    original_gather = torch.gather

    def _tracking_gather(input: torch.Tensor, dim: int, index: torch.Tensor, **kwargs: object) -> torch.Tensor:
        gather_index_calls.append(index)
        return original_gather(input, dim, index, **kwargs)

    monkeypatch.setattr(torch, "gather", _tracking_gather)

    _, _, memory_ts, boxes_ts = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert gather_index_calls, "expected torch.gather to be called during the two-stage top-k selection"
    for index in gather_index_calls:
        assert index.stride(-1) == 0, (
            "the two-stage top-k gather index must broadcast its last dim via Tensor.expand "
            f"(Transformer.forward two-stage top-k gather); got a materialised index with nonzero "
            f"last-dim stride {index.stride()} for shape {tuple(index.shape)}"
        )

    # Ground truth computed independently of the gather under test: the bbox_reparam=True coordinate
    # formula (cx/cy delta scaled by proposal size, log-space w/h delta), then plain row indexing.
    memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
    mask_flatten = torch.cat([m.flatten(1) for m in masks], 1)
    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, mask_flatten, spatial_shapes_hw, unsigmoid=False
    )
    output_memory_gidx = transformer.enc_output_norm[0](transformer.enc_output[0](output_memory))
    coord_delta = transformer.enc_out_bbox_embed[0](output_memory_gidx)
    coord_cxcy = coord_delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
    coord_wh = coord_delta[..., 2:].exp() * output_proposals[..., 2:]
    coord_unselected = torch.concat([coord_cxcy, coord_wh], dim=-1)
    chosen_idx = scores.squeeze(-1).topk(num_queries, dim=1).indices  # mirrors forward()'s torch.topk call

    assert torch.equal(chosen_idx, torch.tensor([[17, 2, 9], [5, 19, 0]]))  # sanity: out of position order
    expected_memory = torch.stack([output_memory_gidx[b, chosen_idx[b]] for b in range(batch_size)])
    # forward() returns boxes_ts as-is (no sigmoid) when bbox_reparam=True.
    expected_coord = torch.stack([coord_unselected[b, chosen_idx[b]] for b in range(batch_size)])
    assert torch.equal(memory_ts, expected_memory)
    assert torch.equal(boxes_ts, expected_coord)


def test_two_stage_topk_gather_backward_routes_gradient_only_to_selected_rows() -> None:
    """The two-stage top-k gather is a plain row copy, so backward() through it must route gradient only to the
    flattened source positions that were selected -- every unselected position must see exactly zero gradient.

    Regression: the forward-value assertions in the sibling tests confirm memory_ts/boxes_ts *equal* the
    correct rows, but a broadcast-index bug that accidentally selected the wrong rows in a way that still
    passed those value checks (e.g. by coincidence on this fixture) would still corrupt exactly which
    upstream positions receive gradient during training. Every position in this fixture's 4x4+2x2
    spatial grid is a "valid" proposal (see gen_encoder_output_proposals's 0.01-0.99 validity window), so
    output_memory is an untouched pass-through of memory and gradient must localise per-row exactly.
    """
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries = 1, 16, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(batch_size, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
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

    picks = [17, 2, 9]
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(_make_out_of_order_scores(total_hw, picks))])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    _, _, memory_ts, boxes_ts = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
    (memory_ts.sum() + boxes_ts.sum()).backward()

    assert all(src.grad is not None for src in srcs)
    # Reproduce Transformer.forward's memory flatten order: cat(src.flatten(2).transpose(1, 2)).
    flattened_grads = torch.cat([src.grad.flatten(2).transpose(1, 2) for src in srcs], dim=1)  # (1, total_hw, dim)

    selected_mask = torch.zeros(total_hw, dtype=torch.bool)
    selected_mask[torch.tensor(picks)] = True

    assert (flattened_grads[0, selected_mask] != 0).any(dim=-1).all(), (
        "every selected row must receive nonzero gradient through the two-stage top-k gather"
    )
    assert torch.equal(flattened_grads[0, ~selected_mask], torch.zeros_like(flattened_grads[0, ~selected_mask])), (
        "every unselected row must receive exactly zero gradient through the two-stage top-k gather"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_stage_topk_gather_cuda_matches_cpu_for_out_of_position_order_indices() -> None:
    """The two-stage top-k gather (Transformer.forward two-stage top-k gather) is an arithmetic-free row
    copy -- torch.gather with an index broadcast via Tensor.expand -- so CUDA must reproduce the CPU
    result bit-for-bit for the same out-of-order, per-batch-row-distinct indices used by
    test_two_stage_topk_gather_selects_correct_rows_out_of_position_order.

    Mirrors the CPU/CUDA parity twin added for the analogous arithmetic-free row-copy gather in
    PostProcess._gather_and_scale_boxes (PR #1268,
    test_gather_and_scale_boxes_cuda_matches_cpu_for_duplicated_indices): both isolate the bare
    gather+expand pattern rather than running a full model forward, so the comparison is not exposed to
    unrelated CPU/CUDA floating-point non-determinism in surrounding Linear/LayerNorm layers.
    """
    batch_size, hidden_dim, num_positions = 2, 16, 20
    source = torch.randn(batch_size, num_positions, hidden_dim)
    # Same out-of-position-order, per-batch-row-distinct picks as the CPU-only sibling test.
    topk_proposals = torch.tensor([[17, 2, 9], [5, 19, 0]])

    cpu_selected = torch.gather(source, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, hidden_dim))
    cuda_selected = torch.gather(source.cuda(), 1, topk_proposals.cuda().unsqueeze(-1).expand(-1, -1, hidden_dim))

    assert torch.equal(cpu_selected, cuda_selected.cpu())


def _bbox_from_delta(delta: torch.Tensor, proposals: torch.Tensor, bbox_reparam: bool) -> torch.Tensor:
    """Reproduce Transformer.forward's two-stage box construction from a bbox-delta MLP output.

    Args:
        delta: Raw ``enc_out_bbox_embed`` output, shape ``(..., 4)``.
        proposals: Matching ``output_proposals`` rows, shape ``(..., 4)``.
        bbox_reparam: Selects the ``bbox_reparam`` branch (cx/cy/w/h reparam vs. plain unsigmoid add).

    Returns:
        Box tensor, shape ``(..., 4)``.

    Examples:
        >>> import torch
        >>> delta = torch.zeros(1, 1, 4)
        >>> proposals = torch.full((1, 1, 4), 0.5)
        >>> torch.equal(_bbox_from_delta(delta, proposals, bbox_reparam=False), proposals)
        True
    """
    if bbox_reparam:
        return torch.cat(
            [
                delta[..., :2] * proposals[..., 2:] + proposals[..., :2],
                delta[..., 2:].exp() * proposals[..., 2:],
            ],
            dim=-1,
        )
    return delta + proposals


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_bbox_mlp_gather_order_matches_forward_and_gradient_for_real_three_layer_mlp(
    bbox_reparam: bool,
) -> None:
    """Gathering top-k rows before vs. after the bbox-delta MLP must match in both forward value and
    gradient, using the real production MLP (``rfdetr.models.math.MLP(d, d, 4, num_layers=3)``,
    matching ``LWDETR.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)`` in ``lwdetr.py``) -- not the
    single ``nn.Linear`` stand-in ``test_two_stage_topk_gather_selects_correct_rows_out_of_position_order``
    and the ``bbox_embed_input_shapes`` test above use, and covering gradient parity, which those
    two only check indirectly (nonzero/zero row pattern, not old-vs-new equality).

    The bbox MLP has no cross-token mixing (no LayerNorm/attention across the token dimension), so
    ``d(mlp(x)_i)/dx_j`` is zero for every ``j != i`` -- backward is as row-independent as forward,
    and gathering before or after the MLP must produce identical gradients w.r.t. the shared input,
    not just identical box values. Compared with a tolerance, not ``torch.equal``: three chained
    matmuls (this MLP has 3 layers, unlike the single-``nn.Linear`` sibling tests) is enough for the
    CPU BLAS backend's reduction order to differ across platforms (confirmed bit-inexact on
    macOS/Accelerate CI, in the 1e-7-to-1e-5 range for the value; bit-exact on Linux/OpenBLAS) --
    this is the same floating-point non-associativity documented for the full model scale in the PR
    body, reproduced here at a much smaller size than expected because the platform's BLAS choice
    matters more than tensor size for how many layers it takes to diverge.
    """
    torch.manual_seed(0)
    bs, sum_hw, d, num_queries = 2, 20, 16, 3
    bbox_mlp = MLP(d, d, 4, num_layers=3)
    output_memory = torch.randn(bs, sum_hw, d, requires_grad=True)
    output_proposals = torch.rand(bs, sum_hw, 4) * 0.9 + 0.05
    topk_idx = torch.stack([torch.randperm(sum_hw)[:num_queries] for _ in range(bs)])

    # Old: MLP on every row (bs, sum_hw, d), gather the box afterwards.
    box_old_full = _bbox_from_delta(bbox_mlp(output_memory), output_proposals, bbox_reparam)
    box_old = torch.gather(box_old_full, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_old.sum().backward()
    grad_old = output_memory.grad.clone()
    output_memory.grad = None

    # New (this fix): gather the selected rows first, run the MLP only on those.
    tgt_new = torch.gather(output_memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
    proposals_g = torch.gather(output_proposals, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_new = _bbox_from_delta(bbox_mlp(tgt_new), proposals_g, bbox_reparam)
    box_new.sum().backward()
    grad_new = output_memory.grad.clone()

    torch.testing.assert_close(box_old, box_new, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(grad_old, grad_new, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_bbox_mlp_gather_order_matches_on_cuda_for_real_three_layer_mlp(bbox_reparam: bool) -> None:
    """CUDA twin of test_two_stage_bbox_mlp_gather_order_matches_forward_and_gradient_for_real_three_layer_mlp:
    the same gather-before-vs-after-MLP comparison, but running the real bbox MLP itself on CUDA (not
    only the bare ``torch.gather`` isolated by
    test_two_stage_topk_gather_cuda_matches_cpu_for_out_of_position_order_indices).

    Deliberately compares old-order-on-CUDA against new-order-on-CUDA (both on the same device), not
    CUDA against CPU: CPU/CUDA cuBLAS reduction order already differs for this MLP's matmuls
    independently of gather order, which would swamp the gather-order comparison this test exists to
    make. The two paths may use different CUDA kernels and reduction orders, so this test uses a
    small numerical tolerance without mutating process-wide deterministic-algorithm state or requiring
    ``CUBLAS_WORKSPACE_CONFIG``.
    """
    torch.manual_seed(0)
    bs, sum_hw, d, num_queries = 2, 20, 16, 3
    bbox_mlp = MLP(d, d, 4, num_layers=3).cuda()
    output_memory = torch.randn(bs, sum_hw, d, device="cuda")
    output_proposals = torch.rand(bs, sum_hw, 4, device="cuda") * 0.9 + 0.05
    topk_idx = torch.stack([torch.randperm(sum_hw, device="cuda")[:num_queries] for _ in range(bs)])

    box_old_full = _bbox_from_delta(bbox_mlp(output_memory), output_proposals, bbox_reparam)
    box_old = torch.gather(box_old_full, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

    tgt_new = torch.gather(output_memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
    proposals_g = torch.gather(output_proposals, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_new = _bbox_from_delta(bbox_mlp(tgt_new), proposals_g, bbox_reparam)

    torch.testing.assert_close(box_old, box_new, atol=1e-5, rtol=1e-4)


class _ShapeRecordingLinear(nn.Module):
    """Wraps a real ``nn.Linear`` and records the shape of every input it is called with.

    Lets a test assert how many rows the wrapped layer actually processed, independent of the values
    it produced (already covered by the ``test_two_stage_topk_gather_selects_correct_rows_*`` tests).

    Examples:
        >>> input_shapes = []
        >>> layer = _ShapeRecordingLinear(nn.Linear(2, 3), input_shapes)
        >>> layer(torch.zeros(1, 2)).shape
        torch.Size([1, 3])
        >>> input_shapes
        [torch.Size([1, 2])]
    """

    def __init__(self, inner: nn.Linear, input_shapes: list[torch.Size]) -> None:
        """Initialize the recording wrapper around a linear layer.

        Args:
            inner: Linear layer whose input shapes should be recorded.
            input_shapes: Mutable list receiving each input shape in call order.
        """
        super().__init__()
        self.inner = inner
        self._input_shapes = input_shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record ``x.shape`` then delegate to the wrapped ``nn.Linear``."""
        self._input_shapes.append(x.shape)
        return self.inner(x)


@pytest.mark.parametrize("group_detr", [1, 3])
@pytest.mark.parametrize("bbox_reparam", [False, True])
@pytest.mark.parametrize("num_queries", [3, 20])
def test_two_stage_bbox_embed_only_runs_on_selected_rows_not_full_encoder_memory(
    group_detr: int, bbox_reparam: bool, num_queries: int
) -> None:
    """The bbox-delta MLP must only run on the rows torch.topk selects, not on every encoder position.

    ``enc_out_bbox_embed`` is a pointwise MLP with no cross-token mixing, so it only needs the
    ``num_queries`` rows that survive ``torch.topk`` selection -- every one of the other
    ``sum(H*W) - num_queries`` encoder positions it used to also run on was discarded by the gather
    that immediately followed, per group.

    Regression: prior to this fix, Transformer.forward ran ``enc_out_bbox_embed[g_idx]`` on the full
    ``output_memory_gidx`` (bs, sum_hw, d) for every one of ``group_detr`` groups and only kept the
    ``num_queries`` gathered rows -- this pins the shape ``enc_out_bbox_embed`` is actually called
    with to the post-topk size, per group. Row *values* are covered separately by
    test_two_stage_topk_gather_selects_correct_rows_out_of_position_order and
    test_two_stage_topk_gather_broadcasts_correctly_across_groups_in_training_mode, which assert
    ``memory_ts``/``boxes_ts`` are bit-identical to gathering after running the MLP on every row --
    this test only pins how much work the MLP itself does, not what it produces.

    Parametrized over ``bbox_reparam`` because it is not merely a config toggle here: the fixed
    (production default per ``ModelConfig.bbox_reparam``, ``config.py``) ``True`` branch runs
    additional pointwise ops (``.exp()``, multiply, ``torch.concat``) on ``enc_out_bbox_embed``'s
    *output* that ``False`` does not, so a test pinning only ``False`` would leave the branch that
    is actually used in production unchecked.
    """
    torch.manual_seed(0)
    hidden_dim = 16
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)
    assert total_hw >= num_queries  # also cover the topk == encoder-memory boundary

    srcs = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(1, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

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
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    assert transformer.training  # default nn.Module state; group_detr>1 only takes effect while training

    num_classes = 5
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    bbox_embed_input_shapes: list[torch.Size] = []
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [_ShapeRecordingLinear(nn.Linear(hidden_dim, 4), bbox_embed_input_shapes) for _ in range(group_detr)]
    )

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert len(bbox_embed_input_shapes) == group_detr
    for shape in bbox_embed_input_shapes:
        assert shape == torch.Size([1, num_queries, hidden_dim]), (
            "enc_out_bbox_embed must only run on the num_queries rows selected by torch.topk, not the "
            f"full sum(H*W)={total_hw} encoder positions (Transformer.forward two-stage top-k gather); "
            f"got input shape {tuple(shape)}"
        )


def _make_cuda_graph_transformer_inputs(
    spatial_shapes_hw: list[tuple[int, int]], batch_size: int = 2, hidden_dim: int = 16, num_queries: int = 3
) -> tuple[Transformer, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build a small two-stage Transformer and matching forward inputs.

    Examples:
        >>> transformer, srcs, *_ = _make_cuda_graph_transformer_inputs([(2, 2)])
        >>> len(srcs), transformer.num_feature_levels
        (1, 1)
    """
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
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 5)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])
    srcs = [torch.randn(batch_size, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    masks = [torch.zeros(batch_size, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
    pos_embeds = [torch.randn(batch_size, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)
    return transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat


def test_cuda_graph_spatial_shapes_cache_reuses_tensor_per_device_and_resolution() -> None:
    """Repeated capture warmups reuse the exact device shape tensor."""
    transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    transformer.enable_cuda_graph_capture()

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)
    assert transformer._cuda_graph_spatial_shapes is not None
    assert len(transformer._cuda_graph_spatial_shapes) == 1
    cached = next(iter(transformer._cuda_graph_spatial_shapes.values()))

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)
    assert next(iter(transformer._cuda_graph_spatial_shapes.values())) is cached


def test_cuda_graph_spatial_shapes_cache_keys_distinct_resolutions() -> None:
    """Each multi-scale resolution gets its own static shape tensor."""
    transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    transformer.enable_cuda_graph_capture()
    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    _, srcs_b, masks_b, pos_embeds_b, _, _ = _make_cuda_graph_transformer_inputs([(6, 6), (3, 3)])
    transformer(srcs_b, masks_b, pos_embeds_b, refpoint_embed, query_feat)

    assert transformer._cuda_graph_spatial_shapes is not None
    assert {key[1] for key in transformer._cuda_graph_spatial_shapes} == {
        ((4, 4), (2, 2)),
        ((6, 6), (3, 3)),
    }


def test_cuda_graph_spatial_shapes_cache_preserves_forward_values() -> None:
    """The cache changes tensor construction, not model arithmetic."""
    torch.manual_seed(7)
    baseline, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    baseline_output = baseline(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    torch.manual_seed(7)
    cached, *_ = _make_cuda_graph_transformer_inputs([(4, 4), (2, 2)])
    cached.enable_cuda_graph_capture()
    cached_output = cached(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    for expected, actual in zip(baseline_output, cached_output):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)
