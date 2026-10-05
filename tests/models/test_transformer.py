# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for transformer utilities, MS deformable attention core, and MSDeformAttn module."""

import copy
import io
from collections.abc import Callable, Iterator
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.hooks import RemovableHandle

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model, build_model_from_config
from rfdetr.models.math import MLP
from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    _AddInDtype,
    _CastThenExpand,
    _InterleavedSinCos,
    _is_tracing,
    _LinearReLU,
    _module_call_is_plain,
    _sineembed_interleaved,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
)
from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner
from rfdetr.utilities.tensors import NestedTensor, _bilinear_grid_sample


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


@pytest.mark.parametrize(
    ("height", "width", "padded"),
    [
        pytest.param(3, 5, False, id="3x5-unpadded"),
        pytest.param(3, 5, True, id="3x5-padded"),
        pytest.param(1, 1, False, id="1x1-unpadded"),
        pytest.param(1, 3, False, id="1x3-unpadded"),
    ],
)
def test_gen_encoder_output_proposals_centres_each_cell(height: int, width: int, padded: bool) -> None:
    """Unsigmoided proposal centres are ``((x + 0.5) / W, (y + 0.5) / H)`` over the unpadded region, 0 where padded.

    The grid behind the proposals feeds every path (eager, ``torch.compile`` and each export format), so its values
    are pinned here independently of how it is built: an off-by-one grid such as ``arange(1, n + 1)`` would shift every
    proposal by one cell without changing any shape. The ``1x1``/``1x3`` cases pin the height/width == 1 boundary,
    where a centring formula that divides by ``(dim - 1)`` instead of ``dim`` would divide by zero or shift the grid.
    """
    valid_height, valid_width = (2, 3) if padded else (height, width)
    padding_mask = torch.ones(1, height, width, dtype=torch.bool)
    padding_mask[:, :valid_height, :valid_width] = False
    expected = torch.tensor(
        [
            [(x + 0.5) / valid_width, (y + 0.5) / valid_height] if y < valid_height and x < valid_width else [0.0, 0.0]
            for y in range(height)
            for x in range(width)
        ]
    )

    _, proposals = gen_encoder_output_proposals(
        torch.randn(1, height * width, 8),
        padding_mask.flatten(1) if padded else None,
        [(height, width)],
        unsigmoid=False,
    )

    torch.testing.assert_close(proposals[0, :, :2], expected)


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

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )

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


def test_two_stage_topk_gather_reads_pre_norm_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval-path two-stage gather must read ``enc_output``'s output, never ``enc_output_norm``'s.

    Regression for an Apple Neural Engine compile failure: in an fp16 CoreML export, an ``enc_output_norm``
    output that feeds the ANE-resident class head and also crosses to the CPU-resident ``topk``-indexed gather
    makes the ANE compiler reject the whole program ("Invalid layer") when that norm still has its identity
    affine and the encoder token count is a multiple of 32. The model then runs entirely on CPU, or fails to
    load with ``ComputeUnit.ALL``. Trained checkpoints have a non-identity affine and are unaffected, so this
    reaches models exported before training — the export tests among them. Gathering the pre-norm rows and
    normalizing only the selected tokens is exact, because LayerNorm acts per token; the selected rows
    themselves are pinned by ``test_two_stage_topk_gather_selects_correct_rows_out_of_position_order``.
    """
    hidden_dim, num_queries, feature_size = 16, 3, 4
    num_tokens = feature_size * feature_size
    srcs = [torch.randn(1, hidden_dim, feature_size, feature_size)]
    masks = [torch.zeros(1, feature_size, feature_size, dtype=torch.bool)]
    pos_embeds = [torch.randn(1, hidden_dim, feature_size, feature_size)]
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

    norm_outputs: list[torch.Tensor] = []
    transformer.enc_output_norm[0].register_forward_hook(lambda _m, _i, out: norm_outputs.append(out))
    gather_inputs: list[torch.Tensor] = []
    original_gather = torch.gather

    def _tracking_gather(input: torch.Tensor, dim: int, index: torch.Tensor, **kwargs: object) -> torch.Tensor:
        gather_inputs.append(input)
        return original_gather(input, dim, index, **kwargs)

    monkeypatch.setattr(torch, "gather", _tracking_gather)

    transformer.eval()
    with torch.no_grad():
        transformer(srcs, masks, pos_embeds, torch.rand(num_queries, 4), torch.randn(num_queries, hidden_dim))

    full_length_norm_outputs = [out for out in norm_outputs if out.shape[1] == num_tokens]
    assert full_length_norm_outputs, "expected enc_output_norm to run over every encoder position"
    assert gather_inputs, "expected torch.gather to be called during the two-stage top-k selection"
    norm_storages = {out.untyped_storage().data_ptr() for out in full_length_norm_outputs}
    # Compare storage, not identity, so a view of the norm output (reshape/slice/expand) is caught as well.
    assert not any(inp.untyped_storage().data_ptr() in norm_storages for inp in gather_inputs), (
        "the two-stage gather reads enc_output_norm's output; gather the pre-norm enc_output rows and "
        "normalize the selected tokens instead (fp16 CoreML ANE compile failure)"
    )


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

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )

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
    hidden_dim: int,
    num_queries: int,
    group_detr: int,
    num_classes: int,
    bbox_reparam: bool,
    num_feature_levels: int = 2,
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
        num_feature_levels: Number of feature-map levels the decoder's deformable attention expects; the caller's
            ``srcs``/``masks``/``pos_embeds`` lists must have this many entries.

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
        num_feature_levels=num_feature_levels,
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


def test_two_stage_group_selection_class_logits_reuse_matches_recomputed_loop() -> None:
    """``LWDETR.forward``'s ``enc_outputs["pred_logits"]`` must match whether it comes from the batched path's gathered
    ``class_logits_all`` (reused, no second forward) or the untouched per-group ``enc_out_class_embed`` recompute loop
    it replaces on the eligible path.

    ``enc_out_class_embed`` is a plain per-position ``nn.Linear``, so gathering its output at the
    positions ``_two_stage_group_selection`` already selected is algebraically the same computation
    the recompute loop performs on the same gathered hidden state -- this only differs in GEMM
    accumulation order (stacked-groups batched matmul vs. one matmul per group), so results are
    compared within float32 tolerance rather than bit-for-bit, matching this file's own established
    convention for the sibling batched-vs-loop comparisons above.

    Also proves ``enc_out_class_embed``'s parameters receive real gradient through this reused value
    directly -- unlike the memory_ts/boxes_ts-only comparisons above, whose own docstrings note
    ``enc_out_class_embed`` structurally never received gradient through that path -- and, following
    this file's ``test_two_stage_group_selection_matches_generic_loop_forward_and_gradient`` convention
    of comparing fast-vs-loop gradients rather than only asserting the fast path's own gradient is
    non-vacuous, that those gradients match the recompute loop's within the same tolerance. A gather
    that silently detached ``class_logits_all`` (or indexed the wrong positions in a way that happened
    to still produce finite, nonzero, but wrong values) would still pass a same-path-only non-vacuity
    check; comparing against the loop's independently computed gradient closes that gap.
    """
    torch.manual_seed(0)
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=7, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    model_fast = build_model(ns)
    assert model_fast.transformer._two_stage_batching_eligible()
    state = copy.deepcopy(model_fast.state_dict())

    model_loop = build_model(ns)
    model_loop.load_state_dict(state)
    model_loop.transformer._two_stage_batching_eligible = lambda: False
    assert not model_loop.transformer._two_stage_batching_eligible()

    model_fast.train()
    model_loop.train()

    samples = NestedTensor(torch.randn(1, 3, 256, 256), torch.zeros(1, 256, 256, dtype=torch.bool))
    out_fast = model_fast(samples)
    out_loop = model_loop(samples)

    torch.testing.assert_close(
        out_fast["enc_outputs"]["pred_logits"], out_loop["enc_outputs"]["pred_logits"], atol=1e-4, rtol=1e-4
    )

    fast_class_embed_parameters = [p for m in model_fast.transformer.enc_out_class_embed for p in m.parameters()]
    loop_class_embed_parameters = [p for m in model_loop.transformer.enc_out_class_embed for p in m.parameters()]
    fast_gradients = torch.autograd.grad(out_fast["enc_outputs"]["pred_logits"].sum(), fast_class_embed_parameters)
    loop_gradients = torch.autograd.grad(out_loop["enc_outputs"]["pred_logits"].sum(), loop_class_embed_parameters)
    for fast_gradient, loop_gradient in zip(fast_gradients, loop_gradients, strict=True):
        assert torch.isfinite(fast_gradient).all()
        assert fast_gradient.abs().sum() > 0
        torch.testing.assert_close(fast_gradient, loop_gradient, atol=1e-4, rtol=1e-4)


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

    _, _, memory_fast, boxes_fast, _ = transformer(
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
    _, _, memory_loop, boxes_loop, _ = transformer_loop(
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
        _, _, memory_ts, boxes_ts, cls_ts = transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
        )
        loss = memory_ts.float().sum() + boxes_ts.float().sum() + cls_ts.float().sum()

    assert torch.isfinite(memory_ts).all()
    assert torch.isfinite(boxes_ts).all()
    # cls_ts is enc_out_class_embed's output gathered by the batched path (module-level docstring
    # above), the same value the class-logits-reuse mechanism forwards as enc_outputs["pred_logits"]
    # instead of recomputing -- the mechanism this benchmark's BF16 training step actually exercises.
    assert torch.isfinite(cls_ts).all()

    # memory_ts/boxes_ts/cls_ts are the two-stage encoder outputs only (the decoder's own parameters
    # correctly get no gradient from this loss), so only check the modules _two_stage_group_selection
    # actually uses for a DIFFERENTIABLE output. Unlike the memory_ts/boxes_ts-only version of this
    # test's loss, cls_ts is now included, so enc_out_class_embed's parameters are no longer excluded:
    # this loss is the first one in this function to route gradient through the reused gather.
    two_stage_modules = [
        *transformer.enc_output,
        *transformer.enc_output_norm,
        *transformer.enc_out_bbox_embed,
        *transformer.enc_out_class_embed,
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
    class_embed_parameter_ids = {id(p) for m in transformer.enc_out_class_embed for p in m.parameters()}
    class_embed_gradients = [
        gradient
        for (name, parameter), gradient in zip(named_parameters, gradients[len(srcs) :], strict=True)
        if id(parameter) in class_embed_parameter_ids
    ]
    assert class_embed_gradients, "enc_out_class_embed contributed no parameters to this check"
    assert any(gradient.abs().sum() > 0 for gradient in class_embed_gradients), (
        "enc_out_class_embed must receive real gradient through the reused cls_ts under bf16 autocast"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "spatial_shapes_hw",
    [
        pytest.param([(8, 8), (4, 4)], id="2-level"),
        pytest.param([(8, 8)], id="1-level"),
    ],
)
def test_two_stage_group_selection_compiles_with_finite_gradients(spatial_shapes_hw: list[tuple[int, int]]) -> None:
    """The batched fast path must be compatible with torch.compile, at both 1 and 2 feature levels.

    Unlike ONNX/TorchScript export (which always forces `group_detr=1` and never reaches this code), `module_model.py`
    applies `torch.compile` directly to the training model, where `group_detr>1` and this new path are exactly what
    training exercises. This asserts compile-time and run-time compatibility (finite outputs and gradients through a
    compiled call, using the same `capture_scalar_outputs` config `module_model.py` itself sets before compiling) -- not
    a timing claim; a local single-GPU smoke run is not a substitute for this project's separately reported L4 step-time
    evidence. Production RF-DETR models (Base, Small, Nano, Medium, Large) use a single feature level, so the 1-level
    case exercises the `is_compiling()` stacked-scalar_tensor spatial_shapes branch at the level count training actually
    compiles most, not only the 2-level case the other real-compile coverage in this file used exclusively.
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 5, 4
    device = "cuda"

    srcs = [torch.randn(2, hidden_dim, ht, wd, device=device, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool, device=device) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd, device=device) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4, device=device)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim, device=device)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=True, num_feature_levels=len(spatial_shapes_hw)
    ).to(device)
    assert transformer._two_stage_batching_eligible()
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled_transformer = torch.compile(transformer, dynamic=True)

        _, _, memory_ts, boxes_ts, _ = compiled_transformer(
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


def test_dynamic_compile_reuses_graph_across_resolutions() -> None:
    """A new input resolution must not compile a new Transformer graph under ``dynamic=True``.

    Multi-scale training feeds up to 11 resolutions through one ``torch.compile(dynamic=True)`` model. Building
    ``spatial_shapes`` with ``torch.as_tensor`` of the (H, W) pairs, or the encoder-proposal grid with
    ``torch.linspace(0, n - 1, n)``, specialises H and W to their traced values, so every resolution recompiled this
    frame until Dynamo's recompile limit (8) left the remaining resolutions eager. The ``eager`` backend exercises the
    Dynamo guards that decide recompilation without paying for Inductor, and ``capture_scalar_outputs`` mirrors the
    training compile setup in ``module_model.py``. The first-traced sizes avoid 0 and 1, which Dynamo would specialise
    to constants, and differ from the other input dimensions, which duck sizing would tie to a shared symbol. Counts are
    taken relative to the global Dynamo counter, and the first resolution must add a graph, so the test also fails if
    nothing gets compiled at all. Every resolution is non-square (H != W) and each level within a resolution swaps
    which axis is larger, so a [W, H] axis swap anywhere in the compiled ``spatial_shapes`` construction --
    undetectable by the graph-count assertion alone -- would still change the numeric output compared against the
    eager, uncompiled ``transformer`` (built with the default ``dropout=0.0``, so its forward is deterministic and
    directly comparable against the compiled call on the same inputs).
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=True
    )
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)
    graph_counts = []
    baseline = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled_transformer = torch.compile(transformer, dynamic=True, backend="eager")
        for spatial_shapes_hw in ([(10, 14), (5, 7)], [(12, 20), (6, 10)], [(18, 14), (9, 7)]):
            srcs = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
            masks = [torch.zeros(2, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
            pos_embeds = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
            compiled_out = compiled_transformer(
                srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
            )
            graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"] - baseline)
            uncompiled_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
            torch.testing.assert_close(compiled_out, uncompiled_out)

    assert graph_counts[0] > 0 and graph_counts[1:] == [graph_counts[0]] * 2, f"graphs per resolution: {graph_counts}"


def test_is_exporting_branch_builds_spatial_shapes_with_as_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Transformer.forward`` must build ``spatial_shapes`` via ``torch.as_tensor`` while ``is_exporting()`` is True.

    ``torch.export`` (ExecuTorch) cannot trace ``torch._shape_as_tensor`` (the eager branch's op), and the
    ``is_compiling()`` branch stacks per-size ``torch.scalar_tensor`` calls to stay symbolic under ``torch.compile`` --
    neither is what ``torch.export`` needs. The ``is_exporting()`` branch exists to route around both by baking
    ``spatial_shapes`` from the concrete (H, W) pairs with a single ``torch.as_tensor`` call. This branch had zero test
    coverage before and after this PR. Mirrors the ``is_compiling()`` polyfill regression
    (``test_spatial_shapes_survives_dynamo_shape_as_tensor_polyfill``): monkeypatch the compile-state predicate, spy on
    both candidate tensor constructors to pin exactly which branch ran, and check the output still matches an
    unpatched eager run.
    """
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    spatial_shapes_hw = [(6, 8), (3, 4)]
    srcs = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    masks = [torch.zeros(2, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    with torch.no_grad():
        eager_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    original_as_tensor = torch.as_tensor
    as_tensor_calls = 0
    scalar_tensor_calls = 0

    def _spy_as_tensor(*args, **kwargs):
        nonlocal as_tensor_calls
        as_tensor_calls += 1
        return original_as_tensor(*args, **kwargs)

    def _spy_scalar_tensor(*args, **kwargs):
        nonlocal scalar_tensor_calls
        scalar_tensor_calls += 1
        raise AssertionError("is_exporting() branch must not build spatial_shapes with torch.scalar_tensor")

    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True, raising=False)
    monkeypatch.setattr(torch, "as_tensor", _spy_as_tensor)
    monkeypatch.setattr(torch, "scalar_tensor", _spy_scalar_tensor)

    with torch.no_grad():
        exporting_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert as_tensor_calls == 1, "is_exporting() branch must build spatial_shapes with exactly one torch.as_tensor call"
    assert scalar_tensor_calls == 0
    torch.testing.assert_close(eager_out, exporting_out)


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

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )

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

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
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


# ---------------------------------------------------------------------------------------------------
# Eager decoder-layer elementwise diet: sine embedding, fused positional add, grouped self-attention.
# ---------------------------------------------------------------------------------------------------


def _reference_sineembed(pos_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """The interleaved-slice formulation ``gen_sineembed_for_position`` replaced, kept as the oracle.

    Examples:
        >>> _reference_sineembed(torch.zeros(1, 2, 4), 4).shape
        torch.Size([1, 2, 16])
    """
    scale = 2 * torch.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    def embed(coord: torch.Tensor) -> torch.Tensor:
        pos = (coord * scale)[:, :, None] / dim_t
        return torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)

    parts = [embed(pos_tensor[:, :, 1]), embed(pos_tensor[:, :, 0])]
    if pos_tensor.size(-1) == 4:
        parts += [embed(pos_tensor[:, :, 2]), embed(pos_tensor[:, :, 3])]
    return torch.cat(parts, dim=2)


#: Marks for a case that needs CUDA: excluded from the CPU job, skipped without a GPU.
_CUDA_MARKS = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")]


@pytest.fixture(params=["highest", "high"])
def float32_matmul_precision(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run a test under each float32 matmul precision a process can be left in, then restore the previous one.

    ``"highest"`` is true fp32 and ``"high"`` allows TF32 GEMMs; ``build_trainer`` sets ``"high"`` for the process, so
    production runs under it and a bitwise comparison of two fp32 GEMM routes must hold under both.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``request``):

        >>> float32_matmul_precision  # doctest: +SKIP
        <pytest_fixture(<function float32_matmul_precision at 0x...>)>
    """
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(request.param)
    yield request.param
    torch.set_float32_matmul_precision(previous_precision)


def _cpu_addmm_activation_available(dtype: torch.dtype) -> bool:
    """Return whether ``torch._addmm_activation`` runs on CPU in ``dtype`` on the installed torch.

    ``_LinearReLU`` is built on this private op, whose CPU kernels are not part of torch's documented surface, so the
    CPU cases of its test are collected only where the op exists.

    Examples:
        >>> isinstance(_cpu_addmm_activation_available(torch.float32), bool)
        True
    """
    try:
        torch._addmm_activation(
            torch.zeros(2, dtype=dtype), torch.zeros(2, 2, dtype=dtype), torch.zeros(2, 2, dtype=dtype)
        )
    except (AttributeError, RuntimeError, NotImplementedError):
        return False
    return True


def _interleaved_sineembed(
    pos_tensor: torch.Tensor, dim: int = 128, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Call the CUDA-eager ``_sineembed_interleaved`` kernel directly, with ``gen_sineembed_for_position``'s defaults.

    The kernel is plain torch ops plus a custom autograd function, so it also runs on CPU, where the public function
    never takes it.

    Examples:
        >>> _interleaved_sineembed(torch.zeros(1, 2, 4), dim=4).shape
        torch.Size([1, 2, 16])
    """
    return _sineembed_interleaved(pos_tensor, dim, out_dtype)


#: ``(device, sine-embedding function)`` pairs. On CPU the public function takes the plain ops, so the kernel is also
#: called directly there; on CUDA the public function takes the kernel itself.
_SINEEMBED_ROUTES = [
    pytest.param("cpu", gen_sineembed_for_position, id="cpu-plain"),
    pytest.param("cpu", _interleaved_sineembed, id="cpu-interleaved-kernel"),
    pytest.param("cuda", gen_sineembed_for_position, id="cuda-public", marks=_CUDA_MARKS),
]

#: Every hook ``Module.__call__`` runs around ``forward``, as ``(module, hook) -> handle``: the four per-module
#: kinds and the four registered for all modules through ``torch.nn.modules.module``.
_MODULE_HOOK_REGISTRARS: dict[str, Callable[[nn.Module, Callable[..., object]], RemovableHandle]] = {
    "forward_hook": lambda module, hook: module.register_forward_hook(hook),
    "forward_pre_hook": lambda module, hook: module.register_forward_pre_hook(hook),
    "backward_hook": lambda module, hook: module.register_full_backward_hook(hook),
    "backward_pre_hook": lambda module, hook: module.register_full_backward_pre_hook(hook),
    "global_forward_hook": lambda _module, hook: nn.modules.module.register_module_forward_hook(hook),
    "global_forward_pre_hook": lambda _module, hook: nn.modules.module.register_module_forward_pre_hook(hook),
    "global_backward_hook": lambda _module, hook: nn.modules.module.register_module_full_backward_hook(hook),
    "global_backward_pre_hook": lambda _module, hook: nn.modules.module.register_module_full_backward_pre_hook(hook),
}


@pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
@pytest.mark.parametrize("box_width", [2, 4])
@pytest.mark.parametrize("dim", [4, 128])
def test_gen_sineembed_for_position_is_bitwise_the_interleaved_slice_formulation(
    box_width: int, dim: int, device: str, sineembed: Callable[..., torch.Tensor]
) -> None:
    """Dividing by the ``dim // 2`` distinct frequencies once must reproduce every bit of the strided-slice
    version: the even and odd entries of ``dim_t`` are the same float, so the angles are the same values."""
    pos_tensor = torch.rand(3, 517, box_width, dtype=torch.float32, device=device)

    actual = sineembed(pos_tensor, dim=dim)

    assert actual.shape == (3, 517, box_width * dim)
    assert torch.equal(actual, _reference_sineembed(pos_tensor, dim))


@pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
def test_gen_sineembed_for_position_backward_matches_reference(
    device: str, sineembed: Callable[..., torch.Tensor]
) -> None:
    """The contiguous formulation changes the summation order of the coordinate gradient, nothing else."""
    pos_tensor = torch.rand(2, 64, 4, dtype=torch.float64, device=device)
    grad_out = torch.randn(2, 64, 4 * 32, dtype=torch.float64, device=device)
    expected_input = pos_tensor.clone().requires_grad_(True)
    actual_input = pos_tensor.clone().requires_grad_(True)

    _reference_sineembed(expected_input, 32).backward(grad_out)
    sineembed(actual_input, dim=32).backward(grad_out)

    assert expected_input.grad is not None and actual_input.grad is not None
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "pos_in_compute_dtype", [pytest.param(False, id="fp32-pos"), pytest.param(True, id="compute-pos")]
)
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")])
def test_add_in_dtype_is_bitwise_the_add_then_cast_graph(dtype: torch.dtype, pos_in_compute_dtype: bool) -> None:
    """One kernel, same bits: forward equals ``(a + b).to(dtype)``, backward equals that graph's gradients."""
    pos_dtype = dtype if pos_in_compute_dtype else torch.float32
    tensor = torch.randn(2, 30, 8, dtype=torch.float32, requires_grad=True)
    pos = torch.randn(2, 30, 8).to(pos_dtype).requires_grad_(True)
    grad_out = torch.randn(2, 30, 8).to(dtype)

    expected = (tensor + pos).to(dtype)
    expected.backward(grad_out)
    expected_grads = (tensor.grad, pos.grad)
    tensor.grad = pos.grad = None

    actual = _AddInDtype.apply(tensor, pos, dtype)
    actual.backward(grad_out)

    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert tensor.grad is not None and pos.grad is not None
    assert expected_grads[0] is not None and expected_grads[1] is not None
    assert tensor.grad.dtype == torch.float32 and pos.grad.dtype == pos_dtype
    assert torch.equal(tensor.grad, expected_grads[0])
    assert torch.equal(pos.grad, expected_grads[1])


def _decoder_layer(
    dropout: float = 0.0, *, d_model: int = 16, heads: int = 4, group_detr: int = 3
) -> TransformerDecoderLayer:
    """A small decoder layer in training mode, three groups of 4-head 16-wide attention by default.

    Args:
        dropout: Dropout probability of the layer.
        d_model: Embedding width; must be divisible by ``heads``.
        heads: Head count of both the self- and the cross-attention.
        group_detr: Number of query groups.

    Examples:
        >>> _decoder_layer().group_detr
        3
        >>> _decoder_layer(d_model=12, heads=1, group_detr=1).self_attn.num_heads
        1
    """
    return TransformerDecoderLayer(
        d_model=d_model,
        sa_nhead=heads,
        ca_nhead=heads,
        dim_feedforward=32,
        dropout=dropout,
        group_detr=group_detr,
        num_feature_levels=2,
    ).train()


@pytest.mark.parametrize("case", [*_MODULE_HOOK_REGISTRARS, "instance_forward", "compiled_call_impl"])
def test_module_call_is_plain_is_false_when_the_call_is_observed_or_overridden(
    case: str, request: pytest.FixtureRequest
) -> None:
    """Everything ``Module.__call__`` can do beyond ``forward`` makes the call non-plain.

    The predicate is device independent, so this runs on CPU. A per-module feature spares an unrelated module (the
    negative counterexample) while a global hook applies to every module. The plain control comes first so that no case
    passes because the predicate is always ``False``.
    """
    observed, bystander = nn.Linear(4, 4), nn.Linear(4, 4)
    assert _module_call_is_plain(observed, bystander) is True
    if case == "instance_forward":
        observed.forward = Mock(side_effect=observed.forward)  # type: ignore[method-assign]
    elif case == "compiled_call_impl":
        observed._compiled_call_impl = Mock()
    else:
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[case](observed, Mock(return_value=None)).remove)

    assert _module_call_is_plain(observed, bystander) is False
    assert _module_call_is_plain(bystander) is (not case.startswith("global_"))


def test_pos_embed_for_linear_is_the_plain_add_outside_cuda_autocast() -> None:
    """Without CUDA autocast there is no cast to fold, so the helper is exactly ``with_pos_embed``."""
    layer = _decoder_layer()
    tensor = torch.randn(2, 12, 16)
    pos = torch.randn(2, 12, 16)

    assert layer._pos_embed_for_linear(tensor, None) is tensor
    assert torch.equal(layer._pos_embed_for_linear(tensor, pos), tensor + pos)
    assert layer._pos_embed_for_linear(tensor, pos).dtype == torch.float32


def _module_path_self_attention(
    layer: TransformerDecoderLayer, tgt: torch.Tensor, query_pos: torch.Tensor
) -> torch.Tensor:
    """The ``forward_post`` self-attention block as written around the ``nn.MultiheadAttention`` call.

    Examples:
        >>> _module_path_self_attention(_decoder_layer(), torch.zeros(2, 12, 16), torch.zeros(2, 12, 16)).shape
        torch.Size([2, 12, 16])
    """
    bs, num_queries, _ = tgt.shape
    q = layer.with_pos_embed(tgt, query_pos)
    q = torch.cat(q.split(num_queries // layer.group_detr, dim=1), dim=0)
    v = torch.cat(tgt.split(num_queries // layer.group_detr, dim=1), dim=0)
    tgt2 = layer.self_attn(q, q, v, need_weights=False)[0]
    return torch.cat(tgt2.split(bs, dim=0), dim=1)


@pytest.mark.parametrize(
    ("group_detr", "batch", "queries_per_group", "heads", "d_model"),
    [
        pytest.param(3, 2, 4, 4, 16, id="baseline"),
        pytest.param(1, 2, 4, 4, 16, id="single-group"),
        pytest.param(3, 1, 4, 4, 16, id="single-image"),
        pytest.param(3, 2, 1, 4, 16, id="one-query-per-group"),
        pytest.param(3, 2, 4, 1, 16, id="single-head"),
        pytest.param(3, 2, 4, 4, 12, id="width-not-a-multiple-of-8"),
        pytest.param(13, 2, 5, 8, 32, id="many-groups-odd-queries"),
        pytest.param(1, 1, 1, 1, 8, id="all-singletons"),
    ],
)
def test_grouped_self_attention_is_bitwise_the_module_path(
    group_detr: int, batch: int, queries_per_group: int, heads: int, d_model: int
) -> None:
    """Projections on the ungrouped layout plus a batch-major view must reproduce the group-major ``cat``
    path bit for bit in the forward and in the input gradients: every kernel sees the same rows in a
    different order. The weight gradients reduce over those rows, so their summation order changes;
    they are checked to fp32 rounding here and bitwise on CUDA in the GPU test.

    The group count, batch size, queries per group, head count and width each take their singleton or an
    odd value in turn, since the regrouping views are where a boundary shape would go wrong.
    """
    layer = _decoder_layer(group_detr=group_detr, heads=heads, d_model=d_model)
    num_queries = group_detr * queries_per_group
    tgt = torch.randn(batch, num_queries, d_model, requires_grad=True)
    query_pos = torch.randn(batch, num_queries, d_model, requires_grad=True)
    grad_out = torch.randn(batch, num_queries, d_model)

    expected = _module_path_self_attention(layer, tgt, query_pos)
    expected.backward(grad_out)
    expected_input_grads = [tgt.grad, query_pos.grad]
    expected_param_grads = [p.grad for p in layer.self_attn.parameters()]
    tgt.grad = query_pos.grad = None
    layer.zero_grad(set_to_none=True)

    actual = layer._grouped_self_attention(tgt, query_pos)
    actual.backward(grad_out)
    actual_input_grads = [tgt.grad, query_pos.grad]
    actual_param_grads = [p.grad for p in layer.self_attn.parameters()]

    assert torch.equal(actual, expected)
    for expected_grad, actual_grad in zip(expected_input_grads, actual_input_grads, strict=True):
        assert expected_grad is not None and actual_grad is not None
        assert torch.equal(actual_grad, expected_grad)
    for expected_grad, actual_grad in zip(expected_param_grads, actual_param_grads, strict=True):
        assert expected_grad is not None and actual_grad is not None
        # Reassociating a sum of ``batch * num_queries`` fp32 terms moves it by rounding proportional to the
        # gradient's own magnitude, which grows with the row count, so the bound scales with the largest entry.
        atol = 1e-6 + 1e-6 * float(expected_grad.abs().max())
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=atol)


def test_grouped_self_attention_uses_a_view_for_the_regrouping() -> None:
    """A wrong batch order (group-major view) would scramble rows across images: pin the batch-major one by checking
    that image 1's queries only attend within image 1."""
    layer = _decoder_layer()
    tgt = torch.randn(2, 12, 16)
    query_pos = torch.zeros(2, 12, 16)
    tgt_swapped = tgt.flip(0)

    with torch.no_grad():
        out = layer._grouped_self_attention(tgt, query_pos)
        out_swapped = layer._grouped_self_attention(tgt_swapped, query_pos)

    assert torch.equal(out_swapped, out.flip(0))


class _MultiheadAttentionSubclass(nn.MultiheadAttention):
    """Test double that must keep the generic multi-head-attention call path."""

    pass


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "case",
    [
        "eval",
        "tgt_mask",
        "key_padding_mask",
        "subclass",
        "forward_hook",
        "forward_pre_hook",
        "backward_hook",
        "backward_pre_hook",
        "global_forward_hook",
        "global_backward_hook",
        "instance_forward",
        "compiled_call_impl",
        "dropout",
        "indivisible_queries",
        "compiling",
        "tracing",
        "no_in_proj_bias",
        "add_bias_kv",
        "add_zero_attn",
        "sequence_first",
        "distinct_kdim",
    ],
)
def test_grouped_self_attention_eligibility_falls_back_to_the_module_call(
    case: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Everything the explicit path cannot reproduce keeps the generic ``self_attn(...)`` call.

    CUDA only, with a positive control: on CPU every case is ineligible for its device alone, so a CPU run would pass
    whether or not the case's own guard exists.
    """
    device = torch.device("cuda")
    layer = _decoder_layer().to(device)
    tgt = torch.randn(2, 12, 16, device=device)
    assert layer._grouped_self_attention_eligible(tgt, None, None) is True
    tgt_mask = key_padding_mask = None
    if case == "eval":
        layer.eval()
    elif case == "tgt_mask":
        tgt_mask = torch.zeros(4, 4, device=device, dtype=torch.bool)
    elif case == "key_padding_mask":
        key_padding_mask = torch.zeros(6, 4, device=device, dtype=torch.bool)
    elif case == "subclass":
        layer.self_attn = _MultiheadAttentionSubclass(16, 4, batch_first=True).to(device)
    elif case in _MODULE_HOOK_REGISTRARS:
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[case](layer.self_attn, Mock(return_value=None)).remove)
    elif case == "instance_forward":
        layer.self_attn.forward = layer.self_attn.forward  # type: ignore[method-assign]
    elif case == "compiled_call_impl":
        layer.self_attn._compiled_call_impl = Mock()
    elif case == "dropout":
        layer.self_attn.dropout = 0.1
    elif case == "indivisible_queries":
        tgt = torch.randn(2, 11, 16, device=device)
    elif case == "compiling":
        monkeypatch.setattr("rfdetr.models.transformer.is_compiling", lambda: True)
    elif case == "tracing":
        monkeypatch.setattr("rfdetr.models.transformer._is_tracing", lambda: True)
    elif case == "no_in_proj_bias":
        layer.self_attn = nn.MultiheadAttention(16, 4, bias=False, batch_first=True).to(device)
    elif case == "add_bias_kv":
        layer.self_attn = nn.MultiheadAttention(16, 4, add_bias_kv=True, batch_first=True).to(device)
    elif case == "add_zero_attn":
        layer.self_attn = nn.MultiheadAttention(16, 4, add_zero_attn=True, batch_first=True).to(device)
    elif case == "sequence_first":
        layer.self_attn = nn.MultiheadAttention(16, 4, batch_first=False).to(device)
    elif case == "distinct_kdim":
        layer.self_attn = nn.MultiheadAttention(16, 4, kdim=8, vdim=8, batch_first=True).to(device)

    assert layer._grouped_self_attention_eligible(tgt, tgt_mask, key_padding_mask) is False


def test_grouped_self_attention_is_cuda_only() -> None:
    """The layout savings are a CUDA measurement; CPU keeps the module call."""
    layer = _decoder_layer()

    assert layer._grouped_self_attention_eligible(torch.randn(2, 12, 16), None, None) is False


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "autocast_dtype",
    [pytest.param(None, id="fp32"), pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")],
)
def test_forward_post_grouped_path_matches_module_path_on_cuda(
    autocast_dtype: torch.dtype | None, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """On CUDA in training the explicit path is taken, and ``forward_post`` returns the module path's bits.

    The oracle turns off every eager rewrite of the layer (grouped self-attention, the folded positional add and the
    fused FFN epilogue) and the gradients of every layer parameter, not only the self-attention ones, are compared, so
    ``linear1``, ``linear2``, the norms and ``cross_attn`` are checked against the two-op graph under autocast too.

    Shapes follow RF-DETR Nano's decoder (256-wide, 8 heads, 13 groups of 300 queries) at batch 2, so the kernels
    exercised are the ones training runs. The gradients are compared with tolerances: the weight gradients reduce over
    the tokens in a different row order, and the flash-attention and deformable-attention backward kernels are not
    deterministic on their own (fp32 to rounding, bf16 and fp16 to one ulp of the largest entry).
    """
    from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn as _MSDeformAttn

    # build_trainer leaves TF32 matmuls on for the process; fp32 here means fp32.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(previous_precision))
    torch.manual_seed(0)
    layer = TransformerDecoderLayer(
        d_model=256, sa_nhead=8, ca_nhead=16, dim_feedforward=2048, dropout=0.0, group_detr=13, num_feature_levels=1
    ).cuda()
    layer.train()
    assert type(layer.cross_attn) is _MSDeformAttn
    tgt = torch.randn(2, 13 * 300, 256, device="cuda")
    memory = torch.randn(2, 24 * 24, 256, device="cuda")
    query_pos = torch.randn(2, 13 * 300, 256, device="cuda")
    reference_points = torch.rand(2, 13 * 300, 1, 4, device="cuda")
    spatial_shapes = torch.tensor([[24, 24]], device="cuda")
    level_start_index = torch.tensor([0], device="cuda")
    grad_out = torch.randn(2, 13 * 300, 256, device="cuda")

    compute_dtype = torch.float32 if autocast_dtype is None else autocast_dtype

    def run(use_explicit_path: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not use_explicit_path:
            monkeypatch.setattr(
                TransformerDecoderLayer, "_grouped_self_attention_eligible", lambda *_args, **_kwargs: False
            )
            monkeypatch.setattr(
                TransformerDecoderLayer, "_pos_embed_for_linear", TransformerDecoderLayer.with_pos_embed
            )
            monkeypatch.setattr(
                TransformerDecoderLayer, "_ffn_hidden", lambda self, tgt: self.activation(self.linear1(tgt))
            )
        else:
            monkeypatch.undo()
        layer.zero_grad(set_to_none=True)
        tgt_in = tgt.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=compute_dtype, enabled=autocast_dtype is not None):
            out = layer.forward_post(
                tgt_in,
                memory.to(compute_dtype),
                query_pos=query_pos.to(compute_dtype),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                spatial_shapes_hw=[(24, 24)],
            )
        assert isinstance(out, torch.Tensor)
        out.backward(grad_out.to(out.dtype))
        assert tgt_in.grad is not None
        grads = [tgt_in.grad.clone(), *(p.grad.clone() for p in layer.parameters() if p.grad is not None)]
        return out.detach(), grads

    assert layer._grouped_self_attention_eligible(tgt, None, None) is True
    expected, expected_grads = run(use_explicit_path=False)
    actual, actual_grads = run(use_explicit_path=True)

    assert torch.equal(actual, expected)
    ulp = {None: 2**-16, torch.bfloat16: 2**-7, torch.float16: 2**-10}[autocast_dtype]
    for expected_grad, actual_grad in zip(expected_grads, actual_grads, strict=True):
        atol = float(expected_grad.abs().max()) * ulp
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=atol)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pos_embed_for_linear_emits_the_autocast_dtype_under_cuda_autocast() -> None:
    """Under bf16 autocast the positional add is produced in bf16 in one step, bitwise the cast sum."""
    layer = _decoder_layer().cuda()
    tensor = torch.randn(2, 12, 16, device="cuda")
    pos = torch.randn(2, 12, 16, device="cuda").to(torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = layer._pos_embed_for_linear(tensor, pos)
    layer.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        in_eval = layer._pos_embed_for_linear(tensor, pos)

    assert fused.dtype == torch.bfloat16
    assert torch.equal(fused, (tensor + pos).to(torch.bfloat16))
    assert in_eval.dtype == torch.float32
    assert torch.equal(in_eval, tensor + pos)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("target", ["ref_point_head", "first_linear"])
@pytest.mark.parametrize("feature", ["forward_pre_hook", "instance_forward"])
def test_ref_point_head_observers_keep_the_full_precision_embedding_input(
    target: str, feature: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An observer on the embedding consumer must receive the fp32 input the original module path exposed.

    The first call is a positive control proving the unobserved production path writes the sine embedding directly in
    bf16. The second call installs a real forward-pre hook on either consumer boundary and requires both the write and
    the hook-visible input to stay fp32.
    """
    decoder = TransformerDecoder(_decoder_layer(), num_layers=1, d_model=16, lite_refpoint_refine=True).cuda()
    tgt = torch.randn(2, 12, 16, device="cuda")
    memory = torch.randn(2, 20, 16, device="cuda")
    refpoints = torch.randn(2, 12, 4, device="cuda")
    spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
    level_start_index = torch.tensor([0, 16], device="cuda")
    valid_ratios = torch.ones(2, 2, 2, device="cuda")
    written_dtypes: list[torch.dtype] = []
    real_apply = _InterleavedSinCos.apply
    monkeypatch.setattr(
        _InterleavedSinCos,
        "apply",
        lambda angle, dtype: (written_dtypes.append(dtype), real_apply(angle, dtype))[1],
    )

    def run() -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            decoder(
                tgt,
                memory,
                refpoints_unsigmoid=refpoints,
                spatial_shapes=spatial_shapes,
                spatial_shapes_hw=[(4, 4), (2, 2)],
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

    run()
    assert written_dtypes == [torch.bfloat16]
    written_dtypes.clear()

    watched = decoder.ref_point_head if target == "ref_point_head" else decoder.ref_point_head.layers[0]
    observed_dtypes: list[torch.dtype] = []
    forward_mock = None
    if feature == "forward_pre_hook":
        watched.register_forward_pre_hook(lambda _module, args: observed_dtypes.append(args[0].dtype))
    else:
        forward_mock = Mock(side_effect=watched.forward)
        watched.forward = forward_mock  # type: ignore[method-assign]
    run()
    if forward_mock is not None:
        observed_dtypes.append(forward_mock.call_args.args[0].dtype)

    assert written_dtypes == [torch.float32]
    assert observed_dtypes == [torch.float32]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("target", ["cross_attn", "sampling_offsets", "attention_weights"])
@pytest.mark.parametrize("feature", ["forward_pre_hook", "instance_forward"])
def test_cross_attention_observers_keep_the_full_precision_query(
    target: str, feature: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-attention and its query-consuming linears must observe the original fp32 positional sum.

    A positive control first proves that both the self- and cross-attention positional adds take the fused path. After
    installing a real hook, only self-attention may keep the folded add and the watched cross-attention boundary must
    receive fp32.
    """
    layer = _decoder_layer().cuda()
    tgt = torch.randn(2, 12, 16, device="cuda")
    memory = torch.randn(2, 20, 16, device="cuda")
    query_pos = torch.randn(2, 12, 16, device="cuda")
    reference_points = torch.rand(2, 12, 2, 4, device="cuda")
    spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
    level_start_index = torch.tensor([0, 16], device="cuda")
    folded_adds: list[object] = []
    real_apply = _AddInDtype.apply
    monkeypatch.setattr(
        _AddInDtype,
        "apply",
        lambda *args: (folded_adds.append(args), real_apply(*args))[1],
    )

    def run() -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            layer.forward_post(
                tgt,
                memory,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_hw=[(4, 4), (2, 2)],
                level_start_index=level_start_index,
            )

    run()
    assert len(folded_adds) == 2
    folded_adds.clear()

    watched = layer.cross_attn if target == "cross_attn" else getattr(layer.cross_attn, target)
    observed_dtypes: list[torch.dtype] = []
    forward_mock = None
    if feature == "forward_pre_hook":
        watched.register_forward_pre_hook(lambda _module, args: observed_dtypes.append(args[0].dtype))
    else:
        forward_mock = Mock(side_effect=watched.forward)
        watched.forward = forward_mock  # type: ignore[method-assign]
    run()
    if forward_mock is not None:
        observed_dtypes.append(forward_mock.call_args.args[0].dtype)

    assert len(folded_adds) == 1
    assert observed_dtypes == [torch.float32]


#: ``(device, dtype)`` cases of ``_LinearReLU``: CUDA always, and CPU in every dtype for which
#: the private ``torch._addmm_activation`` op exists on the installed torch, so a CPU job gets a signal on the op's
#: drift across the supported torch range without failing where the op has no CPU kernel.
_LINEAR_RELU_CASES = [
    pytest.param(
        device,
        dtype,
        id=f"{device}-{name}",
        marks=_CUDA_MARKS
        if device == "cuda"
        else pytest.mark.skipif(
            not _cpu_addmm_activation_available(dtype), reason="torch._addmm_activation has no CPU kernel"
        ),
    )
    for device in ("cpu", "cuda")
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16))
]


@pytest.mark.usefixtures("float32_matmul_precision")
@pytest.mark.parametrize("dim_feedforward", [64, 63])
@pytest.mark.parametrize(("device", "dtype"), _LINEAR_RELU_CASES)
def test_linear_relu_is_bitwise_relu_of_linear_forward_and_backward(
    device: str, dtype: torch.dtype, dim_feedforward: int
) -> None:
    """The epilogue ReLU rounds once then clamps, which commutes with the clamp; the backward is autograd's.

    The fp32 cases run under both float32 matmul precisions (production sets ``"high"``, TF32 GEMMs), and the odd
    ``dim_feedforward`` is not a multiple of 8, the alignment cuBLASLt's fused epilogue prefers.
    """
    torch.manual_seed(0)
    x = (torch.randn(3, 40, 32, device=device) * 3).to(dtype).requires_grad_(True)
    weight = (torch.randn(dim_feedforward, 32, device=device) * 0.2).to(dtype).requires_grad_(True)
    bias = torch.randn(dim_feedforward, device=device).to(dtype).requires_grad_(True)
    grad_out = torch.randn(3, 40, dim_feedforward, device=device).to(dtype)

    expected = F.relu(F.linear(x, weight, bias))
    expected.backward(grad_out)
    expected_grads = [t.grad for t in (x, weight, bias)]
    x.grad = weight.grad = bias.grad = None

    actual = _LinearReLU.apply(x, weight, bias)
    actual.backward(grad_out)

    assert actual.dtype == dtype and actual.shape == expected.shape
    assert torch.equal(actual, expected)
    for expected_grad, tensor in zip(expected_grads, (x, weight, bias), strict=True):
        assert expected_grad is not None and tensor.grad is not None
        assert torch.equal(tensor.grad, expected_grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.usefixtures("float32_matmul_precision")
@pytest.mark.parametrize(
    "mode", [pytest.param("train", id="train"), pytest.param("inference", id="eval-inference-mode")]
)
@pytest.mark.parametrize(
    "autocast_dtype",
    [pytest.param(None, id="fp32"), pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")],
)
def test_ffn_hidden_is_bitwise_the_two_op_path_on_cuda(
    autocast_dtype: torch.dtype | None, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_ffn_hidden`` must return the bits of ``activation(linear1(tgt))`` in the dtype that path produces.

    ``_ffn_hidden`` is not tied to training, so ``predict()`` on CUDA takes it too: the ``inference`` mode runs it in
    ``eval()`` under ``torch.inference_mode``. The spy proves the fused path ran instead of a silent fallback.
    """
    layer = _decoder_layer().cuda()
    tgt = torch.randn(2, 12, 16, device="cuda")
    if mode == "inference":
        layer.eval()
    fused_calls: list[object] = []
    real_apply = _LinearReLU.apply
    monkeypatch.setattr(_LinearReLU, "apply", lambda *args: (fused_calls.append(args), real_apply(*args))[1])

    with (
        torch.inference_mode(mode == "inference"),
        torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None),
    ):
        expected = layer.activation(layer.linear1(tgt))
        actual = layer._ffn_hidden(tgt)

    assert len(fused_calls) == 1
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(name, marks=_CUDA_MARKS)
        for name in (
            "gelu",
            "forward_hook",
            "forward_pre_hook",
            "backward_hook",
            "backward_pre_hook",
            "global_forward_hook",
            "global_backward_hook",
            "instance_forward",
            "compiled_call_impl",
            "linear_subclass",
            "no_bias",
            "compiling",
            "tracing",
        )
    ]
    + ["cpu"],
)
def test_ffn_hidden_falls_back_to_the_two_op_path(
    case: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Anything the fused epilogue cannot reproduce keeps ``activation(linear1(tgt))``.

    Every case but ``cpu`` needs CUDA and starts from a positive control: on CPU each one would fall back for its device
    alone and pass whether or not its own guard exists.
    """
    device = torch.device("cpu" if case == "cpu" else "cuda")
    layer = _decoder_layer().to(device)
    tgt = torch.randn(2, 12, 16, device=device)
    fused_calls: list[object] = []
    real_apply = _LinearReLU.apply
    monkeypatch.setattr(_LinearReLU, "apply", lambda *args: (fused_calls.append(args), real_apply(*args))[1])
    if device.type == "cuda":
        layer._ffn_hidden(tgt)
        assert len(fused_calls) == 1
        fused_calls.clear()
    if case == "gelu":
        layer.activation = F.gelu
    elif case in _MODULE_HOOK_REGISTRARS:
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[case](layer.linear1, Mock(return_value=None)).remove)
    elif case == "instance_forward":
        layer.linear1.forward = layer.linear1.forward  # type: ignore[method-assign]
    elif case == "compiled_call_impl":
        layer.linear1._compiled_call_impl = Mock(side_effect=layer.linear1._call_impl)
    elif case == "linear_subclass":

        class _Linear(nn.Linear):
            pass

        layer.linear1 = _Linear(16, 32).to(device)
    elif case == "no_bias":
        layer.linear1 = nn.Linear(16, 32, bias=False).to(device)
    elif case == "compiling":
        monkeypatch.setattr("rfdetr.models.transformer.is_compiling", lambda: True)
    elif case == "tracing":
        monkeypatch.setattr("rfdetr.models.transformer._is_tracing", lambda: True)

    out = layer._ffn_hidden(tgt)

    assert fused_calls == []
    assert torch.equal(out, layer.activation(layer.linear1(tgt)))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "feature",
    [
        "forward_hook",
        "backward_hook",
        "backward_pre_hook",
        "global_forward_hook",
        "global_backward_hook",
        "compiled_call_impl",
    ],
)
@pytest.mark.parametrize("target", ["linear1", "self_attn"])
def test_forward_post_keeps_the_module_call_that_a_feature_observes(
    target: str, feature: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """On CUDA in training ``forward_post`` still calls a module whose call something observes.

    The grouped self-attention and the fused FFN epilogue read a module's parameters instead of calling it, so a hook
    (per module or global, forward or backward) or a ``compile()`` wrapper would silently stop firing. The oracle is the
    feature's own effect during a real ``forward_post`` plus backward; a control run without the feature first proves
    that both fast paths are reached, so the fallback cannot pass merely because they never were.
    """
    layer = _decoder_layer().cuda()
    watched = getattr(layer, target)
    fused: list[object] = []
    grouped: list[object] = []
    real_apply = _LinearReLU.apply
    real_grouped = TransformerDecoderLayer._grouped_self_attention
    monkeypatch.setattr(_LinearReLU, "apply", lambda *args: (fused.append(args), real_apply(*args))[1])
    monkeypatch.setattr(
        TransformerDecoderLayer,
        "_grouped_self_attention",
        lambda *args: (grouped.append(args), real_grouped(*args))[1],
    )
    memory = torch.randn(2, 4 * 4 + 2 * 2, 16, device="cuda")
    reference_points = torch.rand(2, 12, 2, 4, device="cuda")
    spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
    level_start_index = torch.tensor([0, 16], device="cuda")
    fired: list[object] = []

    def run() -> None:
        tgt = torch.randn(2, 12, 16, device="cuda", requires_grad=True)
        out = layer.forward_post(
            tgt,
            memory,
            query_pos=torch.randn(2, 12, 16, device="cuda"),
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            spatial_shapes_hw=[(4, 4), (2, 2)],
        )
        assert isinstance(out, torch.Tensor)
        out.sum().backward()

    run()
    assert (len(fused), len(grouped), fired) == (1, 1, [])
    fused.clear()
    grouped.clear()

    def record_call(*args: object, **kwargs: object) -> object:
        fired.append(watched)
        return watched._call_impl(*args, **kwargs)

    def record_hook(module: nn.Module, *_: object) -> None:
        # Global hooks see every module call in the layer; only the watched module's counts.
        if module is watched:
            fired.append(module)

    if feature == "compiled_call_impl":
        watched._compiled_call_impl = record_call
    else:
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[feature](watched, record_hook).remove)
    run()

    assert fired, f"the {feature} on {target} never fired"
    is_global = feature.startswith("global_")
    assert len(fused) == (0 if target == "linear1" or is_global else 1)
    assert len(grouped) == (0 if target == "self_attn" or is_global else 1)


@pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
    ],
)
def test_gen_sineembed_for_position_out_dtype_is_bitwise_the_cast_result(
    out_dtype: torch.dtype, device: str, sineembed: Callable[..., torch.Tensor]
) -> None:
    """Rounding each sin/cos once equals casting the full-precision embedding of the previous formulation."""
    pos_tensor = torch.rand(2, 300, 4, device=device)

    embedded = sineembed(pos_tensor, dim=128, out_dtype=out_dtype)

    assert embedded.dtype == out_dtype
    assert torch.equal(embedded, _reference_sineembed(pos_tensor, 128).to(out_dtype))


@pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
@pytest.mark.parametrize("out_dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")])
def test_gen_sineembed_for_position_out_dtype_backward_is_the_two_op_graph(
    out_dtype: torch.dtype, device: str, sineembed: Callable[..., torch.Tensor]
) -> None:
    """The interleaved write's backward is ``grad * cos`` and ``-(grad * sin)`` from the upcast gradient."""
    pos_tensor = torch.rand(2, 64, 4, device=device)
    grad_out = torch.randn(2, 64, 4 * 32, device=device).to(out_dtype)
    expected_input = pos_tensor.clone().requires_grad_(True)
    actual_input = pos_tensor.clone().requires_grad_(True)

    reference = _reference_sineembed(expected_input, 32).to(out_dtype)
    reference.backward(grad_out)
    sineembed(actual_input, dim=32, out_dtype=out_dtype).backward(grad_out)

    assert expected_input.grad is not None and actual_input.grad is not None
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-5)


def test_is_tracing_is_true_only_while_torch_jit_trace_records() -> None:
    """``_is_tracing`` reports a real ``torch.jit.trace`` recording, not just the monkeypatched stand-in.

    The fallback tests for the eager CUDA rewrites switch tracing on by replacing ``_is_tracing`` with a constant, which
    keeps passing if the real predicate stops reporting a trace (a torch change, or a swapped-in call). This CPU canary
    ties that stand-in to the real thing: the predicate is ``False`` in plain eager execution and ``True`` while a
    function is being traced.
    """
    seen: list[bool] = []

    def probe(x: torch.Tensor) -> torch.Tensor:
        """Record ``_is_tracing()`` on every call and return ``x + 1`` so the traced graph has a real op."""
        seen.append(_is_tracing())
        return x + 1

    outside = _is_tracing()
    torch.jit.trace(probe, (torch.zeros(1),), check_trace=False)

    assert outside is False
    assert seen == [True]


@pytest.mark.parametrize(
    ("device", "mode", "expected_calls"),
    [
        pytest.param("cpu", "eager", 0, id="cpu-eager"),
        pytest.param("cuda", "eager", 1, id="cuda-eager", marks=_CUDA_MARKS),
        pytest.param("cuda", "compiling", 0, id="cuda-compiling", marks=_CUDA_MARKS),
        pytest.param("cuda", "tracing", 0, id="cuda-tracing", marks=_CUDA_MARKS),
    ],
)
def test_gen_sineembed_for_position_takes_the_interleaved_write_only_on_cuda_eager(
    device: str, mode: str, expected_calls: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU (and so MPS and XLA), compiled and traced graphs keep the plain ops; CUDA eager takes the write."""
    applied: list[object] = []
    real_apply = _InterleavedSinCos.apply
    monkeypatch.setattr(_InterleavedSinCos, "apply", lambda *args: (applied.append(args), real_apply(*args))[1])
    if mode == "compiling":
        monkeypatch.setattr("rfdetr.models.transformer.is_compiling", lambda: True)
    elif mode == "tracing":
        monkeypatch.setattr("rfdetr.models.transformer._is_tracing", lambda: True)

    gen_sineembed_for_position(torch.rand(2, 8, 4, device=device), dim=16, out_dtype=torch.bfloat16)

    assert len(applied) == expected_calls


@pytest.mark.parametrize("out_dtype", [None, pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize("box_width", [2, 4])
def test_gen_sineembed_for_position_off_cuda_keeps_the_previous_graph_bitwise_in_backward(
    box_width: int, out_dtype: torch.dtype | None
) -> None:
    """Off CUDA the coordinate gradient is the previous formulation's bit for bit, not merely close.

    The contiguous-angle rewrite sums the coordinate gradient in a different order, so on CPU (and so MPS and XLA) it
    would move gradients by an fp32 rounding the CUDA-only claim does not cover; the plain-op branch must therefore be
    the previous graph itself. ``float32`` is used because a wider dtype hides that order at a tolerance.
    """
    pos_tensor = torch.rand(2, 300, box_width)
    grad_out = torch.randn(2, 300, box_width * 128)
    if out_dtype is not None:
        grad_out = grad_out.to(out_dtype)
    expected_input = pos_tensor.clone().requires_grad_(True)
    actual_input = pos_tensor.clone().requires_grad_(True)

    expected = _reference_sineembed(expected_input, 128)
    expected = expected if out_dtype is None else expected.to(out_dtype)
    expected.backward(grad_out)
    actual = gen_sineembed_for_position(actual_input, dim=128, out_dtype=out_dtype)
    actual.backward(grad_out)

    assert torch.equal(actual, expected)
    assert expected_input.grad is not None and actual_input.grad is not None
    assert torch.equal(actual_input.grad, expected_input.grad)


@pytest.mark.parametrize("width", [3, 5])
def test_gen_sineembed_for_position_rejects_a_last_dimension_other_than_2_or_4(width: int) -> None:
    """The public function documents boxes of width 2 or 4 and raises ``ValueError`` for any other last dimension.

    Widths 3 and 5 are the smallest ones the plain path can read far enough into (it indexes coordinates 0 and 1 before
    checking the width, so widths 0 and 1 fail there with ``IndexError`` instead).
    """
    pos_tensor = torch.rand(1, 2, width)

    with pytest.raises(ValueError, match=rf"Unknown pos_tensor shape\(-1\):{width}"):
        gen_sineembed_for_position(pos_tensor, dim=8)


@pytest.mark.parametrize("width", [0, 1, 3, 5])
def test_sineembed_interleaved_rejects_a_last_dimension_other_than_2_or_4(width: int) -> None:
    """The interleaved CUDA-eager kernel checks the width before reading any coordinate, so every bad width is a
    ``ValueError`` (the function is runnable on CPU, which is what lets this guard be tested without a GPU)."""
    pos_tensor = torch.rand(1, 2, width)

    with pytest.raises(ValueError, match=rf"Unknown pos_tensor shape\(-1\):{width}"):
        _sineembed_interleaved(pos_tensor, 8, None)


@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")])
def test_cast_then_expand_matches_expand_then_cast_forward_and_backward(dtype: torch.dtype) -> None:
    """Casting once before the expand gives the expanded cast's values; the backward is the fp32 group sum."""
    memory = torch.randn(2, 20, 16, requires_grad=True)
    grad_out = torch.randn(13, 2, 20, 16).to(dtype)

    expected = memory.unsqueeze(0).expand(13, -1, -1, -1).to(dtype)
    expected.backward(grad_out)
    expected_grad = memory.grad
    memory.grad = None

    actual = _CastThenExpand.apply(memory, 13, dtype)
    actual.backward(grad_out)

    assert actual.dtype == dtype and actual.shape == (13, 2, 20, 16)
    assert torch.equal(actual, expected)
    assert expected_grad is not None and memory.grad is not None
    assert memory.grad.dtype == torch.float32
    torch.testing.assert_close(memory.grad, expected_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_stage_group_selection_cast_once_matches_expanded_cast_under_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under bf16 autocast the batched selection's forward is bitwise the expand-then-cast path, and the encoder
    memory's gradient matches it to fp32 rounding (it is summed over groups in fp32 either way)."""
    torch.manual_seed(0)
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    transformer = build_model(ns).transformer.cuda()
    transformer.train()
    assert transformer._two_stage_batching_eligible()
    memory = torch.randn(2, 60, transformer.d_model, device="cuda")
    proposals = torch.rand(2, 60, 4, device="cuda")

    def run(cast_once: bool) -> tuple[list[torch.Tensor], torch.Tensor]:
        if not cast_once:
            monkeypatch.setattr(
                _CastThenExpand,
                "apply",
                lambda x, groups, dtype: x.unsqueeze(0).expand(groups, -1, -1, -1).to(dtype),
            )
        else:
            monkeypatch.undo()
        transformer.zero_grad(set_to_none=True)
        memory_in = memory.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = transformer._two_stage_group_selection(memory_in, proposals, transformer.group_detr)
        outputs[2].float().sum().backward()
        assert memory_in.grad is not None
        return [o.detach() for o in outputs], memory_in.grad.clone()

    expected, expected_grad = run(cast_once=False)
    actual, actual_grad = run(cast_once=True)

    for expected_out, actual_out in zip(expected, actual, strict=True):
        assert torch.equal(actual_out, expected_out)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_stage_group_selection_keeps_expand_then_cast_while_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """TorchScript tracing must not record the custom cast-then-expand autograd function.

    The first call is a positive control proving bf16 CUDA autocast normally takes the rewrite. The tracing call then
    requires the previous expand-then-cast graph, so this fails if the tracing predicate is absent from the guard.
    """
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    ).cuda()
    memory = torch.randn(2, 20, 16, device="cuda")
    proposals = torch.rand(2, 20, 4, device="cuda")
    cast_once_calls: list[object] = []
    real_apply = _CastThenExpand.apply
    monkeypatch.setattr(
        _CastThenExpand,
        "apply",
        lambda *args: (cast_once_calls.append(args), real_apply(*args))[1],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)
    assert len(cast_once_calls) == 1
    cast_once_calls.clear()

    monkeypatch.setattr("rfdetr.models.transformer._is_tracing", lambda: True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)

    assert cast_once_calls == []


def test_two_stage_group_selection_routes_the_shared_memory_through_cast_then_expand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the eager-CUDA gate open and a bf16 autocast dtype, the fp32 memory is cast once, then expanded per group.

    Runs on CPU by forcing the gate, so it asserts routing only: the spy hands back the fp32 expansion the plain path
    builds, because without real autocast the batched GEMM downstream needs fp32 operands.
    """
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    memory = torch.randn(2, 20, 16)
    proposals = torch.rand(2, 20, 4)
    routed: list[tuple[int, torch.dtype]] = []

    def spy(x: torch.Tensor, groups: int, dtype: torch.dtype) -> torch.Tensor:
        routed.append((groups, dtype))
        return x.unsqueeze(0).expand(groups, *x.shape)

    monkeypatch.setattr("rfdetr.models.transformer._eager_cuda", lambda tensor: True)
    monkeypatch.setattr("rfdetr.models.transformer._cuda_autocast_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(_CastThenExpand, "apply", spy)

    transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)

    assert routed == [(2, torch.bfloat16)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("training", "autocast_dtype", "captured"),
    [
        pytest.param(True, torch.bfloat16, False, id="train-bf16"),
        pytest.param(True, torch.float16, False, id="train-fp16"),
        pytest.param(True, None, False, id="train-fp32"),
        pytest.param(False, torch.bfloat16, False, id="eval-bf16"),
        pytest.param(True, torch.bfloat16, True, id="train-bf16-cuda-graph"),
    ],
)
def test_every_eager_decoder_rewrite_is_taken_by_the_production_model_on_cuda(
    training: bool, autocast_dtype: torch.dtype | None, captured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each rewrite runs on the route ``LWDETR.forward`` takes, in exactly the modes it is meant for.

    The rewrites are bitwise the ops they replace, so no output comparison can tell them from their fallbacks: unwiring
    one leaves every parity test green and silently gives the speedup back. Delegating spies on a real Nano built by
    ``build_model`` observe each one at its owner. The FFN epilogue and the sine-embedding write are not tied to
    training and also run in ``eval()``; the grouped self-attention, the fused positional adds and the cast-once memory
    are training-only, and the last two need autocast (there is no cast to fold without it).

    The ``captured`` case drives the model through ``CudaGraphTrainingRunner`` (``cuda_graphs=True``) under bf16
    autocast: a host-side index tensor or synchronisation in any rewrite fails the capture. The capture tests in
    ``test_cuda_graph_step.py`` run without autocast and never reach the two autocast-only rewrites.
    """
    from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner

    calls: dict[str, list[tuple[object, ...]]] = {key: [] for key in ("sine", "add", "ffn", "attn", "memory")}

    def count(owner: type, attr: str, key: str) -> None:
        real = getattr(owner, attr)

        def spy(*args: object, **kwargs: object) -> object:
            calls[key].append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(owner, attr, spy)

    count(_InterleavedSinCos, "apply", "sine")
    count(_AddInDtype, "apply", "add")
    count(_LinearReLU, "apply", "ffn")
    count(TransformerDecoderLayer, "_grouped_self_attention", "attn")
    count(_CastThenExpand, "apply", "memory")
    torch.manual_seed(0)
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=7, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    model = build_model(ns).cuda().train(training)
    transformer = model.transformer
    assert transformer._two_stage_batching_eligible()
    layers = len(transformer.decoder.layers)
    samples = NestedTensor(
        torch.randn(2, 3, 256, 256, device="cuda"), torch.zeros(2, 256, 256, dtype=torch.bool, device="cuda")
    )

    with (
        torch.set_grad_enabled(training),
        torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None),
    ):
        outputs = CudaGraphTrainingRunner(model)(samples) if captured else model(samples)
    if captured:
        loss = outputs["pred_logits"].float().sum()
        loss.backward()
        assert torch.isfinite(loss)

    folds_a_cast = training and autocast_dtype is not None
    per_pass = {
        "sine": 1 if transformer.decoder.lite_refpoint_refine else layers,
        "ffn": layers,
        "attn": layers if training else 0,
        "add": 2 * layers if folds_a_cast else 0,
        "memory": 1 if folds_a_cast else 0,
    }
    # A CUDA graph runs the module for its warm-up passes and the capture, so the counts are a whole number of passes.
    passes = len(calls["ffn"]) // layers
    assert passes >= 1 and (captured or passes == 1)
    assert {key: len(taken) for key, taken in calls.items()} == {key: passes * n for key, n in per_pass.items()}
    # ``ref_point_head`` is the embedding's only consumer, so under autocast it is written in the compute dtype.
    assert {args[1] for args in calls["sine"]} == {autocast_dtype or torch.float32}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("autocast_dtype", "grad_rtol"),
    [pytest.param(None, 1e-4, id="fp32"), pytest.param(torch.bfloat16, 1e-2, id="bf16")],
)
def test_nano_loss_terms_are_bitwise_and_gradients_match_with_the_eager_rewrites_on_and_off(
    autocast_dtype: torch.dtype | None, grad_rtol: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real Nano forward plus criterion gives the same outputs and loss terms with every rewrite on or replaced.

    The rewrites are bitwise the ops they replace in the forward, so the outputs and every loss term must be equal bit
    for bit. The gradients are compared per parameter with a scale-aware norm bound instead: the deformable-attention
    and flash-attention backward kernels are not deterministic on their own, and the rewrites reorder weight-gradient
    sums. The bound is ``grad_rtol`` of the reference gradient's norm (a wrong gradient is off by far more); bf16 rounds
    each of those sums to eight bits, hence the looser constant. Autocast is a case because the folded positional add
    and the cast-once memory only run under it. This test needs a GPU and was written without one.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    model_config = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cuda")
    train_config = TrainConfig(dataset_dir="unused", drop_path=0.0)
    model = build_model_from_config(model_config, train_config).cuda().train()
    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion = criterion.cuda()
    samples = NestedTensor(
        torch.randn(2, 3, 256, 256, device="cuda"), torch.zeros(2, 256, 256, dtype=torch.bool, device="cuda")
    )
    targets = [
        {"labels": torch.tensor([1], device="cuda"), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device="cuda")},
        {
            "labels": torch.tensor([0, 2], device="cuda"),
            "boxes": torch.tensor([[0.3, 0.4, 0.2, 0.2], [0.7, 0.6, 0.3, 0.1]], device="cuda"),
        },
    ]
    fused_calls: list[object] = []
    real_apply = _LinearReLU.apply

    def run(rewrites_on: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with monkeypatch.context() as patch:
            if rewrites_on:
                patch.setattr(_LinearReLU, "apply", lambda *args: (fused_calls.append(args), real_apply(*args))[1])
            else:
                patch.setattr(
                    TransformerDecoderLayer, "_grouped_self_attention_eligible", lambda *_args, **_kwargs: False
                )
                patch.setattr(TransformerDecoderLayer, "_pos_embed_for_linear", TransformerDecoderLayer.with_pos_embed)
                patch.setattr(
                    TransformerDecoderLayer, "_ffn_hidden", lambda self, tgt: self.activation(self.linear1(tgt))
                )
                patch.setattr(
                    _CastThenExpand, "apply", lambda x, groups, dtype: x.unsqueeze(0).expand(groups, *x.shape).to(dtype)
                )
                patch.setattr(
                    _InterleavedSinCos,
                    "apply",
                    lambda angle, dtype: torch.stack((angle.sin(), angle.cos()), -1).to(dtype),
                )
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1)
            with torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
                outputs = model(samples, targets)
            losses = criterion(outputs, targets)
            total = sum(losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses)
            total.backward()
            terms = {"pred_logits": outputs["pred_logits"].detach(), "pred_boxes": outputs["pred_boxes"].detach()}
            terms.update({key: value.detach() for key, value in losses.items()})
            terms["total"] = total.detach()
            return terms, {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    expected_terms, expected_grads = run(rewrites_on=False)
    assert fused_calls == []
    actual_terms, actual_grads = run(rewrites_on=True)

    assert fused_calls, "the rewrites never ran, so the comparison would not test them"
    assert actual_terms.keys() == expected_terms.keys()
    for key, expected_term in expected_terms.items():
        assert torch.equal(actual_terms[key], expected_term), key
    assert actual_grads.keys() == expected_grads.keys()
    for name, reference in expected_grads.items():
        difference_norm = (actual_grads[name] - reference).norm()
        # The absolute floor covers analytically zero gradients (attention key biases) whose residue is rounding noise.
        assert difference_norm <= grad_rtol * reference.norm() + 1e-9, (
            f"{name}: ||diff|| {difference_norm:.3e} vs ||reference|| {reference.norm():.3e}"
        )


class _AutocastRewriteBlock(nn.Module):
    """Smallest module that runs the two autocast-only rewrites, :class:`_AddInDtype` and :class:`_CastThenExpand`.

    A fp32 activation is added to a bf16 projection and rounded once to bf16, and the same fp32 activation is cast once
    to bf16 and expanded over ``groups``; ``rewrites=False`` computes both with the plain two-op graphs they replace.
    The forward takes ``NestedTensor`` and ``targets`` like ``LWDETR`` so ``CudaGraphTrainingRunner`` can capture it.

    Examples:
        >>> block = _AutocastRewriteBlock()
        >>> samples = NestedTensor(torch.zeros(2, 4, 8), torch.zeros(2, 4, dtype=torch.bool))
        >>> with torch.autocast("cpu", dtype=torch.bfloat16):
        ...     block(samples)["pred"].shape
        torch.Size([3, 2, 4, 8])
    """

    def __init__(self, width: int = 8, groups: int = 3, rewrites: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.rand(width) + 0.5)
        self.pos_proj = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.groups = groups
        self.rewrites = rewrites

    def forward(self, samples: NestedTensor, targets: object = None) -> dict[str, torch.Tensor]:
        """Return the mixed bf16 sum plus the group-expanded bf16 copy of the scaled activation."""
        del targets
        x = samples.tensors
        activation = x * self.scale
        pos = self.pos_proj(x)
        if self.rewrites:
            summed = _AddInDtype.apply(activation, pos, torch.bfloat16)
            memory = _CastThenExpand.apply(activation, self.groups, torch.bfloat16)
        else:
            summed = (activation + pos).to(torch.bfloat16)
            memory = activation.unsqueeze(0).expand(self.groups, *activation.shape).to(torch.bfloat16)
        return {"pred": self.mix(summed).unsqueeze(0) + memory}


def _assert_within_norm_bound(actual: torch.Tensor, expected: torch.Tensor, rel_tol: float, name: str) -> None:
    """Assert ``||actual - expected||`` is at most ``rel_tol`` of ``||expected||`` (plus a tiny absolute floor).

    Unlike an element-wise tolerance this separates kernel rounding noise on a large entry from a wrong value on a
    small one: a stale or aliased tensor is off by a large fraction of the norm, rounding noise is not.

    Examples:
        >>> _assert_within_norm_bound(torch.ones(4), torch.ones(4) * 1.001, 1e-2, "ones")
        >>> _assert_within_norm_bound(torch.ones(4), torch.zeros(4), 1e-2, "ones")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        AssertionError: ones: ||diff|| 2.000e+00 vs ||expected|| 0.000e+00...
    """
    difference_norm = (actual.float() - expected.float()).norm()
    expected_norm = expected.float().norm()
    assert difference_norm <= rel_tol * expected_norm + 1e-9, (
        f"{name}: ||diff|| {difference_norm:.3e} vs ||expected|| {expected_norm:.3e}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocast_only_rewrites_match_the_two_op_graph_under_cuda_graph_replay() -> None:
    """Captured and replayed on new inputs, ``_AddInDtype`` and ``_CastThenExpand`` keep the two-op graph's values.

    The capture tests in ``test_cuda_graph_step.py`` run without autocast, and the production-model capture case only
    checks that the loss is finite. Here a minimal module around the two autocast-only rewrites is captured once and
    replayed twice on different inputs, each time against an eager run of the plain two-op graph; a stale saved tensor
    or an aliased output buffer shows as an error of order one. bf16 capture and eager pick different kernels, so the
    bound is a few percent of the norm rather than the fp32 ``1e-4`` of the full-Nano fp32 test. Needs a GPU; written
    without one.
    """
    torch.manual_seed(0)
    reference = _AutocastRewriteBlock(rewrites=False).cuda().train()
    graphed = copy.deepcopy(reference)
    graphed.rewrites = True
    runner = CudaGraphTrainingRunner(graphed)

    def run_once() -> None:
        samples = NestedTensor(torch.randn(2, 8, 8, device="cuda"), torch.zeros(2, 8, dtype=torch.bool, device="cuda"))
        reference.zero_grad(set_to_none=True)
        graphed.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            expected = reference(samples)["pred"]
            actual = runner(samples)["pred"]
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        _assert_within_norm_bound(actual, expected, 3e-2, "pred")
        for (name, parameter), expected_parameter in zip(graphed.named_parameters(), reference.parameters()):
            assert parameter.grad is not None and expected_parameter.grad is not None, name
            _assert_within_norm_bound(parameter.grad, expected_parameter.grad, 3e-2, f"grad of {name}")

    run_once()  # captures
    run_once()  # replays on new values
    run_once()

    assert len(runner._graphed_cache) == 1
