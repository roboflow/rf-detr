# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.transformer import gen_encoder_output_proposals


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
        memory=memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
        unsigmoid=True,
    )

    assert call_count == 1
    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (1, 4, 4)


def test_gen_encoder_output_proposals_accepts_int_tuple_spatial_shapes() -> None:
    """Regression: spatial_shapes as list[tuple[int, int]] with masks=None must not crash.

    Transformer.forward() passes Python int pairs (from bs, c, h, w = src.shape) to
    gen_encoder_output_proposals. The export path (masks=None) triggers the else branch
    which previously called H_.expand(N_) — failing with AttributeError on a Python int.
    """
    batch, h, w, d = 2, 3, 4, 8
    memory = torch.randn(batch, h * w, d)
    spatial_shapes = [(h, w)]  # Python int pairs, as produced by Transformer.forward()

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory=memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
        unsigmoid=True,
    )

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, h * w, 4)


class TestMSDeformAttnCorePytorch:
    """Tests for ms_deform_attn_core_pytorch with Python int pair spatial shapes.

    Regression suite for torch.export.export compatibility: iterating over a
    spatial_shapes tensor yields FakeTensor scalars during FakeTensor tracing,
    which cannot be used as Python int split/view sizes.  The function now
    accepts an optional ``value_spatial_shapes_hw`` list of Python int pairs
    that bypasses tensor iteration.
    """

    def _make_inputs(
        self,
        B: int = 1,
        n_heads: int = 2,
        head_dim: int = 4,
        levels: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Build minimal valid inputs for ms_deform_attn_core_pytorch.

        Args:
            B: Batch size.
            n_heads: Number of attention heads.
            head_dim: Dimension per head.
            levels: List of (H, W) int pairs; defaults to [(4, 4), (2, 2)].

        Returns:
            Tuple of (value, spatial_shapes_tensor, sampling_locations,
                      attention_weights, spatial_shapes_hw).
        """
        if levels is None:
            levels = [(4, 4), (2, 2)]
        L = len(levels)
        P = 1
        Len_q = 3

        total_hw = sum(H * W for H, W in levels)

        spatial_shapes_tensor = torch.tensor(levels, dtype=torch.long)

        value = torch.randn(B, n_heads, head_dim, total_hw)
        # sampling_locations: (B, Len_q, n_heads, L, P, 2) in [0, 1]
        sampling_locations = torch.rand(B, Len_q, n_heads, L, P, 2)
        # attention_weights: (B, Len_q, n_heads, L * P)
        attention_weights = torch.softmax(torch.randn(B, Len_q, n_heads, L * P), dim=-1)

        return value, spatial_shapes_tensor, sampling_locations, attention_weights, levels

    def test_with_tensor_spatial_shapes(self) -> None:
        """Baseline: passing only the tensor spatial_shapes still works."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, _ = self._make_inputs()

        output = ms_deform_attn_core_pytorch(value, spatial_shapes_tensor, sampling_locations, attention_weights)

        B, n_heads, head_dim, _ = value.shape
        Len_q = sampling_locations.shape[1]
        assert output.shape == (B, Len_q, n_heads * head_dim)

    def test_with_python_int_pair_spatial_shapes(self) -> None:
        """Regression: value_spatial_shapes_hw list of Python int pairs must be accepted.

        This is the torch.export.export-compatible code path: tensor scalar values
        (from iterating over a FakeTensor) cannot be used as split/view sizes, so the
        caller passes explicit Python int pairs via value_spatial_shapes_hw instead.
        """
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = self._make_inputs()

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        B, n_heads, head_dim, _ = value.shape
        Len_q = sampling_locations.shape[1]
        assert output.shape == (B, Len_q, n_heads * head_dim)

    def test_tensor_and_hw_paths_produce_identical_outputs(self) -> None:
        """Python int pair path and tensor iteration path must produce the same result."""
        torch.manual_seed(42)
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = self._make_inputs()

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

    def test_single_level(self) -> None:
        """Single-level case with Python int pair path must not crash."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = self._make_inputs(levels=[(8, 8)])

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        assert output.shape[0] == 1
