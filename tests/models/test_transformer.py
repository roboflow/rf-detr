# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for transformer utilities, MS deformable attention core, and MSDeformAttn module."""

from collections.abc import Callable

import pytest
import torch

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import gen_encoder_output_proposals


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


_MSDeformInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]


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
    h, w = 4, 4
    memory = torch.randn(batch, h * w, 8)
    spatial_shapes = torch.tensor([[h, w]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        spatial_shapes=spatial_shapes,
    )

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, h * w, 4)


def test_gen_encoder_output_proposals_accepts_python_int_pair_spatial_shapes() -> None:
    """`gen_encoder_output_proposals` must accept `spatial_shapes` as `list[tuple[int, int]]` with no padding mask.

    Regression: `Transformer.forward` passes Python int pairs derived from `src.shape`, so the
    export-driven call path uses `list[tuple[int, int]]` rather than a tensor.
    """
    batch, h, w, d = 2, 4, 4, 8
    memory = torch.randn(batch, h * w, d)
    spatial_shapes = [(h, w)]  # Python int pairs, as produced by Transformer.forward()

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
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

    @pytest.fixture
    def make_inputs(self) -> Callable[..., _MSDeformInputs]:
        """Return a factory that builds minimal valid inputs for ms_deform_attn_core_pytorch.

        Returns:
            A callable accepting optional keyword arguments ``B``, ``n_heads``,
            ``head_dim``, and ``levels``, and returning a 5-tuple of
            ``(value, spatial_shapes_tensor, sampling_locations,
            attention_weights, spatial_shapes_hw)``.
        """

        def _factory(
            B: int = 1,
            n_heads: int = 2,
            head_dim: int = 4,
            levels: list[tuple[int, int]] | None = None,
        ) -> _MSDeformInputs:
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

        return _factory

    def test_with_tensor_spatial_shapes(self, make_inputs: Callable[..., _MSDeformInputs]) -> None:
        """Baseline: passing only the tensor spatial_shapes still works."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, _ = make_inputs()

        output = ms_deform_attn_core_pytorch(value, spatial_shapes_tensor, sampling_locations, attention_weights)

        B, n_heads, head_dim, _ = value.shape
        Len_q = sampling_locations.shape[1]
        assert output.shape == (B, Len_q, n_heads * head_dim)

    def test_with_python_int_pair_spatial_shapes(self, make_inputs: Callable[..., _MSDeformInputs]) -> None:
        """Regression: value_spatial_shapes_hw list of Python int pairs must be accepted.

        This is the torch.export.export-compatible code path: tensor scalar values
        (from iterating over a FakeTensor) cannot be used as split/view sizes, so the
        caller passes explicit Python int pairs via value_spatial_shapes_hw instead.
        """
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs()

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

    def test_tensor_and_hw_paths_produce_identical_outputs(self, make_inputs: Callable[..., _MSDeformInputs]) -> None:
        """Python int pair path and tensor iteration path must produce the same result."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs()

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

    def test_single_level(self, make_inputs: Callable[..., _MSDeformInputs]) -> None:
        """Single-level case with Python int pair path must not crash."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs(levels=[(8, 8)])

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

    Validates the module-level parameter threading and export-mode assert guard
    introduced in the torch.export.export compatibility fix.
    """

    _D_MODEL = 32
    _N_HEADS = 4
    _N_LEVELS = 2
    _N_POINTS = 1
    _HW_PAIRS: list[tuple[int, int]] = [(4, 4), (2, 2)]

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
        hw_pairs = self._HW_PAIRS
        total_len = sum(H * W for H, W in hw_pairs)
        N, Len_q = 1, 3

        query = torch.randn(N, Len_q, self._D_MODEL)
        reference_points = torch.rand(N, Len_q, self._N_LEVELS, 2)
        input_flatten = torch.randn(N, total_len, self._D_MODEL)
        input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        # Cumulative start index per level: [0, H0*W0]
        starts = [sum(H * W for H, W in hw_pairs[:i]) for i in range(self._N_LEVELS)]
        input_level_start_index = torch.tensor(starts, dtype=torch.long)

        return query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, hw_pairs

    def test_forward_without_hw_param_backward_compat(self) -> None:
        """MSDeformAttn.forward without hw param produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._D_MODEL, n_levels=self._N_LEVELS, n_heads=self._N_HEADS, n_points=self._N_POINTS
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, _ = self._make_module_inputs()

        output = module(query, ref_pts, input_flatten, spatial_shapes, level_start_index)

        N, Len_q, _ = query.shape
        assert output.shape == (N, Len_q, self._D_MODEL)

    def test_forward_with_hw_param_produces_correct_shape(self) -> None:
        """MSDeformAttn.forward with input_spatial_shapes_hw produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._D_MODEL, n_levels=self._N_LEVELS, n_heads=self._N_HEADS, n_points=self._N_POINTS
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        N, Len_q, _ = query.shape
        assert output.shape == (N, Len_q, self._D_MODEL)

    def test_export_mode_forward_with_hw_param(self) -> None:
        """MSDeformAttn.forward in export mode with hw param must not raise."""
        module = MSDeformAttn(
            d_model=self._D_MODEL, n_levels=self._N_LEVELS, n_heads=self._N_HEADS, n_points=self._N_POINTS
        )
        module.export()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        N, Len_q, _ = query.shape
        assert output.shape == (N, Len_q, self._D_MODEL)

    def test_export_flag_set_after_export_call(self) -> None:
        """Calling .export() must set _export=True, enabling the torch._assert guard path."""
        module = MSDeformAttn(
            d_model=self._D_MODEL, n_levels=self._N_LEVELS, n_heads=self._N_HEADS, n_points=self._N_POINTS
        )
        assert not module._export

        module.export()

        assert module._export


def test_ms_deform_attn_core_pytorch_export_compatible() -> None:
    """torch.export.export must succeed on a module using ms_deform_attn_core_pytorch with hw param.

    Regression test for the FakeTensor tracing failure: iterating over spatial_shapes
    and using the scalar elements as split/view sizes fails during torch.export.export
    because FakeTensor data is not allocated. Passing value_spatial_shapes_hw (concrete
    Python ints from a module attribute) bypasses the tensor iteration entirely.
    """
    levels: list[tuple[int, int]] = [(4, 4), (2, 2)]
    B, n_heads, head_dim = 1, 2, 4
    total_hw = sum(H * W for H, W in levels)
    Len_q, L, P = 3, len(levels), 1

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

    value = torch.randn(B, n_heads, head_dim, total_hw)
    spatial_shapes = torch.tensor(levels, dtype=torch.long)
    sampling_locations = torch.rand(B, Len_q, n_heads, L, P, 2)
    attention_weights = torch.softmax(torch.randn(B, Len_q, n_heads, L * P), dim=-1)

    module = _MinimalDeformAttn(hw=levels)

    exported = torch.export.export(module, args=(value, spatial_shapes, sampling_locations, attention_weights))
    assert exported is not None
