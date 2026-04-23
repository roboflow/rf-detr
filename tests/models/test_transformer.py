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

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import gen_encoder_output_proposals


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

    Regression suite for torch.export.export compatibility: iterating over a
    spatial_shapes tensor yields FakeTensor scalars during FakeTensor tracing,
    which cannot be used as Python int split/view sizes.  The function now
    accepts an optional ``value_spatial_shapes_hw`` list of Python int pairs
    that bypasses tensor iteration.
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

        This is the torch.export.export-compatible code path: tensor scalar values
        (from iterating over a FakeTensor) cannot be used as split/view sizes, so the
        caller passes explicit Python int pairs via value_spatial_shapes_hw instead.
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

    Validates the module-level parameter threading and export-mode assert guard
    introduced in the torch.export.export compatibility fix.
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

    def test_export_flag_set_after_export_call(self) -> None:
        """Calling .export() must set _export=True, enabling the torch._assert guard path."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        assert not module._export

        module.export()

        assert module._export


class TestGenEncoderOutputProposalsDynamicBatch:
    """Regression tests for dynamic batch support in gen_encoder_output_proposals.

    Ensures that the ONNX-symbolic refactoring (PR #950 / issue #949) does not bake a
    fixed batch dimension into proposals and that output shapes are correct for varying
    batch sizes.
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

        Regression for PR #950 / issue #949: the masked branch used .reshape(-1, h, w, 1) to
        infer the batch dimension dynamically; this test verifies the branch handles varying
        batch sizes without error.

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

        Regression for issue #949: exporting with a fixed trace batch baked `Reshape([8,...])` as
        a constant ONNX node, causing TRT engines to fail at inference for any batch != 8.
        Skipped when onnx or onnxruntime is not installed.
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

    Regression test for the FakeTensor tracing failure: iterating over spatial_shapes
    and using the scalar elements as split/view sizes fails during torch.export.export
    because FakeTensor data is not allocated. Passing value_spatial_shapes_hw (concrete
    Python ints from a module attribute) bypasses the tensor iteration entirely.
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
