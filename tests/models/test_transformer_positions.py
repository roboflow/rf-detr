# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the transformer's position handling: encoder output proposals, sine position embeddings, and how positions,
padding masks and memory of a single feature level are packed into the decoder inputs."""

import inspect
import io
from collections.abc import Callable

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.models.transformer import (
    _InterleavedSinCos,
    _sineembed_interleaved,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
)
from tests._markers import _cuda_marks, _requires_cuda, _requires_onnx_and_onnxruntime
from tests.models._transformer_support import record_apply_calls, single_level_two_stage_transformer

# Mirrors the production exporter: the dynamo exporter needs onnxscript, so pin the legacy path when torch offers one.
_DYNAMO_KWARG = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def test_gen_encoder_output_proposals_passes_ij_indexing_to_meshgrid(mocker: MockerFixture) -> None:
    """`gen_encoder_output_proposals` should call `torch.meshgrid` once with explicit ij indexing."""
    meshgrid = mocker.patch.object(torch, "meshgrid", wraps=torch.meshgrid)
    memory = torch.randn(1, 4, 8)
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)

    gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    meshgrid.assert_called_once()
    assert meshgrid.call_args.kwargs.get("indexing") == "ij"


@pytest.mark.parametrize("position_layout", ["real", "strided"])
def test_transformer_packs_single_level_position_without_redundant_copy(position_layout: str) -> None:
    """Reuse real contiguous position storage while preserving contiguous output for custom strided input."""
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries, height, width = 1, 16, 3, 4, 4
    transformer = single_level_two_stage_transformer(hidden_dim, num_queries)

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

    transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_decoder_positions.append(kwargs["pos"]), with_kwargs=True
    )
    transformer(
        [torch.randn(batch_size, hidden_dim, height, width)],
        [torch.zeros(batch_size, height, width, dtype=torch.bool)],
        [position],
        torch.rand(num_queries, 4),
        torch.randn(num_queries, hidden_dim),
    )

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
    transformer = single_level_two_stage_transformer(hidden_dim, num_queries)

    if mask_layout == "real":
        mask = torch.zeros(batch_size, height, width, dtype=torch.bool)
    else:
        mask_storage = torch.zeros(batch_size, height, width * 2, dtype=torch.bool)
        mask = mask_storage[:, :, ::2]
    flattened_mask = mask.flatten(1)
    assert flattened_mask.is_contiguous() is (mask_layout == "real")
    seen_decoder_masks: list[torch.Tensor] = []

    transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_decoder_masks.append(kwargs["memory_key_padding_mask"]), with_kwargs=True
    )
    transformer(
        [torch.randn(batch_size, hidden_dim, height, width)],
        [mask],
        [torch.randn(batch_size, hidden_dim, height, width)],
        torch.rand(num_queries, 4),
        torch.randn(num_queries, hidden_dim),
    )

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
    transformer = single_level_two_stage_transformer(hidden_dim, num_queries)

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

    transformer.decoder.register_forward_pre_hook(
        lambda _module, args, _kwargs: seen_decoder_memories.append(args[1]), with_kwargs=True
    )
    transformer(
        [src],
        [torch.zeros(batch_size, height, width, dtype=torch.bool)],
        [torch.randn(batch_size, hidden_dim, height, width)],
        torch.rand(num_queries, 4),
        torch.randn(num_queries, hidden_dim),
    )

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
    transformer = single_level_two_stage_transformer(hidden_dim, num_queries, dual_projector_kp_only=True)

    if memory_layout == "real":
        cross_src_storage = torch.randn(batch_size, height, width, hidden_dim)
        cross_src = cross_src_storage.permute(0, 3, 1, 2)
    else:
        cross_src = torch.randn(batch_size, hidden_dim, height, width)
    cross_src.requires_grad_(True)
    flattened_cross_src = cross_src.flatten(2).transpose(1, 2)
    assert flattened_cross_src.is_contiguous() is (memory_layout == "real")
    seen_cross_attn_memories: list[torch.Tensor] = []

    transformer.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_cross_attn_memories.append(kwargs["kp_cross_attn_memory"]), with_kwargs=True
    )
    transformer(
        [torch.randn(batch_size, hidden_dim, height, width)],
        [torch.zeros(batch_size, height, width, dtype=torch.bool)],
        [torch.randn(batch_size, hidden_dim, height, width)],
        torch.rand(num_queries, 4),
        torch.randn(num_queries, hidden_dim),
        cross_attn_srcs=[cross_src],
    )

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


@pytest.mark.gpu
@_requires_cuda
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [pytest.param(torch.float16, 1e-2, id="float16"), pytest.param(torch.bfloat16, 5e-2, id="bfloat16")],
)
def test_gen_sineembed_for_position_compiles_with_inductor_in_half_precision(
    dtype: torch.dtype, atol: float, width: int
) -> None:
    """``inference(compile_backend="inductor", dtype="float16")`` compiles this function with half-precision positions.

    torch 2.14 lowers a half-precision ``dim_t // 2`` to ``libdevice.isinf`` on a float16 or bfloat16 operand, which
    Triton cannot compile (pytorch/pytorch#197002), so the compile failed. Inductor keeps the arithmetic in float32
    where eager rounds each step to ``dtype``, hence the tolerance of a few half-precision steps.
    """
    # A fresh Dynamo cache per case, so each one compiles instead of falling back to eager at the recompile limit.
    torch._dynamo.reset()
    torch.manual_seed(0)
    pos_tensor = torch.rand(2, 300, width, device="cuda", dtype=dtype)

    compiled = torch.compile(gen_sineembed_for_position)(pos_tensor, 128)

    assert compiled.dtype == dtype
    torch.testing.assert_close(compiled, gen_sineembed_for_position(pos_tensor, 128), rtol=0, atol=atol)


def test_gen_encoder_output_proposals_rejects_non_square_ij_indexing(mocker: MockerFixture) -> None:
    """Wrong meshgrid indexing (xy vs ij) produces different proposals for non-square spatial shapes."""
    original_meshgrid = torch.meshgrid

    def _meshgrid_wrong_indexing(*args, **kwargs):
        kwargs["indexing"] = "xy"
        return original_meshgrid(*args, **kwargs)

    # Use non-square spatial shapes so that ij vs xy indexing produces observably different outputs.
    memory = torch.randn(1, 8, 8)
    spatial_shapes = torch.tensor([[2, 4]], dtype=torch.long)

    correct_memory, correct_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    mocker.patch.object(torch, "meshgrid", side_effect=_meshgrid_wrong_indexing)

    wrong_memory, wrong_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    assert not torch.allclose(correct_proposals, wrong_proposals), (
        "xy indexing must produce different proposals than ij indexing for non-square spatial shapes"
    )


@pytest.mark.parametrize(
    "spatial_shapes",
    [pytest.param(torch.tensor([[4, 4]], dtype=torch.long), id="tensor"), pytest.param([(4, 4)], id="int-pairs")],
)
def test_gen_encoder_output_proposals_accepts_spatial_shapes_container(
    spatial_shapes: torch.Tensor | list[tuple[int, int]],
) -> None:
    """``spatial_shapes`` may be a tensor of int pairs or the Python ``(H, W)`` int pairs ``Transformer.forward``
    passes.

    Regression: the export-driven call path derives int pairs from ``src.shape`` instead of a tensor, with no padding
    mask.
    """
    batch, ht, wd = 2, 4, 4
    memory = torch.randn(batch, ht * wd, 8)

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, memory_padding_mask=None, spatial_shapes=spatial_shapes
    )

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

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_onnx_export_with_dynamic_batch_axis(self, batch_size: int) -> None:
        """ONNX export with dynamic batch axis must run inference for batch sizes other than the trace batch.

        Regression for issue #949: exporting with a fixed trace batch baked `Reshape([8,...])` as a constant ONNX node,
        causing TRT engines to fail at inference for any batch != 8. Skipped when onnx or onnxruntime is not installed.
        """
        import onnxruntime

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
            **_DYNAMO_KWARG,
        )
        buf.seek(0)
        onnx_bytes = buf.read()

        session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        memory_np = np.random.randn(batch_size, ht * wd, dim).astype(np.float32)
        out_memory, out_proposals = session.run(None, {"memory": memory_np})
        assert out_memory.shape == (batch_size, ht * wd, dim), f"wrong memory shape for batch={batch_size}"
        assert out_proposals.shape == (batch_size, ht * wd, 4), f"wrong proposals shape for batch={batch_size}"


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
    pytest.param("cuda", gen_sineembed_for_position, id="cuda-public", marks=_cuda_marks),
]


class TestGenSineembedForPosition:
    """Sine position embedding: reference parity, output dtype, CUDA-eager route and width validation."""

    @pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
    @pytest.mark.parametrize("box_width", [2, 4])
    @pytest.mark.parametrize("dim", [4, 128])
    def test_is_bitwise_the_interleaved_slice_formulation(
        self, box_width: int, dim: int, device: str, sineembed: Callable[..., torch.Tensor]
    ) -> None:
        """Dividing by the ``dim // 2`` distinct frequencies once must reproduce every bit of the strided-slice
        version: the even and odd entries of ``dim_t`` are the same float, so the angles are the same values."""
        pos_tensor = torch.rand(3, 517, box_width, dtype=torch.float32, device=device)

        actual = sineembed(pos_tensor, dim=dim)

        assert actual.shape == (3, 517, box_width * dim)
        assert torch.equal(actual, _reference_sineembed(pos_tensor, dim))

    @pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
    def test_backward_matches_reference(self, device: str, sineembed: Callable[..., torch.Tensor]) -> None:
        """The contiguous formulation changes the summation order of the coordinate gradient, nothing else."""
        pos_tensor = torch.rand(2, 64, 4, dtype=torch.float64, device=device)
        grad_out = torch.randn(2, 64, 4 * 32, dtype=torch.float64, device=device)
        expected_input = pos_tensor.clone().requires_grad_(True)
        actual_input = pos_tensor.clone().requires_grad_(True)

        _reference_sineembed(expected_input, 32).backward(grad_out)
        sineembed(actual_input, dim=32).backward(grad_out)

        assert expected_input.grad is not None and actual_input.grad is not None
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
    @pytest.mark.parametrize(
        "out_dtype",
        [
            pytest.param(torch.float32, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16"),
            pytest.param(torch.float16, id="fp16"),
        ],
    )
    def test_out_dtype_is_bitwise_the_cast_result(
        self, out_dtype: torch.dtype, device: str, sineembed: Callable[..., torch.Tensor]
    ) -> None:
        """Rounding each sin/cos once equals casting the full-precision embedding of the previous formulation."""
        pos_tensor = torch.rand(2, 300, 4, device=device)

        embedded = sineembed(pos_tensor, dim=128, out_dtype=out_dtype)

        assert embedded.dtype == out_dtype
        assert torch.equal(embedded, _reference_sineembed(pos_tensor, 128).to(out_dtype))

    @pytest.mark.parametrize(("device", "sineembed"), _SINEEMBED_ROUTES)
    @pytest.mark.parametrize(
        "out_dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")]
    )
    def test_out_dtype_backward_is_the_two_op_graph(
        self, out_dtype: torch.dtype, device: str, sineembed: Callable[..., torch.Tensor]
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

    @pytest.mark.parametrize(
        ("device", "mode", "expected_calls"),
        [
            pytest.param("cpu", "eager", 0, id="cpu-eager"),
            pytest.param("cuda", "eager", 1, id="cuda-eager", marks=_cuda_marks),
            pytest.param("cuda", "compiling", 0, id="cuda-compiling", marks=_cuda_marks),
            pytest.param("cuda", "tracing", 0, id="cuda-tracing", marks=_cuda_marks),
        ],
    )
    def test_takes_the_interleaved_write_only_on_cuda_eager(
        self, device: str, mode: str, expected_calls: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU (and so MPS and XLA), compiled and traced graphs keep the plain ops; CUDA eager takes the write."""
        applied = record_apply_calls(monkeypatch, _InterleavedSinCos)
        if mode == "compiling":
            monkeypatch.setattr("rfdetr.models.transformer.is_compiling", lambda: True)
        elif mode == "tracing":
            monkeypatch.setattr("rfdetr.models.transformer._is_tracing", lambda: True)

        gen_sineembed_for_position(torch.rand(2, 8, 4, device=device), dim=16, out_dtype=torch.bfloat16)

        assert len(applied) == expected_calls

    @pytest.mark.parametrize("out_dtype", [None, pytest.param(torch.bfloat16, id="bf16")])
    @pytest.mark.parametrize("box_width", [2, 4])
    def test_off_cuda_keeps_the_previous_graph_bitwise_in_backward(
        self, box_width: int, out_dtype: torch.dtype | None
    ) -> None:
        """Off CUDA the coordinate gradient is the previous formulation's bit for bit, not merely close.

        The contiguous-angle rewrite sums the coordinate gradient in a different order, so on CPU (and so MPS and XLA)
        it would move gradients by an fp32 rounding the CUDA-only claim does not cover; the plain-op branch must
        therefore be the previous graph itself. ``float32`` is used because a wider dtype hides that order at a
        tolerance.
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
    def test_rejects_a_last_dimension_other_than_2_or_4(self, width: int) -> None:
        """The public function documents boxes of width 2 or 4 and raises ``ValueError`` for any other last dimension.

        Widths 3 and 5 are the smallest ones the plain path can read far enough into (it indexes coordinates 0 and 1
        before checking the width, so widths 0 and 1 fail there with ``IndexError`` instead).
        """
        pos_tensor = torch.rand(1, 2, width)

        with pytest.raises(ValueError) as exc_info:
            gen_sineembed_for_position(pos_tensor, dim=8)

        assert f"Unknown pos_tensor shape(-1):{width}" in str(exc_info.value)

    @pytest.mark.parametrize("width", [0, 1, 3, 5])
    def test_sineembed_interleaved_rejects_a_last_dimension_other_than_2_or_4(self, width: int) -> None:
        """The interleaved kernel rejects every last dimension other than 2 or 4 with ``ValueError``.

        It checks the width before reading any coordinate, and runs on CPU, which lets this guard be tested without a
        GPU.
        """
        pos_tensor = torch.rand(1, 2, width)

        with pytest.raises(ValueError) as exc_info:
            _sineembed_interleaved(pos_tensor, 8, None)

        assert f"Unknown pos_tensor shape(-1):{width}" in str(exc_info.value)
