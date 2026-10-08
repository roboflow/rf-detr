# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the graph preparation that ``format="executorch", backend="xnnpack"`` lowers."""

from __future__ import annotations

import sys
from unittest import mock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export._executorch.xnnpack import decompose_attention, fold_constants, unmasked_attention

executorch_only = pytest.mark.skipif(not _IS_EXECUTORCH_AVAILABLE, reason="executorch not installed")


def _query_key_value(*batch: int, length: int = 7, source: int = 9, channels: int = 16) -> list[torch.Tensor]:
    """Build random ``(query, key, value)`` with the given leading dimensions.

    Args:
        *batch: Leading dimensions of each tensor.
        length: Number of queries.
        source: Number of keys and values.
        channels: Channels of each query, key and value.

    Returns:
        The query, key and value tensors.

    Examples:
        >>> [tuple(t.shape) for t in _query_key_value(2, 3)]
        [(2, 3, 7, 16), (2, 3, 9, 16), (2, 3, 9, 16)]
    """
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(*batch, rows, channels, generator=generator) for rows in (length, source, source)]


class _EncoderSelfAttention(nn.Module):
    """Multi-head self-attention written the way the DINOv2 encoder writes it.

    The output goes through ``permute(...).contiguous().view(...)``, so the captured graph depends on the memory layout
    of the attention output.
    """

    def __init__(self, channels: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = (nn.Linear(channels, channels) for _ in range(3))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the channels of *x* into attention heads.

        Args:
            x: Tensor of shape ``(batch, length, channels)``.

        Returns:
            A permuted view of shape ``(batch, heads, length, channels // heads)``.

        Examples:
            >>> tuple(_EncoderSelfAttention()._split_heads(torch.zeros(2, 6, 32)).shape)
            (2, 4, 6, 8)
        """
        batch, length, channels = x.shape
        return x.view(batch, length, self.heads, channels // self.heads).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = (self._split_heads(layer(x)) for layer in (self.query, self.key, self.value))
        context = F.scaled_dot_product_attention(query, key, value)
        return context.permute(0, 2, 1, 3).contiguous().view(x.shape)


class _DecoderSelfAttention(nn.Module):
    """``nn.MultiheadAttention`` called the way the RF-DETR decoder calls it.

    The query and key carry a positional embedding and the value does not. For a key that is not the value, PyTorch
    slices the packed ``in_proj_weight`` into the query, key and value weights at run time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.position = nn.Parameter(torch.randn(6, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x + self.position
        return self.attention(query, query, x, need_weights=False)[0]


class _MaskedAttention(nn.Module):
    """Attention with a boolean mask, which must keep the default decomposition."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool).tril()
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


class _Doubler(nn.Module):
    """A program with no weights and no constants, so there is nothing to fold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class _FilledConstant(nn.Module):
    """Adds a filled tensor, which ``fold_constants`` must leave as a call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.full((8, 8), 3.0)


def _export(module: nn.Module, *inputs: torch.Tensor) -> torch.export.ExportedProgram:
    """Capture *module* in eval mode as a program, without strict mode.

    Args:
        module: The module to capture.
        *inputs: Example inputs of the module.

    Returns:
        The captured program.

    Examples:
        >>> isinstance(_export(_Doubler(), torch.zeros(2)), torch.export.ExportedProgram)
        True
    """
    return torch.export.export(module.eval(), inputs, strict=False)


def _call_targets(program: torch.export.ExportedProgram) -> set[str]:
    """Name every operator that *program* calls.

    Args:
        program: The captured program.

    Returns:
        The target of each ``call_function`` node, such as ``"aten.mul.Tensor"``.

    Examples:
        >>> sorted(_call_targets(_export(_Doubler(), torch.zeros(2))))
        ['aten.mul.Tensor']
    """
    return {str(node.target) for node in program.graph.nodes if node.op == "call_function"}


class TestUnmaskedAttention:
    """``unmasked_attention`` computes the same values as ``F.scaled_dot_product_attention`` without a mask."""

    @pytest.mark.parametrize(
        "batch",
        [pytest.param((), id="2d"), pytest.param((5,), id="3d"), pytest.param((2, 3), id="4d")],
    )
    def test_matches_sdpa(self, batch: tuple[int, ...]) -> None:
        query, key, value = _query_key_value(*batch)
        expected = F.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(unmasked_attention(query, key, value), expected)

    def test_matches_sdpa_with_a_scale(self) -> None:
        query, key, value = _query_key_value(2, 3)
        expected = F.scaled_dot_product_attention(query, key, value, scale=0.3)
        torch.testing.assert_close(unmasked_attention(query, key, value, scale=0.3), expected)

    def test_matches_sdpa_for_a_different_value_width(self) -> None:
        query, key, _ = _query_key_value(2, 3)
        value = torch.randn(2, 3, 9, 24)
        expected = F.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(unmasked_attention(query, key, value), expected)

    @pytest.mark.parametrize(
        "memory_order",
        [
            pytest.param((0, 1, 2, 3), id="contiguous"),
            pytest.param((0, 2, 1, 3), id="encoder"),
            pytest.param((2, 0, 1, 3), id="multihead"),
        ],
    )
    def test_4d_output_has_the_memory_layout_of_sdpa(self, memory_order: tuple[int, int, int, int]) -> None:
        restore = sorted(range(4), key=memory_order.__getitem__)
        query, key, value = (t.permute(memory_order).contiguous().permute(restore) for t in _query_key_value(2, 3))
        expected = F.scaled_dot_product_attention(query, key, value)
        assert unmasked_attention(query, key, value).stride() == expected.stride()

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"attn_mask": torch.ones(7, 9, dtype=torch.bool)}, id="mask"),
            pytest.param({"is_causal": True}, id="causal"),
            pytest.param({"dropout_p": 0.1}, id="dropout"),
            pytest.param({"enable_gqa": True}, id="gqa"),
        ],
    )
    def test_other_calls_keep_the_default_decomposition(self, kwargs: dict[str, object]) -> None:
        assert unmasked_attention(*_query_key_value(2, 3), **kwargs) is NotImplemented

    @pytest.mark.parametrize(
        ("shapes", "dtype"),
        [
            pytest.param(((2, 3, 7, 16), (1, 3, 9, 16), (1, 3, 9, 16)), torch.float32, id="broadcast-key-value"),
            pytest.param(((1, 3, 7, 16), (2, 3, 9, 16), (2, 3, 9, 16)), torch.float32, id="broadcast-query"),
            pytest.param(((2, 3, 7, 16), (9, 16), (9, 16)), torch.float32, id="2d-key-value"),
            pytest.param(((2, 3, 7, 16), (2, 3, 0, 16), (2, 3, 0, 16)), torch.float32, id="empty-source"),
            pytest.param(((2, 3, 0, 16), (2, 3, 9, 16), (2, 3, 9, 16)), torch.float32, id="empty-query"),
            pytest.param(((2, 3, 7, 16), (2, 3, 9, 16), (2, 3, 9, 16)), torch.float16, id="float16"),
            pytest.param(((2, 3, 7, 16), (2, 3, 9, 16), (2, 3, 9, 16)), torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_inputs_the_decomposition_cannot_take_keep_the_default_decomposition(
        self, shapes: tuple[tuple[int, ...], ...], dtype: torch.dtype
    ) -> None:
        """Broadcast leading dimensions, empty tensors and low-precision dtypes are left to the operator.

        The decomposition flattens the leading dimensions for ``bmm``, which raised ``RuntimeError`` for the broadcast
        and empty shapes, and it normalizes in the dtype of its inputs, which loses accuracy in float16 and bfloat16.
        """
        query, key, value = (torch.zeros(shape, dtype=dtype) for shape in shapes)
        assert unmasked_attention(query, key, value) is NotImplemented


class TestDecomposeAttention:
    """``decompose_attention`` removes unmasked attention from the captured program and keeps its values."""

    @pytest.mark.parametrize(
        "module",
        [pytest.param(_EncoderSelfAttention, id="encoder"), pytest.param(_DecoderSelfAttention, id="decoder")],
    )
    def test_unmasked_attention_is_decomposed_with_the_same_output(self, module: type[nn.Module]) -> None:
        x = torch.randn(2, 6, 32)
        program = _export(module(), x)
        expected = program.module()(x)
        decomposed = decompose_attention(program)
        targets = _call_targets(decomposed)
        assert not any("scaled_dot_product_attention" in target for target in targets), sorted(targets)
        torch.testing.assert_close(decomposed.module()(x), expected)

    def test_masked_attention_keeps_its_output(self) -> None:
        inputs = _query_key_value(2, 3)
        program = _export(_MaskedAttention(), *inputs)
        expected = program.module()(*inputs)
        torch.testing.assert_close(decompose_attention(program).module()(*inputs), expected)


@executorch_only
@pytest.mark.e2e_executorch
class TestFoldConstants:
    """``fold_constants`` computes the weight slices of ``nn.MultiheadAttention`` at export time.

    Marked ``e2e_executorch`` so the ExecuTorch rows of ``ci-integrations.yml``, which select that marker, run it: the
    CPU jobs install no ExecuTorch, and these tests lower a real program.
    """

    @staticmethod
    def _program() -> tuple[torch.export.ExportedProgram, torch.Tensor]:
        """Capture the decoder attention with its unmasked attention decomposed.

        Returns:
            The captured program and the example input it was captured with.

        Examples:
            >>> program, x = TestFoldConstants._program()
            >>> tuple(x.shape), isinstance(program, torch.export.ExportedProgram)
            ((2, 6, 32), True)
        """
        x = torch.randn(2, 6, 32)
        return decompose_attention(_export(_DecoderSelfAttention(), x)), x

    def test_filled_tensors_are_not_folded(self) -> None:
        """``aten.full`` stays a call instead of becoming a stored constant of its full size.

        ExecuTorch's pass skips ``aten.full`` by default for that reason, but it names the op in the Edge dialect, so a
        plain call to the pass on an ATen program folds it.
        """
        program = fold_constants(_export(_FilledConstant(), torch.randn(8, 8)))
        assert "aten.full.default" in _call_targets(program)

    def test_program_without_constants_is_unchanged(self) -> None:
        """A program with nothing to fold gains no constants and computes the same values."""
        x = torch.randn(4)
        program = fold_constants(_export(_Doubler(), x))
        assert not program.constants
        torch.testing.assert_close(program.module()(x), x * 2)

    def test_weight_slices_become_constants_with_their_own_storage(self) -> None:
        program = fold_constants(self._program()[0])
        slices = [tensor for tensor in program.constants.values() if isinstance(tensor, torch.Tensor)]
        assert slices, "expected the in_proj weight and bias slices as folded constants"
        for tensor in slices:
            assert tensor.storage_offset() == 0
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param(slice(0, 32), id="query"),
            pytest.param(slice(32, 64), id="key"),
            pytest.param(slice(64, 96), id="value"),
        ],
    )
    def test_weight_slices_hold_the_rows_of_in_proj_weight(self, rows: slice) -> None:
        """Each folded weight is the query, key or value rows of the packed ``in_proj_weight``."""
        module = _DecoderSelfAttention()
        program = fold_constants(decompose_attention(_export(module, torch.randn(2, 6, 32))))
        expected = module.attention.in_proj_weight[rows]
        folded = [constant for constant in program.constants.values() if isinstance(constant, torch.Tensor)]
        assert any(torch.equal(constant, expected) for constant in folded)

    def test_output_is_unchanged(self) -> None:
        program, x = self._program()
        expected = program.module()(x)
        actual = fold_constants(program).module()(x)
        # A linear layer with a folded weight can take another matrix-multiplication path than one that slices
        # its weight at run time, so the results agree to float32 rounding, not bit for bit.
        torch.testing.assert_close(actual, expected)

    @executorch_only
    def test_lowered_program_matches_eager(self) -> None:
        """The ``.pte`` holds each weight slice, not the start of the packed ``in_proj_weight``."""
        from executorch import runtime
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        program, x = self._program()
        expected = program.module()(x)
        lowered = to_edge_transform_and_lower(fold_constants(program), partitioner=[XnnpackPartitioner()])
        # The program reads from this buffer while it runs, so the test keeps a reference to it.
        buffer = lowered.to_executorch().buffer
        method = runtime.Runtime.get().load_program(buffer).load_method("forward")
        torch.testing.assert_close(method.execute([x])[0], expected)


class TestFoldConstantsWithoutConstantPropPass:
    """``fold_constants`` names what to upgrade when this ExecuTorch install ships without its pass."""

    def test_missing_pass_raises_an_actionable_import_error(self) -> None:
        """A bare ``ModuleNotFoundError`` after ``torch.export`` ran would say nothing about the install to fix."""
        with mock.patch.dict(sys.modules, {"executorch.exir.passes.constant_prop_pass": None}):
            with pytest.raises(ImportError, match="upgrade executorch"):
                fold_constants(mock.MagicMock())
