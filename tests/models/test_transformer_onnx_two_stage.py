# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX-export regression tests for the two-stage query assembly feeding the decoder.

Eval and export run the two-stage selection with a single group, and when the selected proposals fill every query slot
the remaining learned queries are an empty slice. Concatenating either one put a single-input ``Concat`` and a
``Concat`` with a zero-sized input into every exported detection graph. ONNX Runtime's CoreML execution provider rejects
both, which split the model into extra CoreML partitions with CPU round-trips between them.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from rfdetr.models.math import MLP
from rfdetr.models.transformer import Transformer, gen_encoder_output_proposals

if TYPE_CHECKING:
    import onnx

_D_MODEL = 16
_NUM_CLASSES = 3
# A 4x4 feature map gives the two-stage selection 16 encoder proposals to choose from.
_NUM_PROPOSALS = 16
_DYNAMO_KWARG = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


class _TwoStageExportWrapper(nn.Module):
    """Call ``Transformer.forward`` (one feature level) the way ``LWDETR.forward_export`` does: no padding masks."""

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        refpoint_embed: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the decoder hidden states.

        Args:
            src: Feature map of shape ``(B, C, H, W)``.
            pos: Positional embeddings, same shape as ``src``.
            refpoint_embed: Reference point embeddings of shape ``(num_queries, 4)``.
            query_feat: Query feature embeddings of shape ``(num_queries, C)``.

        Returns:
            Last decoder layer's hidden states of shape ``(B, num_queries, C)`` (export mode keeps only the last layer).

        Examples:
            >>> wrapper = _build_two_stage_wrapper(num_queries=6)
            >>> tuple(wrapper(*_example_inputs(num_queries=6)).shape)
            (1, 6, 16)
        """
        return self.transformer([src], None, [pos], refpoint_embed, query_feat)[0]


def _build_two_stage_wrapper(num_queries: int) -> _TwoStageExportWrapper:
    """Build a two-stage Transformer with the encoder heads LWDETR attaches, in eval and export mode.

    Examples:
        >>> wrapper = _build_two_stage_wrapper(num_queries=6)
        >>> wrapper.transformer.two_stage, wrapper.training
        (True, False)
    """
    transformer = Transformer(
        d_model=_D_MODEL,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=True,
        use_grouppose_keypoints=False,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(_D_MODEL, _NUM_CLASSES)])
    transformer.enc_out_bbox_embed = nn.ModuleList([MLP(_D_MODEL, _D_MODEL, 4, 3)])
    # Same switch LWDETR.export() flips on every submodule before a real export.
    for module in transformer.modules():
        if hasattr(module, "_export") and callable(getattr(module, "export", None)):
            module.export()
    return _TwoStageExportWrapper(transformer).eval()


def _example_inputs(num_queries: int) -> tuple[torch.Tensor, ...]:
    """Return one 4x4 feature level (16 proposals) plus ``num_queries`` learned queries.

    Examples:
        >>> [tuple(t.shape) for t in _example_inputs(num_queries=6)]
        [(1, 16, 4, 4), (1, 16, 4, 4), (6, 4), (6, 16)]
    """
    return (
        torch.randn(1, _D_MODEL, 4, 4),
        torch.randn(1, _D_MODEL, 4, 4),
        torch.rand(num_queries, 4),
        torch.randn(num_queries, _D_MODEL),
    )


@pytest.fixture(scope="module")
def two_stage_onnx(tmp_path_factory: pytest.TempPathFactory) -> onnx.ModelProto:
    """Export a two-stage Transformer whose 6 selected proposals fill all 6 queries, with inferred shapes.

    Examples:
        Pytest fixture functions cannot be called directly outside fixture injection.
        >>> two_stage_onnx(tmp_path_factory)  # doctest: +SKIP
    """
    onnx = pytest.importorskip("onnx", reason="onnx not installed; skip ONNX export tests")
    out = tmp_path_factory.mktemp("onnx_two_stage") / "transformer.onnx"
    torch.onnx.export(
        _build_two_stage_wrapper(num_queries=6),
        _example_inputs(num_queries=6),
        str(out),
        input_names=["src", "pos", "refpoint_embed", "query_feat"],
        output_names=["hs"],
        opset_version=17,
        **_DYNAMO_KWARG,
    )
    return onnx.shape_inference.infer_shapes(onnx.load(str(out)))


def find_single_input_concat_nodes(graph: onnx.GraphProto) -> list[str]:
    """Return the names of ``Concat`` nodes with exactly one input — the shape CoreML rejects.

    Args:
        graph: An ONNX graph, ideally after :func:`onnx.shape_inference.infer_shapes`.

    Returns:
        Names of offending ``Concat`` nodes; empty when none are present.

    Examples:
        >>> import onnx
        >>> from onnx import helper
        >>> node = helper.make_node("Concat", inputs=["a"], outputs=["b"], axis=0, name="bad_concat")
        >>> graph = helper.make_graph([node], "g", [], [])
        >>> find_single_input_concat_nodes(graph)
        ['bad_concat']
    """
    return [n.name for n in graph.node if n.op_type == "Concat" and len(n.input) == 1]


def find_zero_dim_float_concat_nodes(graph: onnx.GraphProto) -> list[str]:
    """Return the names of ``Concat`` nodes with a zero-sized float input — the other shape CoreML rejects.

    Only float data tensors are checked: int64 shape-vector ``Concat``\\s are constant-folded by ONNX Runtime,
    so they never reach CoreML as a real node.

    Args:
        graph: An ONNX graph after :func:`onnx.shape_inference.infer_shapes`, so ``graph.value_info`` carries
            each intermediate tensor's static shape.

    Returns:
        Names of offending ``Concat`` nodes; empty when none are present.

    Examples:
        >>> import onnx
        >>> from onnx import helper, TensorProto
        >>> value = helper.make_tensor_value_info("empty", TensorProto.FLOAT, [1, 0, 4])
        >>> node = helper.make_node("Concat", inputs=["empty", "other"], outputs=["b"], axis=1, name="bad_concat")
        >>> graph = helper.make_graph([node], "g", [], [], value_info=[value])
        >>> find_zero_dim_float_concat_nodes(graph)
        ['bad_concat']
    """
    float_dims = {
        value.name: [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
        for value in graph.value_info
        if value.type.tensor_type.elem_type == 1
    }
    return [
        n.name for n in graph.node if n.op_type == "Concat" and any(0 in float_dims.get(name, []) for name in n.input)
    ]


@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestTwoStageExportGraph:
    """The exported two-stage graph must not contain the degenerate Concats CoreML rejects."""

    def test_has_no_single_input_concat(self, two_stage_onnx: onnx.ModelProto) -> None:
        assert find_single_input_concat_nodes(two_stage_onnx.graph) == []

    def test_has_no_float_concat_with_empty_input(self, two_stage_onnx: onnx.ModelProto) -> None:
        assert find_zero_dim_float_concat_nodes(two_stage_onnx.graph) == []


class TestTwoStageQueryAssembly:
    """Both query-assembly branches keep every query slot."""

    @pytest.mark.parametrize(
        "num_queries",
        [
            pytest.param(6, id="proposals_fill_all_queries"),
            pytest.param(_NUM_PROPOSALS + 4, id="learned_queries_remain"),
        ],
    )
    def test_decoder_receives_every_query(self, num_queries: int) -> None:
        wrapper = _build_two_stage_wrapper(num_queries=num_queries)

        with torch.no_grad():
            hs = wrapper(*_example_inputs(num_queries=num_queries))

        assert hs.shape[-2:] == (num_queries, _D_MODEL)

    def test_learned_query_suffix_matches_raw_input_when_queries_remain(self) -> None:
        """The learned-query suffix (``refpoint_embed[..., ts_len:, :]``) is concatenated as-is, with no transform
        applied, so it must reach the decoder unchanged, in the same order -- not zeroed, reordered, or mis-combined
        with the two-stage-selected slots ahead of it."""
        num_queries = _NUM_PROPOSALS + 4
        ts_len = _NUM_PROPOSALS  # topk = min(num_queries, encoder proposals) = min(20, 16) = 16
        wrapper = _build_two_stage_wrapper(num_queries=num_queries)
        # Sentinel-distinct per-slot values: a reordering or a wrong slice would be caught, not just a shape match.
        refpoint_embed = (torch.arange(num_queries * 4, dtype=torch.float32) / (num_queries * 4)).reshape(
            num_queries, 4
        )
        src, pos, _, query_feat = _example_inputs(num_queries=num_queries)

        original_decoder_forward = wrapper.transformer.decoder.forward
        captured: dict[str, torch.Tensor] = {}

        def _spy_decoder_forward(*args: object, **kwargs: object) -> object:
            captured["refpoints_unsigmoid"] = kwargs["refpoints_unsigmoid"]
            return original_decoder_forward(*args, **kwargs)

        wrapper.transformer.decoder.forward = _spy_decoder_forward

        with torch.no_grad():
            wrapper(src, pos, refpoint_embed, query_feat)

        actual_suffix = captured["refpoints_unsigmoid"][0, ts_len:, :]
        torch.testing.assert_close(actual_suffix, refpoint_embed[ts_len:, :])


class TestSingleLevelProposalsFastPath:
    """``gen_encoder_output_proposals``'s single-level fast path (``proposals[0]``) must match the general
    ``torch.cat(proposals, dim=1)`` path it replaces for exactly one level -- only checked for graph shape/node
    absence in ``TestTwoStageExportGraph`` above, never against the ``torch.cat`` reference values."""

    def test_single_level_fast_path_matches_cat_reference(self) -> None:
        torch.manual_seed(0)
        memory = torch.randn(1, _NUM_PROPOSALS, _D_MODEL)

        _, fast_path_proposals = gen_encoder_output_proposals(memory, None, [(4, 4)], unsigmoid=False)
        reference_proposals = torch.cat([fast_path_proposals], dim=1)

        torch.testing.assert_close(fast_path_proposals, reference_proposals)


class TestGroupDetrOneTrainingMode:
    """``group_detr == 1`` at transformer.py:798 is only exercised via ``.eval()`` in the tests above. The branch
    is logic-identical in train/eval, so this is a coverage addition, not a bug-hunt."""

    def test_group_detr_one_in_training_mode_does_not_error(self) -> None:
        num_queries = 6
        wrapper = _build_two_stage_wrapper(num_queries=num_queries)
        wrapper.train()

        hs = wrapper(*_example_inputs(num_queries=num_queries))

        assert hs.shape[-2:] == (num_queries, _D_MODEL)


class TestNumRegistersWithFullProposalRemainder:
    """Register insertion (``num_registers > 0``) has no regression case combined with the full-remainder branch
    (``ts_len == num_queries``, i.e. ``refpoint_embed_subset.shape[-2] == 0``): both conditions land together
    whenever a model with decoder registers also has every query slot filled by two-stage proposals.

    Exercised in the non-export eval path: register removal (``hs.split(n_with_reg, dim=2)``) indexes the
    decoder's stacked-intermediate-layers output, a shape only the non-export decoder path produces (export mode
    returns the last layer alone, with no layer axis at all). No released config combines
    ``num_decoder_registers > 0`` with export, so the export path is not this test's target.
    """

    def test_registers_and_full_remainder_together(self) -> None:
        num_queries = 6  # ts_len == num_queries -> refpoint_embed_subset.shape[-2] == 0 (full remainder)
        transformer = Transformer(
            d_model=_D_MODEL,
            num_queries=num_queries,
            num_decoder_layers=1,
            sa_nhead=4,
            ca_nhead=4,
            num_feature_levels=1,
            dec_n_points=1,
            return_intermediate_dec=True,
            lite_refpoint_refine=True,
            two_stage=True,
            bbox_reparam=True,
            use_grouppose_keypoints=False,
            num_registers=2,
        ).eval()
        transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(_D_MODEL, _NUM_CLASSES)])
        transformer.enc_out_bbox_embed = nn.ModuleList([MLP(_D_MODEL, _D_MODEL, 4, 3)])
        src, pos, refpoint_embed, query_feat = _example_inputs(num_queries=num_queries)
        # The non-export decoder path requires real valid_ratios, computed from a mask; an all-False mask keeps
        # every position valid, matching the no-padding assumption `_example_inputs`'s other callers rely on.
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)

        with torch.no_grad():
            hs = transformer([src], [mask], [pos], refpoint_embed, query_feat)[0]

        assert hs.shape[-2:] == (num_queries, _D_MODEL)
