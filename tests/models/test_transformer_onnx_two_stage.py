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
from rfdetr.models.transformer import Transformer

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
            Decoder hidden states of shape ``(num_layers, B, num_queries, C)``.
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
    """Export a two-stage Transformer whose 6 selected proposals fill all 6 queries, with inferred shapes."""
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


class TestTwoStageExportGraph:
    """The exported two-stage graph must not contain the degenerate Concats CoreML rejects."""

    def test_has_no_single_input_concat(self, two_stage_onnx: onnx.ModelProto) -> None:
        single_input = [n.name for n in two_stage_onnx.graph.node if n.op_type == "Concat" and len(n.input) == 1]
        assert single_input == []

    def test_has_no_float_concat_with_empty_input(self, two_stage_onnx: onnx.ModelProto) -> None:
        # Only float data tensors: int64 shape-vector Concats are constant-folded by ONNX Runtime.
        float_dims = {
            value.name: [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
            for value in two_stage_onnx.graph.value_info
            if value.type.tensor_type.elem_type == 1
        }
        empty_input = [
            n.name
            for n in two_stage_onnx.graph.node
            if n.op_type == "Concat" and any(0 in float_dims.get(name, []) for name in n.input)
        ]
        assert empty_input == []


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
