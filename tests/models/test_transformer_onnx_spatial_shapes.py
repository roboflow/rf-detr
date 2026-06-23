# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX-export regression tests for the shared Transformer's spatial_shapes path.

These guard the fix that builds ``spatial_shapes`` from symbolic feature-map shapes (``Shape`` -> ``Concat``) instead of
``torch.empty`` + in-place index assignment. The latter (added in #871 to keep the trace symbolic for dynamic-batch
export) emitted a ``ScatterND`` that fed a shape tensor, which TensorRT rejects ("IScatterLayer cannot be used to
compute a shape tensor"). The constant-baking ``torch.as_tensor`` alternative avoids the ScatterND but regresses the
symbolic trace back to a baked constant.

The Transformer is shared by detection, segmentation and keypoint models, so a single low-level export here covers the
spatial_shapes path for all of them.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx", reason="onnx not installed; skip ONNX export tests")

from rfdetr.models.transformer import Transformer  # noqa: E402


class _TransformerExportWrapper(nn.Module):
    """Wrap Transformer.forward with tensor-only args so it can be ONNX-exported."""

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, s0, s1, p0, p1, m0, m1, refpoint_embed, query_feat):
        outputs = self.transformer([s0, s1], [m0, m1], [p0, p1], refpoint_embed, query_feat, cross_attn_srcs=None)
        return outputs[0]


def _build_wrapper() -> _TransformerExportWrapper:
    transformer = Transformer(
        d_model=16,
        num_queries=6,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        use_grouppose_keypoints=False,
    )
    return _TransformerExportWrapper(transformer.eval()).eval()


def _example_inputs(h0: int = 4, w0: int = 4, h1: int = 2, w1: int = 2, batch: int = 1):
    return (
        torch.randn(batch, 16, h0, w0),
        torch.randn(batch, 16, h1, w1),
        torch.randn(batch, 16, h0, w0),
        torch.randn(batch, 16, h1, w1),
        torch.zeros(batch, h0, w0, dtype=torch.bool),
        torch.zeros(batch, h1, w1, dtype=torch.bool),
        torch.rand(6, 4),
        torch.randn(6, 16),
    )


def _export(tmp_path: Path) -> onnx.ModelProto:
    wrapper = _build_wrapper()
    out = tmp_path / "transformer.onnx"
    torch.onnx.export(
        wrapper,
        _example_inputs(),
        str(out),
        input_names=["s0", "s1", "p0", "p1", "m0", "m1", "refpoint_embed", "query_feat"],
        output_names=["hs"],
        opset_version=17,
        dynamo=False,
    )
    return onnx.load(str(out))


def test_spatial_shapes_export_has_no_scatternd(tmp_path: Path) -> None:
    """The exported Transformer must not contain a ScatterND (TRT shape-tensor killer)."""
    model = _export(tmp_path)
    op_types = [n.op_type for n in model.graph.node]
    assert "ScatterND" not in op_types, (
        "ScatterND reintroduced in Transformer export — spatial_shapes is no longer "
        "built from symbolic Shape ops; this breaks TensorRT engine building."
    )


def test_spatial_shapes_export_is_shape_derived(tmp_path: Path) -> None:
    """spatial_shapes should come from Shape ops (dynamic), not a baked constant."""
    model = _export(tmp_path)
    op_types = [n.op_type for n in model.graph.node]
    assert "Shape" in op_types, "Expected Shape nodes — spatial_shapes appears to be baked as a constant."
