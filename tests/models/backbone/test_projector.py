# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for the projector LayerNorm ONNX export fix.

The custom LayerNorm in projector.py performs channel-wise normalization on
(B, C, H, W) inputs. The original implementation passed ``(x.size(3),)`` as
``normalized_shape`` to ``F.layer_norm``, which worked at runtime but caused
ONNX export failures on PyTorch 2.9+ because the exporter treats dynamic
tensor dimensions as symbolic (non-constant) values.

The fix replaces the dynamic ``(x.size(3),)`` with the static attribute
``self.normalized_shape``, which is set once in ``__init__``.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from rfdetr.models.backbone.projector import LayerNorm


def test_layernorm_forward_preserves_shape():
    """LayerNorm should preserve input shape (B, C, H, W)."""
    channels = 64
    ln = LayerNorm(channels)
    x = torch.randn(2, channels, 8, 8)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_normalized_shape_is_static():
    """normalized_shape must be a plain tuple (not derived from a tensor)
    so that ONNX exporters see it as a constant."""
    channels = 128
    ln = LayerNorm(channels)
    assert ln.normalized_shape == (channels,)
    assert isinstance(ln.normalized_shape, tuple)


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed, run: pip install rfdetr[onnxexport]",
)
def test_layernorm_onnx_export(tmp_path: Path):
    """The LayerNorm module should be exportable to ONNX without errors.

    This is the regression test for the fix: prior to the change, PyTorch 2.9+
    raised ``SymbolicValueError`` because ``(x.size(3),)`` was treated as a
    non-constant shape during tracing.
    """
    channels = 64
    ln = LayerNorm(channels)
    ln.train(False)
    dummy = torch.randn(1, channels, 8, 8)

    onnx_path = str(tmp_path / "layernorm.onnx")
    torch.onnx.export(
        ln,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
    )
    assert Path(onnx_path).stat().st_size > 0, "ONNX export should produce non-empty file"
