# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Integration test for segmentation model export.

Tests the fix for the bug where exporting a segmentation model would crash
with 'AttributeError: dict object has no attribute shape' because pred_masks
can be either a tensor or a dictionary depending on the model configuration.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from rfdetr import RFDETRSegNano


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed, run: pip install rfdetr[onnxexport]",
)
def test_segmentation_model_export_no_crash(tmp_path: Path) -> None:
    """
    Test that exporting a segmentation model does not crash.

    This is the actual integration test that exercises the full export path.
    """
    model = RFDETRSegNano()

    # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
    model.export(output_dir=str(tmp_path), simplify=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"
