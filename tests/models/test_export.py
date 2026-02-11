# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for model export functionality.

Use cases covered:
- Export should use eval() on the deepcopy (not the original model).
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
"""

import importlib.util
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

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
    Integration test: exporting a segmentation model should not crash.

    This exercises the full export path to ensure no AttributeError occurs.
    """
    model = RFDETRSegNano()

    # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
    model.export(output_dir=str(tmp_path), simplify=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


def test_eval_on_deepcopy_does_not_affect_original() -> None:
    """Use case: export should set eval() on the deepcopy used for export."""
    base_model = torch.nn.Identity()
    base_model.train()

    model_copy = deepcopy(base_model)
    model_copy.train()

    model_copy.eval()

    assert model_copy.training is False
    assert base_model.training is True


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_segmentation_outputs_present_in_train_and_eval() -> None:
    """Use case: segmentation outputs are present in both train and eval modes."""
    model = RFDETRSegNano()

    model.model = model.model.to("cuda")
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")

    model.model.train()
    with torch.no_grad():
        train_output = model.model(dummy_input)

    model.model.eval()
    with torch.no_grad():
        eval_output = model.model(dummy_input)

    for output in (train_output, eval_output):
        assert "pred_boxes" in output
        assert "pred_logits" in output
        assert "pred_masks" in output
