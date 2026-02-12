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
from unittest.mock import Mock, patch

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


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed, run: pip install rfdetr[onnxexport]",
)
def test_export_calls_eval_on_deepcopy_not_original(tmp_path: Path) -> None:
    """
    Verify that Model.export() calls eval() on the deepcopy, not the original model.
    
    This test patches deepcopy to track whether eval() is called on the copied
    model during export, ensuring the fix in PR #578 is working correctly.
    """
    model = RFDETRSegNano()
    
    # Store the original deepcopy function
    original_deepcopy = deepcopy
    
    # Mock to track eval() calls
    eval_mock = Mock()
    
    def tracking_deepcopy(obj):
        """Deepcopy wrapper that tracks eval() calls on the copy"""
        copied = original_deepcopy(obj)
        
        # Only track eval calls on torch.nn.Module objects
        if isinstance(copied, torch.nn.Module):
            original_eval = copied.eval
            
            def tracked_eval(*args, **kwargs):
                eval_mock()
                return original_eval(*args, **kwargs)
            
            copied.eval = tracked_eval
        
        return copied
    
    # Patch deepcopy in the main module where export is defined
    with patch('rfdetr.main.deepcopy', side_effect=tracking_deepcopy):
        try:
            model.export(output_dir=str(tmp_path), simplify=False)
        except (ImportError, OSError, RuntimeError):
            # Expected failures: missing dependencies, network issues, CUDA errors
            # These are acceptable as we're testing the deepcopy/eval pattern, not the full export
            pass
    
    # Verify that eval() was called on the deepcopy during export
    assert eval_mock.call_count > 0, (
        "export() should call eval() on the deepcopy. "
        "This ensures the exported model is in eval mode without affecting the original."
    )


def test_eval_on_deepcopy_does_not_affect_original() -> None:
    """
    Verify the pattern: deepcopy allows independent train/eval state.
    
    This demonstrates the fundamental behavior that Model.export() relies on:
    calling eval() on a deepcopy doesn't affect the original model's training state.
    """
    # Simulate the pattern used in Model.export()
    original_model = torch.nn.Linear(10, 10)
    original_model.train()  # Start in training mode
    
    # This is what happens in Model.export(): deepcopy then eval on the copy
    model_copy = deepcopy(original_model)
    model_copy.eval()
    
    # Verify the copy is in eval mode but original is still in training mode
    assert model_copy.training is False, "Deepcopy should be in eval mode"
    assert original_model.training is True, "Original should still be in training mode"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_segmentation_outputs_present_in_train_and_eval() -> None:
    """Use case: segmentation outputs are present in both train and eval modes."""
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model)
    torch_model = model.model.model.to("cuda")
    
    # Use resolution compatible with model's patch size (312 for seg-nano)
    resolution = model.model.resolution
    dummy_input = torch.randn(1, 3, resolution, resolution, device="cuda")

    torch_model.train()
    with torch.no_grad():
        train_output = torch_model(dummy_input)

    torch_model.eval()
    with torch.no_grad():
        eval_output = torch_model(dummy_input)

    for output in (train_output, eval_output):
        assert "pred_boxes" in output
        assert "pred_logits" in output
        assert "pred_masks" in output
