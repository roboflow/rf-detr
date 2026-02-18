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
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr import RFDETRSegNano


@contextmanager
def ignore_tracer_warnings() -> Iterator[None]:
    """Suppress torch.jit.TracerWarning during export tests to reduce log spam."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        yield


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
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), simplify=False, verbose=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
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

    # Access the underlying torch module and set it to training mode
    torch_model = model.model.model.to("cuda")
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Store the original deepcopy function
    original_deepcopy = deepcopy

    # Mock to track eval() calls
    eval_mock = Mock()

    def tracking_deepcopy(obj):
        """Deepcopy wrapper that tracks eval() calls on the copy"""
        copied = original_deepcopy(obj)

        # Only track eval calls on torch.nn.Module objects
        if isinstance(copied, torch.nn.Module):
            # Save reference to original eval before replacing it
            original_eval = copied.eval

            def tracked_eval(*args, **kwargs):
                """Wrapper that tracks calls while delegating to the original eval"""
                eval_mock()
                return original_eval(*args, **kwargs)

            # Replace eval with tracked version
            copied.eval = tracked_eval

        return copied

    # Patch deepcopy in the main module where export is defined
    with patch('rfdetr.main.deepcopy', side_effect=tracking_deepcopy):
        try:
            with ignore_tracer_warnings():
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

    # Verify the original model's training state was not changed
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed, run: pip install rfdetr[onnxexport]",
)
def test_export_does_not_change_original_training_state(tmp_path: Path) -> None:
    """
    Verify that calling export() does not change the original model's train/eval state.

    This ensures that export() puts a deepcopy of the model in eval mode without
    mutating the underlying training model used by RF-DETR.
    """
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model), as in other tests
    torch_model = model.model.model.to("cuda")

    # Ensure the original model is in training mode
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Call export() on the high-level model; this should not change the original model's mode
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), simplify=False)

    # After export, the original underlying model should still be in training mode
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["train", "eval"], ids=["train_mode", "eval_mode"])
def test_segmentation_outputs_present_in_train_and_eval(mode: Literal["train", "eval"]) -> None:
    """Use case: segmentation outputs are present in both train and eval modes."""
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model)
    torch_model = model.model.model.to("cuda")

    # Use resolution compatible with model's patch size (312 for seg-nano)
    resolution = model.model.resolution
    dummy_input = torch.randn(1, 3, resolution, resolution, device="cuda")

    if mode == "train":
        torch_model.train()
    else:
        torch_model.eval()

    with torch.no_grad():
        output = torch_model(dummy_input)

    assert "pred_boxes" in output
    assert "pred_logits" in output
    assert "pred_masks" in output
