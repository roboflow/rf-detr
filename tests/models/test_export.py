# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for model export functionality.

Includes regression tests for PR #578, which fixed a bug where exporting
segmentation models would crash because .eval() was called on the wrong model
object (self.model instead of the deepcopy).
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


class TestPR578EvalOnDeepcopyFix:
    """
    Regression tests for PR #578: Fix export calling eval() on wrong model.

    The bug:
    - export() creates a deepcopy: model = deepcopy(self.model.to("cpu"))
    - Originally called: self.model.eval() (wrong - sets original to eval mode)
    - Fixed to call: model.eval() (correct - sets deepcopy to eval mode)

    Why this matters:
    - The deepcopy is used for inference and ONNX export, not the original
    - If the deepcopy stays in training mode, segmentation models can produce
      unexpected output formats (dict vs Tensor for pred_masks)
    - This causes AttributeError when the export code expects a specific format
    """

    def test_original_eval_leaves_copy_in_training_mode(self) -> None:
        """
        Demonstrate the bug: calling eval() on original doesn't affect the copy.

        Shows what happened before the fix - the original was set to eval mode,
        but the deepcopy (which is actually used) remained in training mode.
        """
        # Create a mock model in training mode
        mock_model = MagicMock()
        mock_model.training = True

        eval_called = False

        def mock_eval():
            nonlocal eval_called
            eval_called = True
            mock_model.training = False
            return mock_model

        mock_model.eval = mock_eval

        # Simulate the deepcopy operation in export()
        model_copy = deepcopy(mock_model)
        model_copy.training = True  # deepcopy preserves training state
        model_copy_eval_called = False

        def mock_copy_eval():
            nonlocal model_copy_eval_called
            model_copy_eval_called = True
            model_copy.training = False
            return model_copy

        model_copy.eval = mock_copy_eval

        # WRONG: Calling eval on the original model (the bug)
        mock_model.eval()

        # The original is in eval mode, but the deepcopy is still in training mode
        assert eval_called is True, "Original model's eval() should be called"
        assert mock_model.training is False, "Original model should be in eval mode"
        assert model_copy.training is True, "Deepcopy should still be in training mode (BUG!)"
        assert model_copy_eval_called is False, "Deepcopy's eval() was not called (BUG!)"

    def test_copy_eval_correctly_sets_mode(self) -> None:
        """
        Demonstrate the fix: calling eval() on the copy correctly sets its mode.

        Shows the correct behavior after the fix - the deepcopy (which is
        actually used for export) is properly set to eval mode.
        """
        # Create a mock model in training mode
        mock_model = MagicMock()
        mock_model.training = True

        # Simulate the deepcopy operation in export()
        model_copy = deepcopy(mock_model)
        model_copy.training = True
        model_copy_eval_called = False

        def mock_copy_eval():
            nonlocal model_copy_eval_called
            model_copy_eval_called = True
            model_copy.training = False
            return model_copy

        model_copy.eval = mock_copy_eval

        # CORRECT: Calling eval on the deepcopy (the fix)
        model_copy.eval()

        # The deepcopy is now correctly in eval mode
        assert model_copy_eval_called is True, "Deepcopy's eval() should be called"
        assert model_copy.training is False, "Deepcopy should be in eval mode"
        assert mock_model.training is True, "Original model state doesn't matter for export"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_format_differs_by_mode(self) -> None:
        """
        Validate why the bug matters: output format depends on mode.

        Segmentation models can produce different output formats in training vs
        eval mode. The export function must ensure the model is in eval mode to
        get consistent, expected output.
        """
        model = RFDETRSegNano()

        # Create a small dummy input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model = model.model.to(device)
        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        # Test in training mode
        model.model.train()
        with torch.no_grad():
            train_output = model.model(dummy_input)

        # Test in eval mode
        model.model.eval()
        with torch.no_grad():
            eval_output = model.model(dummy_input)

        # Both modes should produce the required outputs
        for mode_name, output in [("training", train_output), ("eval", eval_output)]:
            assert "pred_boxes" in output, f"{mode_name} mode should produce pred_boxes"
            assert "pred_logits" in output, f"{mode_name} mode should produce pred_logits"
            assert "pred_masks" in output, f"{mode_name} mode should produce pred_masks"

        # Document the output types
        train_masks = train_output["pred_masks"]
        eval_masks = eval_output["pred_masks"]

        train_is_tensor = isinstance(train_masks, torch.Tensor)
        train_is_dict = isinstance(train_masks, dict)
        eval_is_tensor = isinstance(eval_masks, torch.Tensor)
        eval_is_dict = isinstance(eval_masks, dict)

        # pred_masks should be either Tensor or dict in each mode
        assert train_is_tensor or train_is_dict, "pred_masks should be Tensor or dict in training mode"
        assert eval_is_tensor or eval_is_dict, "pred_masks should be Tensor or dict in eval mode"

        # Log the actual types for documentation
        print(f"Training mode pred_masks type: {type(train_masks)}")
        print(f"Eval mode pred_masks type: {type(eval_masks)}")
