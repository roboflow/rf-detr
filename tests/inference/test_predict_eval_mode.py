# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests eval-mode handling for unoptimized inference."""

from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest
import torch

from rfdetr import detr as detr_module

from .helpers import _BaseFakeRFDETR


class _FakeModelWithDropout(torch.nn.Module):
    """Minimal module whose behavior differs between train and eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through dropout, active only in train mode."""
        return self.dropout(x)


class _FakeModelContext:
    """Minimal model context supplying the attributes predict() and train() need."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = _FakeModelWithDropout()
        self.inference_model = None


class _FakeRFDETR(_BaseFakeRFDETR):
    """Concrete test double: provides a dropout-bearing model for eval-mode tests."""

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _FakeModelContext:
        """Return a minimal model context with a dropout-bearing module."""
        return _FakeModelContext()


class TestUnoptimizedInferenceEvalMode:
    """`_ensure_eval_mode_for_unoptimized_inference` must keep the module tree in eval mode."""

    def test_eval_mode_reasserted_after_train_round_trip(self) -> None:
        """Eval mode must be applied to whatever self.model.model currently points to.

        ``train()`` rebinds ``self.model.model`` to a brand-new module left in training mode, so eval must be re-applied
        to the *current* object on every call — not to a cached reference captured at init.
        """
        rfdetr = _FakeRFDETR()

        # First inference call: warns once and switches to eval mode.
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        assert rfdetr.model.model.training is False

        # Simulate train() rebinding self.model.model to a fresh training-mode module.
        rfdetr.model.model = _FakeModelWithDropout()
        assert rfdetr.model.model.training is True  # new object starts in train mode

        # Every subsequent inference call must re-assert eval on the *new* object.
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        assert rfdetr.model.model.training is False

    def test_eval_assignments_skipped_when_module_tree_is_already_in_eval_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A module tree already in eval mode must not have its flags reassigned.

        ``nn.Module.eval()`` reassigns ``training`` on every node of the tree, so re-asserting a mode the tree is
        already in is host time spent to reach the state it is already in. The read-only mode check remains.
        """
        rfdetr = _FakeRFDETR()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        assert rfdetr.model.model.training is False

        train_spy = Mock(wraps=rfdetr.model.model.train)
        monkeypatch.setattr(rfdetr.model.model, "train", train_spy)

        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        train_spy.assert_not_called()
        assert rfdetr.model.model.training is False

    def test_eval_mode_reasserted_when_only_a_submodule_is_training(self) -> None:
        """A mixed-mode tree must be returned completely to eval mode."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.model.eval()
        rfdetr.model.model.dropout.train()
        assert rfdetr.model.model.training is False
        assert rfdetr.model.model.dropout.training is True

        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        assert all(not module.training for module in rfdetr.model.model.modules())

    def test_optimized_model_skips_eval_assertion(self) -> None:
        """When _is_optimized_for_inference is True, the method must be a no-op.

        The compiled inference_model snapshot is already in eval mode; calling eval() on the stale self.model.model
        would target the wrong object.
        """
        rfdetr = _FakeRFDETR()
        rfdetr._is_optimized_for_inference = True
        rfdetr.model.model.train()
        assert rfdetr.model.model.training is True  # confirm starting state

        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        assert rfdetr.model.model.training is True  # must remain unchanged

    def test_not_optimized_warning_emitted_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The not-optimized warning is logged at most once across repeated calls."""
        warnings: list[str] = []
        monkeypatch.setattr(detr_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

        rfdetr = _FakeRFDETR()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        rfdetr.model.model.train()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        assert len(warnings) == 1

    def test_eval_mode_applied_after_warning_when_root_is_training(self) -> None:
        """Eval() must still run after the warning when the current root is in training mode.

        Simulate the code path where the warning has already been emitted
        (``_has_warned_about_not_being_optimized_for_inference=True``) and verify
        that ``eval()`` is still applied to a current module whose root reports training mode.
        """
        rfdetr = _FakeRFDETR()
        rfdetr._has_warned_about_not_being_optimized_for_inference = True
        rfdetr.model.model.train()

        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        assert rfdetr.model.model.training is False

    def test_predict_puts_module_in_eval_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Predict() must delegate to _ensure_eval_mode_for_unoptimized_inference, leaving module in eval mode."""
        rfdetr = _FakeRFDETR()
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))

        monkeypatch.setattr(
            rfdetr.model.model,
            "forward",
            lambda batch: {"pred_logits": torch.zeros(1, 10, 81), "pred_boxes": torch.zeros(1, 10, 4)},
        )
        monkeypatch.setattr(
            rfdetr.model,
            "postprocess",
            lambda preds, target_sizes, score_threshold=None: [
                {"scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}
            ],
            raising=False,
        )

        rfdetr.predict(img)

        assert rfdetr.model.model.training is False
