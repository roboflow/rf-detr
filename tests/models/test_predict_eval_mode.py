# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests that unoptimized inference always runs the module in eval mode."""

from types import SimpleNamespace

import pytest
import torch

from rfdetr import detr as detr_module
from rfdetr.detr import RFDETR


class _FakeModelWithDropout(torch.nn.Module):
    """Minimal module whose behavior differs between train and eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class _FakeModelContext:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = _FakeModelWithDropout()
        self.inference_model = None


class _FakeRFDETR(RFDETR):
    def maybe_download_pretrain_weights(self) -> None:
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(num_channels=3)

    def get_model(self, config: SimpleNamespace) -> _FakeModelContext:
        return _FakeModelContext()


class TestUnoptimizedInferenceEvalMode:
    """`_ensure_eval_mode_for_unoptimized_inference` must keep the module in eval mode."""

    def test_eval_mode_reasserted_after_train_round_trip(self) -> None:
        """eval mode must be (re)applied on every call, not just the first.

        ``train()`` reassigns ``self.model.model`` to a module left in training
        mode, so a subsequent inference call must put it back in eval mode.
        Otherwise inference runs with dropout active and yields nondeterministic,
        degraded predictions.
        """
        rfdetr = _FakeRFDETR()
        module = rfdetr.model.model

        # First inference call: warns once and switches to eval mode.
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        assert module.training is False

        # A train() round-trip leaves the (re)assigned module in training mode.
        module.train()
        assert module.training is True

        # Every later inference call must re-assert eval mode.
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        assert module.training is False

    def test_not_optimized_warning_emitted_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The not-optimized warning is logged once even though eval() runs every call."""
        warnings: list[str] = []
        monkeypatch.setattr(detr_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

        rfdetr = _FakeRFDETR()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        rfdetr.model.model.train()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()
        rfdetr._ensure_eval_mode_for_unoptimized_inference()

        assert len(warnings) == 1
        assert rfdetr.model.model.training is False
