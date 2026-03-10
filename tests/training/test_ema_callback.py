# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit and parity tests for RFDETREMACallback."""

from __future__ import annotations

import math
import warnings
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.model_ema import ModelEma


class TestAvgFnDecayFormula:
    """Verify the tau / no-tau decay formula matches ModelEma."""

    @pytest.mark.parametrize(
        "num_averaged",
        [
            pytest.param(0, id="step-0"),
            pytest.param(5, id="step-5"),
            pytest.param(99, id="step-99"),
        ],
    )
    def test_tau_zero_uses_fixed_decay(self, num_averaged: int) -> None:
        """With tau=0 the effective decay equals the base decay at every step."""
        decay = 0.99
        cb = RFDETREMACallback(decay=decay, tau=0)
        ema_val = torch.tensor(1.0)
        model_val = torch.tensor(2.0)

        result = cb._avg_fn(ema_val, model_val, num_averaged)

        expected = ema_val * decay + model_val * (1.0 - decay)
        assert torch.allclose(result, expected, atol=1e-7)

    def test_tau_warmup_at_step_1(self) -> None:
        """At the first call (num_averaged=0) with tau>0 the effective decay
        uses updates=1 matching ModelEma's 1-indexed counter."""
        decay = 0.993
        tau = 100
        cb = RFDETREMACallback(decay=decay, tau=tau)
        ema_val = torch.tensor(1.0)
        model_val = torch.tensor(2.0)

        result = cb._avg_fn(ema_val, model_val, num_averaged=0)

        updates = 1  # num_averaged + 1
        effective_decay = decay * (1 - math.exp(-updates / tau))
        expected = ema_val * effective_decay + model_val * (1.0 - effective_decay)
        assert torch.allclose(result, expected, atol=1e-7)


class TestModelEmaParity:
    """Ensure N-step EMA weights match ModelEma exactly."""

    def test_avg_fn_matches_modelema_weight_parity(self) -> None:
        """Simulate 500 update steps and compare final EMA weights with
        ModelEma.module to confirm numerical parity."""
        torch.manual_seed(42)
        n_steps = 500
        decay = 0.993
        tau = 100

        model = nn.Linear(4, 4)
        model_ema = ModelEma(model, decay=decay, tau=tau)
        cb = RFDETREMACallback(decay=decay, tau=tau)

        # Initialise manual EMA state from model (same as ModelEma deepcopy)
        ema_weights: dict[str, torch.Tensor] = {name: p.clone() for name, p in model.named_parameters()}

        for step in range(n_steps):
            # Perturb model parameters
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)

            # Update legacy ModelEma
            model_ema.update(model)

            # Replicate update via callback avg_fn
            model_weights = {name: p.clone() for name, p in model.named_parameters()}
            for name in ema_weights:
                ema_weights[name] = cb._avg_fn(ema_weights[name], model_weights[name], step)

        # Compare
        legacy_state = dict(model_ema.module.named_parameters())
        for name, cb_val in ema_weights.items():
            assert torch.allclose(cb_val, legacy_state[name], atol=1e-5), (
                f"Parity failed for {name}: max diff = {(cb_val - legacy_state[name]).abs().max().item()}"
            )


class TestShouldUpdate:
    """Verify should_update triggers on steps and epochs."""

    def test_should_update_on_step(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update(step_idx=42) is True

    def test_should_update_on_epoch(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update(epoch_idx=3) is True

    def test_should_update_neither(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update() is False


class TestInit:
    """Construction and EMA-state access behavior."""

    def test_init_emits_no_user_warning(self) -> None:
        """Instantiation should not emit runtime UserWarnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RFDETREMACallback()
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns

    def test_get_ema_model_state_dict_none_before_setup(self) -> None:
        """EMA state accessor returns None before averaged model is created."""
        cb = RFDETREMACallback()
        assert cb.get_ema_model_state_dict() is None

    def test_get_ema_model_state_dict_returns_model_weights(self) -> None:
        """EMA state accessor returns the wrapped `.model` state dict."""

        class _Container(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(4, 2)

        cb = RFDETREMACallback()
        container = _Container()
        cb._average_model = AveragedModel(container, avg_fn=cb._avg_fn)

        state = cb.get_ema_model_state_dict()

        assert state is not None
        assert "weight" in state
        assert "bias" in state


class TestUpdateInterval:
    """Verify update_interval_steps throttles EMA updates on step hooks."""

    def test_updates_only_on_interval_steps(self) -> None:
        """update_interval_steps=2 updates on steps 2, 4, ... only."""
        cb = RFDETREMACallback(update_interval_steps=2)
        cb._average_model = MagicMock()

        trainer = MagicMock()
        pl_module = MagicMock()

        for step in (1, 2, 3, 4):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=step - 1)

        assert cb._average_model.update_parameters.call_count == 2
