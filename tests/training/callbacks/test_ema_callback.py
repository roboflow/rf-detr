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


class _EMAContainerModule(nn.Module):
    """Minimal module with `.model` to mirror RFDETRModelModule shape."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(4, 2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


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
        """At the first call (num_averaged=0) with tau>0 the effective decay uses updates=1 matching ModelEma's
        1-indexed counter."""
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
        """Simulate 500 update steps and compare final EMA weights with ModelEma.module to confirm numerical parity."""
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
        """update_interval_steps=2 updates on steps 2, 4, ...

        only.
        """
        cb = RFDETREMACallback(update_interval_steps=2)
        cb._average_model = MagicMock()

        trainer = MagicMock()
        pl_module = MagicMock()

        for step in (1, 2, 3, 4):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=step - 1)

        assert cb._average_model.update_parameters.call_count == 2


class TestEpochBoundaryNoDoubleUpdate:
    """Regression test for the epoch-boundary double-update bug.

    ``on_train_epoch_end`` used to call ``update_parameters`` again after the last optimizer step of the epoch, on top
    of that step's own ``on_train_batch_end`` update, double-counting one update per epoch against
    ``update_interval_steps``.
    """

    def test_on_train_epoch_end_is_not_overridden(self) -> None:
        """The callback must not define its own ``on_train_epoch_end`` — PTL's per-step ``on_train_batch_end`` already
        fires for the last batch of every epoch, so a separate epoch-end trigger would update on top of that same
        step."""
        assert "on_train_epoch_end" not in RFDETREMACallback.__dict__

    @pytest.mark.parametrize(
        ("n_epochs", "steps_per_epoch", "update_interval_steps"),
        [
            pytest.param(3, 4, 1, id="3-epochs-4-steps-interval-1"),
            pytest.param(1, 1, 1, id="1-epoch-1-step-interval-1"),
            pytest.param(2, 1, 1, id="2-epochs-1-step-interval-1"),
            pytest.param(2, 2, 2, id="2-epochs-2-steps-interval-2"),
        ],
    )
    def test_multi_epoch_training_updates_exactly_once_per_step(
        self, n_epochs: int, steps_per_epoch: int, update_interval_steps: int
    ) -> None:
        """Simulate ``n_epochs`` of ``steps_per_epoch`` optimizer steps each, including the no-op epoch-end hook.

        Lightning still calls the no-op epoch-end hook. ``update_parameters`` must fire exactly once per configured
        update interval, with no extra update at an epoch boundary.
        """
        cb = RFDETREMACallback(update_interval_steps=update_interval_steps)
        cb._average_model = MagicMock()
        trainer = MagicMock()
        pl_module = MagicMock()

        global_step = 0
        for epoch in range(n_epochs):
            trainer.current_epoch = epoch
            for _ in range(steps_per_epoch):
                global_step += 1
                trainer.global_step = global_step
                cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=global_step - 1)
            # Lightning still calls on_train_epoch_end every epoch; resolve it through the
            # instance so a still-present override (the bug) fires, not just the base no-op.
            cb.on_train_epoch_end(trainer, pl_module)

        total_steps = n_epochs * steps_per_epoch
        assert cb._average_model.update_parameters.call_count == total_steps // update_interval_steps


class TestLegacyEMAResume:
    """Legacy checkpoint EMA payload is consumed by the callback setup path."""

    def test_load_state_dict_ignores_removed_epoch_state(self) -> None:
        """Older callback state with ``latest_update_epoch`` remains loadable after the state was removed."""
        cb = RFDETREMACallback()

        cb.load_state_dict({"latest_update_step": 7, "latest_update_epoch": 4})

        assert cb.state_dict() == {"latest_update_step": 7}

    def test_setup_loads_pending_legacy_ema_state_into_average_model(self) -> None:
        """`_pending_legacy_ema_state` must initialize EMA weights at fit setup."""
        cb = RFDETREMACallback()
        pl_module = _EMAContainerModule()
        trainer = MagicMock()

        legacy_ema_state = {k: torch.full_like(v, 2.0) for k, v in pl_module.model.state_dict().items()}
        pl_module._pending_legacy_ema_state = legacy_ema_state

        cb.setup(trainer, pl_module, stage="fit")

        assert cb._average_model is not None
        restored = cb._average_model.module.model.state_dict()
        for key, expected in legacy_ema_state.items():
            assert torch.allclose(restored[key], expected)
        assert not hasattr(pl_module, "_pending_legacy_ema_state")


class _BufferContainerModule(nn.Module):
    """Container module with a float parameter and an integer buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(4, 2)
        self.register_buffer("step_count", torch.tensor(10, dtype=torch.long))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class TestMultiAvgFn:
    """Foreach ``multi_avg_fn`` path must reproduce the per-tensor ``avg_fn`` numerics exactly."""

    def test_setup_registers_multi_avg_fn(self) -> None:
        """Fit setup must wire multi_avg_fn (foreach branch) and leave avg_fn unset."""
        cb = RFDETREMACallback()
        pl_module = _EMAContainerModule()
        trainer = MagicMock()

        cb.setup(trainer, pl_module, stage="fit")

        assert cb._average_model is not None
        assert cb._average_model.multi_avg_fn is not None
        assert cb._average_model.avg_fn is None

    def test_multi_avg_fn_matches_avg_fn_weight_parity(self) -> None:
        """200 update_parameters steps: foreach multi_avg_fn EMA equals legacy per-tensor avg_fn EMA."""
        torch.manual_seed(42)
        n_steps = 200
        decay = 0.993
        tau = 100
        model = _EMAContainerModule()
        cb = RFDETREMACallback(decay=decay, tau=tau)
        ema_new = AveragedModel(model=model, use_buffers=True, multi_avg_fn=cb._multi_avg_fn)
        ema_old = AveragedModel(model=model, use_buffers=True, avg_fn=cb._avg_fn)

        for _ in range(n_steps):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema_new.update_parameters(model)
            ema_old.update_parameters(model)

        new_state = ema_new.module.state_dict()
        old_state = ema_old.module.state_dict()
        for name, old_val in old_state.items():
            assert torch.allclose(new_state[name], old_val, atol=1e-6), (
                f"Parity failed for {name}: max diff = {(new_state[name].float() - old_val.float()).abs().max().item()}"
            )

    def test_multi_avg_fn_integer_buffer_matches_avg_fn(self) -> None:
        """Integer buffers (non-foreach dtype group) must follow the same cast semantics as avg_fn."""
        torch.manual_seed(42)
        decay = 0.5
        model = _BufferContainerModule()
        cb = RFDETREMACallback(decay=decay, tau=0)
        ema_new = AveragedModel(model=model, use_buffers=True, multi_avg_fn=cb._multi_avg_fn)
        ema_old = AveragedModel(model=model, use_buffers=True, avg_fn=cb._avg_fn)

        for value in (20, 31):
            model.step_count.fill_(value)
            ema_new.update_parameters(model)
            ema_old.update_parameters(model)

        assert torch.equal(ema_new.module.step_count, ema_old.module.step_count)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_avg_fn_matches_avg_fn_weight_parity_on_cuda(self) -> None:
        """CUDA-resident foreach multi_avg_fn EMA must match avg_fn EMA — exercises real GPU group dispatch.

        The CPU-only parity test above verifies numerics but not the actual optimization target:
        torch._foreach_mul_/torch._foreach_add_ dispatching as fused CUDA kernels over a (device, dtype)
        group, collapsing what would otherwise be one .item() GPU->CPU sync per tensor into one per group.
        """
        torch.manual_seed(42)
        n_steps = 50
        decay = 0.993
        tau = 100
        model = _EMAContainerModule().cuda()
        cb = RFDETREMACallback(decay=decay, tau=tau)
        ema_new = AveragedModel(model=model, use_buffers=True, multi_avg_fn=cb._multi_avg_fn)
        ema_old = AveragedModel(model=model, use_buffers=True, avg_fn=cb._avg_fn)

        for _ in range(n_steps):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema_new.update_parameters(model)
            ema_old.update_parameters(model)

        new_state = ema_new.module.state_dict()
        old_state = ema_old.module.state_dict()
        for name, old_val in old_state.items():
            assert new_state[name].is_cuda
            assert torch.allclose(new_state[name], old_val, atol=1e-6), f"CUDA parity failed for {name}"


class TestSuppressTestSwap:
    """suppress_test_swap must disable the test-time EMA weight swap while leaving defaults unchanged."""

    @staticmethod
    def _make_swap_scenario() -> tuple[RFDETREMACallback, _EMAContainerModule]:
        """Build a module at weight 7.0 with an EMA average model captured at weight 5.0."""
        cb = RFDETREMACallback()
        pl_module = _EMAContainerModule()
        with torch.no_grad():
            for p in pl_module.parameters():
                p.fill_(5.0)
        cb._average_model = AveragedModel(model=pl_module, use_buffers=True, avg_fn=cb._avg_fn)
        with torch.no_grad():
            for p in pl_module.parameters():
                p.fill_(7.0)
        return cb, pl_module

    def test_default_flag_is_false(self) -> None:
        """The suppression flag defaults to False so standalone trainer.test() keeps EMA evaluation."""
        cb = RFDETREMACallback()
        assert cb.suppress_test_swap is False

    def test_on_test_epoch_start_swaps_by_default(self) -> None:
        """Without suppression, the test hooks swap live weights (7.0) for EMA weights (5.0)."""
        cb, pl_module = self._make_swap_scenario()
        trainer = MagicMock()

        cb.on_test_epoch_start(trainer, pl_module)

        weight = pl_module.model.weight.detach()
        assert torch.allclose(weight, torch.full_like(weight, 5.0))

    def test_on_test_epoch_start_suppressed_keeps_live_weights(self) -> None:
        """With suppress_test_swap=True the live weights (7.0) must stay in place during test."""
        cb, pl_module = self._make_swap_scenario()
        cb.suppress_test_swap = True
        trainer = MagicMock()

        cb.on_test_epoch_start(trainer, pl_module)

        weight = pl_module.model.weight.detach()
        assert torch.allclose(weight, torch.full_like(weight, 7.0))

    def test_on_test_epoch_end_suppressed_does_not_swap(self) -> None:
        """With suppression active, on_test_epoch_end must not swap EMA weights in unpaired."""
        cb, pl_module = self._make_swap_scenario()
        cb.suppress_test_swap = True
        trainer = MagicMock()

        cb.on_test_epoch_start(trainer, pl_module)
        cb.on_test_epoch_end(trainer, pl_module)

        weight = pl_module.model.weight.detach()
        assert torch.allclose(weight, torch.full_like(weight, 7.0))
