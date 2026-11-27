# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for RFDETRModelModule optimizer configuration, resume-state normalization and CUDA graph lifecycle."""

import logging
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRNanoConfig
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.imports import _IS_PYTORCH_OPTIMIZER_INSTALLED
from tests.training.helpers import (
    _attach_trainer,
    _base_model_config,
    _base_train_config,
    _build_module,
)

_requires_pytorch_optimizer = pytest.mark.skipif(
    not _IS_PYTORCH_OPTIMIZER_INSTALLED, reason="pytorch_optimizer not installed"
)


class _RecordingOptimizer(torch.optim.Optimizer):
    """Optimizer test double that records constructor defaults and kwargs."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        self.extra_kwargs = dict(kwargs)
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, **kwargs})

    def step(self, closure=None):
        """Run an optimizer step for the test double."""
        if closure is not None:
            return closure()
        return None


class _NoWeightDecayOptimizer(torch.optim.Optimizer):
    """Optimizer test double whose constructor does not accept ``weight_decay``."""

    def __init__(self, params, lr=1e-3, momentum=0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    def step(self, closure=None):
        """Run an optimizer step for the test double."""
        if closure is not None:
            return closure()
        return None


class _RaisingOptimizer:
    """Optimizer test double that always fails to construct."""

    def __init__(self, *args, **kwargs):
        raise TypeError("simulated optimizer construction failure")


class TestConfigureOptimizers:
    """Tests for configure_optimizers().

    Covers required output keys, AdamW optimizer type, step-interval scheduler, LR lambda warmup ramp, and step-decay
    behaviour before and after lr_drop.
    """

    def _setup_module(self, tmp_path, **train_overrides):
        tc = _base_train_config(tmp_path, **train_overrides)
        module, _, _, _ = _build_module(train_config=tc)

        _attach_trainer(module, estimated_stepping_batches=1000)

        real_param = nn.Parameter(torch.randn(4, 4))
        param_dicts = [{"params": real_param, "lr": tc.lr}]
        return module, param_dicts

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("optimizer", id="optimizer-key"),
            pytest.param("lr_scheduler", id="lr-scheduler-key"),
        ],
    )
    def test_returns_required_key(self, mocker, key, tmp_path):
        """Lightning requires both 'optimizer' and 'lr_scheduler' keys in the returned config dict."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert key in module.configure_optimizers()

    def test_optimizer_is_adamw(self, mocker, tmp_path):
        """RF-DETR must use AdamW for its decoupled weight decay behavior."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert isinstance(module.configure_optimizers()["optimizer"], torch.optim.AdamW)

    def test_adamw_optimizer_kwargs_forwarded(self, mocker, tmp_path):
        """optimizer_kwargs are forwarded to RF-DETR's default AdamW optimizer."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        optimizer_kwargs = {"betas": (0.8, 0.95), "eps": 1e-7}
        module, param_dicts = self._setup_module(tmp_path, optimizer_kwargs=optimizer_kwargs)
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.defaults["betas"] == optimizer_kwargs["betas"]
        assert optimizer.defaults["eps"] == pytest.approx(optimizer_kwargs["eps"])

    def test_managed_optimizer_resolves_native_class(self, mocker, tmp_path):
        """Managed short names resolve to a native torch.optim class with lr/weight_decay injected."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        mock_resolve = mocker.patch(
            "rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer
        )
        optimizer = module.configure_optimizers()["optimizer"]

        mock_resolve.assert_called_once_with("sgd")
        assert isinstance(optimizer, _RecordingOptimizer)
        assert optimizer.defaults["lr"] == pytest.approx(module.train_config.lr)
        assert optimizer.defaults["weight_decay"] == pytest.approx(module.train_config.weight_decay)

    def test_managed_optimizer_preserves_rfdetr_param_groups(self, mocker, tmp_path):
        """Managed optimizers must receive RF-DETR param groups with layer-wise LR values."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        param_dicts[0]["lr"] = 2.5e-5
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer)
        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(2.5e-5)

    def test_managed_optimizer_kwargs_forwarded(self, mocker, tmp_path):
        """optimizer_kwargs are forwarded to a managed optimizer constructor."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        optimizer_kwargs = {"momentum": 0.9, "nesterov": True}
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd", optimizer_kwargs=optimizer_kwargs)
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer)
        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.extra_kwargs == optimizer_kwargs

    def test_callable_optimizer_called_with_param_groups_only(self, mocker, tmp_path):
        """A non-reconstructable callable optimizer is invoked with the param groups and nothing else."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer=lambda params: _RecordingOptimizer(params))
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, _RecordingOptimizer)
        assert optimizer.extra_kwargs == {}

    def test_dotted_optimizer_built_from_kwargs_without_injection(self, mocker, tmp_path):
        """A dotted-path optimizer is built from optimizer_kwargs only, with no lr/weight_decay injection."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path, optimizer="torch.optim.AdamW", optimizer_kwargs={"weight_decay": 0.5}
        )
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["weight_decay"] == pytest.approx(0.5)
        # Explicit mode must not inject the config lr as the optimizer-level default.
        assert optimizer.defaults["lr"] == pytest.approx(1e-3)

    def test_uninstalled_import_path_optimizer_raises_value_error(self, mocker, tmp_path):
        """A dotted import path that cannot be imported surfaces as a configuration error at train start."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="pytorch_optimizer.Lion")
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model.importlib.import_module", side_effect=ImportError("No module"))
        with pytest.raises(ValueError) as exc_info:
            module.configure_optimizers()
        assert "Could not import optimizer" in str(exc_info.value)

    @_requires_pytorch_optimizer
    def test_real_import_path_optimizer_smoke(self, mocker, tmp_path):
        """A real pytorch-optimizer optimizer can be built via its import path when installed."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="pytorch_optimizer.Lion")
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.__class__.__name__ == "Lion"

    def test_custom_optimizer_omits_weight_decay_when_unsupported(self, mocker, tmp_path):
        """weight_decay must not be injected into optimizers whose constructor does not accept it."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_NoWeightDecayOptimizer)
        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, _NoWeightDecayOptimizer)
        assert "weight_decay" not in optimizer.defaults

    def test_fused_optimizer_warns_for_custom_optimizer_on_bf16(self, mocker, tmp_path):
        """A custom optimizer on a fused-eligible BF16/CUDA run must warn that fused_optimizer is ignored."""
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = "bf16-mixed"

        mocker.patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer)
        mock_warning = mocker.patch("rfdetr.training.module_model.logger.warning")
        module.configure_optimizers()

        assert any("fused_optimizer=True is ignored" in str(call.args[0]) for call in mock_warning.call_args_list)

    def test_no_fused_warning_for_adamw_on_bf16(self, mocker, tmp_path):
        """The built-in AdamW path uses fused and must not emit the fused-ignored warning."""
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = "bf16-mixed"

        mock_warning = mocker.patch("rfdetr.training.module_model.logger.warning")
        module.configure_optimizers()

        assert not any("fused_optimizer=True is ignored" in str(call.args[0]) for call in mock_warning.call_args_list)

    def test_adamw_construction_error_is_rewrapped(self, mocker, tmp_path):
        """An unsupported AdamW kwarg must surface as an RF-DETR-specific initialization error."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer_kwargs={"nonexistent_adamw_kwarg": True})
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(TypeError) as exc_info:
            module.configure_optimizers()
        assert "Failed to initialize optimizer 'adamw'" in str(exc_info.value)

    def test_pytorch_optimizer_construction_error_is_rewrapped(self, mocker, tmp_path):
        """A managed optimizer constructor failure must surface as an RF-DETR-specific initialization error."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RaisingOptimizer)
        with pytest.raises(TypeError) as exc_info:
            module.configure_optimizers()
        assert "Failed to initialize optimizer 'sgd'" in str(exc_info.value)

    def test_use_fused_optimizer_false_for_custom_optimizer(self, mocker, tmp_path):
        """Fused AdamW lifecycle must not activate for custom optimizers, even on BF16/CUDA."""
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        module, _ = self._setup_module(tmp_path, optimizer="sgd")
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_optimizer is False

    def test_use_fused_optimizer_true_for_adamw(self, mocker, tmp_path):
        """Fused AdamW must activate for the built-in AdamW optimizer on BF16/CUDA."""
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        module, _ = self._setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_optimizer is True

    def test_scheduler_interval_is_step(self, mocker, tmp_path):
        """Scheduler must step per batch (not per epoch) for fine-grained warmup."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert module.configure_optimizers()["lr_scheduler"]["interval"] == "step"

    @pytest.mark.parametrize(
        "step, expected_behavior",
        [
            pytest.param(0, "warmup_start", id="warmup-start"),
            pytest.param(50, "warmup_mid", id="warmup-midpoint"),
        ],
    )
    def test_lr_lambda_warmup_phase(self, mocker, step, expected_behavior, tmp_path):
        """LR lambda must produce a linear ramp during the warmup phase."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=1.0, epochs=10)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # steps_per_epoch=100, warmup_steps=100
        expected = float(step) / float(max(1, 100))
        assert lr_lambda(step) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "step, expected_factor",
        [
            pytest.param(500, 1.0, id="before-lr-drop-epoch-8-times-100-steps-per-epoch-equals-800"),
            pytest.param(900, 0.1, id="after-lr-drop-epoch-decays-to-0.1"),
        ],
    )
    def test_lr_lambda_step_decay(self, mocker, tmp_path, step: int, expected_factor: float):
        """The step-decay LR multiplier stays at 1.0 before lr_drop and decays to 0.1 after it."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path, warmup_epochs=0.0, epochs=10, lr_scheduler_kwargs={"lr_drop": 8}
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        assert lr_lambda(step) == pytest.approx(expected_factor)

    def test_lr_lambda_cosine_reads_train_config_fields(self, mocker, tmp_path):
        """Cosine preset must read its floor from lr_scheduler_kwargs['min_factor']."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            epochs=10,
            lr_scheduler="cosine",
            lr_scheduler_kwargs={"min_factor": 0.2},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # At the final step, cosine schedule must end at min_factor.
        assert lr_lambda(1000) == pytest.approx(0.2)

    def test_lr_lambda_step_default_drop_is_epoch_100(self, mocker, tmp_path):
        """With no ``lr_drop`` kwarg the step preset falls back to the managed default of 100 epochs."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=200)
        module._trainer.estimated_stepping_batches = 20000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # steps_per_epoch=100 -> drop at step 10000
        assert lr_lambda(9999) == pytest.approx(1.0)
        assert lr_lambda(10000) == pytest.approx(0.1)

    def test_lr_lambda_cosine_default_floor_is_zero(self, mocker, tmp_path):
        """With no ``min_factor`` kwarg the cosine preset anneals to the managed default floor of 0.0."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=10, lr_scheduler="cosine")
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        assert lr_lambda(1000) == pytest.approx(0.0)

    def test_explicit_dotted_scheduler_builds_from_kwargs(self, mocker, tmp_path):
        """An explicit dotted-path lr_scheduler is built from lr_scheduler_kwargs verbatim."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 30, "gamma": 0.1},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == pytest.approx(0.1)

    def test_per_parameter_lr_lambda_list_is_collapsed_onto_merged_groups(self, mocker, tmp_path):
        """A legacy per-parameter lr_lambda list is regrouped to one callback per merged group."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        mock_build_param_dicts = mocker.patch("rfdetr.training.module_model._build_param_dicts")

        def shared_lambda(step: int) -> float:
            return 1.0

        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.LambdaLR",
            lr_scheduler_kwargs={"lr_lambda": [shared_lambda, shared_lambda]},
        )
        mock_get_param_dict.return_value = param_dicts
        lr = module.train_config.lr
        mock_build_param_dicts.return_value = [
            {"params": nn.Parameter(torch.randn(2)), "lr": lr},
            {"params": nn.Parameter(torch.randn(2)), "lr": lr},
        ]

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert scheduler.lr_lambdas == [shared_lambda]

    def test_managed_preset_builds_lambda_lr(self, mocker, tmp_path):
        """A managed preset still builds a LambdaLR at step interval."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, lr_scheduler="step")
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        config = module.configure_optimizers()["lr_scheduler"]

        assert isinstance(config["scheduler"], torch.optim.lr_scheduler.LambdaLR)
        assert config["interval"] == "step"

    def test_callable_scheduler_built_from_optimizer_only(self, mocker, tmp_path):
        """A non-reconstructable callable scheduler is invoked with the optimizer and nothing else."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=11),
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 11

    def test_explicit_scheduler_auto_wrapped_with_warmup(self, mocker, tmp_path):
        """With warmup_epochs>0 an explicit scheduler is wrapped in a SequentialLR linear warmup."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_epoch_interval_warmup_ramps_over_multiple_epochs(self, mocker, tmp_path):
        """An epoch-interval explicit scheduler gets a real warmup ramp sized in whole epochs (not optimizer steps)."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=3.0,
            epochs=10,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000  # steps_per_epoch = 100
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        # The ramp spans 3 epoch-steps (not 300 optimizer-steps) and actually ramps up (start_factor < 1.0).
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        warmup = scheduler._schedulers[0]
        assert warmup.total_iters == 3
        assert warmup.start_factor < 1.0

    def test_epoch_interval_single_epoch_warmup_warns_and_skips(self, mocker, tmp_path):
        """A single-epoch epoch-interval warmup cannot ramp; it warns and is skipped, not silently flat."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        mock_logger = mocker.patch("rfdetr.training.module_model.logger")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            epochs=10,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        # No degenerate flat SequentialLR wrap; the no-op warmup is skipped loudly instead.
        assert not isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        mock_logger.warning.assert_called_once()

    def test_explicit_scheduler_interval_propagates(self, mocker, tmp_path):
        """lr_scheduler_interval is forwarded to the Lightning scheduler config for explicit schedulers."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        assert module.configure_optimizers()["lr_scheduler"]["interval"] == "epoch"

    def test_reduce_on_plateau_sets_monitor_and_epoch_interval(self, mocker, tmp_path):
        """A plateau loss monitor makes automatic validation loss available each epoch."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
            lr_scheduler_monitor="val/loss",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        config = module.configure_optimizers()["lr_scheduler"]

        assert isinstance(config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert config["monitor"] == "val/loss"
        assert config["interval"] == "epoch"
        assert module._should_compute_val_loss is True

    def test_reduce_on_plateau_records_each_group_initial_lr(self, mocker, tmp_path: Path) -> None:
        """A resumed run rescales its reduced lr from the recorded initial lr when ``lr`` changes (#1613).

        Every other scheduler sets ``initial_lr`` itself.
        """
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
            lr_scheduler_monitor="train/loss",
        )
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.param_groups[0]["initial_lr"] == module.train_config.lr

    @pytest.mark.parametrize(
        ("train_overrides", "expected_base"),
        [
            pytest.param({"warmup_epochs": 1.0}, 1e-4, id="managed-warmup"),
            pytest.param(
                {
                    "warmup_epochs": 0.0,
                    "lr_scheduler": "torch.optim.lr_scheduler.OneCycleLR",
                    "lr_scheduler_kwargs": {"max_lr": 1e-3, "total_steps": 100},
                },
                1e-3 / 25,
                id="one-cycle",
            ),
        ],
    )
    def test_records_each_group_base_as_its_scheduler_set_it(
        self, mocker, train_overrides: dict[str, Any], expected_base: float, tmp_path: Path
    ) -> None:
        """A resumed run compares the recorded base with the restored one (#1613).

        The base is the scheduler's own: not the warmup-scaled current lr and, for a scheduler that sets its own rates
        (OneCycleLR starts at ``max_lr / 25``), not ``lr``.
        """
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, **train_overrides)
        mock_get_param_dict.return_value = param_dicts

        module.configure_optimizers()

        assert module._configured_param_group_settings[0][0] == pytest.approx(expected_base)

    def test_callable_plateau_scheduler_rejects_disabled_val_loss(self, mocker, tmp_path):
        """A closure-built ReduceLROnPlateau monitoring val/loss is rejected when compute_val_loss=False.

        ``TrainConfig.validate_explicit_val_loss_disable`` only string-compares ``lr_scheduler`` against the
        ReduceLROnPlateau dotted path, and a lambda closure — unlike a plain ``functools.partial``, which is desugared
        to that path — never reaches the comparison. The runtime check in ``configure_optimizers`` is therefore the only
        guard left once the concrete scheduler exists.
        """
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        with warnings.catch_warnings():
            # A lambda lr_scheduler cannot round-trip through training_config.json; that reproducibility warning is
            # expected here and unrelated to the conflict under test.
            warnings.simplefilter("ignore", UserWarning)
            module, param_dicts = self._setup_module(
                tmp_path,
                warmup_epochs=0.0,
                lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5),
                lr_scheduler_monitor="val/loss",
                compute_val_loss=False,
            )
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(ValueError) as exc_info:
            module.configure_optimizers()
        assert "incompatible with ReduceLROnPlateau" in str(exc_info.value)

    def test_uninstalled_scheduler_path_raises_value_error(self, mocker, tmp_path):
        """A dotted scheduler path that cannot be imported surfaces as a configuration error at train start."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path, lr_scheduler="nonexistent_pkg.MyScheduler")
        mock_get_param_dict.return_value = param_dicts

        mocker.patch("rfdetr.training.module_model.importlib.import_module", side_effect=ImportError("No module"))
        with pytest.raises(ValueError) as exc_info:
            module.configure_optimizers()
        assert "Could not import lr_scheduler" in str(exc_info.value)

    def test_explicit_scheduler_construction_error_is_reraised(self, mocker, tmp_path):
        """A missing required scheduler kwarg surfaces as an RF-DETR configuration error, not a bare TypeError."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",  # StepLR requires step_size
            lr_scheduler_kwargs={},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(TypeError) as exc_info:
            module.configure_optimizers()
        assert "Failed to initialize lr_scheduler" in str(exc_info.value)

    def test_plateau_with_warmup_warns(self, mocker, tmp_path):
        """warmup_epochs>0 with ReduceLROnPlateau warns; a metric-driven scheduler cannot compose with a warmup ramp."""
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        mock_logger = mocker.patch("rfdetr.training.module_model.logger")
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
            lr_scheduler_monitor="val/loss",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        module.configure_optimizers()

        mock_logger.warning.assert_called_once()

    @pytest.mark.parametrize(
        "precision, expected_fused",
        [
            pytest.param(
                "32-true",
                False,
                id="not-bf16-disables-fused-even-though-the-gpu-supports-bf16",
            ),
            pytest.param(
                "bf16-mixed",
                True,
                id="bf16-mixed-enables-fused-params-grads-and-state-share-dtype-and-layout",
            ),
            pytest.param(
                "transformer-engine",
                True,
                id="transformer-engine-default-fp8-uses-bf16-weights-so-fused-stays-active",
            ),
        ],
    )
    def test_fused_optimizer_gating_by_precision(self, mocker, tmp_path, precision: str, expected_fused: bool):
        """Fused AdamW is enabled only for the precisions whose params/grads/state all stay BF16-compatible.

        On Ampere+ GPUs ``torch.cuda.is_bf16_supported()`` is True even when the trainer is configured for 32-true
        precision. The old code always enabled fused AdamW based on GPU capability alone, crashing with ``params, grads,
        exp_avgs, and exp_avg_sqs must have same dtype, device, and layout`` when DDP gradient bucket views had non-
        matching strides — the fix checks ``trainer.precision`` before enabling fused.
        """
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = precision

        optimizer = module.configure_optimizers()["optimizer"]

        assert bool(optimizer.defaults.get("fused")) is expected_fused

    def test_fused_optimizer_disabled_when_cuda_unavailable(self, mocker, tmp_path):
        """_use_fused_optimizer must return False when CUDA is not available, regardless of precision."""
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=False)
        module, _ = self._setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"

        assert not module._use_fused_optimizer

    def test_total_steps_divided_by_grad_accum_for_keypoint_module(self, mocker, tmp_path):
        """Keypoint (manual-opt) path must divide estimated_stepping_batches by grad_accum_steps for LR scheduling.

        With microbatches=100, grad_accum_steps=4, epochs=1, warmup_epochs=0 the scheduler should span 25 optimizer
        steps (ceil(100/4)).  At step 24 (0-indexed last step) a cosine LR schedule should be nearly at min_factor; if
        total_steps were mistakenly 100 the LR would still be near its peak at step 24.
        """
        mock_get_param_dict = mocker.patch("rfdetr.training.module_model.get_param_dict")
        grad_accum_steps = 4
        microbatches = 100
        min_factor = 0.1
        tc = _base_train_config(
            tmp_path,
            grad_accum_steps=grad_accum_steps,
            warmup_epochs=0,
            epochs=1,
            lr_scheduler="cosine",
            lr_scheduler_kwargs={"min_factor": min_factor},
        )
        module, _, _, _ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=tc,
        )
        _attach_trainer(module, estimated_stepping_batches=microbatches)
        real_param = nn.Parameter(torch.randn(4, 4))
        mock_get_param_dict.return_value = [{"params": real_param, "lr": tc.lr}]

        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        expected_total_steps = max(1, math.ceil(microbatches / grad_accum_steps))  # 25
        # The cosine schedule reaches min_factor exactly at step == total_steps (progress=1.0).
        # If total_steps were wrongly 100, lr at step 25 would still be ~0.87 (near peak).
        lr_at_decay_end = lr_lambda(expected_total_steps)
        assert lr_at_decay_end == pytest.approx(min_factor, abs=1e-6)


class TestCudaGraphLifecycle:
    """Tests for enabling the graph runner after Lightning resolves runtime ownership."""

    def test_on_train_start_enables_single_cuda_detection(
        self, mocker, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supported CUDA fit builds one runner around the registered detector and says so.

        The enable path must be as visible as the fallback warnings: without a log line a graph run
        and an eager run print identical consoles.
        """
        mc = _base_model_config(cuda_graphs=True, fused_optimizer=False)
        module, model, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        trainer = SimpleNamespace(world_size=1, precision="bf16-mixed")
        runner = MagicMock()
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner", return_value=runner)
        with caplog.at_level(logging.INFO, logger="rf-detr"):
            module.on_train_start()

        runner_type.assert_called_once_with(model)
        assert module._cuda_graph_runner is runner
        assert any("CUDA graph replay enabled" in record.getMessage() for record in caplog.records)

    def test_on_train_start_leaves_replay_to_inductor_when_compiled(
        self, mocker, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When Inductor owns CUDA graph replay, the eager runner is never wrapped around the compiled model.

        ``make_graphed_callables`` over an ``OptimizedModule`` would capture Inductor's own launch path a second time;
        cudagraph trees already replay the compiled kernels. The console still announces which runtime replays.
        """
        mc = _base_model_config(cuda_graphs=True, compile=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        module._inductor_cudagraphs = True
        trainer = SimpleNamespace(world_size=1, precision="bf16-mixed")
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        with caplog.at_level(logging.INFO, logger="rf-detr"):
            module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None
        assert any("Inductor" in record.getMessage() for record in caplog.records)

    def test_on_train_start_keeps_compile_only_fallback_out_of_eager_capture(self, mocker, tmp_path: Path) -> None:
        """Gradient accumulation must not wrap a compiled fallback in the eager CUDA graph runner."""
        mc = _base_model_config(cuda_graphs=True, compile=True, fused_optimizer=False)
        tc = _base_train_config(tmp_path, grad_accum_steps=2)
        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("torch._inductor.config", SimpleNamespace(triton=SimpleNamespace())),
            patch("rfdetr.training.module_model.torch.compile", side_effect=lambda model, **_: model),
        ):
            module, _, _, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        trainer = SimpleNamespace(world_size=1, precision="bf16-mixed")
        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        assert module._compile_active is True
        assert module._inductor_cudagraphs is False
        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None

    @pytest.mark.parametrize(
        "trainer",
        [
            pytest.param(SimpleNamespace(world_size=1, accumulate_grad_batches=2), id="accumulation"),
            pytest.param(SimpleNamespace(world_size=2, accumulate_grad_batches=1), id="distributed"),
        ],
    )
    def test_on_train_start_stops_when_trainer_leaves_inductor_scope(self, mocker, trainer, tmp_path: Path) -> None:
        """A trainer that resolved accumulation or several processes stops an Inductor graph run at train start.

        The construction-time gate reads ``TrainConfig``, but ``trainer_kwargs`` can override
        ``accumulate_grad_batches`` and ``devices="auto"`` can resolve to several GPUs. The model is already compiled
        with cudagraphs by then, so the run must stop with a clear message instead of failing on the second microbatch
        inside cudagraph trees.
        """
        mc = _base_model_config(cuda_graphs=True, compile=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        module._inductor_cudagraphs = True

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        with pytest.raises(RuntimeError) as exc_info:
            module.on_train_start()

        assert "validated for single-GPU training without gradient accumulation" in str(exc_info.value)

    def test_on_train_start_disables_inductor_replay_when_trainer_resolves_cpu(self, mocker, tmp_path: Path) -> None:
        """A CUDA-capable construction must not mark graph steps after Lightning places the module on CPU."""
        mc = _base_model_config(cuda_graphs=True, compile=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        module._inductor_cudagraphs = True
        trainer = SimpleNamespace(world_size=1, accumulate_grad_batches=1, precision="bf16-mixed")

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cpu"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        assert module._inductor_cudagraphs is False
        runner_type.assert_not_called()

    def test_on_train_start_rejects_te_precision_override_of_inductor(self, mocker, tmp_path: Path) -> None:
        """A precision-plugin override cannot bypass the FP8 compile/capture separation."""
        mc = RFDETRNanoConfig(
            pretrain_weights=None, device="cpu", cuda_graphs=True, compile=True, fused_optimizer=False
        )
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        module._inductor_cudagraphs = True
        trainer = SimpleNamespace(world_size=1, accumulate_grad_batches=1, precision="transformer-engine")
        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        with pytest.raises(RuntimeError) as exc_info:
            module.on_train_start()
        assert "FP8 CUDA graphs require compile=False" in str(exc_info.value)

    def test_on_train_start_keeps_cpu_training_eager(
        self, mocker, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit graph request on CPU degrades visibly to eager training."""
        mc = _base_model_config(cuda_graphs=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        trainer = SimpleNamespace(world_size=1)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cpu"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None
        assert any("model is on 'cpu'" in record.getMessage() for record in caplog.records)

    def test_on_train_start_keeps_distributed_training_eager(self, mocker, tmp_path: Path) -> None:
        """DDP remains on its established eager path."""
        mc = _base_model_config(cuda_graphs=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        trainer = SimpleNamespace(world_size=2)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None

    @pytest.mark.parametrize(
        "config_overrides",
        [
            pytest.param({"segmentation_head": True}, id="segmentation"),
            pytest.param(
                {"use_grouppose_keypoints": True, "num_keypoints_per_class": [1]},
                id="keypoints",
            ),
            pytest.param({"gradient_checkpointing": True}, id="gradient-checkpointing"),
        ],
    )
    def test_on_train_start_keeps_unsupported_model_modes_eager(
        self, mocker, config_overrides: dict[str, object], tmp_path: Path
    ) -> None:
        """Known-unsupported model modes never enter capture."""
        mc = _base_model_config(cuda_graphs=True, fused_optimizer=False, **config_overrides)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        trainer = SimpleNamespace(world_size=1)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None

    def test_on_train_start_routes_fp8_to_te_with_live_recipe(self, mocker, tmp_path: Path) -> None:
        """FP8 capture receives the actual post-conversion Lightning recipe, not a new default."""
        mc = RFDETRNanoConfig(pretrain_weights=None, device="cpu", cuda_graphs=True, fused_optimizer=False)
        tc = _base_train_config(tmp_path, amp_dtype="fp8")
        module, model, _, _ = _build_module(model_config=mc, train_config=tc)
        recipe = object()
        trainer = SimpleNamespace(
            world_size=1,
            precision="transformer-engine",
            accumulate_grad_batches=1,
            precision_plugin=SimpleNamespace(recipe=recipe),
        )
        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        runner_type.assert_called_once_with(model, fp8_recipe=recipe)
        assert module._cuda_graph_runner is runner_type.return_value

    @pytest.mark.parametrize(
        ("train_overrides", "accumulation"),
        [
            pytest.param({"multi_scale": "per-batch"}, 1, id="per-batch-multi-scale"),
            pytest.param({"square_resize_div_64": False}, 1, id="aspect-ratio-resize"),
            pytest.param({"multi_scale": "per-sample"}, 1, id="per-sample-multi-scale"),
            pytest.param({}, 2, id="trainer-overrides-accumulation"),
        ],
    )
    def test_on_train_start_keeps_fp8_outside_fixed_shape_scope_eager(
        self, mocker, train_overrides: dict[str, object], accumulation: int, tmp_path: Path
    ) -> None:
        """FP8 scope is checked at final trainer placement, including accumulation overrides."""
        mc = RFDETRNanoConfig(pretrain_weights=None, device="cpu", cuda_graphs=True, fused_optimizer=False)
        tc = _base_train_config(tmp_path, amp_dtype="fp8", **train_overrides)
        module, _, _, _ = _build_module(model_config=mc, train_config=tc)
        trainer = SimpleNamespace(
            world_size=1,
            precision="transformer-engine",
            accumulate_grad_batches=accumulation,
            precision_plugin=SimpleNamespace(recipe=object()),
        )
        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None

    def test_on_train_start_rejects_fp8_plugin_without_recipe(self, mocker, tmp_path: Path) -> None:
        """A missing Transformer Engine recipe cannot accidentally select the BF16 capture backend."""
        mc = RFDETRNanoConfig(pretrain_weights=None, device="cpu", cuda_graphs=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path, amp_dtype="fp8"))
        trainer = SimpleNamespace(world_size=1, precision="transformer-engine", precision_plugin=SimpleNamespace())
        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        with pytest.raises(RuntimeError) as exc_info:
            module.on_train_start()
        assert "recipe" in str(exc_info.value)

    @pytest.mark.parametrize("precision", ["16-mixed", "32-true", "transformer-engine-float16"])
    def test_on_train_start_keeps_non_bf16_precision_eager(self, mocker, precision: str, tmp_path: Path) -> None:
        """FP16, FP32 and Transformer Engine with FP16 fallback stay outside the supported capture paths."""
        mc = _base_model_config(cuda_graphs=True, fused_optimizer=False)
        module, _, _, _ = _build_module(model_config=mc, train_config=_base_train_config(tmp_path))
        trainer = SimpleNamespace(world_size=1, precision=precision)

        mocker.patch.object(type(module), "device", new_callable=PropertyMock, return_value=torch.device("cuda"))
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        runner_type = mocker.patch("rfdetr.training.module_model.CudaGraphTrainingRunner")
        module.on_train_start()

        runner_type.assert_not_called()
        assert module._cuda_graph_runner is None


def _module_with_restored_optimizer(
    tmp_path: Path, configured: list[tuple[float, float | None]], *, restarts: bool
) -> tuple[RFDETRModelModule, torch.optim.Optimizer]:
    """Build a module whose optimizer holds one group restored from an older ``ReduceLROnPlateau`` checkpoint.

    The group trained at 0.1 and was reduced to 0.025, and records no ``initial_lr``. The module is marked as resuming
    with an explicit setting, as ``on_load_checkpoint`` leaves it.

    Args:
        tmp_path: Directory for the module's config.
        configured: What ``configure_optimizers`` recorded; empty when a subclass replaced it.
        restarts: Whether a learning-rate setting changed, so groups without a base restart.

    Returns:
        The module and its restored optimizer.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     module, optimizer = _module_with_restored_optimizer(Path(directory), [(0.1, 0.0)], restarts=True)
        >>> optimizer.param_groups[0]["lr"], module._reapplied_param_group_settings
        (0.025, ['lr'])
    """
    module, *_ = _build_module(tmp_path=tmp_path)
    optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=0.025, weight_decay=0.0)
    module.optimizers = MagicMock(return_value=optimizer)
    module._current_lr_scheduler = MagicMock(return_value=None)
    module._configured_param_group_settings = configured
    module._reapplied_param_group_settings = ["lr"]
    module._restarts_groups_without_base = restarts
    return module, optimizer


class TestReapplyConfiguredParamGroupSettings:
    """``on_train_start`` puts the configured settings back over a resumed checkpoint's optimizer state (#1613)."""

    @pytest.fixture(autouse=True)
    def fed_by(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Stand in for the model-dependent mask: every explicit setting feeds every group.

        Examples:
            >>> fed_by  # doctest: +SKIP
            A pytest fixture; it needs monkeypatch.
        """
        mock = MagicMock(side_effect=lambda names, configured, *_: configured)
        monkeypatch.setattr("rfdetr.training.module_model._configured_settings_fed_by", mock)
        return mock

    def test_explicit_settings_mask_configured_ones(self, tmp_path: Path, fed_by: MagicMock) -> None:
        """A group no explicit setting feeds keeps what the checkpoint's optimizer holds, stale ``args`` or not."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=True)
        module._reapplied_param_group_settings = ["weight_decay"]
        fed_by.side_effect = lambda names, configured, *_: [(None, weight_decay) for _, weight_decay in configured]

        module._reapply_configured_param_group_settings()

        assert fed_by.call_args.args[0] == ["weight_decay"]
        assert optimizer.param_groups[0]["lr"] == 0.025

    @pytest.mark.parametrize(("restarts", "expected_lr"), [(True, 0.1), (False, 0.025)])
    def test_group_without_base_restarts_only_for_lr_change(
        self, tmp_path: Path, restarts: bool, expected_lr: float
    ) -> None:
        """A param group with no base lr is restarted only when the reapplied setting is a learning rate."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=restarts)

        module._reapply_configured_param_group_settings()

        assert optimizer.param_groups[0]["lr"] == expected_lr

    def test_unfed_settings_leave_optimizer_alone_and_warn(
        self,
        tmp_path: Path,
        fed_by: MagicMock,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parameters the model does not own (FSDP, DeepSpeed) match no group; the resume must not pass silently."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=True)
        fed_by.side_effect = lambda names, configured, *_: [(None, None)] * len(configured)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        caplog.set_level(logging.WARNING, logger="rf-detr")

        module._reapply_configured_param_group_settings()

        assert optimizer.param_groups[0]["lr"] == 0.025
        assert "No parameter group of the resumed optimizer is fed by lr" in caplog.text

    def test_weight_decay_alone_is_put_back(self, tmp_path: Path, fed_by: MagicMock) -> None:
        """A resume that passes only ``weight_decay`` feeds no learning rate, but its weight decay still applies."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [(0.1, 0.05)], restarts=False)
        module._reapplied_param_group_settings = ["weight_decay"]
        fed_by.side_effect = lambda names, configured, *_: [(None, weight_decay) for _, weight_decay in configured]

        module._reapply_configured_param_group_settings()

        assert optimizer.param_groups[0]["weight_decay"] == 0.05

    def test_on_train_start_reapplies_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second ``fit`` of the module carries no resumed state to put back."""
        module, _ = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=True)
        module._reapply_configured_param_group_settings = MagicMock()
        module._configure_cuda_graph_runner = MagicMock()
        monkeypatch.setattr(RFDETRModelModule, "_use_fused_optimizer", False)

        module.on_train_start()
        module.on_train_start()

        assert module._reapply_configured_param_group_settings.call_count == 1

    def test_logs_how_many_groups_changed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume hit by #1613 recorded the new ``lr`` already, so only this line reports the change."""
        module, _ = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=True)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        caplog.set_level(logging.INFO, logger="rf-detr")

        module._reapply_configured_param_group_settings()

        assert "Gave the learning rate and weight decay set for this run to 1 resumed parameter group(s)" in caplog.text

    def test_replaced_configure_optimizers_keeps_restored_optimizer(self, tmp_path: Path) -> None:
        """A subclass's own ``configure_optimizers`` records no settings, so there is nothing to compare with."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [], restarts=True)

        module._reapply_configured_param_group_settings()

        assert optimizer.param_groups[0]["lr"] == 0.025

    def test_replaced_configure_optimizers_is_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-replaced ``configure_optimizers`` is reported with a warning when settings are reapplied."""
        module, _ = _module_with_restored_optimizer(tmp_path, [], restarts=True)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        caplog.set_level(logging.WARNING, logger="rf-detr")

        module._reapply_configured_param_group_settings()

        assert "configure_optimizers was replaced" in caplog.text

    def test_several_optimizers_from_replaced_configure_optimizers_left_alone(self, tmp_path: Path) -> None:
        """Lightning returns a list when ``configure_optimizers`` built more than one optimizer."""
        module, optimizer = _module_with_restored_optimizer(tmp_path, [(0.1, 0.0)], restarts=True)
        module.optimizers = MagicMock(return_value=[optimizer, optimizer])

        module._reapply_configured_param_group_settings()

        assert optimizer.param_groups[0]["lr"] == 0.025


class TestFusedOptimizerResumeStateNormalization:
    """Resume-time fused AdamW state normalization in on_train_start()."""

    @pytest.fixture
    def fused_module(self, mocker) -> RFDETRModelModule:
        """Return a bare fused-AdamW module on bf16-mixed with CUDA reported available and ``optimizers`` mocked.

        Examples:
            >>> fused_module  # doctest: +SKIP
            A pytest fixture; it needs mocker.
        """
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        module = RFDETRModelModule.__new__(RFDETRModelModule)
        module.model_config = SimpleNamespace(fused_optimizer=True)
        module.train_config = SimpleNamespace(optimizer="adamw")
        module._reapplied_param_group_settings = []
        trainer = MagicMock()
        trainer.precision = "bf16-mixed"
        trainer.is_global_zero = True
        module._trainer = trainer
        module.optimizers = MagicMock()
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        return module

    @pytest.mark.parametrize(
        ("wrapped", "exp_avg_value", "exp_avg_sq_value"),
        [
            pytest.param(False, 2.0, 3.0, id="plain-optimizer"),
            pytest.param(True, 4.0, 5.0, id="lightning-optimizer-wrapper"),
        ],
    )
    def test_on_train_start_normalizes_restored_fused_optimizer_state(
        self, fused_module, wrapped: bool, exp_avg_value: float, exp_avg_sq_value: float
    ) -> None:
        """Resumed fused AdamW state tensors must match the live parameter layout before the first step.

        A Lightning-style optimizer wrapper (exposing ``.optimizer``) must be unwrapped and normalized the same way.
        """
        optimizer_param = nn.Parameter(torch.arange(6.0, dtype=torch.bfloat16).reshape(2, 3).t())
        optimizer = torch.optim.AdamW([optimizer_param], lr=1e-3)
        optimizer.state[optimizer_param] = {
            "exp_avg": torch.full((3, 2), exp_avg_value, dtype=torch.float32),
            "exp_avg_sq": torch.full((3, 2), exp_avg_sq_value, dtype=torch.float64),
        }
        fused_module.optimizers.return_value = SimpleNamespace(optimizer=optimizer) if wrapped else optimizer

        fused_module.on_train_start()

        fused_module.optimizers.assert_called_once_with(use_pl_optimizer=False)
        state = optimizer.state[optimizer_param]
        assert state["exp_avg"].dtype == optimizer_param.dtype
        assert state["exp_avg"].stride() == optimizer_param.stride()
        assert state["exp_avg_sq"].dtype == optimizer_param.dtype
        assert state["exp_avg_sq"].stride() == optimizer_param.stride()
        torch.testing.assert_close(state["exp_avg"], torch.full_like(optimizer_param, exp_avg_value))
        torch.testing.assert_close(state["exp_avg_sq"], torch.full_like(optimizer_param, exp_avg_sq_value))

    def test_on_train_start_leaves_empty_optimizer_state_untouched(self, fused_module) -> None:
        """Fresh fused-optimizer runs with no restored state should remain a no-op."""
        optimizer_param = nn.Parameter(torch.ones((2, 2), dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW([optimizer_param], lr=1e-3)
        fused_module.optimizers.return_value = optimizer

        fused_module.on_train_start()

        assert optimizer.state == {}
