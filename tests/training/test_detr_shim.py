# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Chapter 5 / Phase 7+8 (updated Phase 3):

1. ``TestRFDETRTrainPTL``           — RFDETR.train() delegates to PTL build_trainer().fit()
2. ``TestRFDETRTrainPTLAbsorption`` — Legacy kwargs absorbed by RFDETR.train()
3. ``TestConvertLegacyCheckpoint``  — convert_legacy_checkpoint() round-trip
4. ``TestOnLoadCheckpoint``         — RFDETRModule.on_load_checkpoint() auto-detect
5. ``TestPublicAPIExports``         — rfdetr.__init__ exports RFDETRModule/DataModule/build_trainer
"""

import argparse
import builtins
import importlib
import sys
import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR, RFDETRLarge
from rfdetr.training.auto_batch import AutoBatchResult
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.training.module_model import RFDETRModelModule

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    defaults = dict(pretrain_weights=None, num_classes=3, device="cpu")
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _make_train_config(tmp_path, **overrides):
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_rfdetr_self(tmp_path, **train_overrides):
    """Return a MagicMock shaped like RFDETR with real config objects.

    No spec is used because RFDETR.model is set in __init__ (instance attr)
    and spec=RFDETR would block access to it.
    """
    mock = MagicMock()
    mock.model_config = _make_model_config()
    mock.model = MagicMock()  # exposes mock.model.model for sync-back assertions
    mock.get_train_config.return_value = _make_train_config(tmp_path, **train_overrides)
    return mock


def _patch_lit():
    """Context manager that patches all three rfdetr.training entry points."""
    mock_module_cls = MagicMock(name="RFDETRModule_cls")
    mock_dm_cls = MagicMock(name="RFDETRDataModule_cls")
    mock_build_trainer = MagicMock(name="build_trainer")

    return (
        patch("rfdetr.training.RFDETRModelModule", mock_module_cls),
        patch("rfdetr.training.RFDETRDataModule", mock_dm_cls),
        patch("rfdetr.training.build_trainer", mock_build_trainer),
        mock_module_cls,
        mock_dm_cls,
        mock_build_trainer,
    )


# ---------------------------------------------------------------------------
# 1. RFDETR.train() PTL delegation
# ---------------------------------------------------------------------------


class TestRFDETRTrainPTL:
    """RFDETR.train() delegates to PTL build_trainer().fit()."""

    def test_build_trainer_called_with_config_and_model_config(self, tmp_path):
        """build_trainer receives (train_config, model_config) in the right order."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator=None)

    def test_trainer_fit_called_with_module_and_datamodule(self, tmp_path):
        """trainer.fit() is called with (module_instance, datamodule_instance)."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, mcls, dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        fit_args = trainer.fit.call_args
        assert fit_args[0][0] is mcls.return_value  # module instance
        assert fit_args[0][1] is dmcls.return_value  # datamodule instance

    def test_ckpt_path_none_when_resume_not_set(self, tmp_path):
        """trainer.fit receives ckpt_path=None when config.resume is None."""
        mock_self = _make_rfdetr_self(tmp_path)  # resume defaults to None
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path=None)

    def test_ckpt_path_forwarded_when_resume_set(self, tmp_path):
        """trainer.fit receives ckpt_path when config.resume is a path string."""
        mock_self = _make_rfdetr_self(tmp_path, resume="/some/checkpoint.ckpt")
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path="/some/checkpoint.ckpt")

    def test_ckpt_path_none_when_resume_is_empty_string(self, tmp_path):
        """config.resume='' is coerced to ckpt_path=None via `resume or None`."""
        mock_self = _make_rfdetr_self(tmp_path)
        # Create a real TrainConfig-like object where resume is ""
        mock_config = MagicMock(spec=TrainConfig)
        mock_config.resume = ""
        mock_config.batch_size = 4  # int so auto-batch branch is not taken
        mock_self.get_train_config.return_value = mock_config

        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        _, fit_kwargs = trainer.fit.call_args
        assert fit_kwargs["ckpt_path"] is None

    def test_model_model_synced_back_by_identity(self, tmp_path):
        """self.model.model is reassigned to module.model (identity, not copy)."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, mcls, _dmcls, mock_bt = _patch_lit()
        sentinel_nn_module = object()
        mcls.return_value.model = sentinel_nn_module

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.model is sentinel_nn_module

    def test_returns_none(self, tmp_path):
        """RFDETR.train() has no return value."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            result = RFDETR.train(mock_self)
        assert result is None

    def test_missing_training_extra_raises_install_hint(self, tmp_path, monkeypatch):
        """Missing training dependencies should raise ImportError with extras install hint."""
        mock_self = _make_rfdetr_self(tmp_path)
        real_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rfdetr.training":
                raise ModuleNotFoundError("No module named 'pytorch_lightning'", name="pytorch_lightning")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        with pytest.raises(ImportError, match=r"rfdetr\[train,loggers\]") as exc_info:
            RFDETR.train(mock_self)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.parametrize(
        "missing_name",
        ["rfdetr.training", "rfdetr.training.auto_batch"],
        ids=["training-package", "training-submodule"],
    )
    def test_internal_training_module_import_error_preserved(self, tmp_path, monkeypatch, missing_name):
        """Missing internal training modules should keep original ModuleNotFoundError."""
        mock_self = _make_rfdetr_self(tmp_path)
        real_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == missing_name:
                raise ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        with pytest.raises(ModuleNotFoundError, match=missing_name.replace(".", r"\.")):
            RFDETR.train(mock_self)

    def test_class_names_synced_from_datamodule_after_training(self, tmp_path):
        """self.model.class_names is set from RFDETRDataModule.class_names after train().

        Regression test for #509: custom class names were not synced back from
        RFDETRDataModule after training, causing predict() to return COCO labels
        instead of the dataset's class labels.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = _patch_lit()
        custom_class_names = ["cat", "dog", "bird"]
        dmcls.return_value.class_names = custom_class_names

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == custom_class_names

    def test_class_names_not_synced_when_datamodule_returns_none(self, tmp_path):
        """self.model.class_names is NOT overwritten when datamodule.class_names is None.

        Ensures the sync-back guard does not clobber existing class names
        when the datamodule has no class information (e.g. custom dataset format).
        """
        mock_self = _make_rfdetr_self(tmp_path)
        sentinel_names = ["existing_class"]
        mock_self.model.class_names = sentinel_names
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = _patch_lit()
        dmcls.return_value.class_names = None  # datamodule has no class names

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == sentinel_names

    def test_empty_class_names_synced_from_datamodule_after_training(self, tmp_path):
        """Empty class name lists are synced and overwrite stale model labels.

        Empty list is a valid explicit value and should not be treated as missing.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        sentinel_names = ["stale_label"]
        mock_self.model.class_names = sentinel_names
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = _patch_lit()
        dmcls.return_value.class_names = []

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == []

    def test_device_kwarg_cpu_no_warning(self, tmp_path):
        """device='cpu' is consumed without a DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device="cpu")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_kwarg_non_cpu_emits_deprecation(self, tmp_path):
        """Non-'cpu' device= emits a DeprecationWarning and is not forwarded."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device="cuda")
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 1
        assert "cuda" in str(dep_warns[0].message)
        # device must not have been forwarded to get_train_config
        mock_self.get_train_config.assert_called_once_with()

    def test_callbacks_none_no_warning(self, tmp_path):
        """callbacks=None produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=None)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_empty_dict_no_warning(self, tmp_path):
        """callbacks={} (falsy dict) produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks={})
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_all_empty_lists_no_warning(self, tmp_path):
        """callbacks dict with all-empty lists produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = defaultdict(list)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_non_empty_emits_deprecation_warning(self, tmp_path):
        """callbacks dict with a non-empty list emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        assert "PTL" in str(depr[0].message)

    def test_callbacks_mixed_emits_deprecation_warning(self, tmp_path):
        """Mixed callbacks (some empty, some non-empty) triggers DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [], "on_train_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_do_benchmark_false_no_warning(self, tmp_path):
        """do_benchmark=False (default) emits no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=False)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    @pytest.mark.parametrize("truthy_value", [True, 1, "yes"], ids=["bool_true", "int_1", "str_yes"])
    def test_do_benchmark_truthy_emits_deprecation_warning(self, tmp_path, truthy_value):
        """Any truthy do_benchmark value emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=truthy_value)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        assert "rfdetr benchmark" in str(depr[0].message)

    def test_do_benchmark_not_forwarded_to_get_train_config(self, tmp_path):
        """do_benchmark is popped before calling get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=True)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_not_forwarded_to_get_train_config(self, tmp_path):
        """device= is popped and not passed on to get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cpu")
        # get_train_config must have been called without device=
        assert "device" not in mock_self.get_train_config.call_args.kwargs

    def test_batch_size_auto_resolved_before_module_and_datamodule_build(self, tmp_path):
        """batch_size='auto' is resolved to ints before module/datamodule init."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto", grad_accum_steps=99)
        auto_result = AutoBatchResult(
            safe_micro_batch=3,
            recommended_grad_accum_steps=6,
            effective_batch_size=18,
            device_name="Fake GPU",
        )
        p_mod, p_dm, p_bt, mcls, dmcls, _mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt, patch("rfdetr.training.auto_batch.resolve_auto_batch_config", return_value=auto_result):
            RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        assert config.batch_size == 3
        assert config.grad_accum_steps == 6
        mcls.assert_called_once_with(mock_self.model_config, config)
        dmcls.assert_called_once_with(mock_self.model_config, config)

    def test_batch_size_auto_calls_resolver_with_expected_context(self, tmp_path):
        """Auto-batch resolver receives model context, model config, and train config."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto")
        auto_result = AutoBatchResult(
            safe_micro_batch=2,
            recommended_grad_accum_steps=8,
            effective_batch_size=16,
            device_name="Fake GPU",
        )
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.training.auto_batch.resolve_auto_batch_config", return_value=auto_result) as mock_resolve,
        ):
            RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        mock_resolve.assert_called_once_with(
            model_context=mock_self.model,
            model_config=mock_self.model_config,
            train_config=config,
        )


# ---------------------------------------------------------------------------
# 2. RFDETR.train() legacy kwarg absorption
# ---------------------------------------------------------------------------


class TestRFDETRTrainPTLAbsorption:
    """RFDETR.train() absorbs legacy kwargs and routes through PTL build_trainer()."""

    def test_device_cpu_absorbed_as_accelerator_cpu(self, tmp_path):
        """device='cpu' is absorbed and forwarded to build_trainer as accelerator='cpu'."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cpu")
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="cpu")

    def test_callbacks_empty_dict_no_error(self, tmp_path):
        """callbacks={} is accepted without error."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, callbacks={})  # must not raise

    def test_callbacks_non_empty_emits_deprecation_warning(self, tmp_path):
        """callbacks with non-empty lists emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1

    def test_start_epoch_emits_deprecation_warning(self, tmp_path):
        """start_epoch=1 emits DeprecationWarning and is dropped."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, start_epoch=1)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("start_epoch" in str(d.message) for d in depr)
        # start_epoch must not reach get_train_config
        assert "start_epoch" not in mock_self.get_train_config.call_args.kwargs

    def test_do_benchmark_true_emits_deprecation_warning(self, tmp_path):
        """do_benchmark=True emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=True)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("do_benchmark" in str(d.message) or "rfdetr benchmark" in str(d.message) for d in depr)

    def test_returns_none(self, tmp_path):
        """RFDETR.train() returns None."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            result = RFDETR.train(mock_self)
        assert result is None


# ---------------------------------------------------------------------------
# 3. convert_legacy_checkpoint
# ---------------------------------------------------------------------------


class _CustomArgs:
    """Module-level class so torch.save can pickle instances of it."""

    lr: float
    epochs: int


def _make_legacy_pth(tmp_path, epoch=5, include_ema=False, args_value="namespace") -> str:
    """Write a minimal legacy .pth checkpoint and return its path."""
    path = str(tmp_path / "legacy.pth")
    state = {
        "layer.weight": torch.ones(2, 3),
        "layer.bias": torch.zeros(3),
    }
    ckpt: Dict[str, Any] = {"model": state, "epoch": epoch}

    if args_value == "namespace":
        ns = argparse.Namespace(lr=1e-4, epochs=100)
        ckpt["args"] = ns
    elif args_value == "dict":
        ckpt["args"] = {"lr": 1e-4, "epochs": 100}
    elif args_value is None:
        ckpt["args"] = None
    elif args_value == "missing":
        pass  # no "args" key at all
    else:
        ckpt["args"] = args_value

    if include_ema:
        ckpt["ema_model"] = {k: v.clone() * 0.99 for k, v in state.items()}

    torch.save(ckpt, path)
    return path


class TestConvertLegacyCheckpoint:
    """convert_legacy_checkpoint() produces a valid PTL .ckpt file."""

    def test_state_dict_keys_prefixed_with_model(self, tmp_path):
        """All state_dict keys must be prefixed with 'model.'."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    def test_state_dict_keys_dot_containing_names_prefixed_once(self, tmp_path):
        """Keys already containing dots are prefixed exactly once."""
        path = str(tmp_path / "dot_keys.pth")
        torch.save({"model": {"backbone.layer.weight": torch.zeros(1)}, "epoch": 0}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "model.backbone.layer.weight" in ckpt["state_dict"]
        assert "model.model.backbone.layer.weight" not in ckpt["state_dict"]

    def test_epoch_preserved(self, tmp_path):
        """Epoch value is copied from the source checkpoint."""
        src = _make_legacy_pth(tmp_path, epoch=42)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 42

    def test_epoch_defaults_to_zero_when_missing(self, tmp_path):
        """Missing epoch key in source defaults to 0."""
        path = str(tmp_path / "no_epoch.pth")
        torch.save({"model": {"w": torch.zeros(1)}}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0

    def test_global_step_always_zero(self, tmp_path):
        """global_step is always written as 0."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["global_step"] == 0

    def test_legacy_checkpoint_format_flag_set(self, tmp_path):
        """legacy_checkpoint_format is always True in output."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["legacy_checkpoint_format"] is True

    def test_args_as_namespace_converted_to_dict(self, tmp_path):
        """argparse.Namespace args are converted to a plain dict via vars()."""
        src = _make_legacy_pth(tmp_path, args_value="namespace")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert isinstance(ckpt["hyper_parameters"], dict)
        assert ckpt["hyper_parameters"]["lr"] == pytest.approx(1e-4)

    def test_args_as_dict_kept_as_dict(self, tmp_path):
        """Plain dict args is preserved as-is."""
        src = _make_legacy_pth(tmp_path, args_value="dict")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {"lr": pytest.approx(1e-4), "epochs": 100}

    def test_args_none_gives_empty_hyper_parameters(self, tmp_path):
        """args=None produces an empty hyper_parameters dict."""
        src = _make_legacy_pth(tmp_path, args_value=None)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}

    def test_args_missing_key_gives_empty_hyper_parameters(self, tmp_path):
        """No 'args' key at all also produces empty hyper_parameters."""
        src = _make_legacy_pth(tmp_path, args_value="missing")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}

    def test_args_custom_object_with_dict_converted_via_vars(self, tmp_path):
        """A custom object with __dict__ is converted via vars()."""
        opts = _CustomArgs()
        opts.lr = 2e-4
        opts.epochs = 50

        path = str(tmp_path / "custom_args.pth")
        torch.save({"model": {"w": torch.zeros(1)}, "epoch": 0, "args": opts}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"]["lr"] == pytest.approx(2e-4)

    def test_ema_model_preserved_as_legacy_ema_state_dict(self, tmp_path):
        """ema_model present in source is written as legacy_ema_state_dict."""
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" in ckpt
        assert "layer.weight" in ckpt["legacy_ema_state_dict"]

    def test_no_ema_model_no_legacy_ema_state_dict(self, tmp_path):
        """No ema_model in source means legacy_ema_state_dict is absent."""
        src = _make_legacy_pth(tmp_path, include_ema=False)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" not in ckpt

    def test_round_trip_with_on_load_checkpoint(self, tmp_path):
        """convert_legacy_checkpoint output is handled correctly by on_load_checkpoint.

        After conversion, loading the .ckpt via on_load_checkpoint must NOT
        re-apply the 'model.' prefix because 'state_dict' already exists.
        """
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)

        class _FakeModule:
            pass

        fake = _FakeModule()
        original_state_dict = dict(ckpt["state_dict"])  # copy before mutation

        RFDETRModelModule.on_load_checkpoint(fake, ckpt)

        # state_dict must NOT have been re-prefixed (already had "state_dict")
        assert ckpt["state_dict"] == original_state_dict
        # EMA stashed
        assert hasattr(fake, "_pending_legacy_ema_state")

    def test_missing_model_key_raises_value_error(self, tmp_path):
        """Source file with no 'model' key raises ValueError with a clear message."""
        path = str(tmp_path / "no_model.pth")
        torch.save({"epoch": 5}, path)
        dst = str(tmp_path / "out.ckpt")

        with pytest.raises(ValueError, match="'model' key"):
            convert_legacy_checkpoint(path, dst)

    def test_args_primitive_type_falls_back_to_empty_dict(self, tmp_path):
        """args of a non-dict, non-Namespace type (e.g. string) falls back to {} with a warning."""
        path = str(tmp_path / "prim_args.pth")
        torch.save({"model": {"w": torch.zeros(1)}, "args": "legacy_string_value"}, path)
        dst = str(tmp_path / "out.ckpt")

        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}


# ---------------------------------------------------------------------------
# 4. RFDETRModule.on_load_checkpoint
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal object supporting attribute assignment for on_load_checkpoint tests."""


class TestOnLoadCheckpoint:
    """RFDETRModule.on_load_checkpoint auto-detects legacy formats."""

    def test_raw_pth_writes_state_dict_with_prefix(self):
        """'model' key without 'state_dict' → state_dict written with 'model.' prefix."""
        fake = _FakeModule()
        ckpt = {"model": {"backbone.weight": torch.zeros(2)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "state_dict" in ckpt
        assert "model.backbone.weight" in ckpt["state_dict"]

    def test_raw_pth_original_model_key_preserved(self):
        """Original 'model' key is not deleted after state_dict is written."""
        fake = _FakeModule()
        ckpt = {"model": {"w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "model" in ckpt  # PTL may inspect it; must not be deleted

    def test_empty_model_dict_produces_empty_state_dict(self):
        """Empty 'model' dict without 'state_dict' → empty state_dict written."""
        fake = _FakeModule()
        ckpt = {"model": {}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] == {}

    def test_native_ptl_format_no_op(self):
        """Native PTL checkpoint (has 'state_dict', no 'model') → no mutation."""
        fake = _FakeModule()
        sentinel = {"model.layer.weight": torch.zeros(1)}
        ckpt = {"state_dict": sentinel}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] is sentinel  # not replaced
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_both_model_and_state_dict_present_state_dict_not_overwritten(self):
        """'state_dict' is NOT overwritten when both 'model' and 'state_dict' exist."""
        fake = _FakeModule()
        existing_sd = {"model.existing": torch.zeros(1)}
        ckpt = {
            "state_dict": existing_sd,
            "model": {"new_key": torch.ones(1)},
        }
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] is existing_sd
        assert "model.new_key" not in ckpt["state_dict"]

    def test_legacy_ema_state_dict_stashed(self):
        """'legacy_ema_state_dict' in checkpoint → stashed on _pending_legacy_ema_state."""
        fake = _FakeModule()
        ema_weights = {"layer.weight": torch.ones(2)}
        ckpt = {
            "state_dict": {"model.layer.weight": torch.zeros(2)},
            "legacy_ema_state_dict": ema_weights,
        }
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert fake._pending_legacy_ema_state is ema_weights

    def test_no_legacy_ema_attribute_not_set(self):
        """No 'legacy_ema_state_dict' → _pending_legacy_ema_state not set on module."""
        fake = _FakeModule()
        ckpt = {"state_dict": {"model.w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_empty_checkpoint_is_noop(self):
        """Completely empty checkpoint {} triggers no mutation and no error."""
        fake = _FakeModule()
        ckpt: Dict[str, Any] = {}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)  # must not raise
        assert ckpt == {}
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_second_call_overwrites_pending_ema(self):
        """Calling on_load_checkpoint twice with EMA overwrites the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        second_ema = {"w": torch.ones(1)}
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": second_ema})
        assert fake._pending_legacy_ema_state is second_ema

    def test_second_call_without_ema_leaves_first_stash(self):
        """Second call without 'legacy_ema_state_dict' does not clear the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}})
        assert fake._pending_legacy_ema_state is first_ema


# ---------------------------------------------------------------------------
# 5. Public API exports
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    """rfdetr.__init__ exposes PTL names via __getattr__ (rfdetr[train] extra)."""

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_importable_from_rfdetr(self, name):
        """Each PTL export is accessible as rfdetr.<name> via lazy __getattr__."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} is missing"

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_is_same_object_as_rfdetr_training(self, name):
        """rfdetr.<name> is the identical object to rfdetr.training.<name>."""
        import rfdetr
        import rfdetr.training

        assert getattr(rfdetr, name) is getattr(rfdetr.training, name)

    def test_ptl_names_not_in_all(self):
        """PTL exports are optional (rfdetr[train]) and must not be in rfdetr.__all__."""
        import rfdetr

        for name in ("RFDETRModelModule", "RFDETRDataModule", "build_trainer"):
            assert name not in rfdetr.__all__, f"{name} must not be in __all__ (optional extra)"

    def test_rfdetr_all_no_duplicates(self):
        """rfdetr.__all__ contains no duplicate names."""
        import rfdetr

        assert len(rfdetr.__all__) == len(set(rfdetr.__all__))

    def test_plus_symbol_resolution_does_not_mutate_all(self, monkeypatch):
        """Top-level __all__ remains static when plus-only symbols resolve lazily."""
        import rfdetr
        import rfdetr.platform.models

        sentinel = object()
        monkeypatch.setitem(rfdetr.platform.models.__dict__, "RFDETRXLarge", sentinel)
        monkeypatch.delitem(rfdetr.__dict__, "RFDETRXLarge", raising=False)

        original_all = list(rfdetr.__all__)
        assert getattr(rfdetr, "RFDETRXLarge") is sentinel
        assert rfdetr.__all__ == original_all

    def test_existing_exports_still_present(self):
        """Original RFDETR* class exports are unchanged."""
        import rfdetr

        for name in ["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"]:
            assert hasattr(rfdetr, name), f"rfdetr.{name} unexpectedly missing"

    def test_convert_legacy_checkpoint_not_in_rfdetr_namespace(self):
        """convert_legacy_checkpoint is in rfdetr.training but not the top-level rfdetr namespace."""
        import rfdetr
        from rfdetr.training import convert_legacy_checkpoint  # noqa: F401

        # It is NOT directly on rfdetr (Phase 7.7 spec lists only the three PTL exports)
        assert not hasattr(rfdetr, "convert_legacy_checkpoint")


class TestRemovedLegacyModuleAliases:
    """Removed legacy modules resolve via shims today and via migration hints after removal."""

    @staticmethod
    def _simulate_missing_removed_module_specs(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        """Force the removed-module finder to behave as if shim files no longer exist."""
        import rfdetr

        path_finder = rfdetr._RemovedModuleFinder._PATH_FINDER
        original_find_spec = path_finder.find_spec

        def _fake_find_spec(fullname: str, path: list[str] | None = None, target: object | None = None) -> object:
            if fullname in names:
                return None
            return original_find_spec(fullname, path, target)

        monkeypatch.setattr(path_finder, "find_spec", _fake_find_spec)
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)

        root_names = {name.removeprefix("rfdetr.").split(".", maxsplit=1)[0] for name in names}
        for root_name in root_names:
            monkeypatch.delitem(rfdetr.__dict__, root_name, raising=False)

    def test_removed_util_alias_resolves_via_package_attribute(self) -> None:
        """PEP 562 lookup resolves rfdetr.util while the shim package exists."""
        import rfdetr

        assert getattr(rfdetr, "util").__name__ == "rfdetr.util"

    def test_removed_deploy_alias_resolves_via_package_attribute(self) -> None:
        """PEP 562 lookup resolves rfdetr.deploy while the shim package exists."""
        import rfdetr

        assert rfdetr.deploy.__name__ == "rfdetr.deploy"

    def test_removed_shim_missing_raises_importerror_with_getattr(self) -> None:
        """Missing removed shim should raise ImportError with migration hint."""
        import rfdetr

        missing_name = "rfdetr.missing_removed_shim"
        missing_exc = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
        with (
            patch.dict(rfdetr._REMOVED_IN_V17, {"missing_removed_shim": "migration hint"}),
            patch("rfdetr.importlib.import_module", side_effect=missing_exc),
            pytest.raises(ImportError, match="migration hint"),
        ):
            getattr(rfdetr, "missing_removed_shim")

    def test_nested_module_not_found_is_not_masked_for_package_attribute(self) -> None:
        """Nested ModuleNotFoundError from inside a shim import should propagate."""
        import rfdetr

        with (
            patch.dict(rfdetr._REMOVED_IN_V17, {"missing_dep_shim": "migration hint"}),
            patch(
                "rfdetr.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'torchvision_ops'", name="torchvision_ops"),
            ),
            pytest.raises(ModuleNotFoundError, match="torchvision_ops"),
        ):
            getattr(rfdetr, "missing_dep_shim")

    def test_removed_util_import_raises_migration_hint_when_shim_is_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dotted legacy imports get a migration hint once the util shim package is removed."""
        self._simulate_missing_removed_module_specs(monkeypatch, "rfdetr.util")

        with pytest.raises(ImportError, match=r"rfdetr\.util was removed in v1\.7"):
            importlib.import_module("rfdetr.util")

    def test_removed_deploy_submodule_import_raises_migration_hint_when_shim_is_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dotted legacy submodule imports get a migration hint once the deploy shim is removed."""
        self._simulate_missing_removed_module_specs(monkeypatch, "rfdetr.deploy", "rfdetr.deploy.benchmark")

        with pytest.raises(ImportError, match=r"rfdetr\.deploy was removed in v1\.7"):
            importlib.import_module("rfdetr.deploy.benchmark")

    def test_find_spec_ignores_non_rfdetr_top_level_imports(self) -> None:
        """find_spec must not intercept bare top-level imports like 'util' or 'deploy'."""
        import rfdetr

        finder = rfdetr._RemovedModuleFinder()
        assert finder.find_spec("util", None) is None
        assert finder.find_spec("deploy", None) is None
        assert finder.find_spec("os", None) is None

    def test_meta_path_insertion_is_idempotent_across_reload(self) -> None:
        """importlib.reload(rfdetr) must not insert a second finder into sys.meta_path."""
        import rfdetr

        count_before = sum(type(f).__name__ == "_RemovedModuleFinder" for f in sys.meta_path)
        importlib.reload(rfdetr)
        count_after = sum(type(f).__name__ == "_RemovedModuleFinder" for f in sys.meta_path)
        assert count_after == count_before, (
            f"reload added {count_after - count_before} extra finder(s) to sys.meta_path"
        )


# ---------------------------------------------------------------------------
# 6. RFDETRLarge deprecated-config fallback behaviour
# ---------------------------------------------------------------------------


class TestRFDETRLargeFallback:
    """RFDETRLarge retries only for deprecated-weight compatibility errors."""

    def test_cuda_oom_runtime_error_does_not_retry(self, monkeypatch):
        """CUDA OOM should fail fast without deprecated-config retry."""
        call_count = 0

        def _raise_oom(self, **kwargs):
            del self, kwargs
            nonlocal call_count
            call_count += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 16.00 MiB.")

        monkeypatch.setattr(RFDETR, "__init__", _raise_oom)

        with pytest.raises(RuntimeError, match="out of memory"):
            RFDETRLarge()

        assert call_count == 1

    def test_state_dict_runtime_error_retries_once_with_deprecated_config(self, monkeypatch):
        """State-dict mismatch errors trigger exactly one deprecated-config retry."""
        call_count = 0

        def _raise_then_succeed(self, **kwargs):
            del kwargs
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Error(s) in loading state_dict for Model: size mismatch for backbone.weight")
            self.model = MagicMock()

        monkeypatch.setattr(RFDETR, "__init__", _raise_then_succeed)
        warn_spy = MagicMock()
        monkeypatch.setattr("rfdetr.detr.logger.warning", warn_spy)

        model = RFDETRLarge()

        assert model.is_deprecated is True
        assert call_count == 2
        warn_spy.assert_called_once()


# ---------------------------------------------------------------------------
# 7. _load_pretrain_weights_into — detr.py path (the non-PTL scenario from #806)
# ---------------------------------------------------------------------------


def _make_detr_args(
    pretrain_weights="/fake/weights.pth",
    num_classes=90,
    num_queries=300,
    group_detr=13,
    segmentation_head=False,
    patch_size=14,
):
    """Return a SimpleNamespace shaped like the args passed to _load_pretrain_weights_into."""
    return SimpleNamespace(
        pretrain_weights=pretrain_weights,
        num_classes=num_classes,
        num_queries=num_queries,
        group_detr=group_detr,
        segmentation_head=segmentation_head,
        patch_size=patch_size,
    )


def _make_detr_checkpoint(
    num_classes=91,
    num_queries=300,
    group_detr=13,
    segmentation_head=False,
    patch_size=14,
):
    """Return a minimal checkpoint dict for _load_pretrain_weights_into tests."""
    total_queries = num_queries * group_detr
    state = {
        "class_embed.bias": torch.zeros(num_classes),
        "refpoint_embed.weight": torch.zeros(total_queries, 4),
        "query_feat.weight": torch.zeros(total_queries, 256),
    }
    ckpt_args = SimpleNamespace(
        segmentation_head=segmentation_head,
        patch_size=patch_size,
        class_names=[],
    )
    return {"model": state, "args": ckpt_args}


class TestLoadPretrainWeightsInto:
    """Tests for load_pretrain_weights (models/weights.py) — checkpoint compatibility
    validation exercised when RFDETRNano(pretrain_weights=...) is called (issue #806)."""

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_seg_ckpt_into_detection_model_raises_via_detr_path(self, monkeypatch, tmp_path):
        """Segmentation checkpoint must raise ValueError when loaded into a detection model."""
        from rfdetr.models.weights import load_pretrain_weights

        checkpoint = _make_detr_checkpoint(segmentation_head=True, patch_size=14)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", segmentation_head=False)

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(fake_model, mc)

    def test_patch_size_mismatch_raises_via_detr_path(self, monkeypatch, tmp_path):
        """patch_size mismatch must raise ValueError via the load_pretrain_weights path."""
        from rfdetr.models.weights import load_pretrain_weights

        checkpoint = _make_detr_checkpoint(segmentation_head=False, patch_size=12)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", patch_size=16)

        with pytest.raises(ValueError, match=r"patch_size"):
            load_pretrain_weights(fake_model, mc)


# ---------------------------------------------------------------------------
# 7. RFDETR.class_names property — empty-list identity check
# ---------------------------------------------------------------------------


class TestClassNamesProperty:
    """RFDETR.class_names property returns List[str] (0-indexed)."""

    def test_empty_class_names_returns_empty_list_not_coco(self):
        """class_names property returns [] when model.class_names is [], NOT COCO fallback.

        Regression test for #509: the truthiness check `and self.model.class_names:`
        treated [] as falsy and fell through to return COCO_CLASSES, defeating the
        detr.py sync-back even after training on a dataset that reports empty names.
        The fix uses `is not None` so that [] is preserved.
        """
        mock_self = MagicMock()
        mock_self.model.class_names = []

        result = RFDETR.class_names.fget(mock_self)

        assert result == [], "class_names=[] must return [] (empty list), not COCO fallback"

    def test_none_class_names_returns_coco(self):
        """class_names property falls back to COCO_CLASS_NAMES when model.class_names is None."""
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        mock_self = MagicMock()
        mock_self.model.class_names = None

        result = RFDETR.class_names.fget(mock_self)

        assert result == COCO_CLASS_NAMES
        assert result is not COCO_CLASS_NAMES, "COCO fallback must return a copy, not the mutable global"

    def test_custom_class_names_returned_as_list(self):
        """Non-empty class_names are returned as a 0-indexed list."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)

        assert result == ["cat", "dog"]

    def test_custom_class_names_returns_shallow_copy(self):
        """Mutating the returned class_names list must not mutate model state."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)
        result.append("bird")

        assert result == ["cat", "dog", "bird"]
        assert mock_self.model.class_names == ["cat", "dog"]
