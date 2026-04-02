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
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR, RFDETRLarge
from rfdetr.detr import logger as detr_logger
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


@pytest.fixture
def patch_lit():
    """Provide patched rfdetr.training entry points for tests."""
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

    def test_build_trainer_called_with_config_and_model_config(self, tmp_path, patch_lit):
        """build_trainer receives (train_config, model_config) in the right order."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator=None)

    def test_trainer_fit_called_with_module_and_datamodule(self, tmp_path, patch_lit):
        """trainer.fit() is called with (module_instance, datamodule_instance)."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, mcls, dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        fit_args = trainer.fit.call_args
        assert fit_args[0][0] is mcls.return_value  # module instance
        assert fit_args[0][1] is dmcls.return_value  # datamodule instance

    def test_ckpt_path_none_when_resume_not_set(self, tmp_path, patch_lit):
        """trainer.fit receives ckpt_path=None when config.resume is None."""
        mock_self = _make_rfdetr_self(tmp_path)  # resume defaults to None
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path=None)

    def test_ckpt_path_forwarded_when_resume_set(self, tmp_path, patch_lit):
        """trainer.fit receives ckpt_path when config.resume is a path string."""
        mock_self = _make_rfdetr_self(tmp_path, resume="/some/checkpoint.ckpt")
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path="/some/checkpoint.ckpt")

    def test_ckpt_path_none_when_resume_is_empty_string(self, tmp_path, patch_lit):
        """config.resume='' is coerced to ckpt_path=None via `resume or None`."""
        mock_self = _make_rfdetr_self(tmp_path, resume="")

        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        _, fit_kwargs = trainer.fit.call_args
        assert fit_kwargs["ckpt_path"] is None

    def test_model_model_synced_back_by_identity(self, tmp_path, patch_lit):
        """self.model.model is reassigned to module.model (identity, not copy)."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, mcls, _dmcls, mock_bt = patch_lit
        sentinel_nn_module = object()
        mcls.return_value.model = sentinel_nn_module

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.model is sentinel_nn_module

    def test_returns_none(self, tmp_path, patch_lit):
        """RFDETR.train() has no return value."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            result = RFDETR.train(mock_self)
        assert result is None

    def test_missing_training_extra_raises_install_hint(self, tmp_path, monkeypatch, patch_lit):
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
    def test_internal_training_module_import_error_preserved(self, tmp_path, monkeypatch, missing_name, patch_lit):
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

    def test_class_names_synced_from_datamodule_after_training(self, tmp_path, patch_lit):
        """self.model.class_names is set from RFDETRDataModule.class_names after train().

        Regression test for #509: custom class names were not synced back from
        RFDETRDataModule after training, causing predict() to return COCO labels
        instead of the dataset's class labels.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = patch_lit
        custom_class_names = ["cat", "dog", "bird"]
        dmcls.return_value.class_names = custom_class_names

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == custom_class_names

    def test_class_names_not_synced_when_datamodule_returns_none(self, tmp_path, patch_lit):
        """self.model.class_names is NOT overwritten when datamodule.class_names is None.

        Ensures the sync-back guard does not clobber existing class names
        when the datamodule has no class information (e.g. custom dataset format).
        """
        mock_self = _make_rfdetr_self(tmp_path)
        sentinel_names = ["existing_class"]
        mock_self.model.class_names = sentinel_names
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = patch_lit
        dmcls.return_value.class_names = None  # datamodule has no class names

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == sentinel_names

    def test_empty_class_names_synced_from_datamodule_after_training(self, tmp_path, patch_lit):
        """Empty class name lists are synced and overwrite stale model labels.

        Empty list is a valid explicit value and should not be treated as missing.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        sentinel_names = ["stale_label"]
        mock_self.model.class_names = sentinel_names
        p_mod, p_dm, p_bt, _mcls, dmcls, _mock_bt = patch_lit
        dmcls.return_value.class_names = []

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.class_names == []

    def test_device_kwarg_cpu_no_warning(self, tmp_path, patch_lit):
        """device='cpu' is consumed without a DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device="cpu")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_kwarg_cuda_forwards_gpu_accelerator_without_devices(self, tmp_path, patch_lit):
        """device='cuda' is mapped to accelerator='gpu' without explicit devices override."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device="cuda")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_kwarg_torch_device_cuda_index_forwards_gpu_accelerator_and_devices(self, tmp_path, patch_lit):
        """torch.device('cuda:1') is mapped to accelerator='gpu' and devices=[1]."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device=torch.device("cuda:1"))
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    def test_callbacks_none_no_warning(self, tmp_path, patch_lit):
        """callbacks=None produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=None)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_empty_dict_no_warning(self, tmp_path, patch_lit):
        """callbacks={} (falsy dict) produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks={})
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_all_empty_lists_no_warning(self, tmp_path, patch_lit):
        """Callbacks dict with all-empty lists produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = defaultdict(list)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_non_empty_emits_deprecation_warning(self, tmp_path, patch_lit):
        """Callbacks dict with a non-empty list emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        assert "PTL" in str(depr[0].message)

    def test_callbacks_mixed_emits_deprecation_warning(self, tmp_path, patch_lit):
        """Mixed callbacks (some empty, some non-empty) triggers DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [], "on_train_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_do_benchmark_false_no_warning(self, tmp_path, patch_lit):
        """do_benchmark=False (default) emits no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=False)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    @pytest.mark.parametrize("truthy_value", [True, 1, "yes"], ids=["bool_true", "int_1", "str_yes"])
    def test_do_benchmark_truthy_emits_deprecation_warning(self, tmp_path, truthy_value, patch_lit):
        """Any truthy do_benchmark value emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=truthy_value)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        assert "rfdetr benchmark" in str(depr[0].message)

    def test_do_benchmark_not_forwarded_to_get_train_config(self, tmp_path, patch_lit):
        """do_benchmark is popped before calling get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=True)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_not_forwarded_to_get_train_config(self, tmp_path, patch_lit):
        """device= is popped and not passed on to get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cpu")
        # get_train_config must have been called without device=
        assert "device" not in mock_self.get_train_config.call_args.kwargs

    def test_batch_size_auto_resolved_before_module_and_datamodule_build(self, tmp_path, patch_lit):
        """batch_size='auto' is resolved to ints before module/datamodule init."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto", grad_accum_steps=99)
        auto_result = AutoBatchResult(
            safe_micro_batch=3,
            recommended_grad_accum_steps=6,
            effective_batch_size=18,
            device_name="Fake GPU",
        )
        p_mod, p_dm, p_bt, mcls, dmcls, _mock_bt = patch_lit
        with p_mod, p_dm, p_bt, patch("rfdetr.training.auto_batch.resolve_auto_batch_config", return_value=auto_result):
            RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        assert config.batch_size == 3
        assert config.grad_accum_steps == 6
        mcls.assert_called_once_with(mock_self.model_config, config)
        dmcls.assert_called_once_with(mock_self.model_config, config)

    def test_batch_size_auto_calls_resolver_with_expected_context(self, tmp_path, patch_lit):
        """Auto-batch resolver receives model context, model config, and train config."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto")
        auto_result = AutoBatchResult(
            safe_micro_batch=2,
            recommended_grad_accum_steps=8,
            effective_batch_size=16,
            device_name="Fake GPU",
        )
        p_mod, p_dm, p_bt, *_ = patch_lit
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

    def test_device_cpu_absorbed_as_accelerator_cpu(self, tmp_path, patch_lit):
        """device='cpu' is absorbed and forwarded to build_trainer as accelerator='cpu'."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cpu")
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="cpu")

    def test_device_cuda_absorbed_as_accelerator_gpu(self, tmp_path, patch_lit):
        """device='cuda' forwards accelerator='gpu' without a devices kwarg."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cuda")
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="gpu")
        assert "devices" not in mock_bt.call_args.kwargs

    def test_device_cuda_index_absorbed_as_accelerator_gpu_devices_list(self, tmp_path, patch_lit):
        """device='cuda:1' forwards accelerator='gpu' and devices=[1]."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cuda:1")
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="gpu", devices=[1])

    def test_device_torch_device_cuda_index_absorbed_as_accelerator_gpu_devices_list(self, tmp_path, patch_lit):
        """device=torch.device('cuda:2') forwards accelerator='gpu' and devices=[2]."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device=torch.device("cuda:2"))
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="gpu", devices=[2])

    def test_device_invalid_raises_value_error_with_expected_message(self, tmp_path, patch_lit):
        """Invalid device strings raise a ValueError with the train() device hint."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with (
            p_mod,
            p_dm,
            p_bt,
            pytest.raises(ValueError, match=r"Invalid device specifier for train\(\): 'notadevice'"),
        ):
            RFDETR.train(mock_self, device="notadevice")

    def test_device_unmapped_valid_type_warns_and_falls_back_to_auto_detection(self, tmp_path, patch_lit):
        """Valid but unmapped torch device types warn and use PTL auto-detection."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt, pytest.warns(UserWarning, match="auto-detection"):
            RFDETR.train(mock_self, device="meta")
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator=None)
        assert "devices" not in mock_bt.call_args.kwargs

    def test_callbacks_empty_dict_no_error(self, tmp_path, patch_lit):
        """callbacks={} is accepted without error."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, callbacks={})  # must not raise

    def test_callbacks_non_empty_emits_deprecation_warning(self, tmp_path, patch_lit):
        """Callbacks with non-empty lists emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, callbacks=callbacks)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1

    def test_start_epoch_emits_deprecation_warning(self, tmp_path, patch_lit):
        """start_epoch=1 emits DeprecationWarning and is dropped."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, start_epoch=1)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("start_epoch" in str(d.message) for d in depr)
        # start_epoch must not reach get_train_config
        assert "start_epoch" not in mock_self.get_train_config.call_args.kwargs

    def test_do_benchmark_true_emits_deprecation_warning(self, tmp_path, patch_lit):
        """do_benchmark=True emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, do_benchmark=True)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("do_benchmark" in str(d.message) or "rfdetr benchmark" in str(d.message) for d in depr)

    def test_returns_none(self, tmp_path, patch_lit):
        """RFDETR.train() returns None."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
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
    ckpt: dict[str, Any] = {"model": state, "epoch": epoch}

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

    def test_state_dict_keys_prefixed_with_model(self, tmp_path, patch_lit):
        """All state_dict keys must be prefixed with 'model.'."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    def test_state_dict_keys_dot_containing_names_prefixed_once(self, tmp_path, patch_lit):
        """Keys already containing dots are prefixed exactly once."""
        path = str(tmp_path / "dot_keys.pth")
        torch.save({"model": {"backbone.layer.weight": torch.zeros(1)}, "epoch": 0}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "model.backbone.layer.weight" in ckpt["state_dict"]
        assert "model.model.backbone.layer.weight" not in ckpt["state_dict"]

    def test_epoch_preserved(self, tmp_path, patch_lit):
        """Epoch value is copied from the source checkpoint."""
        src = _make_legacy_pth(tmp_path, epoch=42)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 42

    def test_epoch_defaults_to_zero_when_missing(self, tmp_path, patch_lit):
        """Missing epoch key in source defaults to 0."""
        path = str(tmp_path / "no_epoch.pth")
        torch.save({"model": {"w": torch.zeros(1)}}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0

    def test_global_step_always_zero(self, tmp_path, patch_lit):
        """global_step is always written as 0."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["global_step"] == 0

    def test_legacy_checkpoint_format_flag_set(self, tmp_path, patch_lit):
        """legacy_checkpoint_format is always True in output."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["legacy_checkpoint_format"] is True

    def test_args_as_namespace_converted_to_dict(self, tmp_path, patch_lit):
        """argparse.Namespace args are converted to a plain dict via vars()."""
        src = _make_legacy_pth(tmp_path, args_value="namespace")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert isinstance(ckpt["hyper_parameters"], dict)
        assert ckpt["hyper_parameters"]["lr"] == pytest.approx(1e-4)

    def test_args_as_dict_kept_as_dict(self, tmp_path, patch_lit):
        """Plain dict args is preserved as-is."""
        src = _make_legacy_pth(tmp_path, args_value="dict")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {"lr": pytest.approx(1e-4), "epochs": 100}

    def test_args_none_gives_empty_hyper_parameters(self, tmp_path, patch_lit):
        """args=None produces an empty hyper_parameters dict."""
        src = _make_legacy_pth(tmp_path, args_value=None)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}

    def test_args_missing_key_gives_empty_hyper_parameters(self, tmp_path, patch_lit):
        """No 'args' key at all also produces empty hyper_parameters."""
        src = _make_legacy_pth(tmp_path, args_value="missing")
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}

    def test_args_custom_object_with_dict_converted_via_vars(self, tmp_path, patch_lit):
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

    def test_ema_model_preserved_as_legacy_ema_state_dict(self, tmp_path, patch_lit):
        """ema_model present in source is written as legacy_ema_state_dict."""
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" in ckpt
        assert "layer.weight" in ckpt["legacy_ema_state_dict"]

    def test_no_ema_model_no_legacy_ema_state_dict(self, tmp_path, patch_lit):
        """No ema_model in source means legacy_ema_state_dict is absent."""
        src = _make_legacy_pth(tmp_path, include_ema=False)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" not in ckpt

    def test_round_trip_with_on_load_checkpoint(self, tmp_path, patch_lit):
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

    def test_missing_model_key_raises_value_error(self, tmp_path, patch_lit):
        """Source file with no 'model' key raises ValueError with a clear message."""
        path = str(tmp_path / "no_model.pth")
        torch.save({"epoch": 5}, path)
        dst = str(tmp_path / "out.ckpt")

        with pytest.raises(ValueError, match="'model' key"):
            convert_legacy_checkpoint(path, dst)

    def test_args_primitive_type_falls_back_to_empty_dict(self, tmp_path, patch_lit):
        """Args of a non-dict, non-Namespace type (e.g. string) falls back to {} with a warning."""
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

    def test_raw_pth_writes_state_dict_with_prefix(self, patch_lit):
        """'model' key without 'state_dict' → state_dict written with 'model.' prefix."""
        fake = _FakeModule()
        ckpt = {"model": {"backbone.weight": torch.zeros(2)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "state_dict" in ckpt
        assert "model.backbone.weight" in ckpt["state_dict"]

    def test_raw_pth_original_model_key_preserved(self, patch_lit):
        """Original 'model' key is not deleted after state_dict is written."""
        fake = _FakeModule()
        ckpt = {"model": {"w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "model" in ckpt  # PTL may inspect it; must not be deleted

    def test_empty_model_dict_produces_empty_state_dict(self, patch_lit):
        """Empty 'model' dict without 'state_dict' → empty state_dict written."""
        fake = _FakeModule()
        ckpt = {"model": {}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] == {}

    def test_native_ptl_format_no_op(self, patch_lit):
        """Native PTL checkpoint (has 'state_dict', no 'model') → no mutation."""
        fake = _FakeModule()
        sentinel = {"model.layer.weight": torch.zeros(1)}
        ckpt = {"state_dict": sentinel}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] is sentinel  # not replaced
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_both_model_and_state_dict_present_state_dict_not_overwritten(self, patch_lit):
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

    def test_legacy_ema_state_dict_stashed(self, patch_lit):
        """'legacy_ema_state_dict' in checkpoint → stashed on _pending_legacy_ema_state."""
        fake = _FakeModule()
        ema_weights = {"layer.weight": torch.ones(2)}
        ckpt = {
            "state_dict": {"model.layer.weight": torch.zeros(2)},
            "legacy_ema_state_dict": ema_weights,
        }
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert fake._pending_legacy_ema_state is ema_weights

    def test_no_legacy_ema_attribute_not_set(self, patch_lit):
        """No 'legacy_ema_state_dict' → _pending_legacy_ema_state not set on module."""
        fake = _FakeModule()
        ckpt = {"state_dict": {"model.w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_empty_checkpoint_is_noop(self, patch_lit):
        """Completely empty checkpoint {} triggers no mutation and no error."""
        fake = _FakeModule()
        ckpt: dict[str, Any] = {}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)  # must not raise
        assert ckpt == {}
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_second_call_overwrites_pending_ema(self, patch_lit):
        """Calling on_load_checkpoint twice with EMA overwrites the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        second_ema = {"w": torch.ones(1)}
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": second_ema})
        assert fake._pending_legacy_ema_state is second_ema

    def test_second_call_without_ema_leaves_first_stash(self, patch_lit):
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
    def test_symbol_importable_from_rfdetr(self, name, patch_lit):
        """Each PTL export is accessible as rfdetr.<name> via lazy __getattr__."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} is missing"

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModelModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_is_same_object_as_rfdetr_training(self, name, patch_lit):
        """rfdetr.<name> is the identical object to rfdetr.training.<name>."""
        import rfdetr
        import rfdetr.training

        assert getattr(rfdetr, name) is getattr(rfdetr.training, name)

    def test_ptl_names_not_in_all(self, patch_lit):
        """PTL exports are optional (rfdetr[train]) and must not be in rfdetr.__all__."""
        import rfdetr

        for name in ("RFDETRModelModule", "RFDETRDataModule", "build_trainer"):
            assert name not in rfdetr.__all__, f"{name} must not be in __all__ (optional extra)"

    def test_rfdetr_all_no_duplicates(self, patch_lit):
        """rfdetr.__all__ contains no duplicate names."""
        import rfdetr

        assert len(rfdetr.__all__) == len(set(rfdetr.__all__))

    def test_plus_symbol_resolution_does_not_mutate_all(self, monkeypatch, patch_lit):
        """Top-level __all__ remains static when plus-only symbols resolve lazily."""
        import rfdetr
        import rfdetr.platform.models

        sentinel = object()
        monkeypatch.setitem(rfdetr.platform.models.__dict__, "RFDETRXLarge", sentinel)
        monkeypatch.delitem(rfdetr.__dict__, "RFDETRXLarge", raising=False)

        original_all = list(rfdetr.__all__)
        assert rfdetr.RFDETRXLarge is sentinel
        assert rfdetr.__all__ == original_all

    def test_existing_exports_still_present(self, patch_lit):
        """Original RFDETR* class exports are unchanged."""
        import rfdetr

        for name in ["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"]:
            assert hasattr(rfdetr, name), f"rfdetr.{name} unexpectedly missing"

    def test_convert_legacy_checkpoint_not_in_rfdetr_namespace(self, patch_lit):
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

        assert rfdetr.util.__name__ == "rfdetr.util"

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
            rfdetr.missing_removed_shim

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
            rfdetr.missing_dep_shim

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

    def test_cuda_oom_runtime_error_does_not_retry(self, monkeypatch, patch_lit):
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

    def test_state_dict_runtime_error_retries_once_with_deprecated_config(self, monkeypatch, patch_lit):
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
    validation exercised when RFDETRNano(pretrain_weights=...) is called (issue #806).
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_seg_ckpt_into_detection_model_raises_via_detr_path(self, monkeypatch, tmp_path, patch_lit):
        """Segmentation checkpoint must raise ValueError when loaded into a detection model."""
        from rfdetr.models.weights import load_pretrain_weights

        checkpoint = _make_detr_checkpoint(segmentation_head=True, patch_size=14)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", segmentation_head=False)

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(fake_model, mc)

    def test_patch_size_mismatch_raises_via_detr_path(self, monkeypatch, tmp_path, patch_lit):
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

    def test_empty_class_names_returns_empty_list_not_coco(self, patch_lit):
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

    def test_none_class_names_returns_coco(self, patch_lit):
        """class_names property falls back to COCO_CLASS_NAMES when model.class_names is None."""
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        mock_self = MagicMock()
        mock_self.model.class_names = None

        result = RFDETR.class_names.fget(mock_self)

        assert result == COCO_CLASS_NAMES
        assert result is not COCO_CLASS_NAMES, "COCO fallback must return a copy, not the mutable global"

    def test_custom_class_names_returned_as_list(self, patch_lit):
        """Non-empty class_names are returned as a 0-indexed list."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)

        assert result == ["cat", "dog"]

    def test_custom_class_names_returns_shallow_copy(self, patch_lit):
        """Mutating the returned class_names list must not mutate model state."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)
        result.append("bird")

        assert result == ["cat", "dog", "bird"]
        assert mock_self.model.class_names == ["cat", "dog"]


# ---------------------------------------------------------------------------
# 8. RFDETR.deploy_to_roboflow — class_names.txt and args.class_names
# ---------------------------------------------------------------------------


class TestDeployToRoboflow:
    """deploy_to_roboflow writes class_names.txt and embeds class_names in args.

    Regression tests for the bug where RFDETRSeg models (and any model whose
    args namespace lacks a ``class_names`` attribute) failed to upload to
    Roboflow with a FileNotFoundError from the Roboflow client library.
    """

    @pytest.fixture
    def mock_self(self):
        """Return a minimal RFDETR-like mock for deploy_to_roboflow tests."""
        class_names = ["cat", "dog"]
        mock_self = MagicMock(spec=RFDETR)
        mock_self.size = "rfdetr-small"
        mock_self.class_names = class_names  # the property, resolved to a plain list
        # `model` is an instance attribute (set in __init__), not a class attribute, so
        # MagicMock(spec=RFDETR).__getattr__ would raise AttributeError for it.  Assign
        # it directly via __setattr__ so sub-attribute chaining works correctly.
        mock_self.model = MagicMock()
        mock_self.model.model.state_dict.return_value = {}
        mock_self.model.args = SimpleNamespace(num_classes=len(class_names))
        return mock_self

    @staticmethod
    def _set_class_names(mock_self: MagicMock, class_names: list[str]) -> None:
        """Update class names and keep args.num_classes in sync."""
        mock_self.class_names = class_names
        mock_self.model.args.num_classes = len(class_names)

    def test_class_names_txt_written_with_correct_content(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """deploy_to_roboflow must write class_names.txt with one name per line.

        Regression: RFDETRSeg models were failing with FileNotFoundError from
        the Roboflow client library because class_names.txt was absent.
        """
        monkeypatch.chdir(tmp_path)

        class_names = ["cat", "dog", "bird"]
        self._set_class_names(mock_self, class_names)
        mock_rf = MagicMock()

        captured: dict = {}

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            # Inspect class_names.txt while the temp dir still exists (before cleanup).
            f = (Path(model_path) / "class_names.txt").resolve()
            if f.exists():
                captured["content"] = f.read_text()

        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.side_effect = deploy_side_effect

        with patch("roboflow.Roboflow", return_value=mock_rf):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert "content" in captured, "class_names.txt was not present in the upload directory during deploy"
        assert captured["content"] == "cat\ndog\nbird"

    def test_args_class_names_set_in_checkpoint(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """The saved checkpoint args must contain class_names when args lacks it.

        Regression: args.class_names was absent after switching to PTL training,
        causing the Roboflow client library to raise FileNotFoundError.
        """
        monkeypatch.chdir(tmp_path)

        class_names = ["cat", "dog"]
        # Ensure class_names is absent from args (mimics the regression scenario).
        assert not hasattr(mock_self.model.args, "class_names")

        saved_checkpoints: list = []

        def capturing_save(obj, path, *args, **kwargs):
            # Only capture the object; skip actual disk I/O for this unit test.
            saved_checkpoints.append(obj)

        mock_rf = MagicMock()
        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.return_value = None

        with patch("roboflow.Roboflow", return_value=mock_rf), patch("torch.save", side_effect=capturing_save):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert saved_checkpoints, "torch.save must have been called"
        checkpoint = saved_checkpoints[0]
        assert "args" in checkpoint
        saved_args = checkpoint["args"]
        assert hasattr(saved_args, "class_names"), "class_names must be present in saved args"
        assert saved_args.class_names == class_names

    def test_args_class_names_set_when_none_in_checkpoint(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """class_names must be set when args has the attribute but its value is None."""
        monkeypatch.chdir(tmp_path)

        class_names = ["cat", "dog"]
        # Simulate the case where args has class_names but it is explicitly None.
        mock_self.model.args.class_names = None

        saved_checkpoints: list = []

        def capturing_save(obj, path, *args, **kwargs):
            saved_checkpoints.append(obj)

        mock_rf = MagicMock()
        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.return_value = None

        with patch("roboflow.Roboflow", return_value=mock_rf), patch("torch.save", side_effect=capturing_save):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert saved_checkpoints, "torch.save must have been called"
        saved_args = saved_checkpoints[0]["args"]
        assert saved_args.class_names == class_names, "class_names must be populated when args.class_names is None"

    def test_existing_args_class_names_not_overwritten(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """If args already has class_names set, deploy_to_roboflow must not overwrite it."""
        monkeypatch.chdir(tmp_path)

        existing_names = ["existing_cat", "existing_dog"]
        mock_self.model.args.class_names = existing_names

        saved_checkpoints: list = []

        def capturing_save(obj, path, *args, **kwargs):
            saved_checkpoints.append(obj)

        mock_rf = MagicMock()
        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.return_value = None

        with patch("roboflow.Roboflow", return_value=mock_rf), patch("torch.save", side_effect=capturing_save):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert saved_checkpoints
        saved_args = saved_checkpoints[0]["args"]
        assert saved_args.class_names == existing_names, "existing args.class_names must not be overwritten"

    def test_temp_dir_cleaned_up_after_deploy(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """The temporary upload directory must be removed after a successful deploy."""
        monkeypatch.chdir(tmp_path)

        self._set_class_names(mock_self, ["cat"])
        mock_rf = MagicMock()
        deployed_paths: list[Path] = []

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            deployed_paths.append(Path(model_path))

        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.side_effect = deploy_side_effect

        with patch("roboflow.Roboflow", return_value=mock_rf):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert deployed_paths, "deploy must receive a temporary model_path"
        assert not deployed_paths[0].exists(), "Temporary upload dir must be removed after deploy"
        assert not (tmp_path / ".roboflow_temp_upload").exists(), "Fixed-name temp dir must not be created"

    def test_temp_dir_cleaned_up_after_deploy_failure(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """Temp upload dir must be removed even when deploy() raises an exception."""
        monkeypatch.chdir(tmp_path)

        self._set_class_names(mock_self, ["cat"])
        mock_rf = MagicMock()
        deployed_paths: list[Path] = []

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            deployed_paths.append(Path(model_path))
            raise RuntimeError("upload failed")

        mock_rf.workspace.return_value.project.return_value.version.return_value.deploy.side_effect = deploy_side_effect

        with patch("roboflow.Roboflow", return_value=mock_rf), pytest.raises(RuntimeError, match="upload failed"):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
            )

        assert deployed_paths, "deploy must receive a temporary model_path"
        assert not deployed_paths[0].exists(), "Temporary upload dir must be removed even after a failed deploy"
        assert not (tmp_path / ".roboflow_temp_upload").exists(), "Fixed-name temp dir must not be created"


# ---------------------------------------------------------------------------
# TestSaveTrainingConfig
# ---------------------------------------------------------------------------


class TestSaveTrainingConfig:
    """RFDETR.train() writes training_config.json to output_dir after training."""

    def _run_train(self, tmp_path, patch_lit, class_names=None, **train_overrides):
        """Run RFDETR.train() with patched PTL; return (mock_self, output_dir path).

        class_names is injected via the datamodule mock (the path RFDETR.train uses
        to sync self.model.class_names after trainer.fit completes).
        """
        if class_names is None:
            class_names = ["cat", "dog", "bird"]
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        p_mod, p_dm, p_bt, _, dmcls, _ = patch_lit
        dmcls.return_value.class_names = class_names
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)
        config = mock_self.get_train_config.return_value
        return mock_self, config.output_dir

    def test_training_config_json_created(self, tmp_path, patch_lit):
        """training_config.json is written to output_dir after train() completes."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        assert os.path.exists(os.path.join(output_dir, "training_config.json"))

    def test_training_config_json_has_required_keys(self, tmp_path, patch_lit):
        """Saved JSON contains all expected top-level keys."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert set(saved.keys()) == {"train_config", "model_config", "model_config_type", "class_names", "num_classes"}

    def test_training_config_json_class_names_and_num_classes(self, tmp_path, patch_lit):
        """class_names and num_classes in saved JSON match model state after training."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert saved["class_names"] == ["cat", "dog", "bird"]
        assert saved["num_classes"] == 3

    def test_model_config_type_reflects_class_name(self, tmp_path, patch_lit):
        """model_config_type field matches the actual model config class name."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert saved["model_config_type"] == "RFDETRBaseConfig"

    def test_non_serializable_value_coerced_not_raises(self, tmp_path, patch_lit):
        """Non-JSON-serializable values are coerced via default=str, not raising TypeError."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _, dmcls, _ = patch_lit
        dmcls.return_value.class_names = [Path("/some/class"), None]
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)  # must not raise TypeError
        config = mock_self.get_train_config.return_value
        assert os.path.exists(os.path.join(config.output_dir, "training_config.json"))

    def test_output_dir_created_when_missing(self, tmp_path, patch_lit):
        """output_dir is created by makedirs if it does not exist before training."""
        nested_dir = str(tmp_path / "new" / "nested" / "output")
        _, output_dir = self._run_train(tmp_path, patch_lit, output_dir=nested_dir)
        assert os.path.exists(os.path.join(output_dir, "training_config.json"))


# ---------------------------------------------------------------------------
# TestRFDETRTrainNumClassesAutoDetect
# ---------------------------------------------------------------------------


class TestRFDETRTrainNumClassesAutoDetect:
    """RFDETR.train() auto-detects num_classes from the training dataset.

    When the user did not explicitly override ``num_classes`` (or passed the
    class-config default), the model's ``num_classes`` is automatically aligned
    to the dataset's class count before ``RFDETRModelModule`` is constructed.

    When the user *did* explicitly set a non-default ``num_classes`` that differs
    from the dataset, the configured value is preserved and a warning is logged.

    Dataset detection is best-effort: if ``_load_classes`` raises any of the
    expected exceptions (``FileNotFoundError``, ``ValueError``, ``KeyError``,
    ``OSError``), training proceeds unaffected without raising.
    """

    _FOUR_CLASS_NAMES = ["ball", "goalkeeper", "referee", "player"]

    @pytest.fixture
    def mock_self(self, tmp_path):
        """Return a RFDETR-like mock for num_classes auto-detect tests."""
        mock = MagicMock()
        mock.model_config = RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        mock.model = MagicMock()
        mock.get_train_config.return_value = _make_train_config(tmp_path)
        # Bind the real instance method so train()'s self._align_num_classes_from_dataset
        # call exercises actual logic rather than a no-op MagicMock.
        mock._align_num_classes_from_dataset = lambda ds: RFDETR._align_num_classes_from_dataset(mock, ds)
        return mock

    def _write_coco_categories(self, dataset_dir: Path, categories: list[dict[str, Any]]) -> None:
        """Write a minimal COCO annotation file with provided categories."""
        (dataset_dir / "train").mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "train" / "_annotations.coco.json").open("w", encoding="utf-8") as f:
            json.dump({"images": [], "annotations": [], "categories": categories}, f)

    def test_auto_adjusts_num_classes_when_not_overridden(self, mock_self, patch_lit):
        """When user did not set num_classes, auto-adjust to the dataset class count.

        Scenario: model built without explicit num_classes → default=90.
        Dataset has 4 classes.  Expected: model_config.num_classes becomes 4.
        """
        assert "num_classes" not in mock_self.model_config.model_fields_set

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4

    def test_coco_auto_detect_uses_full_category_mapping_not_leaf_only_names(self, mock_self, patch_lit):
        """COCO class-count detection must follow ``coco.cats`` semantics.

        Regression test for hierarchical COCO datasets where leaf-only class
        names can undercount categories relative to label remapping.
        """
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {"id": 1, "name": "animal", "supercategory": "none"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
                {"id": 3, "name": "cat", "supercategory": "animal"},
            ],
        )

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["dog", "cat"])
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3

    def test_auto_adjusts_when_default_explicitly_passed(self, mock_self, patch_lit):
        """Passing num_classes=<default> is treated the same as not setting it.

        Scenario: user passes num_classes=90 (the ModelConfig default) explicitly.
        Dataset has 4 classes.  Expected: model_config.num_classes becomes 4.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=90)
        mock_self.model_config = mc
        # num_classes is in model_fields_set, but equals the class default (90).
        assert "num_classes" in mock_self.model_config.model_fields_set

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4

    def test_preserves_explicit_non_default_num_classes_when_dataset_differs(
        self,
        tmp_path,
        caplog,
        mock_self,
        patch_lit,
    ):
        """When user explicitly set a non-default num_classes, it is preserved.

        Scenario: user passes num_classes=10 (non-default).  Dataset has 4 classes.
        Expected: model_config.num_classes stays at 10.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=10)
        mock_self.model_config = mc
        dataset_dir = mock_self.get_train_config.return_value.dataset_dir

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            previous_propagate = detr_logger.propagate
            detr_logger.propagate = True
            try:
                with caplog.at_level("WARNING", logger="rf-detr"):
                    RFDETR.train(mock_self)
            finally:
                detr_logger.propagate = previous_propagate

        assert mock_self.model_config.num_classes == 10
        assert any(
            record.levelname == "WARNING"
            and f"Dataset '{dataset_dir}' has 4 classes" in record.message
            and "num_classes=10" in record.message
            for record in caplog.records
        )

    def test_auto_adjust_syncs_model_args_num_classes(self, mock_self, patch_lit):
        """When auto-adjusting, keep ModelContext args.num_classes in sync."""
        mock_self.model.args = SimpleNamespace(num_classes=90)

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4
        assert mock_self.model.args.num_classes == 4

    def test_no_adjustment_when_num_classes_already_matches_dataset(self, mock_self, patch_lit):
        """No adjustment when the model's num_classes already equals the dataset count.

        Scenario: user passes num_classes=4 and dataset has 4 classes.
        Expected: model_config.num_classes remains 4 (no log noise, no error).
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=4)
        mock_self.model_config = mc

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(FileNotFoundError("no such dataset"), id="file-not-found"),
            pytest.param(ValueError("bad dataset"), id="value-error"),
            pytest.param(KeyError("missing key"), id="key-error"),
            pytest.param(OSError("io error"), id="os-error"),
        ],
    )
    def test_no_crash_when_dataset_detection_raises(self, exc, mock_self, patch_lit):
        """Training proceeds even if _load_classes raises a known exception.

        Dataset detection is best-effort; errors must not block training.
        """
        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", side_effect=exc)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)  # must not raise

    def test_no_crash_when_dataset_dir_is_none(self, mock_self, patch_lit):
        """Training proceeds when config.dataset_dir resolves to None.

        Guards against AttributeError if getattr returns None.
        """
        # Override dataset_dir to None on the train config mock.
        mock_self.get_train_config.return_value.dataset_dir = None

        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)  # must not raise
