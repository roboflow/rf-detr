# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Chapter 5 / Phase 7+8:

1. ``TestRFDETRTrainPTLShim``  — RFDETR.train_ptl() delegates to PTL build_trainer().fit()
2. ``TestRFDETRTrainLegacy``   — RFDETR.train() delegates to train_from_config() (baseline)
3. ``TestConvertLegacyCheckpoint`` — convert_legacy_checkpoint() round-trip
4. ``TestOnLoadCheckpoint``    — RFDETRModule.on_load_checkpoint() auto-detect
5. ``TestPublicAPIExports``    — rfdetr.__init__ exports RFDETRModule/DataModule/build_trainer
"""

import argparse
import warnings
from collections import defaultdict
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.lit.checkpoint import convert_legacy_checkpoint
from rfdetr.lit.module import RFDETRModule

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
    """Context manager that patches all three rfdetr.lit entry points."""
    mock_module_cls = MagicMock(name="RFDETRModule_cls")
    mock_dm_cls = MagicMock(name="RFDETRDataModule_cls")
    mock_build_trainer = MagicMock(name="build_trainer")

    return (
        patch("rfdetr.lit.RFDETRModule", mock_module_cls),
        patch("rfdetr.lit.RFDETRDataModule", mock_dm_cls),
        patch("rfdetr.lit.build_trainer", mock_build_trainer),
        mock_module_cls,
        mock_dm_cls,
        mock_build_trainer,
    )


# ---------------------------------------------------------------------------
# 1. RFDETR.train_ptl() shim
# ---------------------------------------------------------------------------


class TestRFDETRTrainPTLShim:
    """RFDETR.train_ptl() must delegate to PTL and absorb legacy kwargs correctly."""

    def test_build_trainer_called_with_config_and_model_config(self, tmp_path):
        """build_trainer receives (train_config, model_config) in the right order."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self)

        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator="auto")

    def test_trainer_fit_called_with_module_and_datamodule(self, tmp_path):
        """trainer.fit() is called with (module_instance, datamodule_instance)."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, mcls, dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self)

        trainer = mock_bt.return_value
        fit_args = trainer.fit.call_args
        assert fit_args[0][0] is mcls.return_value  # module instance
        assert fit_args[0][1] is dmcls.return_value  # datamodule instance

    def test_ckpt_path_none_when_resume_not_set(self, tmp_path):
        """trainer.fit receives ckpt_path=None when config.resume is None."""
        mock_self = _make_rfdetr_self(tmp_path)  # resume defaults to None
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path=None)

    def test_ckpt_path_forwarded_when_resume_set(self, tmp_path):
        """trainer.fit receives ckpt_path when config.resume is a path string."""
        mock_self = _make_rfdetr_self(tmp_path, resume="/some/checkpoint.ckpt")
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self)

        trainer = mock_bt.return_value
        trainer.fit.assert_called_once_with(_mcls.return_value, _dmcls.return_value, ckpt_path="/some/checkpoint.ckpt")

    def test_ckpt_path_none_when_resume_is_empty_string(self, tmp_path):
        """config.resume='' is coerced to ckpt_path=None via `resume or None`."""
        mock_self = _make_rfdetr_self(tmp_path)
        # Create a real TrainConfig-like object where resume is ""
        mock_config = MagicMock(spec=TrainConfig)
        mock_config.resume = ""
        mock_self.get_train_config.return_value = mock_config

        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self)

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
            RFDETR.train_ptl(mock_self)

        assert mock_self.model.model is sentinel_nn_module

    def test_returns_none(self, tmp_path):
        """RFDETR.train_ptl() has no return value."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            result = RFDETR.train_ptl(mock_self)
        assert result is None

    def test_device_kwarg_silently_dropped(self, tmp_path):
        """device= is consumed without error or warning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, device="cuda")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        # device must not have been forwarded to get_train_config
        mock_self.get_train_config.assert_called_once_with()

    def test_callbacks_none_no_warning(self, tmp_path):
        """callbacks=None produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, callbacks=None)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_empty_dict_no_warning(self, tmp_path):
        """callbacks={} (falsy dict) produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, callbacks={})
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_all_empty_lists_no_warning(self, tmp_path):
        """callbacks dict with all-empty lists produces no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = defaultdict(list)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, callbacks=callbacks)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_callbacks_non_empty_emits_deprecation_warning(self, tmp_path):
        """callbacks dict with a non-empty list emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        callbacks = {"on_fit_epoch_end": [lambda: None]}
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, callbacks=callbacks)
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
            RFDETR.train_ptl(mock_self, callbacks=callbacks)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_do_benchmark_false_no_warning(self, tmp_path):
        """do_benchmark=False (default) emits no DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, do_benchmark=False)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    @pytest.mark.parametrize("truthy_value", [True, 1, "yes"], ids=["bool_true", "int_1", "str_yes"])
    def test_do_benchmark_truthy_emits_deprecation_warning(self, tmp_path, truthy_value):
        """Any truthy do_benchmark value emits DeprecationWarning."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, do_benchmark=truthy_value)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        assert "rfdetr benchmark" in str(depr[0].message)

    def test_do_benchmark_not_forwarded_to_get_train_config(self, tmp_path):
        """do_benchmark is popped before calling get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            RFDETR.train_ptl(mock_self, do_benchmark=True)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_not_forwarded_to_get_train_config(self, tmp_path):
        """device= is popped and not passed on to get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = _patch_lit()
        with p_mod, p_dm, p_bt:
            RFDETR.train_ptl(mock_self, device="cpu")
        # get_train_config must have been called without device=
        assert "device" not in mock_self.get_train_config.call_args.kwargs


# ---------------------------------------------------------------------------
# 2. RFDETR.train() legacy shim
# ---------------------------------------------------------------------------


class TestRFDETRTrainLegacy:
    """RFDETR.train() must delegate to train_from_config() (legacy baseline)."""

    def _make_self_with_callbacks(self, tmp_path):
        """Return a mock whose .callbacks is a real defaultdict(list)."""
        mock_self = _make_rfdetr_self(tmp_path)
        mock_self.callbacks = defaultdict(list)
        return mock_self

    def test_train_from_config_called_with_config(self, tmp_path):
        """train_from_config is called once with the TrainConfig returned by get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self)
        config = mock_self.get_train_config.return_value
        mock_self.train_from_config.assert_called_once()
        assert mock_self.train_from_config.call_args[0][0] is config

    def test_get_train_config_called_without_callbacks(self, tmp_path):
        """callbacks kwarg is popped before get_train_config is called."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self, callbacks={"on_fit_epoch_end": [lambda: None]})
        assert "callbacks" not in mock_self.get_train_config.call_args.kwargs

    def test_extra_kwargs_forwarded_to_train_from_config(self, tmp_path):
        """Extra kwargs (e.g. device=) are passed through to train_from_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self, device="cuda", epochs=5)
        _, kwargs = mock_self.train_from_config.call_args
        assert kwargs.get("device") == "cuda"
        assert kwargs.get("epochs") == 5

    def test_callbacks_list_merged_into_self_callbacks(self, tmp_path):
        """A callbacks dict with lists is extended into self.callbacks."""
        mock_self = self._make_self_with_callbacks(tmp_path)
        fn = lambda: None  # noqa: E731
        RFDETR.train(mock_self, callbacks={"on_train_end": [fn]})
        assert fn in mock_self.callbacks["on_train_end"]

    def test_callbacks_callable_appended_into_self_callbacks(self, tmp_path):
        """A callbacks dict where a value is a bare callable gets appended."""
        mock_self = self._make_self_with_callbacks(tmp_path)
        fn = lambda: None  # noqa: E731
        RFDETR.train(mock_self, callbacks={"on_train_end": fn})
        assert fn in mock_self.callbacks["on_train_end"]

    def test_callbacks_none_no_crash(self, tmp_path):
        """callbacks=None is accepted silently."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self, callbacks=None)  # must not raise

    def test_callbacks_empty_dict_no_crash(self, tmp_path):
        """callbacks={} is accepted silently."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self, callbacks={})  # must not raise

    def test_returns_none(self, tmp_path):
        """RFDETR.train() returns None (delegates, doesn't return a value itself)."""
        mock_self = _make_rfdetr_self(tmp_path)
        result = RFDETR.train(mock_self)
        assert result is None


# ---------------------------------------------------------------------------
# 2. convert_legacy_checkpoint
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

        RFDETRModule.on_load_checkpoint(fake, ckpt)

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
# 3. RFDETRModule.on_load_checkpoint
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal object supporting attribute assignment for on_load_checkpoint tests."""


class TestOnLoadCheckpoint:
    """RFDETRModule.on_load_checkpoint auto-detects legacy formats."""

    def test_raw_pth_writes_state_dict_with_prefix(self):
        """'model' key without 'state_dict' → state_dict written with 'model.' prefix."""
        fake = _FakeModule()
        ckpt = {"model": {"backbone.weight": torch.zeros(2)}}
        RFDETRModule.on_load_checkpoint(fake, ckpt)
        assert "state_dict" in ckpt
        assert "model.backbone.weight" in ckpt["state_dict"]

    def test_raw_pth_original_model_key_preserved(self):
        """Original 'model' key is not deleted after state_dict is written."""
        fake = _FakeModule()
        ckpt = {"model": {"w": torch.zeros(1)}}
        RFDETRModule.on_load_checkpoint(fake, ckpt)
        assert "model" in ckpt  # PTL may inspect it; must not be deleted

    def test_empty_model_dict_produces_empty_state_dict(self):
        """Empty 'model' dict without 'state_dict' → empty state_dict written."""
        fake = _FakeModule()
        ckpt = {"model": {}}
        RFDETRModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] == {}

    def test_native_ptl_format_no_op(self):
        """Native PTL checkpoint (has 'state_dict', no 'model') → no mutation."""
        fake = _FakeModule()
        sentinel = {"model.layer.weight": torch.zeros(1)}
        ckpt = {"state_dict": sentinel}
        RFDETRModule.on_load_checkpoint(fake, ckpt)
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
        RFDETRModule.on_load_checkpoint(fake, ckpt)
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
        RFDETRModule.on_load_checkpoint(fake, ckpt)
        assert fake._pending_legacy_ema_state is ema_weights

    def test_no_legacy_ema_attribute_not_set(self):
        """No 'legacy_ema_state_dict' → _pending_legacy_ema_state not set on module."""
        fake = _FakeModule()
        ckpt = {"state_dict": {"model.w": torch.zeros(1)}}
        RFDETRModule.on_load_checkpoint(fake, ckpt)
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_empty_checkpoint_is_noop(self):
        """Completely empty checkpoint {} triggers no mutation and no error."""
        fake = _FakeModule()
        ckpt: Dict[str, Any] = {}
        RFDETRModule.on_load_checkpoint(fake, ckpt)  # must not raise
        assert ckpt == {}
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_second_call_overwrites_pending_ema(self):
        """Calling on_load_checkpoint twice with EMA overwrites the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        second_ema = {"w": torch.ones(1)}
        RFDETRModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": second_ema})
        assert fake._pending_legacy_ema_state is second_ema

    def test_second_call_without_ema_leaves_first_stash(self):
        """Second call without 'legacy_ema_state_dict' does not clear the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        RFDETRModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModule.on_load_checkpoint(fake, {"state_dict": {}})
        assert fake._pending_legacy_ema_state is first_ema


# ---------------------------------------------------------------------------
# 4. Public API exports
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    """rfdetr.__init__ exports RFDETRModule, RFDETRDataModule, and build_trainer."""

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_importable_from_rfdetr(self, name):
        """Each PTL export is accessible as rfdetr.<name>."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} is missing"

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_is_same_object_as_rfdetr_lit(self, name):
        """rfdetr.<name> is the identical object to rfdetr.lit.<name>."""
        import rfdetr
        import rfdetr.lit

        assert getattr(rfdetr, name) is getattr(rfdetr.lit, name)

    @pytest.mark.parametrize(
        "name",
        ["RFDETRModule", "RFDETRDataModule", "build_trainer"],
        ids=["RFDETRModule", "RFDETRDataModule", "build_trainer"],
    )
    def test_symbol_in_rfdetr_all(self, name):
        """Each PTL export appears in rfdetr.__all__."""
        import rfdetr

        assert name in rfdetr.__all__

    def test_rfdetr_all_no_duplicates(self):
        """rfdetr.__all__ contains no duplicate names."""
        import rfdetr

        assert len(rfdetr.__all__) == len(set(rfdetr.__all__))

    def test_existing_exports_still_present(self):
        """Original RFDETR* class exports are unchanged."""
        import rfdetr

        for name in ["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"]:
            assert hasattr(rfdetr, name), f"rfdetr.{name} unexpectedly missing"

    def test_convert_legacy_checkpoint_not_in_rfdetr_namespace(self):
        """convert_legacy_checkpoint is in rfdetr.lit but not the top-level rfdetr namespace."""
        import rfdetr

        # It is available from rfdetr.lit
        from rfdetr.lit import convert_legacy_checkpoint  # noqa: F401

        # It is NOT directly on rfdetr (Phase 7.7 spec lists only the three PTL exports)
        assert not hasattr(rfdetr, "convert_legacy_checkpoint")
