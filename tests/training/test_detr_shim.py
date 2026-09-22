# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Chapter 5 / Phase 7+8 (updated Phase 3):

1. ``TestRFDETRTrainPTL``           — RFDETR.train() delegates to PTL build_trainer().fit()
2. ``TestRFDETRTrainPTLAbsorption`` — Legacy kwargs absorbed by RFDETR.train()
2b. ``TestResolutionKwarg``         — resolution= kwarg validation, sync, and PE update
3. ``TestConvertLegacyCheckpoint``  — convert_legacy_checkpoint() round-trip
4. ``TestOnLoadCheckpoint``         — RFDETRModule.on_load_checkpoint() auto-detect
5. ``TestPublicAPIExports``         — rfdetr.__init__ exports RFDETRModule/DataModule/build_trainer
"""

import argparse
import builtins
import importlib
import json
import logging
import os
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, RFDETRKeypointPreviewConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.datasets.webdataset.index import ShardIndex, index_name
from rfdetr.datasets.webdataset.load import WebDatasetDetection
from rfdetr.detr import RFDETR, _save_training_config
from rfdetr.detr import logger as detr_logger
from rfdetr.training.auto_batch import AutoBatchResult
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.training.module_model import RFDETRModelModule

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    """Build a minimal RFDETRBaseConfig for shim tests.

    Examples:
        >>> config = _make_model_config(num_classes=7)
        >>> config.device, config.num_classes, config.pretrain_weights
        ('cpu', 7, None)
    """
    defaults = dict(pretrain_weights=None, num_classes=3, device="cpu")
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _make_train_config(tmp_path, **overrides):
    """Build a minimal TrainConfig for shim tests.

    Examples:
        >>> from pathlib import Path
        >>> config = _make_train_config(Path("/tmp/example"), epochs=3)
        >>> config.epochs, Path(config.dataset_dir).name, Path(config.output_dir).name
        (3, 'ds', 'out')
    """
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

    No spec is used because RFDETR.model is set in __init__ (instance attr) and spec=RFDETR would block access to it.

    Examples:
        >>> from pathlib import Path
        >>> mock_self = _make_rfdetr_self(Path('/tmp/example'))
        >>> mock_self.model_config.device, Path(mock_self.get_train_config().output_dir).name
        ('cpu', 'out')
    """
    mock = MagicMock()
    mock.model_config = _make_model_config()
    mock.model = MagicMock()  # exposes mock.model.model for sync-back assertions
    mock.get_train_config.return_value = _make_train_config(tmp_path, **train_overrides)
    return mock


def _count_config_write_warnings(records: list[logging.LogRecord]) -> int:
    """Count the distinct ``training_config.json`` warnings among *records*.

    pytest installs its capture handler on the non-propagating ``rf-detr`` logger *and* on the root logger, so
    once a test forces ``propagate = True`` every emission is appended to ``caplog.records`` twice. Both entries
    are the same ``LogRecord`` object, so counting distinct objects recovers how many writes actually failed.

    Examples:
        >>> def _record(message):
        ...     record = logging.LogRecord("rf-detr", logging.WARNING, __file__, 0, message, (), None)
        ...     record.message = record.getMessage()
        ...     return record
        >>> warned, unrelated = _record("Could not save training_config.json to /out."), _record("something else")
        >>> _count_config_write_warnings([warned, warned, unrelated])
        1
        >>> _count_config_write_warnings([])
        0
    """
    return len(
        {id(record) for record in records if record.levelname == "WARNING" and "training_config.json" in record.message}
    )


class _UnevaluableValue:
    """A value that raises when its truthiness is tested, as the payload's ``num_classes`` branch does.

    Examples:
        >>> bool(_UnevaluableValue())
        Traceback (most recent call last):
        ...
        RuntimeError: cannot evaluate truthiness
    """

    def __bool__(self) -> bool:
        raise RuntimeError("cannot evaluate truthiness")


class _UnserializableValue:
    """A value ``json.dumps`` cannot write even with ``default=str``, because coercing it raises.

    Examples:
        >>> str(_UnserializableValue())
        Traceback (most recent call last):
        ...
        RuntimeError: cannot stringify
    """

    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


def _read_training_config(path: str) -> dict[str, Any] | None:
    """Return the parsed training_config.json at *path*, or None when it does not exist.

    Examples:
        >>> import json, tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     written = os.path.join(directory, "training_config.json")
        ...     _ = Path(written).write_text(json.dumps({"num_classes": 2}))
        ...     _read_training_config(written)
        {'num_classes': 2}
        >>> _read_training_config("/nonexistent/training_config.json") is None
        True
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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
        [
            pytest.param("rfdetr.training", id="training-package"),
            pytest.param("rfdetr.training.auto_batch", id="training-submodule"),
        ],
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

        Regression test for #509: custom class names were not synced back from RFDETRDataModule after training, causing
        predict() to return COCO labels instead of the dataset's class labels.
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

        Ensures the sync-back guard does not clobber existing class names when the datamodule has no class information
        (e.g. custom dataset format).
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

    def test_model_args_synced_from_train_config_after_training(self, tmp_path, patch_lit):
        """self.model.args reflects effective training and model config after train().

        Regression test for #1199: model.model.__dict__['args'] retained the construction-time defaults (lr=0.0001,
        lr_encoder=0.00015) instead of the overrides passed to train(), because ``self.model.args`` was built once at
        model-construction time from a dummy config and never refreshed after ``train()`` completed.
        """
        mock_self = _make_rfdetr_self(tmp_path, lr=5e-5, lr_encoder=1e-4, batch_size=7)
        mock_self.model_config.num_classes = 7
        mock_self.model.args = SimpleNamespace(lr=0.0001, lr_encoder=0.00015, resolution=560)
        p_mod, p_dm, p_bt, *_ = patch_lit

        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model.args.lr == 5e-5
        assert mock_self.model.args.lr_encoder == 1e-4
        assert mock_self.model.args.batch_size == 7
        assert mock_self.model.args.num_classes == 7
        assert mock_self.model.args.dataset_dir == str(tmp_path / "ds")
        assert mock_self.model.args.output_dir == str(tmp_path / "out")

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param("cuda", id="cuda"),
            pytest.param(torch.device("cuda:1"), id="torch-device-cuda-index"),
        ],
    )
    def test_device_kwarg_consumed_without_deprecation_warning(self, tmp_path, patch_lit, device: str | torch.device):
        """Device= (string or torch.device) is consumed without a DeprecationWarning or reaching get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device=device)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    def test_device_not_forwarded_to_get_train_config(self, tmp_path, patch_lit):
        """Device= is popped and not passed on to get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device="cpu")
        # get_train_config must have been called without device=
        assert "device" not in mock_self.get_train_config.call_args.kwargs

    def test_skip_best_epochs_forwarded_to_get_train_config(self, tmp_path, patch_lit):
        """Non-absorbed training kwargs must reach get_train_config unchanged."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, skip_best_epochs=3)

        mock_self.get_train_config.assert_called_once_with(skip_best_epochs=3)

    def test_optimizer_config_forwarded_to_get_train_config(self, tmp_path, patch_lit):
        """Optimizer training kwargs must reach get_train_config unchanged."""
        mock_self = _make_rfdetr_self(tmp_path)
        optimizer_kwargs = {"weight_decouple": True}
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(
                mock_self,
                optimizer="torch.optim.AdamW",
                optimizer_kwargs=optimizer_kwargs,
            )

        mock_self.get_train_config.assert_called_once_with(
            optimizer="torch.optim.AdamW",
            optimizer_kwargs=optimizer_kwargs,
        )

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

    @pytest.mark.parametrize(
        "device, expected_kwargs",
        [
            pytest.param("cpu", {"accelerator": "cpu"}, id="cpu"),
            pytest.param("cuda", {"accelerator": "gpu"}, id="cuda"),
            pytest.param("cuda:1", {"accelerator": "gpu", "devices": [1]}, id="cuda-index"),
            pytest.param(torch.device("cuda:2"), {"accelerator": "gpu", "devices": [2]}, id="torch-device-cuda-index"),
            pytest.param("xla", {"accelerator": "tpu"}, id="xla"),
            pytest.param(torch.device("xla:0"), {"accelerator": "tpu", "devices": [0]}, id="torch-device-xla-index"),
        ],
    )
    def test_device_absorbed_as_accelerator_and_devices_kwargs(
        self, tmp_path, patch_lit, device: str | torch.device, expected_kwargs: dict[str, object]
    ):
        """Device= (string or torch.device, with or without an index) is absorbed and forwarded to build_trainer as the
        matching accelerator= (and, when indexed, devices=) kwargs -- and no others, since assert_called_once_with is
        exact on kwargs."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, device=device)
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, **expected_kwargs)

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
        """Callbacks={} is accepted without error."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, callbacks={})  # must not raise

    def test_returns_none(self, tmp_path, patch_lit):
        """RFDETR.train() returns None."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            result = RFDETR.train(mock_self)
        assert result is None

    def test_save_dataset_grids_true_calls_grid_saver(self, tmp_path, patch_lit):
        """save_dataset_grids=True triggers DatasetGridSaver.save_grid() for train and val."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        p_mod, p_dm, p_bt, _mcls, _dmcls, _mock_bt = patch_lit
        mock_saver_cls = MagicMock(name="DatasetGridSaver")
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.datasets.save_grids.DatasetGridSaver", mock_saver_cls),
        ):
            RFDETR.train(mock_self)

        # DatasetGridSaver must be constructed twice (train + val) and save_grid called on each
        assert mock_saver_cls.call_count == 2
        assert mock_saver_cls.return_value.save_grid.call_count == 2

        # setup("fit") must be called on the datamodule before training
        dm_instance = _dmcls.return_value
        dm_instance.setup.assert_called_with("fit")

    def test_save_dataset_grids_false_skips_grid_saver(self, tmp_path, patch_lit):
        """save_dataset_grids=False (default) must not call DatasetGridSaver at all."""
        mock_self = _make_rfdetr_self(tmp_path)  # default save_dataset_grids=False
        p_mod, p_dm, p_bt, *_ = patch_lit
        mock_saver_cls = MagicMock(name="DatasetGridSaver")
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.datasets.save_grids.DatasetGridSaver", mock_saver_cls),
        ):
            RFDETR.train(mock_self)

        mock_saver_cls.assert_not_called()

    def test_save_dataset_grids_uses_output_dir_subdir(self, tmp_path, patch_lit):
        """Grid images are saved to <output_dir>/dataset_grids."""
        from pathlib import Path

        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        config = mock_self.get_train_config.return_value
        p_mod, p_dm, p_bt, _mcls, _dmcls, _mock_bt = patch_lit
        mock_saver_cls = MagicMock(name="DatasetGridSaver")
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.datasets.save_grids.DatasetGridSaver", mock_saver_cls),
        ):
            RFDETR.train(mock_self)

        expected_output_dir = Path(config.output_dir) / "dataset_grids"
        called_dirs = [call.args[1] for call in mock_saver_cls.call_args_list]
        assert all(d == expected_output_dir for d in called_dirs)

    def test_save_dataset_grids_skipped_off_rank_zero(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The grid write shares the launcher guard, so a non-zero rank renders nothing."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        p_mod, p_dm, p_bt, _mcls, _dmcls, _mock_bt = patch_lit
        mock_saver_cls = MagicMock(name="DatasetGridSaver")
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.datasets.save_grids.DatasetGridSaver", mock_saver_cls),
            patch("rfdetr.detr._is_launcher_main_process", return_value=False),
        ):
            RFDETR.train(mock_self)

        mock_saver_cls.assert_not_called()

    def test_save_dataset_grids_failure_does_not_abort_training(self, tmp_path, patch_lit):
        """A save_grid() failure must not abort training — trainer.fit() must still be called."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        p_mod, p_dm, p_bt, _mcls, _dmcls, mock_bt = patch_lit
        mock_saver_cls = MagicMock(name="DatasetGridSaver")
        mock_saver_cls.return_value.save_grid.side_effect = OSError("disk full")
        with (
            p_mod,
            p_dm,
            p_bt,
            patch("rfdetr.datasets.save_grids.DatasetGridSaver", mock_saver_cls),
        ):
            # Must not raise even though save_grid() fails
            RFDETR.train(mock_self)

        # Training must proceed regardless of the grid-save failure
        mock_bt.return_value.fit.assert_called_once()


# ---------------------------------------------------------------------------
# 2b. resolution= kwarg handling
# ---------------------------------------------------------------------------


class TestResolutionKwarg:
    """RFDETR.train(resolution=...) applies, validates, and syncs the resolution override."""

    def test_updates_model_config_resolution(self, tmp_path, patch_lit):
        """Resolution kwarg is applied to model_config.resolution before training."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        valid_resolution = block_size * 11  # guaranteed divisible and different from default
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=valid_resolution)
        assert mock_self.model_config.resolution == valid_resolution

    def test_does_not_implicitly_update_positional_encoding_size(self, tmp_path, patch_lit):
        """Pretrained-specific PE (RFDETRBase DINOv2=37) is preserved when resolution is overridden."""
        mock_self = _make_rfdetr_self(tmp_path)
        # RFDETRBaseConfig: PE=37 (DINOv2 native 518//14), resolution=560, patch_size=14.
        # PE != resolution // patch_size, so the smart PE guard leaves PE unchanged.
        original_pe = mock_self.model_config.positional_encoding_size
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        valid_override_resolution = block_size * 11  # different from default 560
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=valid_override_resolution)
        assert mock_self.model_config.positional_encoding_size == original_pe

    def test_updates_positional_encoding_size_for_formula_derived_config(self, tmp_path, patch_lit):
        """For configs where PE == resolution // patch_size, resolution override updates PE."""
        # RFDETRSmallConfig: patch_size=16, num_windows=2, resolution=512, PE=32=512//16.
        mock_self = _make_rfdetr_self(tmp_path)
        mock_self.model_config = RFDETRSmallConfig(pretrain_weights=None, num_classes=3, device="cpu")
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        new_resolution = block_size * 21  # 672 for Small — valid and different from default 512
        expected_pe = new_resolution // mock_self.model_config.patch_size
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model_config.positional_encoding_size == expected_pe

    def test_does_not_reach_get_train_config(self, tmp_path, patch_lit):
        """Resolution kwarg is popped before get_train_config is called."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=block_size * 10)
        assert "resolution" not in mock_self.get_train_config.call_args.kwargs

    def test_indivisible_raises_value_error(self, tmp_path, patch_lit):
        """Resolution not divisible by patch_size * num_windows raises ValueError."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        indivisible = block_size * 10 + 1  # guaranteed not divisible by block_size
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, pytest.raises(ValueError, match=f"resolution={indivisible}"):
            RFDETR.train(mock_self, resolution=indivisible)

    def test_none_leaves_model_config_unchanged(self, tmp_path, patch_lit):
        """Omitting resolution leaves model_config.resolution unchanged."""
        mock_self = _make_rfdetr_self(tmp_path)
        original_resolution = mock_self.model_config.resolution
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)
        assert mock_self.model_config.resolution == original_resolution

    @pytest.mark.parametrize(
        "bad_resolution",
        [
            pytest.param(0, id="zero"),
            pytest.param(-56, id="negative"),
            pytest.param(True, id="bool_true"),
            pytest.param(False, id="bool_false"),
            pytest.param(1.5, id="non_integer_float"),
            pytest.param(560.0, id="whole_number_float"),
            pytest.param("560", id="string"),
        ],
    )
    def test_invalid_type_or_value_raises_value_error(self, tmp_path, patch_lit, bad_resolution):
        """Non-positive, bool, or non-integer resolution raises ValueError before divisibility check."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, pytest.raises(ValueError, match="resolution must be a positive integer"):
            RFDETR.train(mock_self, resolution=bad_resolution)

    def test_syncs_model_resolution_attribute(self, tmp_path, patch_lit):
        """Resolution kwarg sets model.resolution so predict()/export() see the new resolution.

        Regression test for #952 — keeps the cached inference/export context in sync after a resolution override in
        train().
        """
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        new_resolution = block_size * 11
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model.resolution == new_resolution

    def test_syncs_model_args_resolution_and_pe(self, tmp_path, patch_lit):
        """Resolution kwarg updates model.args.resolution and model.args.positional_encoding_size.

        For formula-derived configs (PE == resolution // patch_size), both fields in model.args must be kept consistent
        with model_config so export/deployment pipelines use the correct values.  Regression test for #952.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        # RFDETRSmallConfig: formula-derived PE (512 // 16 == 32), so PE updates with resolution.
        mock_self.model_config = RFDETRSmallConfig(pretrain_weights=None, num_classes=3, device="cpu")
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        new_resolution = block_size * 21  # 672 for Small — valid, different from default 512
        expected_pe = new_resolution // mock_self.model_config.patch_size
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model.args.resolution == new_resolution
        assert mock_self.model.args.positional_encoding_size == expected_pe


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

    @pytest.mark.parametrize(
        "args_value, expected_hyper_parameters",
        [
            pytest.param("namespace", {"lr": pytest.approx(1e-4), "epochs": 100}, id="namespace-converted-via-vars"),
            pytest.param("dict", {"lr": pytest.approx(1e-4), "epochs": 100}, id="dict-kept-as-dict"),
            pytest.param(None, {}, id="none-gives-empty"),
            pytest.param("missing", {}, id="missing-key-gives-empty"),
        ],
    )
    def test_args_converted_to_hyper_parameters(
        self, tmp_path, patch_lit, args_value: str | None, expected_hyper_parameters: dict[str, object]
    ):
        """convert_legacy_checkpoint() normalizes source 'args' into hyper_parameters: an argparse.Namespace is
        converted via vars(), a dict is kept as-is, and a None or altogether-missing 'args' key produces an empty
        dict."""
        src = _make_legacy_pth(tmp_path, args_value=args_value)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == expected_hyper_parameters

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

        After conversion, loading the .ckpt via on_load_checkpoint must NOT re-apply the 'model.' prefix because
        'state_dict' already exists.
        """
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)

        class _FakeModule:
            model_config = SimpleNamespace(positional_encoding_size=36)

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

    model_config = SimpleNamespace(positional_encoding_size=36)


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

    def test_pre_merge_optimizer_state_is_regrouped(self, patch_lit: Any) -> None:
        """on_load_checkpoint itself regroups one-group-per-parameter optimizer state.

        tests/training/test_param_groups.py exercises regroup_unmerged_optimizer_state() directly; this proves the hook
        actually calls it, since a wiring mistake there (wrong argument, wrong order relative to the other normalisation
        steps, a silently swallowed exception) would not be caught by calling the helper on its own.
        """
        fake = _FakeModule()
        ckpt = {
            "state_dict": {"model.w": torch.zeros(1)},
            "optimizer_states": [
                {
                    "state": {0: {"exp_avg": torch.ones(1)}, 1: {"exp_avg": torch.full((1,), 2.0)}},
                    "param_groups": [
                        {"lr": 0.1, "weight_decay": 0.0, "params": [0]},
                        {"lr": 0.1, "weight_decay": 0.0, "params": [1]},
                    ],
                }
            ],
        }

        RFDETRModelModule.on_load_checkpoint(fake, ckpt)

        groups = ckpt["optimizer_states"][0]["param_groups"]
        assert len(groups) == 1
        assert groups[0]["params"] == [0, 1]


# ---------------------------------------------------------------------------
# 5. Public API exports
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    """rfdetr.__init__ exposes PTL names via __getattr__ (rfdetr[train] extra)."""

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("RFDETRModelModule", id="RFDETRModelModule"),
            pytest.param("RFDETRDataModule", id="RFDETRDataModule"),
            pytest.param("build_trainer", id="build_trainer"),
        ],
    )
    def test_symbol_importable_from_rfdetr(self, name, patch_lit):
        """Each PTL export is accessible as rfdetr.<name> via lazy __getattr__."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} is missing"

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("RFDETRModelModule", id="RFDETRModelModule"),
            pytest.param("RFDETRDataModule", id="RFDETRDataModule"),
            pytest.param("build_trainer", id="build_trainer"),
        ],
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
    """Removed legacy modules raise migration-hint ImportErrors on any access."""

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

    def test_removed_shim_missing_raises_importerror_with_getattr(self) -> None:
        """Missing removed shim should raise ImportError with migration hint."""
        import rfdetr

        missing_name = "rfdetr.missing_removed_shim"
        missing_exc = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
        with (
            patch.dict(rfdetr._REMOVE_IN_VERSION_1_9, {"missing_removed_shim": "migration hint"}),
            patch("rfdetr.importlib.import_module", side_effect=missing_exc),
            pytest.raises(ImportError, match="migration hint"),
        ):
            rfdetr.missing_removed_shim

    def test_nested_module_not_found_is_not_masked_for_package_attribute(self) -> None:
        """Nested ModuleNotFoundError from inside a shim import should propagate."""
        import rfdetr

        with (
            patch.dict(rfdetr._REMOVE_IN_VERSION_1_9, {"missing_dep_shim": "migration hint"}),
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

        with pytest.raises(ImportError, match=r"rfdetr\.util was removed in v1\.9"):
            importlib.import_module("rfdetr.util")

    def test_removed_deploy_submodule_import_raises_migration_hint_when_shim_is_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dotted legacy submodule imports get a migration hint once the deploy shim is removed."""
        self._simulate_missing_removed_module_specs(monkeypatch, "rfdetr.deploy", "rfdetr.deploy.benchmark")

        with pytest.raises(ImportError, match=r"rfdetr\.deploy was removed in v1\.9"):
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
# 6. _load_pretrain_weights_into — detr.py path (the non-PTL scenario from #806)
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
    """Tests for load_pretrain_weights (models/weights.py) — checkpoint compatibility validation exercised when
    RFDETRNano(pretrain_weights=...) is called (issue #806)."""

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

        Regression test for #509: the truthiness check `and self.model.class_names:` treated [] as falsy and fell
        through to return COCO_CLASSES, defeating the detr.py sync-back even after training on a dataset that reports
        empty names. The fix uses `is not None` so that [] is preserved.
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

    Regression tests for the bug where RFDETRSeg models (and any model whose args namespace lacks a ``class_names``
    attribute) failed to upload to Roboflow with a FileNotFoundError from the Roboflow client library.
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
        # deploy_to_roboflow now delegates bundle-writing to export_for_roboflow; bind the
        # real method so these end-to-end tests exercise it (a bare MagicMock attribute
        # would no-op the class_names.txt / torch.save side effects).
        mock_self.export_for_roboflow = lambda output_dir: RFDETR.export_for_roboflow(mock_self, output_dir)
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

    @staticmethod
    def _deploy(mock_self, size=None):
        """Call deploy_to_roboflow with a mocked Roboflow client; return the captured deploy mock."""
        mock_rf = MagicMock()
        deploy_mock = mock_rf.workspace.return_value.project.return_value.version.return_value.deploy
        kwargs = {} if size is None else {"size": size}
        with patch("roboflow.Roboflow", return_value=mock_rf):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                version=1,
                api_key="dummy-key",
                **kwargs,
            )
        return deploy_mock

    def test_explicit_size_overrides_model_size(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """An explicitly passed size must win over self.size (documented precedence).

        Regression: ``size = self.size or size`` inverted the precedence, silently ignoring the user's argument.
        """
        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            deploy_mock = self._deploy(mock_self, size="rfdetr-medium")

        assert deploy_mock.call_args.kwargs["model_type"] == "rfdetr-medium"

    def test_size_defaults_to_model_size_when_not_provided(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """Without an explicit size the model's own size is deployed."""
        monkeypatch.chdir(tmp_path)
        deploy_mock = self._deploy(mock_self)

        assert deploy_mock.call_args.kwargs["model_type"] == "rfdetr-small"

    def test_warns_when_explicit_size_differs_from_model_size(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """A UserWarning is emitted when the explicit size conflicts with the model's own size."""
        monkeypatch.chdir(tmp_path)
        with pytest.warns(UserWarning, match="rfdetr-medium.*rfdetr-small"):
            self._deploy(mock_self, size="rfdetr-medium")

    def test_no_warning_when_explicit_size_matches_model_size(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """No size-conflict warning is emitted when the explicit size equals the model's own size."""
        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._deploy(mock_self, size="rfdetr-small")

        conflict_warnings = [w for w in caught if "deploy_to_roboflow" in str(w.message)]
        assert not conflict_warnings

    @staticmethod
    def _deploy_with_versions(
        mock_self: MagicMock,
        version_info: list[dict[str, Any]],
        **deploy_kwargs: Any,
    ) -> MagicMock:
        """Call deploy_to_roboflow against a mocked project preloaded with version_info; return the project mock.

        Examples:
            >>> TestDeployToRoboflow._deploy_with_versions(model, [{"id": "ws/proj/1"}])  # doctest: +SKIP
            (needs the mocked RFDETR instance built by the ``mock_self`` fixture)
        """
        mock_rf = MagicMock()
        project_mock = mock_rf.workspace.return_value.project.return_value
        project_mock.get_version_information.return_value = version_info
        with patch("roboflow.Roboflow", return_value=mock_rf):
            RFDETR.deploy_to_roboflow(
                mock_self,
                workspace="test-workspace",
                project_id="test-project",
                api_key="dummy-key",
                **deploy_kwargs,
            )
        return project_mock

    def test_omitted_version_resolves_to_latest(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """Omitting version deploys to the highest existing dataset version.

        Project has versions 1 and 3 (server list order not guaranteed); auto-resolution must pick 3 so the model lands
        on the newest dataset snapshot without the caller tracking version numbers.
        """
        monkeypatch.chdir(tmp_path)

        project_mock = self._deploy_with_versions(mock_self, [{"id": "ws/proj/9"}, {"id": "ws/proj/10"}])

        project_mock.version.assert_called_once_with(10)

    def test_omitted_version_falls_back_to_one_for_empty_project(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """Omitting version on a project with no versions falls back to version 1.

        The SDK's Version lookup then raises its own "Version number 1 is not found." for a genuinely empty project, so
        the fallback never silently deploys anywhere unexpected.
        """
        monkeypatch.chdir(tmp_path)

        project_mock = self._deploy_with_versions(mock_self, [])

        project_mock.version.assert_called_once_with(1)

    def test_explicit_version_skips_lookup(self, tmp_path, monkeypatch, mock_self, patch_lit):
        """An explicitly passed version is used verbatim without any version-list lookup.

        Guards the no-surprise contract: existing callers pinning a version must not trigger the extra
        get_version_information network round-trip nor have their choice overridden by a newer version.
        """
        monkeypatch.chdir(tmp_path)

        project_mock = self._deploy_with_versions(mock_self, [{"id": "ws/proj/9"}], version=2)

        project_mock.get_version_information.assert_not_called()
        project_mock.version.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# TestSaveTrainingConfig
# ---------------------------------------------------------------------------


class TestSaveTrainingConfig:
    """RFDETR.train() writes training_config.json to output_dir when training starts and again when it ends."""

    def _run_train(self, tmp_path, patch_lit, class_names=None, **train_overrides):
        """Run RFDETR.train() with patched PTL; return (mock_self, output_dir path).

        class_names is injected via the datamodule mock (the path RFDETR.train uses to sync self.model.class_names after
        trainer.fit completes).
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

    def _run_train_capturing_pre_fit(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        dataset_class_names: list[str] | None = None,
        fit_exception: BaseException | None = None,
        post_fit_exception: BaseException | None = None,
        load_classes_patch: Any = None,
        **train_overrides: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Run RFDETR.train() and read training_config.json both at fit() time and after train() returns.

        trainer.fit is a MagicMock, so its side_effect is the only code that runs at the exact moment the start-of-run
        write must already have happened. Returns (pre_fit_payload, final_payload), each None when the file did not
        exist at that point. load_classes_patch defaults to an unreadable dataset so the start-of-run class-name lookup
        is deterministic instead of depending on what tmp_path happens to contain. post_fit_exception, when set, makes
        remove_optimized_model() raise after a successful fit() but before the post-fit write, simulating a
        housekeeping step (distinct from fit() itself) failing on the way to the second write.

        Examples:
            Needs the ``tmp_path`` and ``patch_lit`` fixtures, so it cannot run standalone:

            >>> self._run_train_capturing_pre_fit(tmp_path, patch_lit)  # doctest: +SKIP
        """
        if dataset_class_names is None:
            dataset_class_names = ["cat", "dog", "bird"]
        if load_classes_patch is None:
            load_classes_patch = patch.object(RFDETR, "_load_classes", side_effect=FileNotFoundError("no dataset"))
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        if post_fit_exception is not None:
            mock_self.remove_optimized_model.side_effect = post_fit_exception
        p_mod, p_dm, p_bt, _, dmcls, mock_bt = patch_lit
        dmcls.return_value.class_names = dataset_class_names
        config_path = os.path.join(mock_self.get_train_config.return_value.output_dir, "training_config.json")
        captured = {}

        def _capture(*args: Any, **kwargs: Any) -> None:
            captured["pre_fit"] = _read_training_config(config_path)
            if fit_exception is not None:
                raise fit_exception

        mock_bt.return_value.fit.side_effect = _capture
        expected_exception = fit_exception or post_fit_exception
        with p_mod, p_dm, p_bt, load_classes_patch:
            if expected_exception is None:
                RFDETR.train(mock_self)
            else:
                with pytest.raises(type(expected_exception)):
                    RFDETR.train(mock_self)
        return captured.get("pre_fit"), _read_training_config(config_path)

    def test_training_config_json_written_before_fit(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The file already exists by the time trainer.fit() is entered."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit is not None

    def test_pre_fit_payload_carries_the_full_schema(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The start-of-run copy carries the same five keys as the final one, not a reduced subset."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert set(pre_fit.keys()) == {
            "train_config",
            "model_config",
            "model_config_type",
            "class_names",
            "num_classes",
        }

    def test_post_fit_write_replaces_pre_fit_class_names(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The second write wins: the dataset's class names overwrite the start-of-run ones."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["placeholder"])
        _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert final["class_names"] == ["cat", "dog", "bird"]

    @pytest.mark.parametrize(
        "fit_exception",
        [
            pytest.param(RuntimeError("boom"), id="crash"),
            pytest.param(KeyboardInterrupt(), id="interrupt"),
        ],
    )
    def test_training_config_json_survives_interrupted_fit(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], fit_exception: BaseException
    ) -> None:
        """A run killed inside fit() still leaves a record of how it was configured (#1493)."""
        _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, fit_exception=fit_exception)
        assert final is not None

    def test_post_fit_step_failure_keeps_pre_fit_training_config_intact(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """A post-fit step that raises before the final write must not lose the start-of-run copy.

        Unlike test_training_config_json_survives_interrupted_fit, trainer.fit() itself succeeds here --
        remove_optimized_model(), one of the housekeeping steps between fit() returning and the final
        _save_training_config() call, raises instead. The code never reaches the post-fit write, so the on-disk
        file must still hold exactly the pre-fit payload, untouched.
        """
        post_fit_exception = RuntimeError("optimized model teardown failed")
        pre_fit, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, post_fit_exception=post_fit_exception)
        assert final == pre_fit

    def test_pre_fit_class_names_taken_from_config_when_set(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """An explicit TrainConfig.class_names is recorded in preference to the dataset's."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["from-dataset"])
        pre_fit, _ = self._run_train_capturing_pre_fit(
            tmp_path,
            patch_lit,
            load_classes_patch=load_classes_patch,
            class_names=["from-config"],
        )
        assert pre_fit["class_names"] == ["from-config"]

    def test_pre_fit_class_names_read_from_dataset_when_config_has_none(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """Without an explicit setting the label space is read straight off the dataset directory."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["cat", "dog"])
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert pre_fit["class_names"] == ["cat", "dog"]

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(FileNotFoundError("no such dataset"), id="file-not-found"),
            pytest.param(ValueError("bad dataset"), id="value-error"),
            pytest.param(KeyError("missing key"), id="key-error"),
            pytest.param(OSError("io error"), id="os-error"),
        ],
    )
    def test_pre_fit_class_names_null_when_dataset_unreadable(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], exc: Exception
    ) -> None:
        """An unreadable or unsupported dataset layout records null rather than blocking training."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", side_effect=exc)
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert pre_fit["class_names"] is None

    def test_pre_fit_num_classes_zero_when_dataset_unreadable(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """num_classes stays the count of resolved names, so it is 0 when there are none."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit["num_classes"] == 0

    def test_pre_fit_payload_keeps_empty_class_names_list_not_null(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """An explicit empty class_names list is recorded as [], not coerced to null like an unset one.

        TrainConfig(class_names=[]) is not None, so the pre-fit write must record it verbatim rather than falling
        through to the dataset-lookup branch that only fires when class_names is unset -- locking in the [] vs None
        distinction the payload relies on to tell "explicitly no classes" apart from "not yet resolved".
        """
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, class_names=[])
        assert pre_fit["class_names"] == []
        assert pre_fit["num_classes"] == 0

    def test_pre_fit_write_skipped_off_rank_zero(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """Distributed workers must not race on the file before fit() initializes torch.distributed."""
        with patch("rfdetr.detr._is_launcher_main_process", return_value=False):
            pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit is None

    def test_post_fit_write_keeps_its_own_rank_guard(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The post-fit write guards on is_main_process() instead, so the launcher guard does not suppress it."""
        with patch("rfdetr.detr._is_launcher_main_process", return_value=False):
            _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert final is not None

    def test_write_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An output_dir that cannot be created warns from both writes and still lets training run."""
        blocked = tmp_path / "blocked"
        blocked.write_text("a regular file where output_dir should be")
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, output_dir=str(blocked))
        assert _count_config_write_warnings(caplog.records) == 2

    def test_pre_fit_serialization_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A value json refuses to serialize is a warning, not an exception out of train()."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=[_UnserializableValue()])
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert _count_config_write_warnings(caplog.records) == 1

    def test_payload_assembly_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Assembling the payload is guarded too, not just writing it: both steps run inside the same try.

        The value is synthetic -- `_load_classes` is annotated `-> list[str]`, so nothing real reaches the assembly step
        this way -- but it is the only handle a test has on that step in isolation.
        """
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=_UnevaluableValue())
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert _count_config_write_warnings(caplog.records) == 1

    def test_post_fit_serialization_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The post-fit write tolerates it too: before this it only caught OSError, and a completed run died here."""
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_class_names=[_UnserializableValue()])
        assert _count_config_write_warnings(caplog.records) == 1

    def test_torn_write_keeps_prior_training_config_intact(self, tmp_path: Path) -> None:
        """A write that fails partway through must not corrupt the previously saved good copy.

        Pre-seeds output_dir with a valid training_config.json from an earlier run, then makes the new write raise
        OSError right after committing half its bytes to disk -- the shape of a disk-full or killed-process failure mid-
        write. ``_save_training_config``'s ``except Exception`` path must swallow the error without letting it escape,
        and the atomic-write contract (tempfile + os.replace) requires the on-disk file to still parse as the untouched
        prior payload afterward, with the abandoned temp file cleaned up.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config_path = output_dir / "training_config.json"
        prior_payload = {"marker": "prior-good-copy", "run": 1}
        config_path.write_text(json.dumps(prior_payload))
        real_named_temporary_file = tempfile.NamedTemporaryFile

        def _torn_temporary_file(*args: Any, **kwargs: Any) -> Any:
            handle = real_named_temporary_file(*args, **kwargs)
            if str(kwargs.get("dir")) != str(output_dir):
                return handle
            real_write = handle.write

            def _torn_write(data: str) -> int:
                real_write(data[: len(data) // 2])
                handle.flush()
                raise OSError("disk full mid-write")

            handle.write = _torn_write
            return handle

        with patch("tempfile.NamedTemporaryFile", side_effect=_torn_temporary_file):
            _save_training_config(
                _make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), ["cat"]
            )

        assert _read_training_config(str(config_path)) == prior_payload
        assert [p.name for p in output_dir.iterdir()] == ["training_config.json"]

    def test_interleaved_writers_leave_valid_json_not_a_byte_level_merge(self, tmp_path: Path) -> None:
        """Two writers both cleared past the launcher-rank guard must never leave a byte-interleaved file.

        Deterministically reproduces the race the guard exists to prevent, without relying on real OS-thread scheduling
        (rejected in an earlier review round as non-deterministic): writer "a" starts writing, pauses once its first
        half is flushed to disk, writer "b" then opens and fully writes a different, longer payload to the same path,
        and only then does "a" resume and flush its second half. Ordering is pinned with ``threading.Event``s with
        bounded waits, never a bare block. The atomic tempfile+os.replace write makes each writer's file appear whole or
        not at all, so the final file is exactly one writer's payload.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config_path = output_dir / "training_config.json"
        payload_a = ["a"]
        payload_b = ["bb"] * 100  # much longer serialized payload than payload_a
        a_paused = threading.Event()
        b_done = threading.Event()
        real_named_temporary_file = tempfile.NamedTemporaryFile
        errors: list[BaseException] = []

        def _paced_temporary_file(*args: Any, **kwargs: Any) -> Any:
            handle = real_named_temporary_file(*args, **kwargs)
            if str(kwargs.get("dir")) != str(output_dir) or threading.current_thread().name != "writer-a":
                return handle
            real_write = handle.write

            def _paced_write(data: str) -> int:
                half = len(data) // 2
                written = real_write(data[:half])
                handle.flush()
                a_paused.set()
                assert b_done.wait(timeout=5), "writer b did not finish in time"
                written += real_write(data[half:])
                handle.flush()
                return written

            handle.write = _paced_write
            return handle

        def _writer_a() -> None:
            try:
                with patch("tempfile.NamedTemporaryFile", side_effect=_paced_temporary_file):
                    _save_training_config(
                        _make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), payload_a
                    )
            except BaseException as exc:
                errors.append(exc)

        def _writer_b() -> None:
            try:
                assert a_paused.wait(timeout=5), "writer a did not pause in time"
                with patch("tempfile.NamedTemporaryFile", side_effect=_paced_temporary_file):
                    _save_training_config(
                        _make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), payload_b
                    )
            except BaseException as exc:
                errors.append(exc)
            finally:
                b_done.set()

        thread_a = threading.Thread(target=_writer_a, name="writer-a")
        thread_b = threading.Thread(target=_writer_b, name="writer-b")
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)

        assert not thread_a.is_alive(), "writer a did not finish in time"
        assert not thread_b.is_alive(), "writer b did not finish in time"
        assert not errors
        on_disk = _read_training_config(str(config_path))
        assert on_disk["class_names"] in (payload_a, payload_b)

    def test_training_config_json_written_before_the_model_is_built(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """The write precedes module construction, so a run that dies loading weights still leaves a record."""
        mock_self = _make_rfdetr_self(tmp_path)
        p_mod, p_dm, p_bt, modcls, _, _ = patch_lit
        modcls.side_effect = RuntimeError("checkpoint is corrupt")
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["cat"])
        with p_mod, p_dm, p_bt, load_classes_patch, pytest.raises(RuntimeError):
            RFDETR.train(mock_self)
        output_dir = mock_self.get_train_config.return_value.output_dir
        assert _read_training_config(os.path.join(output_dir, "training_config.json")) is not None


# ---------------------------------------------------------------------------
# TestRFDETRTrainNumClassesAutoDetect
# ---------------------------------------------------------------------------


class TestRFDETRTrainNumClassesAutoDetect:
    """RFDETR.train() auto-detects num_classes from the training dataset.

    When the user did not explicitly override ``num_classes`` (or passed the class-config default), the model's
    ``num_classes`` is automatically aligned to the dataset's class count before ``RFDETRModelModule`` is constructed.

    When the user *did* explicitly set a non-default ``num_classes`` that differs from the dataset, the configured value
    is preserved and a warning is logged.

    Dataset detection is best-effort: if ``_load_classes`` raises any of the expected exceptions (``FileNotFoundError``,
    ``ValueError``, ``KeyError``, ``OSError``), training proceeds unaffected without raising.
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

    def _write_coco_categories(
        self,
        dataset_dir: Path,
        categories: list[dict[str, Any]],
        annotated_ids: list[int] | None = None,
    ) -> None:
        """Write a minimal COCO annotation file with provided categories and one annotation per annotated id."""
        (dataset_dir / "train").mkdir(parents=True, exist_ok=True)
        annotations = [
            {"id": index, "image_id": 1, "category_id": category_id, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}
            for index, category_id in enumerate(annotated_ids or [])
        ]
        # Annotations reference image_id 1, so a matching "images" entry keeps the fixture COCO-consistent whenever
        # any annotation is written.
        images = [{"id": 1, "file_name": "0.jpg", "width": 10, "height": 10}] if annotated_ids else []
        with (dataset_dir / "train" / "_annotations.coco.json").open("w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    def _write_roboflow_keypoint_categories(self, dataset_dir: Path, keypoint_count: int) -> None:
        """Write a minimal Roboflow COCO keypoint annotation file."""
        keypoint_names = [f"kp_{idx}" for idx in range(keypoint_count)]
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {
                    "id": 0,
                    "name": "person",
                    "supercategory": "none",
                    "keypoints": keypoint_names,
                    "skeleton": [],
                }
            ],
        )

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

    def test_coco_auto_detect_skips_unannotated_grouping_category(self, mock_self, patch_lit):
        """COCO class-count detection must follow the same category basis as ``cat2label``.

        Roboflow COCO exports prepend a grouping category that owns no annotations; it consumes no label index, so it
        must not inflate the detected class count either.
        """
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {"id": 1, "name": "animal", "supercategory": "none"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
                {"id": 3, "name": "cat", "supercategory": "animal"},
            ],
            annotated_ids=[2, 3],
        )

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["dog", "cat"])
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 2

    def test_coco_auto_detect_counts_annotated_parent_category(self, mock_self, patch_lit):
        """A parent category that owns annotations keeps its label index, so it is counted."""
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {"id": 1, "name": "animal", "supercategory": "none"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
                {"id": 3, "name": "cat", "supercategory": "animal"},
            ],
            annotated_ids=[1, 2, 3],
        )

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["animal", "dog", "cat"])
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3

    def test_keypoint_coco_auto_detect_uses_active_first_schema_slots(self, mock_self, patch_lit):
        """Keypoint COCO class-count detection should count active-first RF-DETR schema slots."""
        mock_self.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {
                    "id": 0,
                    "name": "person",
                    "keypoints": ["nose", "left_eye"],
                    "skeleton": [],
                },
            ],
        )

        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 1
        assert mock_self.model.args.num_classes == 1

    def test_preserves_explicit_default_num_classes_when_dataset_differs(
        self,
        caplog,
        mock_self,
        monkeypatch,
        patch_lit,
    ):
        """An explicitly-passed num_classes is preserved even when it equals the default.

        Scenario: user passes num_classes=90 (the ModelConfig default) explicitly.  Dataset has
        4 classes.  Expected: model_config.num_classes stays at 90 and a warning is logged —
        identical to the non-default case below, so an explicit setting always wins regardless of
        whether it happens to equal the class default.
        """
        default_nc = RFDETRBaseConfig.model_fields["num_classes"].default
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=default_nc)
        mock_self.model_config = mc
        # num_classes equals the class default but was set explicitly.
        assert "num_classes" in mock_self.model_config.model_fields_set
        dataset_dir = mock_self.get_train_config.return_value.dataset_dir

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        monkeypatch.setattr(detr_logger, "propagate", True)
        with p_mod, p_dm, p_bt, load_classes_patch:
            with caplog.at_level("WARNING", logger="rf-detr"):
                RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == default_nc
        expected_fragment = (
            f"Dataset '{dataset_dir}' has 4 classes but model was initialized with num_classes={default_nc}"
        )
        assert any(record.levelname == "WARNING" and expected_fragment in record.message for record in caplog.records)

    def test_preserves_explicit_non_default_num_classes_when_dataset_differs(
        self,
        tmp_path,
        caplog,
        mock_self,
        monkeypatch,
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
        monkeypatch.setattr(detr_logger, "propagate", True)
        with p_mod, p_dm, p_bt, load_classes_patch:
            with caplog.at_level("WARNING", logger="rf-detr"):
                RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 10
        expected_fragment = f"Dataset '{dataset_dir}' has 4 classes but model was initialized with num_classes=10"
        assert any(record.levelname == "WARNING" and expected_fragment in record.message for record in caplog.records)

    def test_auto_adjust_syncs_model_args_num_classes(self, mock_self, patch_lit):
        """When auto-adjusting, keep ModelContext args.num_classes in sync."""
        mock_self.model.args = SimpleNamespace(num_classes=90)

        p_mod, p_dm, p_bt, *_ = patch_lit
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4
        assert mock_self.model.args.num_classes == 4

    def test_keypoint_schema_inferred_when_not_explicitly_overridden(self, mock_self, patch_lit):
        """Roboflow keypoint metadata should populate model_config.num_keypoints_per_class."""
        mock_self.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        mock_self.model.args = SimpleNamespace(num_classes=90, num_keypoints_per_class=[17])
        mock_self._align_keypoint_schema_from_dataset = lambda config: RFDETR._align_keypoint_schema_from_dataset(
            mock_self, config
        )
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_roboflow_keypoint_categories(dataset_dir, keypoint_count=25)

        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_keypoints_per_class == [25]
        assert mock_self.model.args.num_keypoints_per_class == [25]
        assert mock_self.model_config.num_classes == 1

    def test_keypoint_flip_pairs_inferred_from_roboflow_coco_metadata(self, mock_self, patch_lit):
        """Roboflow COCO keypoint names should populate train_config.keypoint_flip_pairs."""
        mock_self.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        mock_self.model.args = SimpleNamespace(num_classes=90, num_keypoints_per_class=[17])
        mock_self._align_keypoint_schema_from_dataset = lambda config: RFDETR._align_keypoint_schema_from_dataset(
            mock_self, config
        )
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_coco_categories(
            dataset_dir,
            categories=[
                {
                    "id": 0,
                    "name": "person",
                    "supercategory": "none",
                    "keypoints": ["nose", "left_eye", "right_eye"],
                    "skeleton": [],
                }
            ],
        )

        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)

        assert mock_self.get_train_config.return_value.keypoint_flip_pairs == [1, 2]

    def test_keypoint_schema_and_flip_pairs_inferred_from_native_coco_metadata(self, tmp_path: Path) -> None:
        """Native COCO person-keypoint annotations should use the same symmetry inference as Roboflow COCO."""
        annotation_dir = tmp_path / "annotations"
        annotation_dir.mkdir(parents=True)
        annotation_path = annotation_dir / "person_keypoints_train2017.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {
                            "id": 1,
                            "name": "person",
                            "supercategory": "person",
                            "keypoints": ["nose", "left_eye", "right_eye"],
                            "skeleton": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        from rfdetr.config import KeypointTrainConfig

        model = object.__new__(RFDETR)
        model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        model.model = SimpleNamespace(args=SimpleNamespace(num_classes=90, num_keypoints_per_class=[17]))
        train_config = KeypointTrainConfig(dataset_dir=str(tmp_path), dataset_file="coco", tensorboard=False)

        model._align_keypoint_schema_from_dataset(train_config)

        assert model.model_config.num_keypoints_per_class == [3]
        assert model.model.args.num_keypoints_per_class == [3]
        assert train_config.keypoint_flip_pairs == [1, 2]

    def test_explicit_keypoint_flip_pairs_are_preserved_when_dataset_metadata_has_pairs(self, tmp_path: Path) -> None:
        """Dataset-inferred pairs must not override an explicit user mapping."""
        annotation_path = tmp_path / "train" / "_annotations.coco.json"
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {
                            "id": 0,
                            "name": "person",
                            "supercategory": "person",
                            "keypoints": ["nose", "left_eye", "right_eye"],
                            "skeleton": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        from rfdetr.config import KeypointTrainConfig

        model = object.__new__(RFDETR)
        model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        model.model = SimpleNamespace(args=SimpleNamespace(num_classes=90, num_keypoints_per_class=[17]))
        train_config = KeypointTrainConfig(
            dataset_dir=str(tmp_path),
            dataset_file="roboflow",
            tensorboard=False,
            keypoint_flip_pairs=[2, 0],
        )

        model._align_keypoint_schema_from_dataset(train_config)

        assert model.model_config.num_keypoints_per_class == [3]
        assert model.model.args.num_keypoints_per_class == [3]
        assert train_config.keypoint_flip_pairs == [2, 0]

    def test_explicit_keypoint_schema_mismatch_warns_and_uses_dataset(self, mock_self, patch_lit, caplog):
        """Explicit num_keypoints_per_class mismatches should warn and use dataset metadata."""
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None,
            device="cpu",
            num_keypoints_per_class=[17],
        )
        mock_self.model.args = SimpleNamespace(num_classes=90, num_keypoints_per_class=[17])
        mock_self._align_keypoint_schema_from_dataset = lambda config: RFDETR._align_keypoint_schema_from_dataset(
            mock_self, config
        )
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_roboflow_keypoint_categories(dataset_dir, keypoint_count=25)

        p_mod, p_dm, p_bt, *_ = patch_lit
        previous_propagate = detr_logger.propagate
        detr_logger.propagate = True
        try:
            with p_mod, p_dm, p_bt, caplog.at_level("WARNING", logger="rf-detr"):
                RFDETR.train(mock_self)
        finally:
            detr_logger.propagate = previous_propagate

        assert mock_self.model_config.num_keypoints_per_class == [25]
        assert mock_self.model.args.num_keypoints_per_class == [25]
        assert any(
            record.levelname == "WARNING"
            and "Configured num_keypoints_per_class=[17]" in record.message
            and "dataset keypoint metadata [25]" in record.message
            for record in caplog.records
        )

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
        object.__setattr__(mock_self.get_train_config.return_value, "dataset_dir", None)

        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)  # must not raise

    def test_keypoint_schema_padded_when_num_classes_bumped_for_custom_dataset(self, mock_self, patch_lit):
        """num_keypoints_per_class is zero-padded when auto-adjust bumps num_classes beyond schema length.

        Regression test for the warning "Keypoint class-logit boost has N classes but detection head has M" on custom
        (non-Roboflow) datasets.  Root cause: _align_num_classes_from_dataset bumped num_classes but did not pad the
        keypoint schema.  After the fix the schema length equals num_classes and _aggregate_keypoint_class_logits no
        longer fires the mismatch warning.
        """
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None,
            device="cpu",
            num_keypoints_per_class=[17, 0],
        )
        mock_self.model.args = SimpleNamespace(num_classes=2, num_keypoints_per_class=[17, 0])

        # Dataset has 3 classes; schema covers 2 → triggers the padding path.
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 0]
        assert mock_self.model.args.num_keypoints_per_class == [17, 0, 0]

    def test_keypoint_schema_not_padded_when_already_covers_all_classes(self, mock_self, patch_lit):
        """Schema is left untouched when it already spans all detection classes.

        Scenario: schema [17, 0, 14] (len=3), dataset has 2 classes.  The schema expansion at
        max(2, len(schema))=3 sets dataset_num_classes=3, auto-adjust fires (90→3), then the
        padding guard len(3)<3 evaluates False and the schema is preserved unchanged.
        Using a 2-class dataset (not 3) ensures the guard is actually reached and evaluated
        rather than being vacuously bypassed by an early return.
        """
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None,
            device="cpu",
            num_keypoints_per_class=[17, 0, 14],
        )
        mock_self.model.args = SimpleNamespace(num_classes=3, num_keypoints_per_class=[17, 0, 14])

        # 2-class dataset so max(2, 3)=3; auto-adjust fires (90→3); guard 3<3=False (no padding).
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["player", "ball"])
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 14]
        assert mock_self.model.args.num_keypoints_per_class == [17, 0, 14]

    def test_keypoint_schema_padded_when_model_args_absent(self, mock_self, patch_lit):
        """model_config schema is padded even when model.args is absent (model_args=None path).

        Scenario: schema [17, 0], 3-class dataset, model has no args attribute.
        Expected: model_config.num_keypoints_per_class padded to [17, 0, 0]; no AttributeError.
        """
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None,
            device="cpu",
            num_keypoints_per_class=[17, 0],
        )
        mock_self.model = MagicMock(spec=[])  # no 'args' attr → getattr(model, "args", None) = None

        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 0]

    def test_keypoint_schema_not_padded_when_schema_empty(self, mock_self, patch_lit):
        """Padding is skipped when num_keypoints_per_class is an empty list.

        Scenario: use_grouppose_keypoints=True but schema is [], 3-class dataset.
        Expected: auto-adjust fires (90→3) but schema stays [] — the truthiness guard
        ``if keypoint_schema and ...`` short-circuits before evaluating the length check.
        """
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None,
            device="cpu",
        )
        mock_self.model_config.num_keypoints_per_class = []  # override default [17]
        mock_self.model.args = SimpleNamespace(num_classes=90, num_keypoints_per_class=[])

        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        p_mod, p_dm, p_bt, *_ = patch_lit
        with p_mod, p_dm, p_bt, load_classes_patch:
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3  # auto-adjust still fires
        assert mock_self.model_config.num_keypoints_per_class == []  # empty schema not padded


# ---------------------------------------------------------------------------
# Start-of-run class-name resolution (PR #1496 review follow-ups)
# ---------------------------------------------------------------------------


class TestPreFitClassNamesResolution:
    """The start-of-run training_config.json resolves class names from every layout the datamodule will read.

    ``_run_train_capturing_pre_fit`` is shared with ``TestSaveTrainingConfig`` — it never touches ``self``, so it is
    aliased here rather than inherited, which would re-collect that class's tests under this one.
    """

    _run_train_capturing_pre_fit = TestSaveTrainingConfig._run_train_capturing_pre_fit

    @staticmethod
    def _write_train_shard_index(dataset_dir: Path, category_ids: str, *, version: int | None = None) -> Path:
        """Write a packed ``train`` shard index whose grouping root carries no annotation.

        Only the index is written -- the pre-fit read and ``WebDatasetDetection.__init__`` both stop at the JSON,
        so no shard tar and no ``webdataset`` package is needed. ``version`` overrides the schema stamp to produce
        an index the current reader rejects.

        Examples:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     written = TestPreFitClassNamesResolution._write_train_shard_index(Path(directory), "remap")
            ...     json.loads((written / "train-index.json").read_text())["category_ids"]
            'remap'
        """
        categories = (
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 3, "name": "cat", "supercategory": "root"},
            {"id": 9, "name": "dog", "supercategory": "root"},
        )
        index = ShardIndex("train", ("train-000000.tar",), 4, categories, (3, 9), category_ids, (4,))
        payload = index.to_json()
        if version is not None:
            payload["version"] = version
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / index_name("train")).write_text(json.dumps(payload), encoding="utf-8")
        return dataset_dir

    @pytest.mark.parametrize(
        ("category_ids", "expected"),
        [
            pytest.param("remap", ["cat", "dog"], id="remap-drops-the-unannotated-parent"),
            pytest.param("raw", ["root", "", "", "cat", "", "", "", "", "", "dog"], id="raw-keeps-every-id-slot"),
        ],
    )
    def test_pre_fit_class_names_match_the_webdataset_datamodule(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], category_ids: str, expected: list[str]
    ) -> None:
        """A packed directory records the same names the streaming dataset will report after fit()."""
        dataset_dir = self._write_train_shard_index(tmp_path / "ds", category_ids)
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        dataset = WebDatasetDetection(dataset_dir, "train", transforms=None)
        assert pre_fit["class_names"] == dataset.class_names == expected

    def test_pre_fit_num_classes_counts_the_shard_index_names(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """num_classes follows the recorded list instead of staying 0 for a packed directory."""
        self._write_train_shard_index(tmp_path / "ds", "remap")
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        assert pre_fit["num_classes"] == 2

    def test_unreadable_shard_index_records_null_and_does_not_block_training(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """An index the reader rejects (ValueError from ShardIndex.from_json) degrades to null, never an exception."""
        self._write_train_shard_index(tmp_path / "ds", "remap", version=99)
        pre_fit, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        assert (pre_fit["class_names"], final is not None) == (None, True)

    @staticmethod
    def _train_interrupted_keypoint_run(tmp_path: Path, patch_lit: tuple[Any, ...], **train_overrides: Any) -> dict:
        """Run train() on a background-first keypoint model whose fit() dies; return the file it left behind.

        The dataset reader is patched to hand back the detection basis ``['person']``, so a recorded null proves the
        keypoint gate bypassed the reader rather than the dataset merely being unreadable.

        Examples:
            Needs the ``tmp_path`` and ``patch_lit`` fixtures, so it cannot run standalone:

            >>> TestPreFitClassNamesResolution._train_interrupted_keypoint_run(tmp_path, patch_lit)  # doctest: +SKIP
        """
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None, device="cpu", num_keypoints_per_class=[0, 17]
        )
        p_mod, p_dm, p_bt, _, _, mock_bt = patch_lit
        mock_bt.return_value.fit.side_effect = RuntimeError("interrupted")
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["person"])
        with p_mod, p_dm, p_bt, load_classes_patch, pytest.raises(RuntimeError):
            RFDETR.train(mock_self)
        output_dir = mock_self.get_train_config.return_value.output_dir
        return _read_training_config(os.path.join(output_dir, "training_config.json")) or {}

    def test_interrupted_keypoint_run_records_null_class_names(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """A bg-first keypoint run killed in fit() leaves class_names null, not the detection-basis ['person']."""
        final = self._train_interrupted_keypoint_run(tmp_path, patch_lit)
        assert (final["class_names"], final["num_classes"]) == (None, 0)

    def test_keypoint_run_still_records_an_explicit_config_class_names(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """Only the dataset read is gated: a TrainConfig.class_names the user set is recorded as before."""
        final = self._train_interrupted_keypoint_run(tmp_path, patch_lit, class_names=["", "person"])
        assert final["class_names"] == ["", "person"]


class TestSharedCocoCategoryParse:
    """num_classes alignment and the start-of-run config write read ``train/_annotations.coco.json`` once.

    ``_write_coco_categories`` is shared with ``TestRFDETRTrainNumClassesAutoDetect``; it never touches ``self``, so it
    is aliased rather than inherited (see ``TestPreFitClassNamesResolution``).
    """

    _write_coco_categories = TestRFDETRTrainNumClassesAutoDetect._write_coco_categories

    _CATEGORIES = [
        {"id": 1, "name": "animal", "supercategory": "none"},
        {"id": 2, "name": "dog", "supercategory": "animal"},
        {"id": 3, "name": "cat", "supercategory": "animal"},
    ]

    @staticmethod
    def _train_on_coco_dataset(tmp_path: Path, patch_lit: tuple[Any, ...]) -> tuple[MagicMock, dict[str, Any] | None]:
        """Run train() with the real num_classes alignment bound and a real memo dict on the mock self.

        ``_make_rfdetr_self`` leaves ``_align_num_classes_from_dataset`` as a MagicMock, which would make the pre-fit
        write the only reader and a call count of one trivially true. Returns ``(mock_self, pre_fit_payload)``.

        Examples:
            Needs the ``tmp_path`` and ``patch_lit`` fixtures, so it cannot run standalone:

            >>> TestSharedCocoCategoryParse._train_on_coco_dataset(tmp_path, patch_lit)  # doctest: +SKIP
        """
        mock_self = MagicMock()
        mock_self.model_config = RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        mock_self.model = MagicMock()
        mock_self.get_train_config.return_value = _make_train_config(tmp_path)
        mock_self._align_num_classes_from_dataset = lambda ds: RFDETR._align_num_classes_from_dataset(mock_self, ds)
        mock_self._coco_categories_cache = {}
        config_path = os.path.join(mock_self.get_train_config.return_value.output_dir, "training_config.json")
        captured: dict[str, Any] = {}
        p_mod, p_dm, p_bt, _, _, mock_bt = patch_lit

        def _capture(*args: Any, **kwargs: Any) -> None:
            captured["pre_fit"] = _read_training_config(config_path)

        mock_bt.return_value.fit.side_effect = _capture
        with p_mod, p_dm, p_bt:
            RFDETR.train(mock_self)
        return mock_self, captured.get("pre_fit")

    def test_annotation_file_is_parsed_once_across_alignment_and_pre_fit(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """Both readers go through one _filtered_coco_categories call instead of two back-to-back json.loads."""
        self._write_coco_categories(tmp_path / "ds", categories=self._CATEGORIES, annotated_ids=[2, 3])
        parse = patch.object(RFDETR, "_filtered_coco_categories", side_effect=RFDETR._filtered_coco_categories)
        with parse as parse_mock:
            self._train_on_coco_dataset(tmp_path, patch_lit)
        assert parse_mock.call_count == 1

    def test_shared_parse_feeds_both_num_classes_and_pre_fit_class_names(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """The single parse still lands in both places: the aligned count and the recorded label space agree."""
        self._write_coco_categories(tmp_path / "ds", categories=self._CATEGORIES, annotated_ids=[2, 3])
        mock_self, pre_fit = self._train_on_coco_dataset(tmp_path, patch_lit)
        assert (mock_self.model_config.num_classes, pre_fit["class_names"]) == (2, ["dog", "cat"])

    def test_memo_replaces_its_entry_for_a_different_dataset_dir(self, tmp_path: Path) -> None:
        """One directory at a time: pointing at a second dataset evicts the first instead of accumulating."""
        first, second = tmp_path / "first", tmp_path / "second"
        self._write_coco_categories(first, categories=self._CATEGORIES, annotated_ids=[2, 3])
        self._write_coco_categories(second, categories=self._CATEGORIES[:2], annotated_ids=[2])
        cache: dict[str, list[dict[str, Any]]] = {}
        RFDETR._memoized_coco_categories(cache, str(first))
        RFDETR._memoized_coco_categories(cache, str(second))
        assert list(cache) == [str(second.resolve())]

    def test_memo_returns_the_cached_entry_without_reparsing(self, tmp_path: Path) -> None:
        """A second lookup of the same directory is served from the memo."""
        dataset_dir = tmp_path / "ds"
        self._write_coco_categories(dataset_dir, categories=self._CATEGORIES, annotated_ids=[2, 3])
        cache: dict[str, list[dict[str, Any]]] = {}
        first = RFDETR._memoized_coco_categories(cache, str(dataset_dir))
        with patch.object(RFDETR, "_filtered_coco_categories", side_effect=AssertionError("re-parsed")):
            second = RFDETR._memoized_coco_categories(cache, str(dataset_dir))
        assert second is first
