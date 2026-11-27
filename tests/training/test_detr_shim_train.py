# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETR.train(): PTL delegation, legacy kwarg absorption, resolution and class-count auto-detection."""

import builtins
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.config import (
    KeypointTrainConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRSmallConfig,
)
from rfdetr.detr import RFDETR
from rfdetr.detr import logger as detr_logger
from rfdetr.models.postprocess import PostProcess
from rfdetr.training.auto_batch import AutoBatchResult
from tests.inference.helpers import _DummyRFDETR
from tests.training._shim_support import (
    _DATASET_READ_ERRORS,
    _make_rfdetr_self,
    _make_train_config,
    _read_training_config,
)

# ---------------------------------------------------------------------------
# 1. RFDETR.train() PTL delegation
# ---------------------------------------------------------------------------


class TestRFDETRTrainPTL:
    """RFDETR.train() delegates to PTL build_trainer().fit()."""

    def test_build_trainer_called_with_config_and_model_config(self, tmp_path, patch_lit):
        """build_trainer receives (train_config, model_config) in the right order."""
        mock_self = _make_rfdetr_self(tmp_path)
        _mcls, _dmcls, mock_bt = patch_lit
        RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator=None)

    def test_trainer_fit_called_with_module_and_datamodule(self, tmp_path, patch_lit):
        """trainer.fit() is called with (module_instance, datamodule_instance)."""
        mock_self = _make_rfdetr_self(tmp_path)
        mcls, dmcls, mock_bt = patch_lit
        RFDETR.train(mock_self)

        trainer = mock_bt.return_value
        fit_args = trainer.fit.call_args
        assert fit_args[0][0] is mcls.return_value  # module instance
        assert fit_args[0][1] is dmcls.return_value  # datamodule instance

    @pytest.mark.parametrize(
        ("train_overrides", "expected_ckpt_path"),
        [
            pytest.param({}, None, id="resume-unset"),
            pytest.param({"resume": ""}, None, id="resume-empty-string"),
            pytest.param({"resume": "/some/checkpoint.ckpt"}, "/some/checkpoint.ckpt", id="resume-path"),
        ],
    )
    def test_ckpt_path_follows_resume(self, tmp_path, patch_lit, train_overrides, expected_ckpt_path):
        """trainer.fit receives ckpt_path=config.resume, with an unset or empty resume coerced to None."""
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        mcls, dmcls, mock_bt = patch_lit

        RFDETR.train(mock_self)

        mock_bt.return_value.fit.assert_called_once_with(
            mcls.return_value, dmcls.return_value, ckpt_path=expected_ckpt_path
        )

    def test_model_model_synced_back_by_identity(self, tmp_path, patch_lit):
        """self.model.model is reassigned to module.model (identity, not copy)."""
        mock_self = _make_rfdetr_self(tmp_path)
        mcls, _dmcls, mock_bt = patch_lit
        sentinel_nn_module = object()
        mcls.return_value.model = sentinel_nn_module

        RFDETR.train(mock_self)

        assert mock_self.model.model is sentinel_nn_module

    @pytest.mark.usefixtures("patch_lit")
    def test_returns_none(self, tmp_path):
        """RFDETR.train() has no return value."""
        mock_self = _make_rfdetr_self(tmp_path)
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

        with pytest.raises(ImportError) as exc_info:
            RFDETR.train(mock_self)
        assert "rfdetr[train,loggers]" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.parametrize(
        "missing_name",
        ["rfdetr.training", "rfdetr.training.auto_batch"],
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

        with pytest.raises(ModuleNotFoundError) as exc_info:
            RFDETR.train(mock_self)
        assert missing_name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("model_class_names", "datamodule_class_names", "expected"),
        [
            pytest.param(["stale"], ["cat", "dog", "bird"], ["cat", "dog", "bird"], id="datamodule-names-overwrite"),
            pytest.param(["existing_class"], None, ["existing_class"], id="none-keeps-existing"),
            pytest.param(["stale_label"], [], [], id="empty-list-overwrites-stale"),
        ],
    )
    def test_class_names_synced_back_from_datamodule(
        self, tmp_path, patch_lit, model_class_names, datamodule_class_names, expected
    ):
        """self.model.class_names takes the datamodule's names after train(); None is skipped, [] is a real value.

        Regression test for #509: custom class names were not synced back from RFDETRDataModule after training, causing
        predict() to return COCO labels instead of the dataset's class labels. The sync-back guard must not clobber
        existing names when the datamodule has none (None), yet an explicit empty list is a valid value.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        mock_self.model.class_names = model_class_names
        _, dmcls, _ = patch_lit
        dmcls.return_value.class_names = datamodule_class_names

        RFDETR.train(mock_self)

        assert mock_self.model.class_names == expected

    @pytest.mark.usefixtures("patch_lit")
    def test_model_args_synced_from_train_config_after_training(self, tmp_path):
        """self.model.args reflects effective training and model config after train().

        Regression test for #1199: model.model.__dict__['args'] retained the construction-time defaults (lr=0.0001,
        lr_encoder=0.00015) instead of the overrides passed to train(), because ``self.model.args`` was built once at
        model-construction time from a dummy config and never refreshed after ``train()`` completed.
        """
        mock_self = _make_rfdetr_self(tmp_path, lr=5e-5, lr_encoder=1e-4, batch_size=7)
        mock_self.model_config.num_classes = 7
        mock_self.model.args = SimpleNamespace(lr=0.0001, lr_encoder=0.00015, resolution=560)

        RFDETR.train(mock_self)

        assert mock_self.model.args.lr == 5e-5
        assert mock_self.model.args.lr_encoder == 1e-4
        assert mock_self.model.args.batch_size == 7
        assert mock_self.model.args.num_classes == 7
        assert mock_self.model.args.dataset_dir == str(tmp_path / "ds")
        assert mock_self.model.args.output_dir == str(tmp_path / "out")

    @pytest.mark.usefixtures("patch_lit")
    def test_postprocess_keypoint_schema_synced_after_training(self, tmp_path):
        """self.model.postprocess decodes with the keypoint schema the run trained with after train().

        train() aligns ``model_config.num_keypoints_per_class`` to the dataset, while the postprocessor was built at
        construction from the pretrained schema, so predict() raised on the slot count of the retrained head.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        stale = PostProcess(num_keypoints_per_class=[0, 17])
        mock_self.model.postprocess = stale
        mock_self.model_config.num_keypoints_per_class = [3]

        RFDETR.train(mock_self)

        assert mock_self.model.postprocess.num_keypoints_per_class == [3]
        assert stale.num_keypoints_per_class == [0, 17]

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda", pytest.param(torch.device("cuda:1"), id="torch-device-cuda-index")],
    )
    @pytest.mark.usefixtures("patch_lit")
    def test_device_kwarg_consumed_without_deprecation_warning(self, tmp_path, device: str | torch.device):
        """Device= (string or torch.device) is consumed without a DeprecationWarning or reaching get_train_config."""
        mock_self = _make_rfdetr_self(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFDETR.train(mock_self, device=device)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        mock_self.get_train_config.assert_called_once_with()

    @pytest.mark.parametrize(
        "train_kwargs",
        [
            pytest.param({"skip_best_epochs": 3}, id="skip-best-epochs"),
            pytest.param(
                {"optimizer": "torch.optim.AdamW", "optimizer_kwargs": {"weight_decouple": True}}, id="optimizer-config"
            ),
        ],
    )
    @pytest.mark.usefixtures("patch_lit")
    def test_non_absorbed_kwargs_forwarded_to_get_train_config(self, tmp_path, train_kwargs):
        """Training kwargs that train() does not absorb must reach get_train_config unchanged."""
        mock_self = _make_rfdetr_self(tmp_path)

        RFDETR.train(mock_self, **train_kwargs)

        mock_self.get_train_config.assert_called_once_with(**train_kwargs)

    def test_batch_size_auto_resolved_before_module_and_datamodule_build(self, tmp_path, patch_lit, mocker):
        """batch_size='auto' is resolved to ints before module/datamodule init."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto", grad_accum_steps=99)
        auto_result = AutoBatchResult(
            safe_micro_batch=3,
            recommended_grad_accum_steps=6,
            effective_batch_size=18,
            device_name="Fake GPU",
        )
        mcls, dmcls, _ = patch_lit
        mocker.patch("rfdetr.training.auto_batch.resolve_auto_batch_config", return_value=auto_result)
        RFDETR.train(mock_self)

        config = mock_self.get_train_config.return_value
        assert config.batch_size == 3
        assert config.grad_accum_steps == 6
        mcls.assert_called_once_with(mock_self.model_config, config)
        dmcls.assert_called_once_with(mock_self.model_config, config)

    @pytest.mark.parametrize(
        ("train_kwargs", "expected_devices"),
        [
            pytest.param({}, None, id="no-device"),
            pytest.param({"device": "cuda:1"}, [1], id="cuda-1"),
        ],
    )
    @pytest.mark.usefixtures("patch_lit")
    def test_batch_size_auto_calls_resolver_with_expected_context(
        self, tmp_path, mocker, train_kwargs: dict[str, str], expected_devices: list[int] | None
    ):
        """Auto-batch resolver receives model context, model config, train config, and the GPUs the run trains on."""
        mock_self = _make_rfdetr_self(tmp_path, batch_size="auto")
        auto_result = AutoBatchResult(
            safe_micro_batch=2,
            recommended_grad_accum_steps=8,
            effective_batch_size=16,
            device_name="Fake GPU",
        )
        mock_resolve = mocker.patch("rfdetr.training.auto_batch.resolve_auto_batch_config", return_value=auto_result)
        RFDETR.train(mock_self, **train_kwargs)

        config = mock_self.get_train_config.return_value
        mock_resolve.assert_called_once_with(
            model_context=mock_self.model,
            model_config=mock_self.model_config,
            train_config=config,
            devices=expected_devices,
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
        """Device= is absorbed and forwarded to build_trainer as accelerator= (and devices= when indexed).

        Accepts a string or torch.device, with or without an index. No other kwargs may reach build_trainer, since
        assert_called_once_with is exact on kwargs.
        """
        mock_self = _make_rfdetr_self(tmp_path)
        _mcls, _dmcls, mock_bt = patch_lit
        RFDETR.train(mock_self, device=device)
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, **expected_kwargs)

    @pytest.mark.usefixtures("patch_lit")
    def test_invalid_device_raises_value_error(self, tmp_path):
        """Invalid device strings raise a ValueError with the train() device hint."""
        mock_self = _make_rfdetr_self(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            RFDETR.train(mock_self, device="notadevice")

        assert "Invalid device specifier for train(): 'notadevice'" in str(exc_info.value)

    def test_device_unmapped_valid_type_warns_and_falls_back_to_auto_detection(self, tmp_path, patch_lit):
        """Valid but unmapped torch device types warn and use PTL auto-detection."""
        mock_self = _make_rfdetr_self(tmp_path)
        _mcls, _dmcls, mock_bt = patch_lit
        with pytest.warns(UserWarning) as record:
            RFDETR.train(mock_self, device="meta")
        assert any("auto-detection" in str(warning.message) for warning in record)
        config = mock_self.get_train_config.return_value
        mock_bt.assert_called_once_with(config, mock_self.model_config, accelerator=None)
        assert "devices" not in mock_bt.call_args.kwargs

    @pytest.mark.usefixtures("patch_lit")
    def test_empty_callbacks_accepted(self, tmp_path):
        """Callbacks={} is accepted without error."""
        mock_self = _make_rfdetr_self(tmp_path)
        RFDETR.train(mock_self, callbacks={})  # must not raise

    def test_shifted_class_names_raise_before_fit(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """A class_names read from every category of a Roboflow export stops the run before it leaves any trace.

        The datamodule's ``setup("fit")`` hook rejects the same list, but PTL's ``_call_setup_hook`` starts each
        configured logger's experiment before calling that hook, so waiting for it leaves an orphan crashed wandb/MLflow
        run behind — and the start-of-run ``training_config.json`` has already recorded the list.
        """
        annotations = tmp_path / "ds" / "train"
        annotations.mkdir(parents=True)
        (annotations / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": 1, "name": "animals", "supercategory": "none"},
                        {"id": 2, "name": "cat", "supercategory": "animals"},
                        {"id": 3, "name": "dog", "supercategory": "animals"},
                    ],
                    "annotations": [{"category_id": 2}, {"category_id": 3}],
                }
            ),
            encoding="utf-8",
        )
        mock_self = _make_rfdetr_self(tmp_path, dataset_file="roboflow", class_names=["animals", "cat", "dog"])
        _mcls, _dmcls, mock_bt = patch_lit

        with pytest.raises(ValueError) as exc_info:
            RFDETR.train(mock_self)
        assert "'animals'" in str(exc_info.value)

        mock_bt.return_value.fit.assert_not_called()
        assert not (Path(mock_self.get_train_config().output_dir) / "training_config.json").exists()

    @pytest.fixture
    def mock_saver_cls(self, mocker) -> MagicMock:
        """Replace ``DatasetGridSaver`` with a mock class and return it."""
        return mocker.patch("rfdetr.datasets.save_grids.DatasetGridSaver", MagicMock(name="DatasetGridSaver"))

    def test_save_dataset_grids_true_calls_grid_saver(self, tmp_path, patch_lit, mock_saver_cls):
        """save_dataset_grids=True triggers DatasetGridSaver.save_grid() for train and val."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        _mcls, _dmcls, _mock_bt = patch_lit
        RFDETR.train(mock_self)

        # DatasetGridSaver must be constructed twice (train + val) and save_grid called on each
        assert mock_saver_cls.call_count == 2
        assert mock_saver_cls.return_value.save_grid.call_count == 2

        # setup("fit") must be called on the datamodule before training
        dm_instance = _dmcls.return_value
        dm_instance.setup.assert_called_with("fit")

    @pytest.mark.usefixtures("patch_lit")
    def test_save_dataset_grids_false_skips_grid_saver(self, tmp_path, mock_saver_cls):
        """save_dataset_grids=False (default) must not call DatasetGridSaver at all."""
        mock_self = _make_rfdetr_self(tmp_path)  # default save_dataset_grids=False
        RFDETR.train(mock_self)

        mock_saver_cls.assert_not_called()

    def test_save_dataset_grids_uses_output_dir_subdir(self, tmp_path, patch_lit, mock_saver_cls):
        """Grid images are saved to <output_dir>/dataset_grids."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        config = mock_self.get_train_config.return_value
        _mcls, _dmcls, _mock_bt = patch_lit
        RFDETR.train(mock_self)

        expected_output_dir = Path(config.output_dir) / "dataset_grids"
        called_dirs = [call.args[1] for call in mock_saver_cls.call_args_list]
        assert all(d == expected_output_dir for d in called_dirs)

    def test_save_dataset_grids_skipped_off_rank_zero(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mock_saver_cls: MagicMock, mocker: MockerFixture
    ) -> None:
        """The grid write shares the launcher guard, so a non-zero rank renders nothing."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        _mcls, _dmcls, _mock_bt = patch_lit
        mocker.patch("rfdetr.detr._is_launcher_main_process", return_value=False)

        RFDETR.train(mock_self)

        mock_saver_cls.assert_not_called()

    def test_dataset_setup_failure_aborts_instead_of_becoming_a_grid_warning(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mock_saver_cls: MagicMock
    ) -> None:
        """A ``setup("fit")`` failure propagates as itself rather than as a failed grid render.

        Building the datasets is fatal to ``trainer.fit()`` regardless, so catching it here used to report the real
        cause — an unusable dataset, or a rejected ``class_names`` — as "failed to save dataset grids", then let
        ``fit()`` raise the same thing again.
        """
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        _mcls, _dmcls, mock_bt = patch_lit
        _dmcls.return_value.setup.side_effect = ValueError("dataset is unusable")

        with pytest.raises(ValueError) as exc_info:
            RFDETR.train(mock_self)
        assert "dataset is unusable" in str(exc_info.value)

        mock_bt.return_value.fit.assert_not_called()

    def test_save_dataset_grids_failure_does_not_abort_training(self, tmp_path, patch_lit, mock_saver_cls):
        """A save_grid() failure must not abort training — trainer.fit() must still be called."""
        mock_self = _make_rfdetr_self(tmp_path, save_dataset_grids=True)
        _mcls, _dmcls, mock_bt = patch_lit
        mock_saver_cls.return_value.save_grid.side_effect = OSError("disk full")
        # Must not raise even though save_grid() fails
        RFDETR.train(mock_self)

        # Training must proceed regardless of the grid-save failure
        mock_bt.return_value.fit.assert_called_once()


# ---------------------------------------------------------------------------
# 2b. resolution= kwarg handling
# ---------------------------------------------------------------------------


class TestResolutionKwarg:
    """RFDETR.train(resolution=...) applies, validates, and syncs the resolution override."""

    @pytest.mark.usefixtures("patch_lit")
    def test_updates_model_config_resolution(self, tmp_path):
        """Resolution kwarg is applied to model_config.resolution before training."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        valid_resolution = block_size * 11  # guaranteed divisible and different from default
        RFDETR.train(mock_self, resolution=valid_resolution)
        assert mock_self.model_config.resolution == valid_resolution

    @pytest.mark.usefixtures("patch_lit")
    def test_does_not_implicitly_update_positional_encoding_size(self, tmp_path):
        """Pretrained-specific PE (RFDETRBase DINOv2=37) is preserved when resolution is overridden."""
        mock_self = _make_rfdetr_self(tmp_path)
        # RFDETRBaseConfig: PE=37 (DINOv2 native 518//14), resolution=560, patch_size=14.
        # PE != resolution // patch_size, so the smart PE guard leaves PE unchanged.
        original_pe = mock_self.model_config.positional_encoding_size
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        valid_override_resolution = block_size * 11  # different from default 560
        RFDETR.train(mock_self, resolution=valid_override_resolution)
        assert mock_self.model_config.positional_encoding_size == original_pe

    @pytest.mark.usefixtures("patch_lit")
    def test_updates_positional_encoding_size_for_formula_derived_config(self, tmp_path):
        """For configs where PE == resolution // patch_size, resolution override updates PE."""
        # RFDETRSmallConfig: patch_size=16, num_windows=2, resolution=512, PE=32=512//16.
        mock_self = _make_rfdetr_self(tmp_path)
        mock_self.model_config = RFDETRSmallConfig(pretrain_weights=None, num_classes=3, device="cpu")
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        new_resolution = block_size * 21  # 672 for Small — valid and different from default 512
        expected_pe = new_resolution // mock_self.model_config.patch_size
        RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model_config.positional_encoding_size == expected_pe

    @pytest.mark.usefixtures("patch_lit")
    def test_does_not_reach_get_train_config(self, tmp_path):
        """Resolution kwarg is popped before get_train_config is called."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        RFDETR.train(mock_self, resolution=block_size * 10)
        assert "resolution" not in mock_self.get_train_config.call_args.kwargs

    @pytest.mark.usefixtures("patch_lit")
    def test_indivisible_raises_value_error(self, tmp_path):
        """Resolution not divisible by patch_size * num_windows raises ValueError."""
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        indivisible = block_size * 10 + 1  # guaranteed not divisible by block_size
        with pytest.raises(ValueError) as exc_info:
            RFDETR.train(mock_self, resolution=indivisible)
        assert f"resolution={indivisible}" in str(exc_info.value)

    @pytest.mark.usefixtures("patch_lit")
    def test_none_leaves_model_config_unchanged(self, tmp_path):
        """Omitting resolution leaves model_config.resolution unchanged."""
        mock_self = _make_rfdetr_self(tmp_path)
        original_resolution = mock_self.model_config.resolution
        RFDETR.train(mock_self)
        assert mock_self.model_config.resolution == original_resolution

    @pytest.mark.parametrize(
        "bad_resolution",
        [0, -56, True, False, 1.5, 560.0, "560"],
    )
    @pytest.mark.usefixtures("patch_lit")
    def test_invalid_type_or_value_raises_value_error(self, tmp_path, bad_resolution):
        """Non-positive, bool, or non-integer resolution raises ValueError before divisibility check."""
        mock_self = _make_rfdetr_self(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            RFDETR.train(mock_self, resolution=bad_resolution)
        assert "resolution must be a positive integer" in str(exc_info.value)

    @pytest.mark.usefixtures("patch_lit")
    def test_syncs_model_resolution_attribute(self, tmp_path):
        """Resolution kwarg sets model.resolution so predict()/export() see the new resolution.

        Regression test for #952 — keeps the cached inference/export context in sync after a resolution override in
        train().
        """
        mock_self = _make_rfdetr_self(tmp_path)
        block_size = mock_self.model_config.patch_size * mock_self.model_config.num_windows
        new_resolution = block_size * 11
        RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model.resolution == new_resolution

    @pytest.mark.usefixtures("patch_lit")
    def test_syncs_model_args_resolution_and_pe(self, tmp_path):
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
        RFDETR.train(mock_self, resolution=new_resolution)
        assert mock_self.model.args.resolution == new_resolution
        assert mock_self.model.args.positional_encoding_size == expected_pe


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

    @pytest.mark.usefixtures("patch_lit")
    def test_auto_adjusts_num_classes_when_not_overridden(self, mock_self, mocker: MockerFixture):
        """When user did not set num_classes, auto-adjust to the dataset class count.

        Scenario: model built without explicit num_classes → default=90.
        Dataset has 4 classes.  Expected: model_config.num_classes becomes 4.
        """
        assert "num_classes" not in mock_self.model_config.model_fields_set

        mocker.patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4

    @pytest.mark.usefixtures("patch_lit")
    def test_coco_auto_detect_skips_unannotated_grouping_category(self, mock_self, mocker: MockerFixture):
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

        mocker.patch.object(RFDETR, "_load_classes", return_value=["dog", "cat"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 2

    @pytest.mark.usefixtures("patch_lit")
    def test_coco_auto_detect_counts_annotated_parent_category(self, mock_self, mocker: MockerFixture):
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

        mocker.patch.object(RFDETR, "_load_classes", return_value=["animal", "dog", "cat"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_coco_auto_detect_uses_active_first_schema_slots(self, mock_self):
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

        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 1
        assert mock_self.model.args.num_classes == 1

    @pytest.mark.usefixtures("patch_lit")
    def test_preserves_explicit_default_num_classes_when_dataset_differs(
        self,
        caplog,
        mock_self,
        monkeypatch,
        mocker: MockerFixture,
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

        mocker.patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == default_nc
        expected_fragment = (
            f"Dataset '{dataset_dir}' has 4 classes but model was initialized with num_classes={default_nc}"
        )
        assert any(record.levelname == "WARNING" and expected_fragment in record.message for record in caplog.records)

    @pytest.mark.usefixtures("patch_lit")
    def test_preserves_explicit_non_default_num_classes_when_dataset_differs(
        self,
        tmp_path,
        caplog,
        mock_self,
        monkeypatch,
        mocker: MockerFixture,
    ):
        """When user explicitly set a non-default num_classes, it is preserved.

        Scenario: user passes num_classes=10 (non-default).  Dataset has 4 classes.
        Expected: model_config.num_classes stays at 10.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=10)
        mock_self.model_config = mc
        dataset_dir = mock_self.get_train_config.return_value.dataset_dir

        mocker.patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 10
        expected_fragment = f"Dataset '{dataset_dir}' has 4 classes but model was initialized with num_classes=10"
        assert any(record.levelname == "WARNING" and expected_fragment in record.message for record in caplog.records)

    @pytest.mark.usefixtures("patch_lit")
    def test_auto_adjust_syncs_model_args_num_classes(self, mock_self, mocker: MockerFixture):
        """When auto-adjusting, keep ModelContext args.num_classes in sync."""
        mock_self.model.args = SimpleNamespace(num_classes=90)

        mocker.patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4
        assert mock_self.model.args.num_classes == 4

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_schema_inferred_when_not_explicitly_overridden(self, mock_self):
        """Roboflow keypoint metadata should populate model_config.num_keypoints_per_class."""
        mock_self.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        mock_self.model.args = SimpleNamespace(num_classes=90, num_keypoints_per_class=[17])
        mock_self._align_keypoint_schema_from_dataset = lambda config: RFDETR._align_keypoint_schema_from_dataset(
            mock_self, config
        )
        dataset_dir = Path(mock_self.get_train_config.return_value.dataset_dir)
        self._write_roboflow_keypoint_categories(dataset_dir, keypoint_count=25)

        RFDETR.train(mock_self)

        assert mock_self.model_config.num_keypoints_per_class == [25]
        assert mock_self.model.args.num_keypoints_per_class == [25]
        assert mock_self.model_config.num_classes == 1

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_flip_pairs_inferred_from_roboflow_coco_metadata(self, mock_self):
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

        RFDETR.train(mock_self)

        assert mock_self.get_train_config.return_value.keypoint_flip_pairs == [1, 2]

    @pytest.mark.parametrize(
        ("annotation_file", "dataset_file", "category_id", "config_overrides", "expected_flip_pairs"),
        [
            pytest.param(
                "annotations/person_keypoints_train2017.json", "coco", 1, {}, [1, 2], id="native-coco-infers-pairs"
            ),
            pytest.param(
                "train/_annotations.coco.json",
                "roboflow",
                0,
                {"keypoint_flip_pairs": [2, 0]},
                [2, 0],
                id="explicit-pairs-preserved",
            ),
        ],
    )
    def test_keypoint_schema_and_flip_pairs_from_dataset_metadata(
        self,
        tmp_path: Path,
        annotation_file: str,
        dataset_file: str,
        category_id: int,
        config_overrides: dict[str, Any],
        expected_flip_pairs: list[int],
    ) -> None:
        """Dataset keypoint metadata sets the schema; its symmetry pairs apply only when the user gave none.

        Native COCO person-keypoint annotations use the same symmetry inference as Roboflow COCO, and dataset-inferred
        pairs must not override an explicit user mapping.
        """
        annotation_path = tmp_path / annotation_file
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {
                            "id": category_id,
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
        model = object.__new__(RFDETR)
        model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        model.model = SimpleNamespace(args=SimpleNamespace(num_classes=90, num_keypoints_per_class=[17]))
        train_config = KeypointTrainConfig(
            dataset_dir=str(tmp_path), dataset_file=dataset_file, tensorboard=False, **config_overrides
        )

        model._align_keypoint_schema_from_dataset(train_config)

        assert model.model_config.num_keypoints_per_class == [3]
        assert model.model.args.num_keypoints_per_class == [3]
        assert train_config.keypoint_flip_pairs == expected_flip_pairs

    @pytest.mark.usefixtures("patch_lit")
    def test_explicit_keypoint_schema_mismatch_warns_and_uses_dataset(self, mock_self, caplog, monkeypatch):
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

        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            RFDETR.train(mock_self)

        assert mock_self.model_config.num_keypoints_per_class == [25]
        assert mock_self.model.args.num_keypoints_per_class == [25]
        assert any(
            record.levelname == "WARNING"
            and "Configured num_keypoints_per_class=[17]" in record.message
            and "dataset keypoint metadata [25]" in record.message
            for record in caplog.records
        )

    @pytest.mark.usefixtures("patch_lit")
    def test_no_adjustment_when_num_classes_already_matches_dataset(self, mock_self, mocker: MockerFixture):
        """No adjustment when the model's num_classes already equals the dataset count.

        Scenario: user passes num_classes=4 and dataset has 4 classes.
        Expected: model_config.num_classes remains 4 (no log noise, no error).
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=4)
        mock_self.model_config = mc

        mocker.patch.object(RFDETR, "_load_classes", return_value=self._FOUR_CLASS_NAMES)
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 4

    @pytest.mark.parametrize("exc", _DATASET_READ_ERRORS)
    @pytest.mark.usefixtures("patch_lit")
    def test_dataset_detection_error_does_not_block_training(self, exc, mock_self, mocker: MockerFixture):
        """Training proceeds even if _load_classes raises a known exception.

        Dataset detection is best-effort; errors must not block training.
        """
        mocker.patch.object(RFDETR, "_load_classes", side_effect=exc)
        RFDETR.train(mock_self)  # must not raise

    @pytest.mark.usefixtures("patch_lit")
    def test_none_dataset_dir_does_not_block_training(self, mock_self):
        """Training proceeds when config.dataset_dir resolves to None.

        Guards against AttributeError if getattr returns None.
        """
        # Override dataset_dir to None on the train config mock.
        object.__setattr__(mock_self.get_train_config.return_value, "dataset_dir", None)

        RFDETR.train(mock_self)  # must not raise

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_schema_padded_when_num_classes_bumped_for_custom_dataset(self, mock_self, mocker: MockerFixture):
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
        mocker.patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 0]
        assert mock_self.model.args.num_keypoints_per_class == [17, 0, 0]

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_schema_not_padded_when_already_covers_all_classes(self, mock_self, mocker: MockerFixture):
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
        mocker.patch.object(RFDETR, "_load_classes", return_value=["player", "ball"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 14]
        assert mock_self.model.args.num_keypoints_per_class == [17, 0, 14]

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_schema_padded_when_model_args_absent(self, mock_self, mocker: MockerFixture):
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

        mocker.patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3
        assert mock_self.model_config.num_keypoints_per_class == [17, 0, 0]

    @pytest.mark.usefixtures("patch_lit")
    def test_keypoint_schema_not_padded_when_schema_empty(self, mock_self, mocker: MockerFixture):
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

        mocker.patch.object(RFDETR, "_load_classes", return_value=["player", "ball", "referee"])
        RFDETR.train(mock_self)

        assert mock_self.model_config.num_classes == 3  # auto-adjust still fires
        assert mock_self.model_config.num_keypoints_per_class == []  # empty schema not padded


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
        _, _, mock_bt = patch_lit

        def _capture(*args: Any, **kwargs: Any) -> None:
            captured["pre_fit"] = _read_training_config(config_path)

        mock_bt.return_value.fit.side_effect = _capture
        RFDETR.train(mock_self)
        return mock_self, captured.get("pre_fit")

    def test_annotation_file_is_parsed_once_across_alignment_and_pre_fit(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture
    ) -> None:
        """Both readers go through one _filtered_coco_categories call instead of two back-to-back json.loads."""
        self._write_coco_categories(tmp_path / "ds", categories=self._CATEGORIES, annotated_ids=[2, 3])
        parse_mock = mocker.patch.object(
            RFDETR, "_filtered_coco_categories", side_effect=RFDETR._filtered_coco_categories
        )
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

    def test_memo_returns_the_cached_entry_without_reparsing(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A second lookup of the same directory is served from the memo."""
        dataset_dir = tmp_path / "ds"
        self._write_coco_categories(dataset_dir, categories=self._CATEGORIES, annotated_ids=[2, 3])
        cache: dict[str, list[dict[str, Any]]] = {}
        first = RFDETR._memoized_coco_categories(cache, str(dataset_dir))
        mocker.patch.object(RFDETR, "_filtered_coco_categories", side_effect=AssertionError("re-parsed"))

        second = RFDETR._memoized_coco_categories(cache, str(dataset_dir))
        assert second is first


class TestTrainAlreadyTrainedWarning:
    """Calling ``train()`` on an already-trained/loaded model emits a loud warning."""

    def test_second_train_warns(self, monkeypatch) -> None:
        """A model flagged as trained warns that training restarts from pretrain_weights."""
        model = _DummyRFDETR()
        model._has_been_trained = True

        class _StopTrainError(Exception):
            pass

        def _stub_get_train_config(self, **kwargs: object) -> None:
            raise _StopTrainError

        # Short-circuit train() right after the warning so no real training runs.
        monkeypatch.setattr(RFDETR, "get_train_config", _stub_get_train_config, raising=False)
        with pytest.warns(UserWarning) as record:
            with pytest.raises(_StopTrainError):
                model.train()
        assert any("already been trained" in str(warning.message) for warning in record)
