# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for build_trainer() — PTL Ch3/T5 (callbacks) and Ch4/T1 (precision, loggers, trainer kwargs)."""

import warnings

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.drop_schedule import DropPathCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback


def _mc(**kwargs):
    """Minimal RFDETRBaseConfig for tests."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kwargs)
    return RFDETRBaseConfig(**defaults)


def _find_resume_checkpoints(trainer):
    """Return ModelCheckpoint callbacks that are NOT BestModelCallback."""
    return [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint) and not isinstance(cb, BestModelCallback)]


def _tc(tmp_path, **kwargs):
    """Minimal TrainConfig for tests.

    Loggers are disabled by default to avoid requiring optional deps (tensorboard,
    wandb, mlflow) in the CPU test environment.  Logger-specific tests override these
    explicitly via kwargs or mocking.
    """
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


class TestBuildTrainerReturnType:
    """build_trainer() must return a PTL Trainer."""

    def test_returns_trainer_instance(self, tmp_path):
        """Return value must be a pytorch_lightning.Trainer."""
        from pytorch_lightning import Trainer

        trainer = build_trainer(_tc(tmp_path), _mc())
        assert isinstance(trainer, Trainer)


class TestBuildTrainerCallbacks:
    """build_trainer() must wire the correct callback set."""

    def test_coco_eval_always_present(self, tmp_path):
        """COCOEvalCallback is always included regardless of config flags."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, early_stopping=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert COCOEvalCallback in types

    def test_coco_eval_uses_eval_interval_and_per_class_flags(self, tmp_path):
        """COCOEvalCallback receives eval_interval and log_per_class_metrics from TrainConfig."""
        trainer = build_trainer(
            _tc(tmp_path, use_ema=False, eval_interval=3, log_per_class_metrics=False),
            _mc(),
        )
        coco_cb = next(cb for cb in trainer.callbacks if isinstance(cb, COCOEvalCallback))
        assert coco_cb._eval_interval == 3
        assert coco_cb._log_per_class_metrics is False

    def test_best_model_always_present(self, tmp_path):
        """BestModelCallback is always included."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert BestModelCallback in types

    def test_latest_model_checkpoint_present(self, tmp_path):
        """A ModelCheckpoint (not BestModelCallback) with every_n_epochs==1 is included when checkpoint_interval > 1."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=2), _mc())
        resume_cbs = _find_resume_checkpoints(trainer)
        assert any(cb._every_n_epochs == 1 for cb in resume_cbs)

    def test_latest_model_checkpoint_absent_when_checkpoint_interval_one(self, tmp_path):
        """No separate latest checkpoint callback when interval already saves every epoch."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=1), _mc())
        resume_cbs = _find_resume_checkpoints(trainer)
        assert resume_cbs
        assert not any(cb._every_n_epochs == 1 and cb.save_top_k == 1 for cb in resume_cbs)
        interval_cb = next(
            (cb for cb in resume_cbs if cb._every_n_epochs == 1 and cb.save_top_k == -1),
            None,
        )
        assert interval_cb is not None
        assert interval_cb.filename == "checkpoint_{epoch}"
        assert str(interval_cb.dirpath) == str(tmp_path / "out")

    def test_interval_model_checkpoint_present(self, tmp_path):
        """A ModelCheckpoint (not BestModelCallback) with every_n_epochs==checkpoint_interval is always included."""
        tc = _tc(tmp_path, use_ema=False)
        trainer = build_trainer(tc, _mc())
        resume_cbs = _find_resume_checkpoints(trainer)
        assert any(cb._every_n_epochs == tc.checkpoint_interval for cb in resume_cbs)

    def test_checkpoint_interval_one_has_single_resume_checkpoint_callback(self, tmp_path):
        """checkpoint_interval=1 config creates only one non-best ModelCheckpoint callback."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=1), _mc())
        resume_cbs = _find_resume_checkpoints(trainer)
        assert len(resume_cbs) == 1
        only_cb = resume_cbs[0]
        assert only_cb._every_n_epochs == 1
        assert only_cb.save_top_k == -1

    @pytest.mark.parametrize(
        "checkpoint_interval",
        [
            pytest.param(1, id="interval_1"),
            pytest.param(2, id="interval_2"),
            pytest.param(7, id="interval_7"),
        ],
    )
    def test_all_model_checkpoints_have_unique_state_keys(self, tmp_path, checkpoint_interval):
        """All ModelCheckpoint callbacks (including BestModelCallback) always have unique state keys."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=checkpoint_interval), _mc())
        all_mc_cbs = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
        state_keys = [cb.state_key for cb in all_mc_cbs]
        assert len(state_keys) == len(set(state_keys)), (
            f"Duplicate state_key with checkpoint_interval={checkpoint_interval}: "
            f"{[k for k in state_keys if state_keys.count(k) > 1]}"
        )

    def test_interval_checkpoint_uses_interval_from_config(self, tmp_path):
        """Interval ModelCheckpoint receives checkpoint_interval=7 from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=7), _mc())
        resume_cbs = _find_resume_checkpoints(trainer)
        assert any(cb._every_n_epochs == 7 for cb in resume_cbs)

    def test_checkpoint_interval_validation(self, tmp_path):
        """TrainConfig(checkpoint_interval=0) raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _tc(tmp_path, checkpoint_interval=0)

    def test_ema_callback_when_use_ema_true(self, tmp_path):
        """RFDETREMACallback is added when use_ema=True."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback in types

    def test_ema_callback_uses_update_interval(self, tmp_path):
        """RFDETREMACallback receives ema_update_interval from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True, ema_update_interval=4), _mc())
        ema_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RFDETREMACallback))
        assert ema_cb._update_interval_steps == 4

    def test_no_ema_callback_when_use_ema_false(self, tmp_path):
        """RFDETREMACallback is absent when use_ema=False."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback not in types

    def test_drop_path_callback_when_drop_path_nonzero(self, tmp_path):
        """DropPathCallback is added when drop_path > 0."""
        trainer = build_trainer(_tc(tmp_path, drop_path=0.1), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert DropPathCallback in types

    def test_no_drop_path_callback_when_drop_path_zero(self, tmp_path):
        """DropPathCallback is absent when drop_path == 0."""
        trainer = build_trainer(_tc(tmp_path, drop_path=0.0), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert DropPathCallback not in types

    def test_early_stopping_when_enabled(self, tmp_path):
        """RFDETREarlyStopping is added when early_stopping=True."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREarlyStopping in types

    def test_no_early_stopping_when_disabled(self, tmp_path):
        """RFDETREarlyStopping is absent when early_stopping=False."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREarlyStopping not in types

    def test_segmentation_config_accepted(self, tmp_path):
        """SegmentationTrainConfig is accepted without error."""
        seg_tc = SegmentationTrainConfig(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
        )
        trainer = build_trainer(seg_tc, _mc(segmentation_head=True))
        assert isinstance(trainer, __import__("pytorch_lightning").Trainer)


class TestBuildTrainerPrecision:
    """build_trainer() must resolve training precision from model_config.amp + device caps."""

    def test_amp_false_gives_32_true(self, tmp_path):
        """amp=False always produces '32-true' regardless of device."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_amp_true_cpu_gives_32_true(self, tmp_path):
        """amp=True on CPU (no CUDA, no MPS) must fall back to '32-true'."""
        import unittest.mock as mock

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert trainer.precision == "32-true"

    def test_amp_true_cuda_no_bf16_gives_16_mixed(self, tmp_path):
        """amp=True with CUDA but no bf16 support must produce '16-mixed'.

        We mock the Trainer constructor to capture the precision kwarg rather
        than inspecting trainer.precision after construction: PTL may re-detect
        hardware bf16 support during __init__ and normalise the precision string
        on machines that happen to have a bf16-capable GPU.
        """
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_bf16_supported", return_value=False),
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
        ):
            build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert captured["precision"] == "16-mixed"

    def test_amp_true_cuda_bf16_supported_gives_bf16_mixed(self, tmp_path):
        """amp=True with CUDA + bf16 hardware produces 'bf16-mixed' (scaler-free).

        On Ampere+ GPUs (bf16 supported) we select bf16-mixed to eliminate
        GradScaler overhead.  Fine-tuning from pretrained weights is safe with
        BF16; callers training from scratch can override via trainer_kwargs.
        """
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_bf16_supported", return_value=True),
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
        ):
            build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert captured["precision"] == "bf16-mixed"


class TestBuildTrainerEMAShardingGuard:
    """EMA must be disabled and a UserWarning emitted for sharded strategies.

    PTL validates strategy+accelerator compatibility at Trainer construction time,
    so tests that exercise sharded strategies mock Trainer to capture the callback
    list without triggering platform-specific validation.
    """

    @pytest.mark.parametrize(
        "strategy",
        [
            pytest.param("fsdp", id="fsdp"),
            pytest.param("deepspeed", id="deepspeed"),
            pytest.param("deepspeed_stage_2", id="deepspeed_stage_2"),
        ],
    )
    def test_ema_disabled_for_sharded_strategy(self, tmp_path, strategy):
        """EMA callback must be absent when a sharded strategy is requested."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=True)
        # Inject strategy via monkey-patch (field not yet in TrainConfig until T4-2).
        tc.__dict__["strategy"] = strategy

        captured_callbacks = []

        def _fake_trainer(**kwargs):
            captured_callbacks.extend(kwargs.get("callbacks", []))
            return mock.MagicMock()

        with (
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            build_trainer(tc, _mc())

        types = [type(cb) for cb in captured_callbacks]
        assert RFDETREMACallback not in types

    def test_ema_sharding_emits_user_warning(self, tmp_path):
        """A UserWarning is emitted when EMA is requested with a sharded strategy."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=True)
        tc.__dict__["strategy"] = "fsdp"

        with (
            mock.patch("rfdetr.training.trainer.Trainer", return_value=mock.MagicMock()),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            build_trainer(tc, _mc())

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("EMA disabled" in str(w.message) for w in user_warns)

    def test_ema_enabled_for_non_sharded_strategy(self, tmp_path):
        """EMA callback must be present for non-sharded strategies."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback in types


class TestBuildTrainerLoggers:
    """build_trainer() must wire loggers from TrainConfig flags."""

    def test_no_loggers_always_has_csv_logger(self, tmp_path):
        """CSVLogger is always present even when all optional logger flags are off."""
        from pytorch_lightning.loggers import CSVLogger

        trainer = build_trainer(
            _tc(tmp_path, use_ema=False),  # _tc already sets all loggers to False
            _mc(),
        )
        assert any(isinstance(lg, CSVLogger) for lg in trainer.loggers)

    def test_tensorboard_logger_wired(self, tmp_path):
        """TensorBoardLogger is added when tensorboard=True (dep mocked)."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import TensorBoardLogger

        fake_logger = mock.MagicMock(spec=TensorBoardLogger)
        with mock.patch("rfdetr.training.trainer.TensorBoardLogger", return_value=fake_logger):
            trainer = build_trainer(
                _tc(tmp_path, tensorboard=True, use_ema=False),
                _mc(),
            )
        assert fake_logger in trainer.loggers

    def test_mlflow_logger_wired(self, tmp_path):
        """MLFlowLogger is added when mlflow=True (dep mocked)."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import MLFlowLogger

        fake_logger = mock.MagicMock(spec=MLFlowLogger)
        with mock.patch("rfdetr.training.trainer.MLFlowLogger", return_value=fake_logger):
            trainer = build_trainer(
                _tc(tmp_path, mlflow=True, use_ema=False),
                _mc(),
            )
        assert fake_logger in trainer.loggers

    def test_missing_tensorboard_dep_warns_not_crashes(self, tmp_path):
        """If tensorboard package is absent, a warning is logged and training continues."""
        import unittest.mock as mock

        with mock.patch("rfdetr.training.trainer.TensorBoardLogger", side_effect=ModuleNotFoundError("no tensorboard")):
            with mock.patch("rfdetr.training.trainer._logger") as mock_logger:
                trainer = build_trainer(
                    _tc(tmp_path, tensorboard=True, use_ema=False),
                    _mc(),
                )
        mock_logger.warning.assert_called_once()
        assert "TensorBoard" in mock_logger.warning.call_args[0][0]
        # CSVLogger is always present; TensorBoard was not added due to missing dep
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

        assert all(not isinstance(lg, TensorBoardLogger) for lg in trainer.loggers)
        assert any(isinstance(lg, CSVLogger) for lg in trainer.loggers)

    def test_clearml_flag_raises_not_implemented(self, tmp_path):
        """clearml=True must raise NotImplementedError (not yet supported)."""
        with pytest.raises(NotImplementedError, match="ClearML"):
            build_trainer(
                _tc(tmp_path, clearml=True, use_ema=False),
                _mc(),
            )

    def test_multiple_loggers_combined(self, tmp_path):
        """Multiple loggers can be wired simultaneously."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

        fake_tb = mock.MagicMock(spec=TensorBoardLogger)
        fake_mlflow = mock.MagicMock(spec=MLFlowLogger)
        with (
            mock.patch("rfdetr.training.trainer.TensorBoardLogger", return_value=fake_tb),
            mock.patch("rfdetr.training.trainer.MLFlowLogger", return_value=fake_mlflow),
        ):
            trainer = build_trainer(
                _tc(tmp_path, tensorboard=True, mlflow=True, use_ema=False),
                _mc(),
            )
        assert fake_tb in trainer.loggers
        assert fake_mlflow in trainer.loggers


class TestBuildTrainerKwargs:
    """build_trainer() must pass the correct kwargs to Trainer."""

    def test_gradient_clip_val_default(self, tmp_path):
        """gradient_clip_val defaults to 0.1 when clip_max_norm is not yet in TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        assert trainer.gradient_clip_val == pytest.approx(0.1)

    def test_accumulate_grad_batches(self, tmp_path):
        """accumulate_grad_batches maps from grad_accum_steps."""
        trainer = build_trainer(_tc(tmp_path, grad_accum_steps=8, use_ema=False), _mc())
        assert trainer.accumulate_grad_batches == 8

    def test_max_epochs(self, tmp_path):
        """max_epochs maps from config.epochs."""
        trainer = build_trainer(_tc(tmp_path, epochs=42, use_ema=False), _mc())
        assert trainer.max_epochs == 42

    def test_log_every_n_steps(self, tmp_path):
        """log_every_n_steps is fixed at 50."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        assert trainer.log_every_n_steps == 50

    def test_default_root_dir(self, tmp_path):
        """default_root_dir maps from config.output_dir."""
        out = str(tmp_path / "my_output")
        trainer = build_trainer(_tc(tmp_path, output_dir=out, use_ema=False), _mc())
        assert str(trainer.default_root_dir) == out

    def test_trainer_kwargs_can_override_precision(self, tmp_path):
        """Explicit trainer kwargs must override default precision without raising."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(
                _tc(tmp_path, use_ema=False),
                _mc(amp=True),
                precision="32-true",
            )
        assert captured["precision"] == "32-true"


class TestBuildTrainerSeed:
    """build_trainer() must not mutate global RNG state."""

    def test_seed_is_not_applied_in_factory(self, tmp_path):
        """Seeding is deferred to RFDETRModule.on_fit_start (no factory side-effect)."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=False, seed=42)

        with mock.patch("pytorch_lightning.seed_everything") as mock_seed:
            build_trainer(tc, _mc())
        mock_seed.assert_not_called()


class TestBuildTrainerDDPFields:
    """build_trainer() must thread devices/num_nodes/strategy from TrainConfig to Trainer."""

    def test_devices_threaded_from_train_config(self, tmp_path):
        """TrainConfig.devices is forwarded to Trainer(devices=...)."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, devices=4)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["devices"] == 4

    def test_num_nodes_threaded_from_train_config(self, tmp_path):
        """TrainConfig.num_nodes is forwarded to Trainer(num_nodes=...)."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, num_nodes=2)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["num_nodes"] == 2

    def test_strategy_threaded_from_train_config(self, tmp_path):
        """TrainConfig.strategy is forwarded to Trainer(strategy=...)."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="ddp")
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["strategy"] == "ddp"

    def test_default_devices_is_1(self, tmp_path):
        """Default TrainConfig.devices must produce devices=1 (single-GPU default)."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["devices"] == 1

    def test_default_num_nodes_is_1(self, tmp_path):
        """Default TrainConfig.num_nodes must produce num_nodes=1."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["num_nodes"] == 1

    def test_devices_string_accepted(self, tmp_path):
        """TrainConfig.devices accepts a string value (e.g. '0,1')."""
        tc = _tc(tmp_path, use_ema=False, devices="auto")
        # Should not raise during config construction.
        assert tc.devices == "auto"
