# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for build_trainer() — PTL Ch3/T5 (callbacks) and Ch4/T1 (precision, loggers, trainer kwargs)."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from rfdetr.config import (
    KeypointTrainConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
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

    Loggers are disabled by default to avoid requiring optional deps (tensorboard, wandb, mlflow) in the CPU test
    environment.  Logger-specific tests override these explicitly via kwargs or mocking.
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


def _kp_tc(tmp_path, **kwargs):
    """Minimal KeypointTrainConfig for tests that exercise keypoint model paths."""
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
    return KeypointTrainConfig(**defaults)


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

    def test_coco_eval_uses_keypoint_oks_sigmas(self, tmp_path):
        """COCOEvalCallback receives custom keypoint OKS sigmas from TrainConfig."""
        sigmas = [0.05] * 25
        trainer = build_trainer(
            _kp_tc(tmp_path, use_ema=False, keypoint_oks_sigmas=sigmas),
            RFDETRKeypointPreviewConfig(pretrain_weights=None),
        )
        coco_cb = next(cb for cb in trainer.callbacks if isinstance(cb, COCOEvalCallback))
        assert coco_cb._keypoint_oks_sigmas == sigmas

    def test_best_model_always_present(self, tmp_path):
        """BestModelCallback is always included."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert BestModelCallback in types

    def test_skip_best_epochs_forwarded_to_best_model_callback(self, tmp_path):
        """BestModelCallback receives skip_best_epochs from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, skip_best_epochs=3), _mc())
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb._skip_best_epochs == 3

    def test_keypoint_best_model_monitors_keypoint_map(self, tmp_path):
        """Keypoint training checkpoints should rank models by keypoint AP, not bbox mAP."""
        trainer = build_trainer(_kp_tc(tmp_path, use_ema=True), RFDETRKeypointPreviewConfig(pretrain_weights=None))
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb.monitor == "val/keypoint_map_50_95"
        assert best_cb._monitor_ema == "val/ema_keypoint_map_50_95"

    def test_segmentation_best_model_monitors_segmentation_map(self, tmp_path):
        """Segmentation training checkpoints should rank models by segmentation AP, not bbox AP."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True), _mc(segmentation_head=True))
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb.monitor == "val/segm_mAP_50_95"
        assert best_cb._monitor_ema == "val/ema_segm_mAP_50_95"

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

    def test_skip_best_epochs_forwarded_to_early_stopping(self, tmp_path):
        """RFDETREarlyStopping receives skip_best_epochs from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=True, skip_best_epochs=4), _mc())
        early_stop_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RFDETREarlyStopping))
        assert early_stop_cb._skip_best_epochs == 4

    def test_keypoint_early_stopping_monitors_keypoint_map(self, tmp_path):
        """Keypoint early stopping should use keypoint AP as the regular metric."""
        trainer = build_trainer(
            _kp_tc(tmp_path, early_stopping=True, early_stopping_use_ema=True),
            RFDETRKeypointPreviewConfig(pretrain_weights=None),
        )
        early_stop_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RFDETREarlyStopping))
        assert early_stop_cb._monitor_regular == "val/keypoint_map_50_95"
        assert early_stop_cb._monitor_ema == "val/ema_keypoint_map_50_95"

    def test_segmentation_early_stopping_monitors_segmentation_map(self, tmp_path):
        """Segmentation early stopping should use segmentation AP as the regular metric."""
        trainer = build_trainer(
            _tc(tmp_path, early_stopping=True, early_stopping_use_ema=True),
            _mc(segmentation_head=True),
        )
        early_stop_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RFDETREarlyStopping))
        assert early_stop_cb._monitor_regular == "val/segm_mAP_50_95"
        assert early_stop_cb._monitor_ema == "val/ema_segm_mAP_50_95"

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


class TestBuildTrainerKeypointDefaults:
    """Verify build_trainer() applies keypoint-specific defaults for noisy fine-tuning metrics."""

    def test_keypoint_default_skip_best_epochs_is_ten(self, tmp_path):
        """KeypointTrainConfig defaults skip_best_epochs to 10; build_trainer forwards it to callbacks."""
        trainer = build_trainer(
            _kp_tc(tmp_path, use_ema=False, early_stopping=True),
            RFDETRKeypointPreviewConfig(pretrain_weights=None),
        )
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        early_stop_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RFDETREarlyStopping))
        assert best_cb._skip_best_epochs == 10
        assert early_stop_cb._skip_best_epochs == 10

    def test_keypoint_explicit_skip_best_epochs_overrides_default(self, tmp_path):
        """An explicitly-set skip_best_epochs on a keypoint config overrides the class default of 10."""
        trainer = build_trainer(
            _kp_tc(tmp_path, use_ema=False, skip_best_epochs=3),
            RFDETRKeypointPreviewConfig(pretrain_weights=None),
        )
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb._skip_best_epochs == 3

    def test_non_keypoint_default_skip_best_epochs_is_zero(self, tmp_path):
        """For detection models, skip_best_epochs default remains 0."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb._skip_best_epochs == 0

    def test_keypoint_smooth_alpha_is_half(self, tmp_path):
        """BestModelCallback receives smooth_alpha=0.5 for keypoint models to dampen noisy mAP swings."""
        trainer = build_trainer(_kp_tc(tmp_path, use_ema=False), RFDETRKeypointPreviewConfig(pretrain_weights=None))
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb._smooth_alpha == pytest.approx(0.5)

    def test_non_keypoint_smooth_alpha_is_zero(self, tmp_path):
        """Detection / segmentation BestModelCallback keeps smooth_alpha=0.0 (no smoothing)."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        best_cb = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert best_cb._smooth_alpha == 0.0


class TestBuildTrainerPrecision:
    """build_trainer() must resolve training precision from model_config.amp + device caps."""

    def test_amp_false_gives_32_true(self, tmp_path):
        """Amp=False always produces '32-true' regardless of device."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_amp_true_cpu_gives_32_true(self, tmp_path):
        """Amp=True on CPU (no CUDA, no MPS) must fall back to '32-true'."""
        import unittest.mock as mock

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert trainer.precision == "32-true"

    def test_amp_true_explicit_cpu_accelerator_gives_32_true_even_with_mps(self, tmp_path):
        """Amp=True with explicit accelerator='cpu' must produce '32-true' even when MPS is present.

        bf16 autocast on macOS CPU (Apple Silicon) is ~13x slower than fp32 — no hardware support for bfloat16 in CPU
        kernels causes software emulation.  When the caller explicitly opts into CPU (e.g. for test isolation), mixed
        precision must not be used.
        """
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
        ):
            build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator="cpu")
        assert captured["precision"] == "32-true"

    def test_amp_true_cuda_no_bf16_gives_16_mixed(self, tmp_path):
        """Amp=True with CUDA but no bf16 support must produce '16-mixed'."""
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
        """Amp=True with CUDA + bf16 hardware produces 'bf16-mixed'."""
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

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.is_bf16_supported", return_value=False)
    @patch("rfdetr.training.trainer.Trainer")
    def test_amp_true_ddp_notebook_probes_bf16_normally(
        self, mock_trainer: MagicMock, _mock_bf16: MagicMock, _mock_cuda: MagicMock, tmp_path
    ):
        """ddp_notebook uses standard precision probing (spawn makes CUDA init safe).

        With spawn-based DDP, child processes start fresh — CUDA init in the parent does not propagate.  So
        ``is_bf16_supported()`` is safe to call and pre-Ampere GPUs correctly get ``16-mixed`` instead of the slower
        bf16 emulation path.  Simulates pre-Ampere GPU: CUDA available, bf16 NOT supported.
        """
        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mock_trainer.side_effect = _fake_trainer
        build_trainer(
            _tc(tmp_path, use_ema=False, strategy="ddp_notebook"),
            _mc(amp=True),
        )
        assert captured["precision"] == "16-mixed"

    @pytest.mark.parametrize("strategy_name", ["ddp_notebook", "ddp_spawn"])
    def test_ddp_notebook_and_spawn_use_interactive_spawn(self, tmp_path, strategy_name):
        """ddp_notebook and ddp_spawn must be replaced with interactive spawn DDPStrategy.

        Fork-based DDP inherits the parent's OpenMP thread pool which is invalid after fork, causing SIGABRT in the
        autograd engine. ddp_spawn is blocked by PTL in notebooks without the override.
        """
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(
                _tc(tmp_path, use_ema=False, strategy=strategy_name),
                _mc(amp=True),
            )
        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._start_method == "spawn"
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    @patch("rfdetr.training.trainer._InteractiveSpawnLauncher", None)
    def test_ddp_notebook_raises_clear_error_when_private_launcher_is_missing(self, tmp_path):
        """Missing private PTL launcher should raise a targeted compatibility error."""
        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(
                _tc(tmp_path, use_ema=False, strategy="ddp_notebook"),
                _mc(amp=True),
            )

        strategy = captured["strategy"]
        strategy.cluster_environment = object()
        with pytest.raises(RuntimeError, match="private API"):
            strategy._configure_launcher()


class TestBuildTrainerAmpDtype:
    """``TrainConfig.amp_dtype`` (a ``train()`` kwarg) lets callers pin the AMP autocast dtype (fp16 vs bf16) — #1132.

    Precision is resolved inside ``build_trainer``; these tests mock the CUDA/MPS capability probes and assert the
    Lightning precision string captured at ``Trainer`` construction time.
    """

    @staticmethod
    def _resolved_precision(tmp_path, *, cuda: bool, bf16: bool = False, mps: bool = False, amp_dtype: str = "auto"):
        """Resolve the Lightning precision string for a mocked device capability and ``amp_dtype``.

        Args:
            tmp_path: pytest temporary directory fixture.
            cuda: Value returned by the mocked ``torch.cuda.is_available``.
            bf16: Value returned by the mocked ``torch.cuda.is_bf16_supported``.
            mps: Value returned by the mocked ``torch.backends.mps.is_available``.
            amp_dtype: The ``TrainConfig.amp_dtype`` value under test.

        Returns:
            The ``precision`` string passed to the (mocked) ``Trainer``.
        """
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with (
            mock.patch("torch.cuda.is_available", return_value=cuda),
            mock.patch("torch.cuda.is_bf16_supported", return_value=bf16),
            mock.patch("torch.backends.mps.is_available", return_value=mps),
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
        ):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype=amp_dtype), _mc(amp=True))
        return captured["precision"]

    def test_amp_dtype_is_a_train_kwarg_not_dropped(self, tmp_path):
        """amp_dtype is a real TrainConfig field (reachable via train(**kwargs)), not silently dropped."""
        assert _tc(tmp_path, amp_dtype="fp16").amp_dtype == "fp16"

    @pytest.mark.parametrize(
        "cuda, bf16, mps, amp_dtype, expected",
        [
            pytest.param(True, True, False, "auto", "bf16-mixed", id="auto-cuda-bf16"),
            pytest.param(True, True, False, "fp16", "16-mixed", id="fp16-cuda-bf16"),
            pytest.param(True, True, False, "bf16", "bf16-mixed", id="bf16-cuda-bf16"),
            pytest.param(True, False, False, "auto", "16-mixed", id="auto-cuda-no-bf16"),
            pytest.param(False, False, True, "fp16", "16-mixed", id="fp16-mps"),
        ],
    )
    def test_resolved_precision(self, tmp_path, cuda, bf16, mps, amp_dtype, expected):
        """amp_dtype + hardware caps resolve to the correct Lightning precision string."""
        assert self._resolved_precision(tmp_path, cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype) == expected

    @pytest.mark.parametrize(
        "cuda, bf16, mps, amp_dtype, warn_match",
        [
            pytest.param(True, False, False, "bf16", "bf16", id="bf16-cuda-no-hw-support"),
            pytest.param(False, False, True, "bf16", "MPS", id="bf16-mps"),
        ],
    )
    def test_resolved_precision_warns(self, tmp_path, cuda, bf16, mps, amp_dtype, warn_match):
        """amp_dtype falls back to '16-mixed' and emits a UserWarning when hardware cannot satisfy the request."""
        with pytest.warns(UserWarning, match=warn_match):
            precision = self._resolved_precision(tmp_path, cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype)
        assert precision == "16-mixed"

    def test_amp_false_overrides_amp_dtype(self, tmp_path):
        """Amp=False wins over any amp_dtype: precision is '32-true'."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp16"), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_cpu_accelerator_ignores_amp_dtype(self, tmp_path):
        """Explicit accelerator='cpu' yields '32-true' regardless of amp_dtype."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp16"), _mc(amp=True), accelerator="cpu")
        assert captured["precision"] == "32-true"

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param("float8", id="string-float8"),
            pytest.param(None, id="none"),
            pytest.param(42, id="int"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_invalid_amp_dtype_falls_back_to_auto_with_warning(self, tmp_path, bad_value):
        """An unrecognised or wrong-typed amp_dtype falls back to 'auto' with a warning rather than raising."""
        with pytest.warns(UserWarning, match="amp_dtype"):
            tc = _tc(tmp_path, amp_dtype=bad_value)
        assert tc.amp_dtype == "auto"


class TestBuildTrainerEMAShardingGuard:
    """EMA must be disabled and a UserWarning emitted for sharded strategies.

    PTL validates strategy+accelerator compatibility at Trainer construction time, so tests that exercise sharded
    strategies mock Trainer to capture the callback list without triggering platform-specific validation.
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
        with (
            mock.patch("rfdetr.training.trainer._try_import_tensorboard_summary_writer"),
            mock.patch("rfdetr.training.trainer.TensorBoardLogger", return_value=fake_logger),
        ):
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

        with mock.patch(
            "rfdetr.training.trainer._try_import_tensorboard_summary_writer",
            side_effect=ModuleNotFoundError("no module named 'tensorboard'"),
        ):
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

    def test_numpy2_tensorboard_incompatibility_warns_not_crashes(self, tmp_path):
        """AttributeError from NumPy 2.0/tensorflow incompatibility falls back to CSV logger."""
        import unittest.mock as mock

        numpy2_error = AttributeError("`np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.")
        with mock.patch(
            "rfdetr.training.trainer._try_import_tensorboard_summary_writer",
            side_effect=numpy2_error,
        ):
            with mock.patch("rfdetr.training.trainer._logger") as mock_logger:
                trainer = build_trainer(
                    _tc(tmp_path, tensorboard=True, use_ema=False),
                    _mc(),
                )
        mock_logger.warning.assert_called_once()
        assert "TensorBoard" in mock_logger.warning.call_args[0][0]
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

        assert all(not isinstance(lg, TensorBoardLogger) for lg in trainer.loggers)
        assert any(isinstance(lg, CSVLogger) for lg in trainer.loggers)

    def test_clearml_flag_raises_not_implemented(self, tmp_path):
        """Clearml=True must raise NotImplementedError (not yet supported)."""
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
            mock.patch("rfdetr.training.trainer._try_import_tensorboard_summary_writer"),
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

    def test_gradient_clip_val_disabled_for_keypoint_manual_optimization(self, tmp_path):
        """Trainer-owned clipping is disabled for keypoint models because RFDETRModelModule clips manually."""
        trainer = build_trainer(
            _kp_tc(tmp_path, use_ema=False, clip_max_norm=0.25),
            _mc(use_grouppose_keypoints=True),
        )
        assert trainer.gradient_clip_val is None

    def test_gradient_clip_val_forwarded_for_detection_automatic_optimization(self, tmp_path):
        """Detection models use Lightning's automatic optimization; trainer-owned clipping must flow through."""
        trainer = build_trainer(
            _tc(tmp_path, use_ema=False, clip_max_norm=0.25),
            _mc(),
        )
        assert trainer.gradient_clip_val == pytest.approx(0.25)

    def test_accumulate_grad_batches_disabled_for_keypoint_manual_optimization(self, tmp_path):
        """Trainer-owned accumulation is disabled for keypoint models because RFDETRModelModule accumulates manually."""
        trainer = build_trainer(
            _kp_tc(tmp_path, grad_accum_steps=8, use_ema=False),
            _mc(use_grouppose_keypoints=True),
        )
        assert trainer.accumulate_grad_batches == 1

    def test_accumulate_grad_batches_forwarded_for_detection_automatic_optimization(self, tmp_path):
        """Detection models use Lightning's automatic optimization; ``accumulate_grad_batches`` must flow through."""
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

    def test_keypoint_trainer_kwargs_cannot_override_manual_optimization_ownership(self, tmp_path):
        """Keypoint accumulation and clipping remain disabled even when passed as trainer kwargs, and the override emits
        a UserWarning so the caller can spot the silent coercion."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with (
            mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer),
            pytest.warns(UserWarning, match="manual optimization"),
        ):
            build_trainer(
                _kp_tc(tmp_path, use_ema=False),
                _mc(use_grouppose_keypoints=True),
                accumulate_grad_batches=8,
                gradient_clip_val=0.25,
            )

        assert captured["accumulate_grad_batches"] == 1
        assert captured["gradient_clip_val"] is None

    def test_detection_trainer_kwargs_override_takes_effect(self, tmp_path):
        """Detection models use automatic optimization; trainer kwargs must override the built-in defaults."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(
                _tc(tmp_path, use_ema=False),
                _mc(),
                accumulate_grad_batches=8,
                gradient_clip_val=0.25,
            )

        assert captured["accumulate_grad_batches"] == 8
        assert captured["gradient_clip_val"] == pytest.approx(0.25)


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

        tc = _tc(tmp_path, use_ema=False, strategy="auto")
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, _mc())

        assert captured["strategy"] == "auto"

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


class TestBuildTrainerKeypointDistributedGuard:
    """Keypoint mode must fail fast for unsupported distributed training settings."""

    def test_keypoint_ddp_strategy_raises_clear_error(self, tmp_path):
        """Keypoint mode rejects explicit distributed strategy requests with a clear error."""
        tc = _kp_tc(tmp_path, use_ema=False, strategy="ddp")
        mc = _mc(use_grouppose_keypoints=True)

        with pytest.raises(NotImplementedError, match="Keypoint training currently does not support distributed"):
            build_trainer(tc, mc)

    def test_keypoint_auto_devices_raises_when_cuda_has_multiple_devices(self, tmp_path):
        """Keypoint mode rejects devices='auto' when it would resolve to multi-GPU execution."""
        tc = _kp_tc(tmp_path, use_ema=False, devices="auto")
        mc = _mc(use_grouppose_keypoints=True)

        with (
            patch("rfdetr.training.trainer.torch.cuda.is_available", return_value=True),
            patch("rfdetr.training.trainer.torch.cuda.device_count", return_value=2),
            pytest.raises(NotImplementedError, match="Keypoint training currently does not support distributed"),
        ):
            build_trainer(tc, mc)

    def test_non_keypoint_ddp_strategy_wrapped_with_find_unused_parameters(self, tmp_path):
        """Non-keypoint mode with strategy='ddp' produces DDPStrategy(find_unused_parameters=True)."""
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="ddp")
        mc = _mc(use_grouppose_keypoints=False)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True


class TestBuildTrainerDDPFindUnusedParameters:
    """build_trainer() must enable find_unused_parameters for strategy='ddp' on both detection and segmentation."""

    def test_auto_strategy_multiple_devices_enables_find_unused_parameters(self, tmp_path):
        """Strategy='auto' + devices > 1 must produce DDPStrategy(find_unused_parameters=True).

        This covers the default strategy path where Lightning would otherwise select a distributed strategy without RF-
        DETR's unused-parameter guard.
        """
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="auto", devices=2)
        mc = _mc(segmentation_head=False)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True
        assert captured["devices"] == 2

    def test_ddp_segmentation_enables_find_unused_parameters(self, tmp_path):
        """Strategy='ddp' + segmentation_head=True must produce DDPStrategy(find_unused_parameters=True).

        One case of the broader unconditional rule: find_unused_parameters is enabled for all strategy='ddp'
        requests.  The segmentation head's sparse_forward() is one source of conditionally-unused parameters under
        DDP.
        """
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="ddp")
        mc = _mc(segmentation_head=True)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    def test_ddp_no_segmentation_enables_find_unused_parameters(self, tmp_path):
        """Strategy='ddp' for detection-only must produce DDPStrategy(find_unused_parameters=True).

        Detection models can leave parameters unused under DDP (two-stage group_detr ModuleLists, conditional aux_loss
        branches), so find_unused_parameters is enabled unconditionally for strategy='ddp' regardless of
        segmentation_head. Regression test for
        https://github.com/roboflow/rf-detr/issues/1093.
        """
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="ddp")
        mc = _mc(segmentation_head=False)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    def test_ddp_spawn_segmentation_preserves_find_unused_parameters(self, tmp_path):
        """strategy='ddp_spawn' + segmentation_head=True must keep find_unused_parameters=True.

        ddp_spawn is already replaced with an interactive-spawn DDPStrategy that has find_unused_parameters=True for
        notebook compatibility.  Segmentation must not accidentally drop that flag when the ddp_spawn path is taken
        instead of the plain 'ddp' path.
        """
        import unittest.mock as mock

        from pytorch_lightning.strategies import DDPStrategy

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="ddp_spawn")
        mc = _mc(segmentation_head=True)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        strategy_obj = captured["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    def test_non_ddp_strategy_with_segmentation_is_unchanged(self, tmp_path):
        """Strategies other than 'ddp' must not be wrapped even when segmentation is on."""
        import unittest.mock as mock

        captured: dict = {}

        def _fake_trainer(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        tc = _tc(tmp_path, use_ema=False, strategy="auto")
        mc = _mc(segmentation_head=True)
        with mock.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer):
            build_trainer(tc, mc)

        assert captured["strategy"] == "auto"
