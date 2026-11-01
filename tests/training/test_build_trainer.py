# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for build_trainer() — PTL Ch3/T5 (callbacks) and Ch4/T1 (precision, loggers, trainer kwargs)."""

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from pytest_mock import MockerFixture
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import XLAAccelerator
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy, ModelParallelStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

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
from rfdetr.training.trainer import (
    _accelerator_resolves_to_xla,
    _ForceLastEpochValidationCallback,
    _requests_multiple_devices,
    _xla_resolves_to_single_device,
)
from rfdetr.utilities.imports import _IS_HOTCOCO_INSTALLED, _IS_TRANSFORMER_ENGINE_INSTALLED
from tests._markers import requires_torch_xla

hotcoco_only = pytest.mark.skipif(not _IS_HOTCOCO_INSTALLED, reason="hotcoco not installed")
transformer_engine_only = pytest.mark.skipif(
    not _IS_TRANSFORMER_ENGINE_INSTALLED, reason="requires the 'cuda' extra (transformer-engine)"
)


def _mc(**kwargs):
    """Minimal RFDETRBaseConfig for tests.

    Examples:
        >>> config = _mc(num_classes=7)
        >>> config.device, config.num_classes
        ('cpu', 7)
    """
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kwargs)
    return RFDETRBaseConfig(**defaults)


def _kp_preview_mc():
    """Keypoint preview model config without pretrained weights.

    Examples:
        >>> _kp_preview_mc().use_grouppose_keypoints
        True
    """
    return RFDETRKeypointPreviewConfig(pretrain_weights=None)


def _tc(tmp_path, config_cls=TrainConfig, **kwargs):
    """Minimal TrainConfig (or ``config_cls``) for tests.

    Loggers are disabled by default to avoid requiring optional deps (tensorboard, wandb, mlflow) in the CPU test
    environment.  Logger-specific tests override these explicitly via kwargs or mocking.

    Examples:
        >>> from pathlib import Path
        >>> config = _tc(Path("/tmp/example"), epochs=3)
        >>> config.epochs, Path(config.dataset_dir).name, Path(config.output_dir).name
        (3, 'ds', 'out')
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
    return config_cls(**defaults)


def _kp_tc(tmp_path, **kwargs):
    """Minimal KeypointTrainConfig for tests that exercise keypoint model paths.

    Examples:
        >>> from pathlib import Path
        >>> config = _kp_tc(Path('/tmp/example'), batch_size=4)
        >>> config.batch_size, Path(config.dataset_dir).name
        (4, 'ds')
    """
    return _tc(tmp_path, config_cls=KeypointTrainConfig, **kwargs)


def _configs(tmp_path, *, keypoint=False, segmentation=False, **tc_kwargs):
    """Return a ``(train_config, model_config)`` pair for a detection, segmentation or keypoint run without EMA.

    Examples:
        >>> from pathlib import Path
        >>> tc, mc = _configs(Path('/tmp/example'), keypoint=True, strategy='ddp')
        >>> type(tc).__name__, tc.strategy, mc.use_grouppose_keypoints
        ('KeypointTrainConfig', 'ddp', True)
    """
    tc_kwargs.setdefault("use_ema", False)
    if keypoint:
        return _kp_tc(tmp_path, **tc_kwargs), _mc(use_grouppose_keypoints=True)
    return _tc(tmp_path, **tc_kwargs), _mc(segmentation_head=segmentation)


def _find_callback(trainer, callback_type):
    """Return the first callback of ``callback_type`` wired into ``trainer``.

    Examples:
        >>> callbacks = [ModelCheckpoint(dirpath='.'), LearningRateMonitor()]
        >>> _find_callback(MagicMock(callbacks=callbacks), LearningRateMonitor) is callbacks[1]
        True
    """
    return next(cb for cb in trainer.callbacks if isinstance(cb, callback_type))


def _find_resume_checkpoints(trainer):
    """Return ModelCheckpoint callbacks that are NOT BestModelCallback.

    Examples:
        >>> resume_cb = ModelCheckpoint(dirpath='.')
        >>> best_cb = BestModelCallback(output_dir='.')
        >>> trainer = MagicMock(callbacks=[resume_cb, best_cb])
        >>> _find_resume_checkpoints(trainer) == [resume_cb]
        True
    """
    return [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint) and not isinstance(cb, BestModelCallback)]


@pytest.fixture
def captured_trainer_kwargs(mocker: MockerFixture) -> dict[str, Any]:
    """Patch ``rfdetr.training.trainer.Trainer`` and return the kwargs its calls capture.

    Every test in this module calls ``build_trainer()`` and inspects what it passed to ``Trainer(**kwargs)`` instead of
    constructing a real one; this fixture centralizes that capture in place of a per-test ``_fake_trainer`` closure.

    Examples:
        >>> captured_trainer_kwargs(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    captured: dict[str, Any] = {}

    def _fake_trainer(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    mocker.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer)
    return captured


@pytest.fixture
def mock_xla_precision(mocker: MockerFixture) -> MagicMock:
    """Replace ``XLAPrecision`` so XLA dispatch tests run without ``torch_xla``; returns the mocked class.

    Examples:
        >>> mock_xla_precision(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    return mocker.patch("pytorch_lightning.plugins.XLAPrecision")


@pytest.fixture
def bf16_cuda_host(mocker: MockerFixture) -> None:
    """Mock a CUDA host whose GPU natively supports bfloat16 (Ampere).

    Examples:
        >>> bf16_cuda_host(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
    mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))


@pytest.fixture
def resolved_precision(
    mocker: MockerFixture, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
) -> Callable[..., str]:
    """Return a resolver mapping a mocked device capability and ``amp_dtype`` to the Lightning precision string.

    The capability probes are mocked so the expected precision is a property of the config under test rather than of
    whichever machine runs the suite -- an unmocked call resolves to ``"16-mixed"`` on an MPS host and ``"32-true"`` on
    a CPU-only one. A callable is returned because the mocks are installed when the test chooses its hardware.

    Examples:
        >>> resolved_precision(mocker, captured_trainer_kwargs, tmp_path)  # doctest: +SKIP
        # Needs pytest's mocker/tmp_path fixtures; cannot run standalone.
    """

    def _resolve(
        *,
        cuda: bool,
        bf16: bool = False,
        bf16_emulated: bool = False,
        mps: bool = False,
        amp_dtype: str | None = "auto",
        amp: bool = True,
    ) -> str:
        """Resolve the precision for a mocked device and return the string the ``Trainer`` receives.

        Args:
            cuda: Value returned by the mocked ``torch.cuda.is_available``.
            bf16: Whether the mocked CUDA device supports bfloat16 natively.
            bf16_emulated: Whether ``torch.cuda.is_bf16_supported`` also reports bfloat16 through emulation, as PyTorch
                does on pre-Ampere GPUs such as the T4 (``including_emulation=True`` is its default).
            mps: Value returned by the mocked ``torch.backends.mps.is_available``.
            amp_dtype: The ``TrainConfig.amp_dtype`` value under test.
            amp: The deprecated ``ModelConfig.amp`` value under test.

        Returns:
            The ``precision`` string passed to the (mocked) ``Trainer``.
        """

        def _is_bf16_supported(including_emulation: bool = True) -> bool:
            return bf16 or (bf16_emulated and including_emulation)

        capability = (8, 0) if bf16 else (7, 5) if bf16_emulated else (6, 1)
        mocker.patch("torch.cuda.is_available", return_value=cuda)
        mocker.patch("torch.cuda.is_bf16_supported", side_effect=_is_bf16_supported)
        mocker.patch("torch.cuda.get_device_capability", return_value=capability)
        mocker.patch("torch.backends.mps.is_available", return_value=mps)
        mocker.patch("torch.version.hip", None)
        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype=amp_dtype), _mc(amp=amp))
        return captured_trainer_kwargs["precision"]

    return _resolve


class TestBuildTrainerReturnType:
    """build_trainer() must return a PTL Trainer."""

    def test_returns_trainer_instance(self, tmp_path):
        """Return value must be a pytorch_lightning.Trainer."""
        assert isinstance(build_trainer(_tc(tmp_path), _mc()), Trainer)


class TestBuildTrainerCallbacks:
    """build_trainer() must wire the correct callback set."""

    @pytest.mark.parametrize(
        ("tc_kwargs", "callback_type", "expected"),
        [
            pytest.param({"use_ema": False, "early_stopping": False}, COCOEvalCallback, True, id="coco-eval-always"),
            pytest.param({"use_ema": False}, BestModelCallback, True, id="best-model-always"),
            pytest.param({"use_ema": True}, RFDETREMACallback, True, id="ema-when-use-ema"),
            pytest.param({"use_ema": False}, RFDETREMACallback, False, id="no-ema-without-use-ema"),
            pytest.param({"drop_path": 0.1}, DropPathCallback, True, id="drop-path-when-nonzero"),
            pytest.param({"drop_path": 0.0}, DropPathCallback, False, id="no-drop-path-when-zero"),
            pytest.param({"early_stopping": True}, RFDETREarlyStopping, True, id="early-stopping-when-enabled"),
            pytest.param({"early_stopping": False}, RFDETREarlyStopping, False, id="no-early-stopping-when-disabled"),
        ],
    )
    def test_callback_presence_follows_config(self, tmp_path, tc_kwargs, callback_type, expected):
        """Each optional callback is wired exactly when its TrainConfig flag asks for it; the eval and best-model ones
        always."""
        trainer = build_trainer(_tc(tmp_path, **tc_kwargs), _mc())
        assert any(type(cb) is callback_type for cb in trainer.callbacks) is expected

    def test_coco_eval_uses_eval_interval_and_per_class_flags(self, tmp_path):
        """COCOEvalCallback receives eval_interval and log_per_class_metrics from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, eval_interval=3, log_per_class_metrics=False), _mc())
        coco_cb = _find_callback(trainer, COCOEvalCallback)
        assert coco_cb._eval_interval == 3
        assert coco_cb._log_per_class_metrics is False

    def test_coco_eval_default_skips_per_class_metrics(self, tmp_path):
        """The default TrainConfig disables the costly per-class metric path."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        assert _find_callback(trainer, COCOEvalCallback)._log_per_class_metrics is False

    @pytest.mark.parametrize("backend", ["hotcoco", "hotcoco_streaming", "faster_coco_eval", "ufcoco", "vernier"])
    def test_coco_eval_uses_eval_backend(self, tmp_path: Path, backend: str) -> None:
        """COCOEvalCallback receives every eval_backend value TrainConfig accepts."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, eval_backend=backend), _mc())
        assert _find_callback(trainer, COCOEvalCallback)._eval_backend == backend

    def test_extra_callbacks_join_the_default_callbacks(self, tmp_path: Path) -> None:
        """``extra_callbacks`` reach the ``Trainer`` with the RF-DETR callbacks instead of replacing them."""
        monitor = LearningRateMonitor()
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(), extra_callbacks=[monitor])

        assert monitor in trainer.callbacks and any(isinstance(cb, COCOEvalCallback) for cb in trainer.callbacks)

    def test_extra_callbacks_are_validated_with_the_default_callbacks(self, tmp_path: Path) -> None:
        """Lightning checks ``extra_callbacks`` at construction, so a second progress bar is refused there."""
        with pytest.raises(MisconfigurationException, match="multiple progress bar"):
            build_trainer(_tc(tmp_path, use_ema=False, progress_bar="tqdm"), _mc(), extra_callbacks=[TQDMProgressBar()])

    @hotcoco_only
    def test_streaming_backend_receives_the_model_class_count(self, tmp_path: Path) -> None:
        """``hotcoco_streaming`` gets ``num_classes`` from the model config, so it streams rather than falling back.

        ``StreamingEval`` needs every category before the first batch; without the class count the metric would evaluate
        in one batch on every run and the backend would never stream in production.
        """
        model_config = _mc()
        trainer = build_trainer(_tc(tmp_path, use_ema=False, eval_backend="hotcoco_streaming"), model_config)
        coco_cb = _find_callback(trainer, COCOEvalCallback)

        coco_cb.setup(trainer, SimpleNamespace(model_config=model_config), stage="fit")

        assert len(coco_cb.map_metric._stream_categories) == model_config.num_classes + 1

    @hotcoco_only
    def test_streaming_backend_receives_the_model_class_count_on_the_ema_metric(self, tmp_path: Path) -> None:
        """The EMA metric streams too: dropping its ``num_classes`` would silently make it evaluate in one batch."""
        model_config = _mc()
        trainer = build_trainer(_tc(tmp_path, use_ema=True, eval_backend="hotcoco_streaming"), model_config)
        coco_cb = _find_callback(trainer, COCOEvalCallback)
        coco_cb.setup(trainer, SimpleNamespace(model_config=model_config), stage="fit")

        coco_cb._prepare_ema_metric(trainer)

        assert len(coco_cb.map_metric_ema._stream_categories) == model_config.num_classes + 1

    @pytest.mark.parametrize(
        ("num_classes", "expected"),
        [
            pytest.param(np.int64(3), 3, id="numpy-int64"),
            pytest.param(np.int32(3), 3, id="numpy-int32"),
            (0, 0),
            (True, None),
            pytest.param(np.bool_(True), None, id="numpy-bool"),
            (3.0, None),
            pytest.param(torch.tensor(3), None, id="zero-dim-tensor"),
            ("3", None),
            (None, None),
        ],
    )
    def test_class_count_coercion(self, tmp_path: Path, num_classes: Any, expected: int | None) -> None:
        """An integer-like class count reaches the metric as an ``int``; a ``bool`` or non-integer becomes ``None``."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        coco_cb = _find_callback(trainer, COCOEvalCallback)

        coco_cb.setup(trainer, SimpleNamespace(model_config=SimpleNamespace(num_classes=num_classes)), stage="fit")

        assert coco_cb._num_classes == expected
        assert type(coco_cb._num_classes) is type(expected)

    @hotcoco_only
    def test_numpy_class_count_keeps_the_streaming_backend_streaming(self, tmp_path: Path) -> None:
        """A NumPy class count still declares every category to the streaming metric, instead of disabling streaming."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, eval_backend="hotcoco_streaming"), _mc())
        coco_cb = _find_callback(trainer, COCOEvalCallback)

        coco_cb.setup(trainer, SimpleNamespace(model_config=SimpleNamespace(num_classes=np.int64(3))), stage="fit")

        assert len(coco_cb.map_metric._stream_categories) == 4

    def test_coco_eval_uses_keypoint_oks_sigmas(self, tmp_path):
        """COCOEvalCallback receives custom keypoint OKS sigmas from TrainConfig."""
        sigmas = [0.05] * 25
        trainer = build_trainer(_kp_tc(tmp_path, use_ema=False, keypoint_oks_sigmas=sigmas), _kp_preview_mc())
        assert _find_callback(trainer, COCOEvalCallback)._keypoint_oks_sigmas == sigmas

    @pytest.mark.parametrize(
        ("use_ema", "eval_base_model", "expected"),
        [
            pytest.param(True, False, False, id="ema_default_evaluates_ema_only"),
            pytest.param(True, True, True, id="ema_with_opt_in_evaluates_both"),
            pytest.param(False, False, True, id="no_ema_evaluates_base"),
        ],
    )
    def test_eval_policy_is_wired_to_both_callbacks(self, tmp_path, use_ema, eval_base_model, expected):
        """The eval policy must reach COCOEvalCallback and BestModelCallback consistently.

        The two callbacks have to agree: whenever the base model is not evaluated, COCOEvalCallback mirrors the
        EMA score onto the primary key and BestModelCallback must stop checkpointing base weights against it.
        Wiring only one of the pair reintroduces the metric/weights mismatch this policy exists to avoid.
        """
        trainer = build_trainer(_tc(tmp_path, use_ema=use_ema, eval_base_model=eval_base_model), _mc())

        assert _find_callback(trainer, COCOEvalCallback)._eval_base_model is eval_base_model
        assert _find_callback(trainer, BestModelCallback)._evaluates_base_model is expected

    def test_skip_best_epochs_forwarded_to_best_model_callback(self, tmp_path):
        """BestModelCallback receives skip_best_epochs from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, skip_best_epochs=3), _mc())
        assert _find_callback(trainer, BestModelCallback)._skip_best_epochs == 3

    @pytest.mark.parametrize(
        ("make_tc", "make_mc", "expected_monitor", "expected_monitor_ema"),
        [
            pytest.param(
                lambda p: _kp_tc(p, use_ema=True),
                _kp_preview_mc,
                "val/keypoint_map_50_95",
                "val/ema_keypoint_map_50_95",
                id="keypoint-ranks-by-keypoint-ap-not-bbox-map",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True),
                lambda: _mc(segmentation_head=True),
                "val/segm_mAP_50_95",
                "val/ema_segm_mAP_50_95",
                id="segmentation-ranks-by-segmentation-ap-not-bbox-ap",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True, best_model_metric="mar"),
                _mc,
                "val/mAR",
                "val/ema_mAR",
                id="detection-mar-metric-ranks-by-mar-not-map",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=False, best_model_metric="mar"),
                _mc,
                "val/mAR",
                None,
                id="detection-mar-metric-without-ema-has-no-ema-monitor",
            ),
            pytest.param(
                lambda p: _kp_tc(p, use_ema=True, best_model_metric="mar"),
                _kp_preview_mc,
                "val/keypoint_mAR",
                "val/ema_keypoint_mAR",
                id="keypoint-mar-metric-ranks-by-oks-keypoint-mar",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True, best_model_metric="mar"),
                lambda: _mc(segmentation_head=True),
                "val/mAR",
                "val/ema_mAR",
                id="segmentation-mar-metric-has-no-dedicated-mask-mar-falls-back-to-bbox-mar",
            ),
        ],
    )
    def test_best_model_monitor_selection(self, tmp_path, make_tc, make_mc, expected_monitor, expected_monitor_ema):
        """BestModelCallback.monitor/._monitor_ema select the metric key matching task and best_model_metric.

        Detection ranks by mAP by default and mAR when best_model_metric='mar'; keypoint and segmentation swap in their
        own AP/AR metric families; the EMA monitor is only set when EMA is enabled.
        """
        best_cb = _find_callback(build_trainer(make_tc(tmp_path), make_mc()), BestModelCallback)
        assert best_cb.monitor == expected_monitor
        assert best_cb._monitor_ema == expected_monitor_ema

    def test_latest_model_checkpoint_present(self, tmp_path):
        """A ModelCheckpoint (not BestModelCallback) with every_n_epochs==1 is included when checkpoint_interval > 1."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=2), _mc())
        assert any(cb._every_n_epochs == 1 for cb in _find_resume_checkpoints(trainer))

    def test_checkpoint_interval_one_has_single_interval_checkpoint_callback(self, tmp_path):
        """checkpoint_interval=1 creates one non-best ModelCheckpoint that archives every epoch, with no separate latest
        one."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=1), _mc())
        resume_cbs = _find_resume_checkpoints(trainer)

        assert len(resume_cbs) == 1
        only_cb = resume_cbs[0]
        assert only_cb._every_n_epochs == 1
        assert only_cb.save_top_k == -1
        assert only_cb.filename == "checkpoint_{epoch}"
        assert str(only_cb.dirpath) == str(tmp_path / "out")

    @pytest.mark.parametrize(
        "tc_kwargs",
        [pytest.param({}, id="default-interval"), pytest.param({"checkpoint_interval": 7}, id="interval-7")],
    )
    def test_interval_checkpoint_uses_interval_from_config(self, tmp_path, tc_kwargs):
        """A ModelCheckpoint (not BestModelCallback) with every_n_epochs==checkpoint_interval is always included."""
        tc = _tc(tmp_path, use_ema=False, **tc_kwargs)
        trainer = build_trainer(tc, _mc())
        assert any(cb._every_n_epochs == tc.checkpoint_interval for cb in _find_resume_checkpoints(trainer))

    @pytest.mark.parametrize("checkpoint_interval", [1, 2, 7])
    def test_all_model_checkpoints_have_unique_state_keys(self, tmp_path, checkpoint_interval):
        """All ModelCheckpoint callbacks (including BestModelCallback) always have unique state keys."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, checkpoint_interval=checkpoint_interval), _mc())
        state_keys = [cb.state_key for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(state_keys) == len(set(state_keys)), (
            f"Duplicate state_key with checkpoint_interval={checkpoint_interval}: "
            f"{[k for k in state_keys if state_keys.count(k) > 1]}"
        )

    def test_checkpoint_interval_validation(self, tmp_path):
        """TrainConfig(checkpoint_interval=0) raises ValidationError."""
        with pytest.raises(ValidationError):
            _tc(tmp_path, checkpoint_interval=0)

    def test_ema_callback_uses_update_interval(self, tmp_path):
        """RFDETREMACallback receives ema_update_interval from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True, ema_update_interval=4), _mc())
        assert _find_callback(trainer, RFDETREMACallback)._update_interval_steps == 4

    def test_skip_best_epochs_forwarded_to_early_stopping(self, tmp_path):
        """RFDETREarlyStopping receives skip_best_epochs from TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=True, skip_best_epochs=4), _mc())
        assert _find_callback(trainer, RFDETREarlyStopping)._skip_best_epochs == 4

    @pytest.mark.parametrize(
        ("make_tc", "make_mc", "expected_monitor_regular", "expected_monitor_ema"),
        [
            pytest.param(
                lambda p: _kp_tc(p, early_stopping=True, early_stopping_use_ema=True),
                _kp_preview_mc,
                "val/keypoint_map_50_95",
                "val/ema_keypoint_map_50_95",
                id="keypoint-uses-keypoint-ap-as-the-regular-metric",
            ),
            pytest.param(
                lambda p: _tc(p, early_stopping=True, early_stopping_use_ema=True),
                lambda: _mc(segmentation_head=True),
                "val/segm_mAP_50_95",
                "val/ema_segm_mAP_50_95",
                id="segmentation-uses-segmentation-ap-as-the-regular-metric",
            ),
            pytest.param(
                lambda p: _tc(p, early_stopping=True, early_stopping_use_ema=True, best_model_metric="mar"),
                _mc,
                "val/mAR",
                "val/ema_mAR",
                id="detection-mar-metric-watches-mar-not-map",
            ),
            pytest.param(
                lambda p: _kp_tc(p, early_stopping=True, early_stopping_use_ema=True, best_model_metric="mar"),
                _kp_preview_mc,
                "val/keypoint_mAR",
                "val/ema_keypoint_mAR",
                id="keypoint-mar-metric-watches-oks-keypoint-mar",
            ),
            pytest.param(
                lambda p: _tc(
                    p, use_ema=True, early_stopping=True, early_stopping_use_ema=True, best_model_metric="mar"
                ),
                lambda: _mc(segmentation_head=True),
                "val/mAR",
                "val/ema_mAR",
                id="segmentation-mar-metric-uses-bbox-mar-keys-when-ema-enabled",
            ),
        ],
    )
    def test_early_stopping_monitor_selection(
        self, tmp_path, make_tc, make_mc, expected_monitor_regular, expected_monitor_ema
    ):
        """RFDETREarlyStopping's regular/EMA monitors select the metric key matching task and best_model_metric.

        Mirrors BestModelCallback's monitor selection: detection watches mAP by default and mAR when
        best_model_metric='mar'; keypoint and segmentation swap in their own AP/AR metric families.
        """
        early_stop_cb = _find_callback(build_trainer(make_tc(tmp_path), make_mc()), RFDETREarlyStopping)
        assert early_stop_cb._monitor_regular == expected_monitor_regular
        assert early_stop_cb._monitor_ema == expected_monitor_ema

    def test_segmentation_config_accepted(self, tmp_path):
        """SegmentationTrainConfig is accepted without error."""
        trainer = build_trainer(_tc(tmp_path, config_cls=SegmentationTrainConfig), _mc(segmentation_head=True))
        assert isinstance(trainer, Trainer)


class TestBuildTrainerCallbackOrdering:
    """Callback order after PTL moves every ``Checkpoint`` callback to the end of ``trainer.callbacks``.

    COCOEvalCallback is appended after BestModelCallback/RFDETREarlyStopping in source order (trainer.py), but PTL's
    ``_CallbackConnector._reorder_callbacks`` moves every ``Checkpoint`` subclass -- including ``BestModelCallback`` --
    to the end, while ``RFDETREarlyStopping`` (an ``EarlyStopping`` subclass, not ``Checkpoint``) keeps its relative
    position. Regression for the PR #1134 append-order move: this asserts the *actual* PTL-resolved execution order
    matches the safety argument in the trainer.py comment, not just the raw append order.
    """

    def test_early_stopping_and_coco_eval_fire_before_best_model_checkpoint(self, tmp_path):
        """After PTL's callback reordering, EarlyStopping and COCOEvalCallback still precede BestModelCallback."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, early_stopping=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        early_stop_idx = types.index(RFDETREarlyStopping)
        coco_eval_idx = types.index(COCOEvalCallback)
        best_model_idx = types.index(BestModelCallback)
        assert early_stop_idx < best_model_idx, (
            "RFDETREarlyStopping must fire before BestModelCallback on every on_validation_end "
            "(BestModelCallback's try/finally restore relies on EarlyStopping reading the raw metric first)"
        )
        assert coco_eval_idx < best_model_idx, (
            "COCOEvalCallback must write its metrics before BestModelCallback reads them on_validation_end"
        )

    def test_best_model_callback_is_among_the_reordered_checkpoint_group(self, tmp_path):
        """BestModelCallback (a ModelCheckpoint subclass) is moved into the trailing checkpoint group by PTL."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, early_stopping=True, checkpoint_interval=2), _mc())
        checkpoint_types = {type(cb) for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)}
        non_checkpoint_types = [type(cb) for cb in trainer.callbacks if not isinstance(cb, ModelCheckpoint)]
        assert BestModelCallback in checkpoint_types
        # No non-Checkpoint callback (EarlyStopping, COCOEvalCallback, ...) is displaced past a checkpoint.
        last_non_checkpoint_idx = max(i for i, cb in enumerate(trainer.callbacks) if type(cb) in non_checkpoint_types)
        first_checkpoint_idx = min(i for i, cb in enumerate(trainer.callbacks) if isinstance(cb, ModelCheckpoint))
        assert last_non_checkpoint_idx < first_checkpoint_idx


class TestBuildTrainerKeypointDefaults:
    """Verify build_trainer() applies keypoint-specific defaults for noisy fine-tuning metrics."""

    @pytest.mark.parametrize(
        ("make_tc", "make_mc", "expected_skip", "expected_alpha"),
        [
            pytest.param(_kp_tc, _kp_preview_mc, 10, 0.5, id="keypoint-skips-ten-epochs-and-smooths-by-half"),
            pytest.param(_tc, _mc, 0, 0.0, id="detection-skips-none-and-does-not-smooth"),
        ],
    )
    def test_default_skip_best_epochs_and_smooth_alpha(self, tmp_path, make_tc, make_mc, expected_skip, expected_alpha):
        """Task defaults reach both callbacks: keypoint skips 10 epochs and smooths mAP by 0.5 to dampen noisy
        swings."""
        trainer = build_trainer(make_tc(tmp_path, use_ema=False, early_stopping=True), make_mc())
        best_cb = _find_callback(trainer, BestModelCallback)

        assert best_cb._skip_best_epochs == expected_skip
        assert _find_callback(trainer, RFDETREarlyStopping)._skip_best_epochs == expected_skip
        assert best_cb._smooth_alpha == pytest.approx(expected_alpha)

    def test_keypoint_explicit_skip_best_epochs_overrides_default(self, tmp_path):
        """An explicitly-set skip_best_epochs on a keypoint config overrides the class default of 10."""
        trainer = build_trainer(_kp_tc(tmp_path, use_ema=False, skip_best_epochs=3), _kp_preview_mc())
        assert _find_callback(trainer, BestModelCallback)._skip_best_epochs == 3


class TestBuildTrainerPrecision:
    """build_trainer() must resolve training precision from model_config.amp + device caps."""

    def test_amp_false_gives_32_true(self, tmp_path):
        """Amp=False always produces '32-true' regardless of device."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_amp_true_cpu_gives_32_true(self, mocker: MockerFixture, tmp_path):
        """Amp=True on CPU (no CUDA, no MPS) must fall back to '32-true'."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)

        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))

        assert trainer.precision == "32-true"

    def test_amp_true_explicit_cpu_accelerator_gives_32_true_even_with_mps(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Amp=True with explicit accelerator='cpu' must produce '32-true' even when MPS is present.

        bf16 autocast on macOS CPU (Apple Silicon) is ~13x slower than fp32 — no hardware support for bfloat16 in CPU
        kernels causes software emulation.  When the caller explicitly opts into CPU (e.g. for test isolation), mixed
        precision must not be used.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator="cpu")

        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.parametrize(("accelerator", "expected_precision"), [("xla", "32-true"), ("tpu", "bf16-true")])
    def test_xla_accelerator_uses_xla_precision_plugin_not_precision_string(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        bf16_cuda_host: None,
        tmp_path,
        accelerator,
        expected_precision,
    ):
        """An XLA accelerator uses XLAPrecision instead of a precision string.

        XLAStrategy's precision_plugin setter only accepts the XLAPrecision plugin and raises TypeError for standard
        precision strings like 'bf16-mixed'. ``XLAPrecision.__init__`` itself raises ``ModuleNotFoundError`` unless
        torch_xla is importable (see ``lightning_fabric.accelerators.xla._XLA_AVAILABLE``), so the class is patched
        here to keep this test backend-neutral -- the behaviour under test is build_trainer's accelerator dispatch,
        not PTL's own package-availability guard.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator=accelerator)

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with(expected_precision)
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.parametrize("amp_dtype", ["bf16", "auto"])
    def test_bf16_or_auto_on_tpu_uses_bf16_true(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
        amp_dtype: str,
    ) -> None:
        """Explicit BF16 and the default auto mode select TPU BF16 true precision.

        'auto' matters here as much as the explicit case: it is the amp_dtype every caller gets by just passing
        ``amp=True`` -- the exact recipe issue #1058 itself documents for TPU training -- and without CUDA/MPS on the
        host it used to fall through to the CPU-only "32-true" default, silently training in FP32 on TPU by default.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype=amp_dtype), _mc(amp=True), accelerator="tpu")

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with("bf16-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    def test_auto_on_available_tpu_uses_bf16_true(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Automatic accelerator selection retains TPU BF16 true precision."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=False)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator="auto")

        mock_xla_precision.assert_called_once_with("bf16-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    def test_explicit_xla_bf16_stays_fp32_without_backend_evidence(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Explicit XLA never assumes GPU PJRT can execute BF16 true precision."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="bf16"), _mc(amp=True), accelerator="xla")

        mock_xla_precision.assert_called_once_with("32-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.parametrize("amp", [True, False])
    def test_xla_accelerator_preserves_plugins_and_rejects_precision_kwargs(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        bf16_cuda_host: None,
        tmp_path,
        amp,
    ):
        """XLA appends its precision plugin after the caller's and ignores an incompatible ``precision`` kwarg."""
        caller_plugin = object()

        build_trainer(
            _tc(tmp_path, use_ema=False), _mc(amp=amp), accelerator="xla", plugins=[caller_plugin], precision="16-mixed"
        )

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with("32-true")
        assert captured_trainer_kwargs["plugins"] == [caller_plugin, mock_xla_precision.return_value]

    def test_non_xla_accelerator_sets_no_plugins_key(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """A non-XLA accelerator does not add a 'plugins' key -- only the XLA path does."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="cpu")

        assert "plugins" not in captured_trainer_kwargs
        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.xla
    @requires_torch_xla
    def test_tpu_accelerator_refuses_to_launch_off_real_tpu(self, tmp_path) -> None:
        """Resolves plan Sec 1.3 caveat #1: PTL's 'tpu' accelerator needs real TPU chips, not just torch_xla+PJRT.

        ``XLAAccelerator.is_available()`` returns ``False`` under ``PJRT_DEVICE=CPU`` (no TPU silicon), so
        ``Trainer.__init__`` raises ``MisconfigurationException`` naming ``XLAAccelerator`` as unavailable and listing
        ``cpu`` as the only available accelerator (confirmed against the live message from ``ci-tests-xla.yml``'s CPU-
        PJRT run, not just source inspection). This confirms the full ``model.train(accelerator="tpu")`` entry point is
        not launchable under the T1 (CPU-PJRT) CI lane -- only the device-gated unit tests (Tasks 1.1/1.3/1.6/1.7/1.8,
        which move tensors to ``xm.xla_device()`` directly) validate Phase 1 correctness there.
        """
        if XLAAccelerator.is_available():
            pytest.skip(
                "the refusal this pins holds only while ``XLAAccelerator.is_available()`` is False, so the skip is "
                "gated on that same predicate rather than on the configured backend: ``xr.device_type()`` reports "
                "``PJRT_DEVICE`` and would also skip on a host that sets ``PJRT_DEVICE=TPU`` without chips, where "
                "``auto_device_count()`` is 0, the Trainer still raises, and the assertion is still meaningful. "
                "Gating on availability keeps NEURON in the refusal path for the same reason."
            )

        with pytest.raises(MisconfigurationException, match="XLAAccelerator"):
            build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu")

    def test_amp_true_ddp_notebook_probes_bf16_normally(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """ddp_notebook uses standard precision probing (spawn makes CUDA init safe).

        With spawn-based DDP, child processes start fresh — CUDA init in the parent does not propagate.  So
        ``is_bf16_supported()`` is safe to call and pre-Ampere GPUs correctly get ``16-mixed`` instead of the slower
        bf16 emulation path.  Simulates a pre-Ampere GPU as real PyTorch reports it: CUDA available, bf16 supported only
        through emulation.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation)
        mocker.patch("torch.cuda.get_device_capability", return_value=(7, 5))

        build_trainer(_tc(tmp_path, use_ema=False, strategy="ddp_notebook"), _mc(amp=True))

        assert captured_trainer_kwargs["precision"] == "16-mixed"


class TestBuildTrainerAmpDtype:
    """``TrainConfig.amp_dtype`` (a ``train()`` kwarg) lets callers pin the AMP autocast dtype (fp16 vs bf16) — #1132.

    Precision is resolved inside ``build_trainer``; these tests mock the CUDA/MPS capability probes and assert the
    Lightning precision string captured at ``Trainer`` construction time.
    """

    def test_amp_dtype_is_a_train_kwarg_not_dropped(self, tmp_path):
        """amp_dtype is a real TrainConfig field (reachable via train(**kwargs)), not silently dropped."""
        assert _tc(tmp_path, amp_dtype="fp16").amp_dtype == "fp16"

    @pytest.mark.parametrize(
        ("cuda", "bf16", "mps", "amp_dtype", "expected"),
        [
            (True, True, False, "auto", "bf16-mixed"),
            (True, True, False, "fp16", "16-mixed"),
            (True, True, False, "bf16", "bf16-mixed"),
            (True, True, False, "fp8", "transformer-engine"),
            (True, False, False, "auto", "16-mixed"),
            (False, False, True, "fp16", "16-mixed"),
        ],
    )
    def test_resolved_precision(self, resolved_precision, cuda, bf16, mps, amp_dtype, expected):
        """amp_dtype + hardware caps resolve to the correct Lightning precision string."""
        assert resolved_precision(cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype) == expected

    @pytest.mark.parametrize(
        ("cuda", "bf16", "mps", "amp_dtype", "warn_match"),
        [(True, False, False, "bf16", "bf16"), (False, False, True, "bf16", "MPS")],
    )
    def test_resolved_precision_warns(self, resolved_precision, cuda, bf16, mps, amp_dtype, warn_match):
        """amp_dtype falls back to '16-mixed' and emits a UserWarning when hardware cannot satisfy the request."""
        with pytest.warns(UserWarning, match=warn_match):
            precision = resolved_precision(cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype)
        assert precision == "16-mixed"

    def test_auto_uses_fp16_on_gpu_with_emulated_bf16(self, resolved_precision) -> None:
        """``amp_dtype="auto"`` on a GPU whose bfloat16 is only emulated (T4, V100) trains in fp16.

        ``torch.cuda.is_bf16_supported()`` counts emulation, so it returns ``True`` on such a GPU and ``"auto"`` used to
        pick ``bf16-mixed``. On a Colab T4 that was about 2x slower per RF-DETR Nano training step than ``16-mixed``.
        """
        precision = resolved_precision(cuda=True, bf16_emulated=True, amp_dtype="auto")
        assert precision == "16-mixed", f"amp_dtype='auto' on an emulated-bf16 GPU resolved to {precision!r}"

    def test_explicit_bf16_on_gpu_with_emulated_bf16_is_kept_with_a_warning(self, resolved_precision) -> None:
        """An explicit ``amp_dtype="bf16"`` is honoured on an emulated-bf16 GPU, with a warning naming faster
        options."""
        with pytest.warns(UserWarning, match="emulation"):
            precision = resolved_precision(cuda=True, bf16_emulated=True, amp_dtype="bf16")
        assert precision == "bf16-mixed", f"explicit bf16 on an emulated-bf16 GPU resolved to {precision!r}"

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            pytest.param("1,", "bf16-mixed", id="a100-at-index-1"),
            pytest.param("0,", "16-mixed", id="t4-at-index-0"),
            pytest.param(2, "16-mixed", id="both-gpus"),
            pytest.param([1], "bf16-mixed", id="a100-by-index-list"),
        ],
    )
    def test_auto_checks_the_gpus_the_run_trains_on(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mocker: MockerFixture,
        tmp_path: Path,
        devices: int | str | list[int],
        expected: str,
    ) -> None:
        """On a host with a T4 at index 0 and an A100 at index 1, ``"auto"`` follows the GPUs the run trains on.

        Selecting a GPU by index (``devices="1,"``, or the ``[1]`` that ``RFDETR.train(device="cuda:1")`` forwards) does
        not change the current device, so checking only the current device would pick fp16 for the A100 run here, and
        bf16 for a T4 run on a host whose index 0 is an A100. A run across both GPUs uses fp16.
        """
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda device=None: capabilities[device or 0])
        mocker.patch("torch.version.hip", None)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(), devices=devices)

        assert captured_trainer_kwargs["precision"] == expected, (
            f"devices={devices!r} resolved to {captured_trainer_kwargs['precision']!r}"
        )

    def test_explicit_bf16_checks_the_gpus_the_run_trains_on(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """An explicit ``amp_dtype="bf16"`` is granted on the strength of the training GPU, not the current device.

        The host here has a pre-Ampere GPU at index 0 and an A100 at index 1 on a CUDA build without bfloat16 emulation,
        so ``torch.cuda.is_bf16_supported()`` — which only ever looks at the current device — says no. A run pinned to
        the A100 with ``devices=[1]`` must still get ``bf16-mixed``, and no emulation warning.
        """
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=False)
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda device=None: capabilities[device or 0])
        mocker.patch("torch.version.hip", None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="bf16"), _mc(), devices=[1])

        assert captured_trainer_kwargs["precision"] == "bf16-mixed"
        assert not [warning for warning in caught if "bf16" in str(warning.message)]

    def test_explicit_amp_dtype_overrides_deprecated_amp_false(self, resolved_precision):
        """An explicit amp_dtype wins over the deprecated amp flag: the stale amp=False is ignored.

        Mocked onto a bf16-capable CUDA device, so the fp16 request is distinguishable both from the '32-true' the old
        amp=False precedence produced and from the 'bf16-mixed' the hardware would otherwise select.
        """
        assert resolved_precision(cuda=True, bf16=True, amp_dtype="fp16", amp=False) == "16-mixed"

    def test_deprecated_amp_false_applies_when_amp_dtype_is_default(self, resolved_precision):
        """Amp=False still disables AMP while amp_dtype is left at its default, and warns.

        The mocked device is bf16-capable CUDA, which would resolve to 'bf16-mixed' on its own — so '32-true' can only
        come from the legacy toggle still being honored.
        """
        with pytest.warns(FutureWarning, match="ModelConfig.amp is deprecated"):
            resolved = resolved_precision(cuda=True, bf16=True, amp_dtype="auto", amp=False)
        assert resolved == "32-true"

    def test_amp_dtype_none_disables_amp(self, resolved_precision):
        """amp_dtype=None is the replacement for the deprecated amp=False and needs no model flag.

        Mocked onto bf16-capable CUDA with amp left at its default True, so '32-true' is attributable to amp_dtype=None
        alone rather than to absent hardware.
        """
        assert resolved_precision(cuda=True, bf16=True, amp_dtype=None, amp=True) == "32-true"

    def test_cpu_accelerator_ignores_amp_dtype(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Explicit accelerator='cpu' yields '32-true' regardless of amp_dtype."""
        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp16"), _mc(amp=True), accelerator="cpu")
        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.parametrize("amp", [True, False])
    def test_fp8_rejects_non_cuda_accelerator(self, tmp_path, amp):
        """FP8 must fail clearly on a non-CUDA accelerator, never fall back or be disabled by the deprecated AMP
        flag."""
        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=amp), accelerator="cpu")

    @pytest.mark.parametrize("accelerator", ["cpu", "mps", "xla", "tpu"])
    def test_fp8_rejects_non_cuda_with_cuda_visible(
        self, tmp_path: Path, accelerator: str, mocker: MockerFixture, mock_xla_precision: MagicMock
    ) -> None:
        """Reject FP8 before an explicitly selected XLA backend loads its precision plugin."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), accelerator=accelerator)
        mock_xla_precision.assert_not_called()

    def test_fp8_rejects_auto_resolving_to_xla(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Lightning selects XLA before CUDA for auto; FP8 must honor that choice."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), accelerator="auto")

    def test_fp8_rejects_deepspeed_strategy(self, tmp_path, mocker: MockerFixture):
        """Lightning cannot combine its Transformer Engine precision plugin with DeepSpeed precision."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="amp_dtype='fp8'.*DeepSpeed"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8", strategy="deepspeed_stage_2"), _mc(amp=True))

    @pytest.mark.parametrize(
        ("capabilities", "device_names", "devices", "expected_label"),
        [
            pytest.param({0: (8, 0)}, "NVIDIA A100", None, "cuda:0 (NVIDIA A100)", id="single-pre-ada-gpu"),
            pytest.param({0: (9, 0), 1: (7, 5)}, "NVIDIA T4", 2, "cuda:1 (NVIDIA T4)", id="hopper-plus-t4-uses-both"),
        ],
    )
    def test_fp8_rejects_unsupported_training_device(
        self, tmp_path, mocker: MockerFixture, capabilities, device_names, devices, expected_label
    ):
        """FP8 must fail clearly if any training GPU is CUDA-visible but below Transformer Engine's compute capability.

        ``torch.cuda.is_available()`` alone does not establish FP8/Transformer Engine hardware support: an A100 or T4 is
        CUDA-visible but below the compute capability (8.9, Ada) Transformer Engine requires, and a multi-GPU run must
        validate every device it trains on, not just the first.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda index: capabilities[index])
        mocker.patch("torch.cuda.get_device_name", return_value=device_names)
        device_kwargs = {} if devices is None else {"devices": devices}

        with pytest.raises(ValueError) as exc_info:
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), **device_kwargs)

        assert "compute capability >= 8.9" in str(exc_info.value)
        assert expected_label in str(exc_info.value)

    @pytest.mark.parametrize(
        ("capabilities", "devices"),
        [
            pytest.param({0: (8, 9)}, None, id="single-ada-gpu"),
            pytest.param({0: (9, 0), 1: (7, 5)}, [0], id="hopper-pinned-on-a-mixed-host"),
        ],
    )
    def test_fp8_accepts_capable_training_devices(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, mocker: MockerFixture, capabilities, devices
    ):
        """An Ada-or-newer training device resolves to the Transformer Engine precision string, not an error.

        A run pinned to the fp8-capable GPU is not rejected for an older GPU it never touches: with a Hopper at index 0
        and a T4 at index 1, ``devices=[0]`` -- the shape ``RFDETR.train(device="cuda:0")`` forwards -- trains only on
        hardware Transformer Engine supports. The real ``Trainer`` is mocked, so only the capability gate is exercised,
        not Transformer Engine's plugin construction, which needs real hardware.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda index: capabilities[index])
        device_kwargs = {} if devices is None else {"devices": devices}

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), **device_kwargs)

        assert captured_trainer_kwargs["precision"] == "transformer-engine"

    @pytest.mark.parametrize("bad_value", ["float8", 42, True])
    def test_invalid_amp_dtype_falls_back_to_auto_with_warning(self, tmp_path, bad_value):
        """An unrecognised or wrong-typed amp_dtype falls back to 'auto' with a warning rather than raising."""
        with pytest.warns(UserWarning, match="amp_dtype"):
            tc = _tc(tmp_path, amp_dtype=bad_value)
        assert tc.amp_dtype == "auto"


def _cuda_supports_fp8() -> bool:
    """Return whether device 0 meets Transformer Engine's FP8 minimum compute capability.

    Examples:
        >>> isinstance(_cuda_supports_fp8(), bool)
        True
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)


class TestBuildTrainerFP8Smoke:
    """Real CUDA + Transformer Engine integration smoke test for FP8 training.

    ``TestBuildTrainerAmpDtype.test_resolved_precision`` (its ``fp8`` case) only mocks ``torch.cuda.is_available`` and
    ``torch.cuda.is_bf16_supported`` and captures the precision string passed to the (also mocked) ``Trainer`` — it
    never imports or instantiates Transformer Engine's PyTorch extension. A missing/mismatched ``cuda`` extra or a
    plugin setup failure would therefore pass every other test in this module and only surface on the first real
    CUDA run. This test builds a real trainer and Lightning module with an actual ``nn.Linear`` layer and executes
    one training step under FP8, skipped everywhere the hardware or the ``transformer-engine`` package is absent.
    """

    @pytest.mark.gpu
    @transformer_engine_only
    @pytest.mark.skipif(
        not _cuda_supports_fp8(),
        reason="FP8 requires a Transformer Engine-supported GPU (Ada, Hopper, or newer; compute capability >= 8.9)",
    )
    def test_fp8_smoke_builds_and_steps(self, base_model_config, base_train_config, mocker: MockerFixture):
        """A real CUDA + Transformer Engine environment must build the plugin and run one FP8 training step.

        Guards against the gap the mocked ``fp8-cuda`` precision-resolution test cannot close: the actual
        ``transformer_engine.pytorch`` extension being unimportable, unbuilt, or incompatible with the installed
        CUDA/PyTorch stack only fails here, on real hardware, never in the CPU-only unit tests above.
        """
        from rfdetr.training.module_data import RFDETRDataModule
        from rfdetr.training.module_model import RFDETRModelModule

        from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts

        class _LinearModel(torch.nn.Module):
            """Minimal real model with an ``nn.Linear`` layer for Transformer Engine to convert."""

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 32 * 32, 64)

            def forward(self, samples, targets=None):
                images = samples.tensors if hasattr(samples, "tensors") else samples
                return {"dummy": self.linear(images.flatten(1)).sum()}

            def update_drop_path(self, *args, **kwargs) -> None:
                pass

            def update_dropout(self, *args, **kwargs) -> None:
                pass

            def reinitialize_detection_head(self, *args, **kwargs) -> None:
                pass

        mc = base_model_config(device="cuda", amp=True)
        tc = base_train_config(use_ema=False, run_test=False, amp_dtype="fp8", batch_size=2)

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_LinearModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4))
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        )

        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cuda",
            fast_dev_run=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        assert trainer.precision == "transformer-engine"
        trainer.fit(module, datamodule=datamodule)


class TestBuildTrainerEMAShardingGuard:
    """EMA must be disabled and a UserWarning emitted for sharded strategies.

    PTL validates strategy+accelerator compatibility at Trainer construction time, so these tests mock Trainer to
    capture the callback list without triggering platform-specific validation.
    """

    @pytest.mark.parametrize("strategy", ["fsdp", "deepspeed", "deepspeed_stage_2"])
    def test_ema_disabled_with_warning_for_sharded_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, strategy
    ):
        """The EMA callback is absent and a UserWarning says so when a sharded strategy is requested."""
        with pytest.warns(UserWarning, match="EMA disabled"):
            build_trainer(_tc(tmp_path, use_ema=True, strategy=strategy), _mc())

        assert not any(isinstance(cb, RFDETREMACallback) for cb in captured_trainer_kwargs["callbacks"])


class TestBuildTrainerEMAXLAGuard:
    """Multi-device XLA training must disable EMA and its checkpoint/evaluation bookkeeping."""

    @pytest.mark.parametrize(
        ("devices", "num_nodes"),
        [
            pytest.param(4, 1, id="four_devices_one_node"),
            pytest.param(1, 2, id="one_device_per_host_two_hosts"),
        ],
    )
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_device_xla_disables_ema_and_uses_regular_checkpoint_track(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, devices: int, num_nodes: int
    ) -> None:
        """Multi-device XLA, one device per host across nodes included, must omit EMA callbacks and metrics.

        The regular-model checkpoint/evaluation path stays in place.
        """
        with pytest.warns(UserWarning, match="EMA disabled on multi-device XLA"):
            build_trainer(
                _tc(tmp_path, use_ema=True, eval_base_model=False, devices=devices, num_nodes=num_nodes),
                _mc(),
                accelerator="xla",
            )

        callbacks = captured_trainer_kwargs["callbacks"]
        assert not any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is None
        assert best_callback._evaluates_base_model is True

    @pytest.mark.parametrize("accelerator", ["xla", "tpu"])
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_single_device_xla_keeps_ema_enabled(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, accelerator: str
    ) -> None:
        """One-device XLA keeps EMA after the shipped callback completed the one-chip validation runs.

        ``"tpu"`` is the accelerator string ``RFDETR.train(device="xla")`` forwards to ``build_trainer`` (see
        ``test_device_xla_absorbed_as_accelerator_tpu``); the guard must treat it exactly like ``"xla"``.
        """
        build_trainer(_tc(tmp_path, use_ema=True, eval_base_model=False, devices=1), _mc(), accelerator=accelerator)

        assert captured_trainer_kwargs["accelerator"] == accelerator
        assert "precision" not in captured_trainer_kwargs
        callbacks = captured_trainer_kwargs["callbacks"]
        assert any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is not None
        assert best_callback._evaluates_base_model is False

    @pytest.mark.parametrize(
        ("tc_kwargs", "build_kwargs"),
        [
            pytest.param({"devices": 1}, {}, id="single-device-keeps-ema"),
            pytest.param({"devices": 4}, {"include_training_callbacks": False}, id="evaluation-only-builds-no-ema"),
        ],
    )
    @pytest.mark.usefixtures("captured_trainer_kwargs", "mock_xla_precision")
    def test_ema_kept_or_not_built_does_not_warn(self, tmp_path: Path, tc_kwargs, build_kwargs) -> None:
        """The EMA-disabled warning fires neither when EMA is kept nor when evaluation-only trainers never build it.

        The evaluation-only case uses the multi-device configuration that *does* warn when training callbacks are built,
        so the assertion is about ``include_training_callbacks=False`` rather than a device count that would stay silent
        either way.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_trainer(_tc(tmp_path, use_ema=True, **tc_kwargs), _mc(), accelerator="xla", **build_kwargs)

        assert not any("EMA disabled" in str(warning.message) for warning in caught)

    @pytest.mark.parametrize("auto_device_count", [1, 4])
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_auto_devices_on_xla_follows_the_resolved_chip_count(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path: Path, auto_device_count: int
    ) -> None:
        """``devices='auto'`` must resolve through the XLA chip count, not CUDA's auto branch."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.auto_device_count", return_value=auto_device_count)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_trainer(_tc(tmp_path, use_ema=True, eval_base_model=False, devices="auto"), _mc(), accelerator="xla")

        callbacks = captured_trainer_kwargs["callbacks"]
        has_ema = any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        assert has_ema is (auto_device_count == 1)

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_xla_device_index_list_keeps_ema_disabled_instead_of_raising(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        """A device-index list, as ``RFDETR.train(device='xla:N')`` forwards it, must not raise in the EMA guard.

        A distributed ``strategy`` skips the earlier ``_requests_multiple_devices`` call, so this is the first place the
        list is inspected; the guard has to answer "not provably one device" like it did before.
        """
        with pytest.warns(UserWarning, match="EMA disabled"):
            build_trainer(
                _tc(tmp_path, use_ema=True, eval_base_model=False, strategy="ddp"),
                _mc(),
                accelerator="tpu",
                devices=[0],
            )

        callbacks = captured_trainer_kwargs["callbacks"]
        assert not any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is None
        assert best_callback._evaluates_base_model is True


class TestRequestsMultipleDevices:
    """``_requests_multiple_devices`` gates the distributed branch, so every ``devices`` form must reach an answer."""

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            pytest.param([1], False, id="single-index-list"),
            pytest.param([0, 1], True, id="two-index-list"),
            pytest.param((0, 1), True, id="two-index-tuple"),
            pytest.param([], False, id="empty-list"),
        ],
    )
    def test_an_index_sequence_is_measured_by_its_length(self, devices: Sequence[int], expected: bool) -> None:
        """A sequence of device indices is counted instead of being parsed as a string.

        ``devices=[1]`` is what ``RFDETR.train(device="cuda:1")`` forwards; it used to reach the string branch's
        ``.strip()`` and raise ``AttributeError: 'list' object has no attribute 'strip'`` before training started.
        """
        assert _requests_multiple_devices(devices) is expected


class TestXlaResolvesToSingleDevice:
    """``_xla_resolves_to_single_device`` decides the EMA guard, so each ``int | str`` device form is covered."""

    @pytest.fixture
    def forbidden_auto_device_count(self, mocker: MockerFixture) -> MagicMock:
        """Make any query of the XLA runtime's chip count fail, so a test proves the answer needed no runtime.

        Examples:
            >>> forbidden_auto_device_count(mocker)  # doctest: +SKIP
            # Needs pytest's mocker fixture; cannot run standalone.
        """
        return mocker.patch(
            "pytorch_lightning.accelerators.XLAAccelerator.auto_device_count",
            side_effect=AssertionError("auto_device_count must not be consulted"),
        )

    @pytest.mark.usefixtures("forbidden_auto_device_count")
    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            (1, True),
            (2, False),
            (4, False),
            ("1", True),
            ("4", False),
            ("0,1", False),
            ("", False),
            ("all", False),
            pytest.param([0], False, id="one_index_list"),
            pytest.param([0, 1], False, id="two_index_list"),
            pytest.param((0,), False, id="one_index_tuple"),
            pytest.param([], False, id="empty_list"),
        ],
    )
    def test_explicit_device_forms_do_not_consult_the_accelerator(
        self, devices: int | str | list[int] | tuple[int, ...], expected: bool
    ) -> None:
        """An explicitly named device count resolves without querying the XLA runtime.

        A device-index sequence, as ``train(device='xla:N')`` forwards it, answers ``False``: only the ``int | str``
        count forms are validated on one chip, so a sequence keeps the disabled path instead of raising
        ``AttributeError`` on ``.strip()``.
        """
        assert _xla_resolves_to_single_device(devices) is expected

    @pytest.mark.parametrize("devices", [-1, "auto", "-1", "AUTO"])
    @pytest.mark.parametrize("auto_device_count", [1, 8])
    def test_auto_forms_resolve_through_the_xla_chip_count(
        self, mocker: MockerFixture, devices: int | str, auto_device_count: int
    ) -> None:
        """``auto``/``-1`` must count TPU chips; CUDA's auto branch reports none for XLA."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.auto_device_count", return_value=auto_device_count)

        assert _xla_resolves_to_single_device(devices) is (auto_device_count == 1)

    @pytest.mark.usefixtures("forbidden_auto_device_count")
    @pytest.mark.parametrize("devices", [1, "1", -1, "auto"])
    @pytest.mark.parametrize("num_nodes", [2, 4])
    def test_multiple_nodes_are_never_a_single_device(self, devices: int | str, num_nodes: int) -> None:
        """One device per host across several hosts is a multi-replica run, whatever ``devices`` says."""
        assert _xla_resolves_to_single_device(devices, num_nodes) is False


class TestBuildTrainerLoggers:
    """build_trainer() must wire loggers from TrainConfig flags."""

    @pytest.fixture
    def fake_loggers(self, mocker: MockerFixture) -> dict[str, MagicMock]:
        """Replace the TensorBoard, MLflow and WandB logger classes (and the TensorBoard import probe) with fakes.

        Examples:
            >>> fake_loggers(mocker)  # doctest: +SKIP
            # Needs pytest's mocker fixture; cannot run standalone.
        """
        fakes = {
            "tensorboard": MagicMock(spec=TensorBoardLogger),
            "mlflow": MagicMock(spec=MLFlowLogger),
            "wandb": MagicMock(spec=WandbLogger),
        }
        mocker.patch("rfdetr.training.trainer._try_import_tensorboard_summary_writer")
        mocker.patch("rfdetr.training.trainer.TensorBoardLogger", return_value=fakes["tensorboard"])
        mocker.patch("rfdetr.training.trainer.MLFlowLogger", return_value=fakes["mlflow"])
        mocker.patch("rfdetr.training.trainer.WandbLogger", return_value=fakes["wandb"])
        return fakes

    def test_no_loggers_always_has_csv_logger(self, tmp_path):
        """CSVLogger is always present even when all optional logger flags are off."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())  # _tc already sets all loggers to False
        assert any(isinstance(lg, CSVLogger) for lg in trainer.loggers)

    @pytest.mark.parametrize(
        "flags",
        [
            pytest.param(("tensorboard",), id="tensorboard"),
            pytest.param(("mlflow",), id="mlflow"),
            pytest.param(("wandb",), id="wandb"),
            pytest.param(("tensorboard", "mlflow"), id="tensorboard-and-mlflow-combined"),
        ],
    )
    def test_enabled_loggers_are_wired(self, tmp_path, fake_loggers, flags):
        """Each logger flag adds its logger (dep mocked) to the trainer, and several can be wired simultaneously."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, **dict.fromkeys(flags, True)), _mc())

        assert all(fake_loggers[flag] in trainer.loggers for flag in flags)

    @pytest.mark.parametrize(
        ("target", "error", "flag", "label", "logger_cls"),
        [
            pytest.param(
                "rfdetr.training.trainer._try_import_tensorboard_summary_writer",
                ModuleNotFoundError("no module named 'tensorboard'"),
                "tensorboard",
                "TensorBoard",
                TensorBoardLogger,
                id="tensorboard-missing",
            ),
            pytest.param(
                "rfdetr.training.trainer._try_import_tensorboard_summary_writer",
                AttributeError("`np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead."),
                "tensorboard",
                "TensorBoard",
                TensorBoardLogger,
                id="tensorboard-numpy2-incompatible",
            ),
            pytest.param(
                "rfdetr.training.trainer.WandbLogger",
                ModuleNotFoundError("no module named 'wandb'"),
                "wandb",
                "WandB",
                WandbLogger,
                id="wandb-missing",
            ),
            pytest.param(
                "rfdetr.training.trainer.MLFlowLogger",
                ModuleNotFoundError("no module named 'mlflow'"),
                "mlflow",
                "MLflow",
                MLFlowLogger,
                id="mlflow-missing",
            ),
        ],
    )
    def test_unavailable_logger_dep_warns_and_keeps_csv_logger(
        self, tmp_path, mocker, target, error, flag, label, logger_cls
    ):
        """An absent or incompatible logger dependency logs one warning and falls back to the CSV logger."""
        mocker.patch(target, side_effect=error)
        mock_logger = mocker.patch("rfdetr.training.trainer._logger")

        trainer = build_trainer(_tc(tmp_path, use_ema=False, **{flag: True}), _mc())

        mock_logger.warning.assert_called_once()
        assert label in mock_logger.warning.call_args[0][0]
        assert all(not isinstance(lg, logger_cls) for lg in trainer.loggers)
        assert any(isinstance(lg, CSVLogger) for lg in trainer.loggers)

    def test_clearml_flag_raises_not_implemented(self, tmp_path):
        """Clearml=True must raise NotImplementedError (not yet supported)."""
        with pytest.raises(NotImplementedError, match="ClearML"):
            build_trainer(_tc(tmp_path, clearml=True, use_ema=False), _mc())


class TestBuildTrainerKwargs:
    """build_trainer() must pass the correct kwargs to Trainer."""

    @pytest.mark.parametrize(
        ("keypoint", "expected"),
        [
            pytest.param(True, None, id="keypoint-manual-optimization-clips-itself"),
            pytest.param(False, 0.25, id="detection-automatic-optimization-forwards-it"),
        ],
    )
    def test_gradient_clip_val_follows_optimization_mode(self, tmp_path, keypoint, expected):
        """Trainer-owned clipping is off for keypoint models, whose RFDETRModelModule clips manually, and on
        otherwise."""
        tc, mc = _configs(tmp_path, keypoint=keypoint, clip_max_norm=0.25)
        assert build_trainer(tc, mc).gradient_clip_val == expected

    @pytest.mark.parametrize(
        ("keypoint", "expected"),
        [
            pytest.param(True, 1, id="keypoint-manual-optimization-accumulates-itself"),
            pytest.param(False, 8, id="detection-automatic-optimization-forwards-it"),
        ],
    )
    def test_accumulate_grad_batches_follows_optimization_mode(self, tmp_path, keypoint, expected):
        """Trainer-owned accumulation is off for keypoint models, whose RFDETRModelModule accumulates manually."""
        tc, mc = _configs(tmp_path, keypoint=keypoint, grad_accum_steps=8)
        assert build_trainer(tc, mc).accumulate_grad_batches == expected

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

    def test_trainer_kwargs_can_override_precision(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Explicit trainer kwargs must override default precision without raising."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), precision="32-true")
        assert captured_trainer_kwargs["precision"] == "32-true"

    def test_keypoint_trainer_kwargs_cannot_override_manual_optimization_ownership(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ):
        """Keypoint accumulation and clipping stay disabled when passed as trainer kwargs, with a UserWarning."""
        tc, mc = _configs(tmp_path, keypoint=True)
        with pytest.warns(UserWarning, match="manual optimization"):
            build_trainer(tc, mc, accumulate_grad_batches=8, gradient_clip_val=0.25)

        assert captured_trainer_kwargs["accumulate_grad_batches"] == 1
        assert captured_trainer_kwargs["gradient_clip_val"] is None

    def test_detection_trainer_kwargs_override_takes_effect(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Detection models use automatic optimization; trainer kwargs must override the built-in defaults."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(), accumulate_grad_batches=8, gradient_clip_val=0.25)

        assert captured_trainer_kwargs["accumulate_grad_batches"] == 8
        assert captured_trainer_kwargs["gradient_clip_val"] == pytest.approx(0.25)


class TestBuildTrainerSeed:
    """build_trainer() must not mutate global RNG state."""

    def test_seed_is_not_applied_in_factory(self, mocker: MockerFixture, tmp_path):
        """Seeding is deferred to RFDETRModule.on_fit_start (no factory side-effect)."""
        mock_seed = mocker.patch("pytorch_lightning.seed_everything")

        build_trainer(_tc(tmp_path, use_ema=False, seed=42), _mc())

        mock_seed.assert_not_called()


class TestBuildTrainerDDPFields:
    """build_trainer() must thread devices/num_nodes/strategy from TrainConfig to Trainer."""

    @pytest.mark.parametrize(
        ("extra_kwargs", "captured_key", "expected"),
        [
            pytest.param({"devices": 4}, "devices", 4, id="devices-forwarded-from-train-config"),
            pytest.param({"num_nodes": 2}, "num_nodes", 2, id="num-nodes-forwarded-from-train-config"),
            pytest.param({"strategy": "auto"}, "strategy", "auto", id="strategy-forwarded-from-train-config"),
            pytest.param({}, "devices", 1, id="default-devices-is-1-single-gpu-default"),
            pytest.param({}, "num_nodes", 1, id="default-num-nodes-is-1"),
        ],
    )
    def test_ddp_fields_threaded_from_train_config(
        self,
        captured_trainer_kwargs: dict[str, Any],
        tmp_path,
        extra_kwargs: dict[str, Any],
        captured_key: str,
        expected: object,
    ):
        """TrainConfig.devices/num_nodes/strategy are forwarded to Trainer(...) unchanged, defaults included."""
        build_trainer(_tc(tmp_path, use_ema=False, **extra_kwargs), _mc())

        assert captured_trainer_kwargs[captured_key] == expected

    def test_devices_string_accepted(self, tmp_path):
        """TrainConfig.devices accepts a string value such as 'auto'."""
        assert _tc(tmp_path, use_ema=False, devices="auto").devices == "auto"


class TestBuildTrainerDDPStrategy:
    """build_trainer() builds ``DDPStrategy(find_unused_parameters=True)`` for distributed runs and rejects sharded
    ones."""

    @pytest.mark.parametrize(
        ("keypoint", "segmentation", "tc_kwargs"),
        [
            pytest.param(True, False, {"strategy": "ddp"}, id="keypoint-ddp"),
            pytest.param(True, False, {"strategy": "auto", "devices": 2}, id="keypoint-auto-with-multiple-devices"),
            pytest.param(True, False, {"strategy": "ddp", "num_nodes": 2}, id="keypoint-ddp-across-nodes"),
            pytest.param(False, False, {"strategy": "auto", "devices": 2}, id="detection-auto-with-multiple-devices"),
            pytest.param(False, True, {"strategy": "ddp"}, id="segmentation-ddp-sparse-forward"),
            pytest.param(False, False, {"strategy": "ddp"}, id="detection-ddp-regression-1093"),
            pytest.param(False, True, {"strategy": "ddp_spawn"}, id="segmentation-ddp-spawn"),
        ],
    )
    def test_distributed_strategy_enables_find_unused_parameters(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, keypoint, segmentation, tc_kwargs
    ):
        """Every distributed request becomes DDPStrategy(find_unused_parameters=True), and devices/num_nodes reach
        Trainer.

        Detection can leave parameters unused under DDP (two-stage group_detr ModuleLists, conditional aux_loss branches
        -- regression test for https://github.com/roboflow/rf-detr/issues/1093), segmentation's sparse_forward() is
        another source, and ``strategy='auto'`` with more than one device would otherwise let Lightning pick a
        distributed strategy without RF-DETR's unused-parameter guard. ``ddp_spawn`` must not drop the flag the
        interactive-spawn path already sets.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, segmentation=segmentation, **tc_kwargs)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True
        assert captured_trainer_kwargs["devices"] == tc.devices
        assert captured_trainer_kwargs["num_nodes"] == tc.num_nodes

    @pytest.mark.parametrize("keypoint", [True, False])
    @pytest.mark.parametrize("strategy_name", ["ddp_notebook", "ddp_spawn"])
    def test_spawn_strategies_use_interactive_spawn(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, keypoint, strategy_name
    ):
        """ddp_notebook and ddp_spawn must be replaced with interactive spawn DDPStrategy.

        Fork-based DDP inherits the parent's OpenMP thread pool which is invalid after fork, causing SIGABRT in the
        autograd engine. ddp_spawn is blocked by PTL in notebooks without the override.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, strategy=strategy_name)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._start_method == "spawn"
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    def test_ddp_notebook_raises_clear_error_when_private_launcher_is_missing(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Missing private PTL launcher should raise a targeted compatibility error."""
        mocker.patch("rfdetr.training.trainer._InteractiveSpawnLauncher", None)
        build_trainer(_tc(tmp_path, use_ema=False, strategy="ddp_notebook"), _mc(amp=True))

        strategy = captured_trainer_kwargs["strategy"]
        strategy.cluster_environment = object()
        with pytest.raises(RuntimeError, match="private API"):
            strategy._configure_launcher()

    def test_non_ddp_strategy_with_segmentation_is_unchanged(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Strategies other than 'ddp' must not be wrapped even when segmentation is on."""
        tc, mc = _configs(tmp_path, segmentation=True, strategy="auto")

        build_trainer(tc, mc)

        assert captured_trainer_kwargs["strategy"] == "auto"

    def test_keypoint_auto_devices_multi_gpu_builds_ddp(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Keypoint mode with devices='auto' resolving to multiple CUDA devices builds DDP (no error)."""
        tc, mc = _configs(tmp_path, keypoint=True, strategy="auto", devices="auto", accelerator="cuda")
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    @pytest.mark.parametrize("strategy", ["fsdp", "deepspeed"])
    def test_keypoint_sharded_strategy_string_raises_clear_error(self, tmp_path, strategy):
        """Keypoint mode rejects sharded strategy strings (FSDP/DeepSpeed) with a clear error."""
        tc, mc = _configs(tmp_path, keypoint=True, strategy=strategy)

        with pytest.raises(NotImplementedError, match="sharded distributed strategies"):
            build_trainer(tc, mc)

    def test_keypoint_model_parallel_strategy_object_raises_clear_error(self, tmp_path):
        """Keypoint mode rejects a ModelParallelStrategy (FSDP2) object, whose repr has no 'fsdp' token."""
        tc, mc = _configs(tmp_path, keypoint=True)

        # Strategy objects reach build_trainer via trainer_kwargs (TrainConfig.strategy is a string field).
        with pytest.raises(NotImplementedError, match="sharded distributed strategies"):
            build_trainer(tc, mc, strategy=ModelParallelStrategy(), devices=2)

    def test_keypoint_ddp_strategy_object_without_find_unused_parameters_raises(self, tmp_path):
        """A supplied DDPStrategy() object lacking find_unused_parameters=True is rejected for keypoint DDP."""
        tc, mc = _configs(tmp_path, keypoint=True)

        with pytest.raises(ValueError, match="find_unused_parameters=True"):
            build_trainer(tc, mc, strategy=DDPStrategy(), devices=2)

    def test_keypoint_ddp_strategy_object_with_find_unused_parameters_ok(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ):
        """A supplied DDPStrategy(find_unused_parameters=True) object passes through unchanged for keypoint DDP."""
        supplied = DDPStrategy(find_unused_parameters=True)
        tc, mc = _configs(tmp_path, keypoint=True)

        build_trainer(tc, mc, strategy=supplied, devices=2)

        assert captured_trainer_kwargs["strategy"] is supplied
        assert captured_trainer_kwargs["strategy"]._ddp_kwargs.get("find_unused_parameters") is True


class TestBuildTrainerDDPStaticGraph:
    """build_trainer() enables DDP ``static_graph`` only for the configurations the throughput measurement covered.

    Those are detection and segmentation with ``grad_accum_steps<=1``; keypoint models and ``grad_accum_steps>1`` were
    not exercised by that measurement, so ``static_graph`` and ``gradient_as_bucket_view`` stay off there.
    """

    @pytest.mark.parametrize(
        ("keypoint", "segmentation", "tc_kwargs", "expected"),
        [
            pytest.param(False, False, {"strategy": "ddp"}, True, id="ddp-detection"),
            pytest.param(False, True, {"strategy": "ddp"}, True, id="ddp-segmentation-same-always-unused-mask-token"),
            pytest.param(False, False, {"strategy": "ddp_spawn"}, True, id="ddp-spawn-via-interactive-spawn"),
            pytest.param(False, False, {"strategy": "auto", "devices": 2}, True, id="auto-with-multiple-devices"),
            pytest.param(False, False, {"strategy": "ddp", "grad_accum_steps": 2}, False, id="grad-accum-disables"),
            pytest.param(True, False, {"strategy": "ddp"}, False, id="keypoint-disables"),
        ],
    )
    def test_static_graph_follows_measured_configurations(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, keypoint, segmentation, tc_kwargs, expected
    ) -> None:
        """static_graph and gradient_as_bucket_view are on exactly for the measured configurations.

        An empirical probe (real RFDETRSegNano forward, real/empty-target batches) found the same single always-unused
        parameter (backbone mask_token) for segmentation as for detection-only, so segmentation stays in scope.
        ``strategy='auto'`` with more than one device takes the same ``distributed_requested`` path. Gradient
        accumulation is excluded because DDP's no_sync() across several backward calls interacts with static_graph's
        iteration-counted structure-learning in a way the measurement (which used none) did not verify. Keypoint models'
        manual-optimization DDP path was not measured either.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, segmentation=segmentation, **tc_kwargs)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True
        assert strategy_obj._ddp_kwargs.get("static_graph") is expected
        assert strategy_obj._ddp_kwargs.get("gradient_as_bucket_view") is expected

    def test_ddp_trainer_kwargs_accumulate_grad_batches_disables_static_graph(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        """A caller passing accumulate_grad_batches= directly (bypassing tc.grad_accum_steps) must also disable
        static_graph.

        Regression test: this exact path (build_trainer(..., accumulate_grad_batches=2), tc.grad_accum_steps left
        at its default of 1) crashed a real 2-rank gloo DDP run with ``RuntimeError: expect_autograd_hooks_
        INTERNAL ASSERT FAILED`` in ``reducer.cpp`` when static_graph was gated on tc.grad_accum_steps alone --
        trainer_kwargs["accumulate_grad_batches"] overrides tc.grad_accum_steps later in this same function
        (see the "accumulate_grad_batches" resolution below) and must be read the same way here.
        """
        tc, mc = _configs(tmp_path, strategy="ddp")
        assert tc.grad_accum_steps <= 1

        build_trainer(tc, mc, accumulate_grad_batches=2)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("static_graph") is False
        assert strategy_obj._ddp_kwargs.get("gradient_as_bucket_view") is False


class TestBuildTrainerEvalMode:
    """``include_training_callbacks=False`` builds a lean eval-only trainer (issue #1110).

    The eval path keeps only the metric callback (and progress bar) so ``RFDETR.evaluate()`` does not write checkpoints
    or training logs to ``output_dir`` or run training-only callbacks such as EMA and early stopping.
    """

    def test_coco_eval_callback_present(self, tmp_path):
        """The metric callback is retained — it computes the returned COCO metrics."""
        trainer = build_trainer(_tc(tmp_path), _mc(), include_training_callbacks=False)
        assert any(isinstance(cb, COCOEvalCallback) for cb in trainer.callbacks)

    @pytest.mark.parametrize(
        ("tc_kwargs", "callback_type"),
        [
            pytest.param({}, ModelCheckpoint, id="no-checkpoint-including-best-model"),
            pytest.param({"use_ema": True}, RFDETREMACallback, id="no-ema"),
            pytest.param({"early_stopping": True}, RFDETREarlyStopping, id="no-early-stopping"),
        ],
    )
    def test_training_only_callbacks_are_absent(self, tmp_path, tc_kwargs, callback_type):
        """Checkpointing (BestModelCallback included), EMA and early stopping are training-only and absent in eval
        mode."""
        trainer = build_trainer(_tc(tmp_path, **tc_kwargs), _mc(), include_training_callbacks=False)
        assert not [cb for cb in trainer.callbacks if isinstance(cb, callback_type)]

    def test_loggers_disabled(self, tmp_path):
        """No loggers are attached so no metrics.csv / lightning_logs are written."""
        trainer = build_trainer(_tc(tmp_path), _mc(), include_training_callbacks=False)
        assert trainer.loggers == []

    def test_training_callbacks_present_by_default(self, tmp_path):
        """The default (training) path is unchanged: BestModelCallback is still wired."""
        trainer = build_trainer(_tc(tmp_path), _mc())
        assert any(isinstance(cb, BestModelCallback) for cb in trainer.callbacks)


class TestEvalIntervalValidationGating:
    """eval_interval must gate the whole validation loop, not just metric logging."""

    @pytest.mark.parametrize(
        ("tc_kwargs", "expected_interval", "expects_force_callback"),
        [
            pytest.param({"eval_interval": 3, "epochs": 10}, 3, True, id="interval-above-one-forces-last-epoch"),
            pytest.param({}, 1, False, id="default-validates-every-epoch"),
        ],
    )
    def test_check_val_every_n_epoch_mirrors_eval_interval(
        self, tmp_path, tc_kwargs, expected_interval, expects_force_callback
    ):
        """check_val_every_n_epoch mirrors eval_interval so Lightning skips whole val epochs.

        An interval above 1 also wires the callback guaranteeing last-epoch validation; the default of 1 needs none.
        """
        trainer = build_trainer(_tc(tmp_path, **tc_kwargs), _mc(), accelerator="cpu")

        assert trainer.check_val_every_n_epoch == expected_interval
        has_force_callback = any(isinstance(cb, _ForceLastEpochValidationCallback) for cb in trainer.callbacks)
        assert has_force_callback is expects_force_callback

    @pytest.mark.parametrize(
        ("max_epochs", "current_epoch", "expected_interval"),
        [
            (10, 9, 1),  # final epoch: validation is re-enabled
            (10, 8, 3),  # before the final epoch: the interval is untouched
            (None, 8, 3),  # PTL's not-yet-known sentinel
            (-1, 8, 3),  # PTL's unlimited sentinel
        ],
    )
    def test_force_last_epoch_callback_resets_interval_only_on_final_epoch(
        self, max_epochs, current_epoch, expected_interval
    ):
        """On the final epoch start the callback re-enables validation; otherwise the interval is left untouched.

        The guard is ``isinstance(max_epochs, int) and max_epochs > 0``: ``None`` and ``-1`` are both PTL-permitted
        values with no well-defined "final epoch" to force.
        """
        trainer = MagicMock()
        trainer.max_epochs = max_epochs
        trainer.current_epoch = current_epoch
        trainer.check_val_every_n_epoch = 3

        _ForceLastEpochValidationCallback().on_train_epoch_start(trainer, MagicMock())

        assert trainer.check_val_every_n_epoch == expected_interval

    def test_real_fit_validates_final_epoch_despite_eval_interval_skip(
        self, base_model_config, base_train_config, mocker: MockerFixture
    ):
        """A real trainer.fit() must validate the final epoch even when it isn't an eval_interval multiple.

        epochs=4, eval_interval=3: Lightning's own check_val_every_n_epoch=3 gating would only validate at (0-indexed)
        epoch 2 (current_epoch+1==3). Epoch 3 (the final epoch, 4 % 3 != 0) is only reached because
        _ForceLastEpochValidationCallback resets check_val_every_n_epoch=1 when the final epoch starts — this is a
        behavioral regression guard, not an attribute-level check.
        """
        from rfdetr.training.module_data import RFDETRDataModule
        from rfdetr.training.module_model import RFDETRModelModule

        from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

        mc = base_model_config()
        tc = base_train_config(epochs=4, eval_interval=3, use_ema=False, run_test=False)

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4))
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        )

        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)

        validated_epochs: list[int] = []
        original_validation_step = module.validation_step

        def _recording_validation_step(batch, batch_idx):
            validated_epochs.append(module.trainer.current_epoch)
            return original_validation_step(batch, batch_idx)

        module.validation_step = _recording_validation_step

        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(module, datamodule=datamodule)

        assert validated_epochs == [2, 3], (
            f"expected validation on eval_interval epoch 2 and forced final epoch 3, got {validated_epochs}"
        )


class TestFloat32MatmulPrecision:
    """build_trainer must enable TF32 matmul on every entry path (CLI included)."""

    @pytest.fixture(autouse=True)
    def _restore_matmul_precision(self):
        """Snapshot and restore the process-global matmul precision around each test."""
        previous = torch.get_float32_matmul_precision()
        yield
        torch.set_float32_matmul_precision(previous)

    def test_build_trainer_sets_matmul_precision_high(self, tmp_path):
        """build_trainer upgrades the default 'highest' to 'high' (TF32 on Ampere+)."""
        torch.set_float32_matmul_precision("highest")

        build_trainer(_tc(tmp_path), _mc(), accelerator="cpu")
        assert torch.get_float32_matmul_precision() == "high"


class TestAcceleratorResolvesToXLA:
    """``_accelerator_resolves_to_xla`` must agree with Lightning's own ``accelerator="auto"`` resolution.

    Unlike ``TestMultiDeviceXLAStrategy`` below, this does not need real ``torch_xla`` or a chip: it mocks
    ``XLAAccelerator.is_available`` directly and asserts on the pure helper, not on a constructed ``Trainer``.
    """

    @pytest.mark.parametrize(
        ("accelerator", "xla_available", "expected"),
        [
            ("tpu", False, True),
            ("xla", False, True),
            ("TPU", False, True),
            ("auto", True, True),
            ("auto", False, False),
            ("cpu", True, False),
            ("gpu", True, False),
        ],
    )
    def test_matches_lightnings_own_auto_resolution(
        self, accelerator, xla_available, expected, mocker: MockerFixture
    ) -> None:
        """A caller who leaves ``accelerator="auto"`` resolves to XLA only when XLA is available, as Lightning would.

        ``"auto"`` is TrainConfig's own default, and build_trainer decides the strategy/precision arguments before
        Trainer performs that resolution itself. An explicit ``"tpu"``/``"xla"`` (case-insensitive) is always XLA, and
        an explicit ``"cpu"``/``"gpu"`` never is, even when XLA is available.
        """
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=xla_available)

        assert _accelerator_resolves_to_xla(accelerator) is expected


class TestMultiDeviceXLAStrategy:
    """`XLAAccelerator` pairs only with `SingleDeviceXLAStrategy` or `XLAStrategy`.

    The first four tests are marked ``xla`` and guarded with ``requires_torch_xla`` because they exercise
    ``build_trainer``'s real, unpatched ``XLAPrecision`` construction, which needs ``torch_xla`` present -- no chip is
    touched, so the CPU-PJRT lane runs them. The later tests instead patch ``XLAPrecision`` (and, for the
    ``accelerator="auto"`` case, ``XLAAccelerator.is_available``) the same way ``TestBuildTrainerPrecision`` does, so
    they run on every lane without needing real ``torch_xla``.

    RF-DETR's generic ``strategy="auto"`` distributed branch creates ``DDPStrategy`` before Lightning can resolve an XLA
    accelerator. The guard therefore selects ``"xla"`` only when multiple local XLA devices are requested; one-device-
    per-host XLA is not claimed supported. Single-device XLA is unaffected because Lightning resolves ``"auto"`` to
    `SingleDeviceXLAStrategy`. The guard applies to explicit ``accelerator="tpu"``/``"xla"`` and to
    ``accelerator="auto"`` once it resolves to XLA (``TestAcceleratorResolvesToXLA`` above), for both segmentation and
    plain detection configs; only ``has_keypoints`` is excluded.
    """

    @pytest.mark.xla
    @requires_torch_xla
    def test_multiple_xla_devices_select_the_xla_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """Without this, asking for more than one chip fails with `found DDPStrategy`."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=4)

        assert captured_trainer_kwargs["strategy"] == "xla"

    @pytest.mark.xla
    @requires_torch_xla
    def test_single_xla_device_keeps_auto(self, captured_trainer_kwargs: dict[str, Any], tmp_path) -> None:
        """One chip already resolves to SingleDeviceXLAStrategy, so nothing should be overridden."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=1)

        assert captured_trainer_kwargs["strategy"] == "auto"

    @pytest.mark.xla
    @requires_torch_xla
    def test_an_explicit_strategy_is_never_overridden(self, captured_trainer_kwargs: dict[str, Any], tmp_path) -> None:
        """A caller who names a strategy owns that choice, even on multi-device XLA.

        The new guard only rewrites ``"auto"``, so an explicit ``"ddp"`` string still reaches the pre-existing,
        unrelated ``strategy_name == "ddp"`` branch further down in ``build_trainer`` (see
        ``TestBuildTrainerDDPStrategy`` above), which always turns it into a ``DDPStrategy`` object -- on XLA and off it
        alike. Asserting the literal string ``"ddp"`` here would be wrong regardless of this PR.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=4, strategy="ddp")

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    @pytest.mark.xla
    @requires_torch_xla
    def test_multi_device_xla_strategy_is_not_selected_for_keypoint_models(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """Keypoint models keep resolving to DDPStrategy on multi-device XLA instead of being promoted to `"xla"`.

        Keypoint training uses manual optimization and the DDP-specific ``find_unused_parameters=True`` handling a few
        lines below this guard (see the keypoint block above) has no validated XLA equivalent, so this combination is
        deliberately excluded from the fix and keeps hitting the pre-existing `DDPStrategy`/`XLAAccelerator` mismatch
        rather than running unverified.
        """
        build_trainer(_kp_tc(tmp_path, use_ema=False), _mc(use_grouppose_keypoints=True), accelerator="tpu", devices=4)

        assert isinstance(captured_trainer_kwargs["strategy"], DDPStrategy)

    def test_accelerator_auto_resolves_to_xla_strategy_when_xla_is_available(
        self, captured_trainer_kwargs: dict[str, Any], mock_xla_precision: MagicMock, mocker: MockerFixture, tmp_path
    ) -> None:
        """``accelerator="auto"`` -- TrainConfig's own default -- must be covered too, not just an explicit string.

        A caller who leaves ``accelerator`` unset passes literal ``"auto"`` here. Without
        ``_accelerator_resolves_to_xla``, RF-DETR's generic distributed branch would create ``DDPStrategy`` before
        Lightning resolves that value to XLA, causing the `XLAAccelerator`/`DDPStrategy` mismatch. Not marked
        ``xla``/``requires_torch_xla``: it patches ``XLAPrecision`` and ``XLAAccelerator.is_available`` directly, so
        it runs on every CI lane rather than only the CPU-PJRT one.
        """
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="auto", devices=4)

        assert captured_trainer_kwargs["strategy"] == "xla"
        assert "precision" not in captured_trainer_kwargs
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_node_single_device_keeps_the_existing_ddp_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """One device per host must not be promoted to unsupported ``XLAStrategy``.

        The generic distributed branch still creates ``DDPStrategy`` for ``devices=1, num_nodes=2``. That topology needs
        real multi-host XLA validation before it can select ``XLAStrategy``; this test only prevents the local strategy
        promotion from claiming it is supported. ``accelerator="tpu"`` is explicit here, so only ``XLAPrecision`` needs
        patching.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=1, num_nodes=2)

        assert isinstance(captured_trainer_kwargs["strategy"], DDPStrategy)

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_device_xla_strategy_is_selected_for_segmentation_models(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """A segmentation config reaches the same guard as plain detection, since only ``has_keypoints`` is excluded.

        ``segmentation_head.sparse_forward()`` is one of the documented reasons the pre-existing DDP branch a few lines
        below needs ``find_unused_parameters=True`` (unused parameters on some forward steps); this guard does not
        special-case segmentation, so it must still promote it to ``"xla"`` on multi-device auto XLA rather than
        silently falling through to some other strategy.
        """
        build_trainer(
            _tc(tmp_path, use_ema=False), _mc(amp=False, segmentation_head=True), accelerator="tpu", devices=4
        )

        assert captured_trainer_kwargs["strategy"] == "xla"
