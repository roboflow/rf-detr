# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for build_trainer(): return type, callbacks, keypoint defaults, loggers, trainer kwargs and eval gating."""

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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.trainer.call import _call_callbacks_load_state_dict, _call_callbacks_on_load_checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from rfdetr.config import RFDETRKeypointPreviewConfig, SegmentationTrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.drop_schedule import DropPathCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.trainer import _ForceLastEpochValidationCallback
from tests._markers import _requires_hotcoco
from tests.training._trainer_support import _configs, _kp_tc, _mc, _tc


def _kp_preview_mc():
    """Keypoint preview model config without pretrained weights.

    Examples:
        >>> _kp_preview_mc().use_grouppose_keypoints
        True
    """
    return RFDETRKeypointPreviewConfig(pretrain_weights=None)


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

    @_requires_hotcoco
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

    @_requires_hotcoco
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

    @_requires_hotcoco
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

    @pytest.mark.parametrize(
        ("make_tc", "make_mc", "expected_monitor", "expected_monitor_ema"),
        [
            pytest.param(
                lambda p: _tc(p, use_ema=True, min_class_support=5),
                _mc,
                "val/mAP_50_95_gated",
                "val/ema_mAP_50_95_gated",
                id="detection-map-gated",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=False, min_class_support=5),
                _mc,
                "val/mAP_50_95_gated",
                None,
                id="detection-map-gated-without-ema",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True, min_class_support=5),
                lambda: _mc(segmentation_head=True),
                "val/segm_mAP_50_95_gated",
                "val/ema_segm_mAP_50_95_gated",
                id="segmentation-map-gated",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True, min_class_support=0),
                _mc,
                "val/mAP_50_95",
                "val/ema_mAP_50_95",
                id="no-floor-keeps-ungated-keys",
            ),
            pytest.param(
                lambda p: _tc(p, use_ema=True, min_class_support=5, best_model_metric="mar"),
                _mc,
                "val/mAR",
                "val/ema_mAR",
                id="mar-has-no-gated-variant",
            ),
            pytest.param(
                lambda p: _kp_tc(p, use_ema=True, min_class_support=5),
                _kp_preview_mc,
                "val/keypoint_map_50_95",
                "val/ema_keypoint_map_50_95",
                id="keypoint-oks-map-has-no-gated-variant",
            ),
        ],
    )
    def test_min_class_support_selects_gated_monitor(
        self, tmp_path, make_tc, make_mc, expected_monitor, expected_monitor_ema
    ):
        """Under a support floor, best-model selection and early stopping follow the gated box/mask mAP."""
        trainer = build_trainer(make_tc(tmp_path), make_mc())
        best_cb = _find_callback(trainer, BestModelCallback)
        assert best_cb.monitor == expected_monitor
        assert best_cb._monitor_ema == expected_monitor_ema

    def test_min_class_support_early_stopping_monitors_gated_keys(self, tmp_path):
        """RFDETREarlyStopping reads the same gated keys BestModelCallback selects on."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True, min_class_support=5, early_stopping=True), _mc())
        early = _find_callback(trainer, RFDETREarlyStopping)
        assert early._monitor_regular == "val/mAP_50_95_gated"
        assert early._monitor_ema == "val/ema_mAP_50_95_gated"

    @pytest.mark.parametrize(
        ("saved_floor", "resumed_floor", "restores_state"),
        [
            pytest.param(0, 0, True, id="keep-ungated"),
            pytest.param(0, 5, False, id="enable-gating"),
            pytest.param(5, 8, False, id="change-floor"),
            pytest.param(5, 0, False, id="disable-gating"),
            pytest.param(5, 5, True, id="keep-floor"),
        ],
    )
    def test_support_floor_controls_selection_state_restore(
        self, tmp_path: Path, saved_floor: int, resumed_floor: int, restores_state: bool
    ) -> None:
        """Lightning loads best-score and patience history only for the same support floor."""
        saved = build_trainer(_tc(tmp_path, min_class_support=saved_floor, early_stopping=True), _mc())
        saved_best = _find_callback(saved, BestModelCallback)
        saved_early = _find_callback(saved, RFDETREarlyStopping)
        saved_best.best_model_score = torch.tensor(0.75)
        saved_early.best_score = torch.tensor(0.75)
        saved_early.wait_count = 2
        checkpoint = {
            "callbacks": {
                saved_best.state_key: saved_best.state_dict(),
                saved_early.state_key: saved_early.state_dict(),
            }
        }

        resumed = build_trainer(_tc(tmp_path, min_class_support=resumed_floor, early_stopping=True), _mc())
        _call_callbacks_load_state_dict(resumed, checkpoint)

        best = _find_callback(resumed, BestModelCallback)
        early = _find_callback(resumed, RFDETREarlyStopping)
        assert best.best_model_score == (torch.tensor(0.75) if restores_state else None)
        assert early.best_score.item() == (0.75 if restores_state else -float("inf"))
        assert early.wait_count == (2 if restores_state else 0)

    @pytest.mark.parametrize(
        ("saved_floor", "resumed_floor", "restores_state"),
        [
            pytest.param(5, 0, False, id="disable-gating"),
            pytest.param(5, 8, False, id="change-floor"),
            pytest.param(None, 5, False, id="unknown-floor"),
            pytest.param(5, 5, True, id="match-known-floor"),
        ],
    )
    def test_legacy_gated_checkpoint_restores_only_matching_history(
        self, tmp_path: Path, saved_floor: int | None, resumed_floor: int, restores_state: bool
    ) -> None:
        """Older gated checkpoint keys require the serialized floor to establish a match."""
        saved_best = BestModelCallback(output_dir=str(tmp_path / "out"), monitor_regular="val/mAP_50_95_gated")
        saved_early = RFDETREarlyStopping(monitor_regular="val/mAP_50_95_gated")
        saved_best.best_model_score = torch.tensor(0.75)
        saved_early.best_score = torch.tensor(0.75)
        saved_early.wait_count = 2
        checkpoint = {
            "pytorch-lightning_version": "2.6.0",
            "callbacks": {
                saved_best.state_key: saved_best.state_dict(),
                saved_early.state_key: saved_early.state_dict(),
            },
            "args": {"min_class_support": saved_floor},
        }
        resumed = build_trainer(_tc(tmp_path, min_class_support=resumed_floor, early_stopping=True), _mc())

        with pytest.warns(UserWarning, match="callbacks used to create the checkpoint"):
            _call_callbacks_on_load_checkpoint(resumed, checkpoint)
        _call_callbacks_load_state_dict(resumed, checkpoint)

        best = _find_callback(resumed, BestModelCallback)
        early = _find_callback(resumed, RFDETREarlyStopping)
        assert best.best_model_score == (torch.tensor(0.75) if restores_state else None)
        assert early.best_score.item() == (0.75 if restores_state else -float("inf"))
        assert early.wait_count == (2 if restores_state else 0)

    @pytest.mark.parametrize(
        ("tc_kwargs", "expected_support", "expected_agnostic"),
        [
            pytest.param({}, 0, False, id="defaults"),
            pytest.param({"min_class_support": 7, "class_agnostic_eval": True}, 7, True, id="both-set"),
        ],
    )
    def test_support_floor_and_agnostic_flag_reach_coco_eval_callback(
        self, tmp_path, tc_kwargs, expected_support, expected_agnostic
    ):
        """TrainConfig.min_class_support / class_agnostic_eval are forwarded to COCOEvalCallback."""
        trainer = build_trainer(_tc(tmp_path, **tc_kwargs), _mc())
        coco_cb = _find_callback(trainer, COCOEvalCallback)
        assert coco_cb._min_class_support == expected_support
        assert coco_cb._class_agnostic is expected_agnostic

    def test_min_class_support_rejects_negative_values(self, tmp_path):
        """A negative support floor is a configuration error, not a silently disabled gate."""
        with pytest.raises(ValueError, match="min_class_support"):
            _tc(tmp_path, min_class_support=-1)

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
        from tests.training.helpers import (
            _fake_postprocess,
            _FakeCriterion,
            _FakeDataset,
            _make_param_dicts,
            _TinyModel,
        )

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


class _LongTailDataset(torch.utils.data.Dataset):
    """Four images that each hold one class-1 box; image 0 also holds the only class-2 box.

    ``orig_size`` is ``(32, 32 + idx)``, so a scripted postprocessor can tell from the width which image it scores.
    """

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        boxes = [[0.5, 0.5, 0.5, 0.5]] + ([[0.15, 0.15, 0.2, 0.2]] if idx == 0 else [])
        target = {
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor([1, 2][: len(boxes)]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([32, 32 + idx]),
            "size": torch.tensor([32, 32]),
        }
        return torch.randn(3, 32, 32), target


class TestMinClassSupportModelSelection:
    """A real fit keeps the checkpoint the gated mAP prefers, not the one a two-instance class would pick."""

    @pytest.mark.parametrize(
        ("min_class_support", "expected_monitor", "expected_best_epoch", "expected_best_score"),
        [
            pytest.param(0, "val/mAP_50_95", 0, 0.5, id="ungated-ties-and-keeps-first-epoch"),
            pytest.param(2, "val/mAP_50_95_gated", 1, 1.0, id="gated-prefers-common-class-epoch"),
        ],
    )
    def test_real_fit_selects_best_epoch_on_gated_map(
        self,
        base_model_config,
        base_train_config,
        mocker: MockerFixture,
        min_class_support,
        expected_monitor,
        expected_best_epoch,
        expected_best_score,
    ):
        """Epoch 0 finds only the single class-2 box, epoch 1 only the four class-1 boxes.

        All-class mAP is 0.5 on both epochs, so it cannot tell them apart and keeps the first. With a floor of 2 the
        one-instance class leaves the monitored mean, which then reads 0.0 and 1.0.
        """
        from rfdetr.training.module_data import RFDETRDataModule
        from rfdetr.training.module_model import RFDETRModelModule
        from tests.training.helpers import _FakeCriterion, _make_param_dicts, _TinyModel

        mc = base_model_config()
        tc = base_train_config(epochs=2, use_ema=False, run_test=False, min_class_support=min_class_support)
        module_holder: list[RFDETRModelModule] = []

        def _scripted_postprocess(outputs, orig_sizes):
            epoch = module_holder[0].trainer.current_epoch
            results = []
            for height, width in orig_sizes.tolist():
                if epoch == 0:
                    found = width == 32
                    boxes = [[0.05 * width, 0.05 * height, 0.25 * width, 0.25 * height]] if found else []
                    labels = [2] if found else []
                else:
                    boxes = [[0.25 * width, 0.25 * height, 0.75 * width, 0.75 * height]]
                    labels = [1]
                results.append(
                    {
                        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                        "scores": torch.full((len(labels),), 0.9),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )
            return results

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=_scripted_postprocess)),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_LongTailDataset())
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        )

        module = RFDETRModelModule(mc, tc)
        module_holder.append(module)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        best_cb = _find_callback(trainer, BestModelCallback)
        saved_epochs: list[int] = []
        save_checkpoint = best_cb._save_checkpoint

        def _recording_save_checkpoint(trainer, filepath):
            saved_epochs.append(trainer.current_epoch)
            save_checkpoint(trainer, filepath)

        mocker.patch.object(best_cb, "_save_checkpoint", side_effect=_recording_save_checkpoint)

        trainer.fit(module, datamodule=RFDETRDataModule(mc, tc))

        assert best_cb.monitor == expected_monitor
        assert float(trainer.callback_metrics["val/mAP_50_95"]) == pytest.approx(0.5, abs=1e-4)
        assert ("val/mAP_50_95_gated" in trainer.callback_metrics) is (min_class_support > 0)
        assert float(best_cb.best_model_score) == pytest.approx(expected_best_score, abs=1e-4)
        assert saved_epochs[-1] == expected_best_epoch
