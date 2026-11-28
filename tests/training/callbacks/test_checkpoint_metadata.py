# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for :class:`BestModelCallback` checkpoint lifecycle and metadata, and :class:`RFDETREarlyStopping`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from rfdetr.config import RFDETRLargeDeprecatedConfig, RFDETRMediumConfig, TrainConfig
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping, _save_final_checkpoint
from rfdetr.training.callbacks.ema import RFDETREMACallback
from tests.training.callbacks.helpers import _REGULAR_AND_EMA_CASES, _EmaMetricModule, _make_test_step_module
from tests.training.callbacks.helpers import _make_best_model_pl_module as _make_pl_module
from tests.training.callbacks.helpers import _make_best_model_trainer as _make_trainer


class TestCheckpointLifecycle:
    """Checkpoint files and tracking state belong only to ordinary fit validation."""

    @pytest.mark.parametrize(
        ("fast_dev_run", "trainer_fn", "sanity_checking"),
        [
            pytest.param(True, TrainerFn.FITTING, False, id="fast-dev-run"),
            pytest.param(False, TrainerFn.VALIDATING, False, id="standalone-validate"),
            pytest.param(False, TrainerFn.TESTING, False, id="standalone-test"),
            pytest.param(False, TrainerFn.PREDICTING, False, id="standalone-predict"),
            pytest.param(False, TrainerFn.FITTING, True, id="sanity-check"),
        ],
    )
    def test_suppressed_validation_preserves_tracking_state(
        self, tmp_path: Path, fast_dev_run: bool, trainer_fn: TrainerFn, sanity_checking: bool
    ) -> None:
        """Suppressed hooks must neither write files nor seed resumed metric bookkeeping."""
        callback = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", smooth_alpha=0.5)
        previous_module = _make_pl_module()
        callback._current_pl_module = previous_module
        callback._best_ema = 0.2
        callback._smoothed_regular = 0.1
        before = callback.state_dict()
        trainer = _make_trainer({"val/mAP_50_95": 0.8, "val/ema_mAP_50_95": 0.9})
        trainer.fast_dev_run = fast_dev_run
        trainer.state.fn = trainer_fn
        trainer.sanity_checking = sanity_checking

        callback.on_validation_end(trainer, _make_pl_module())

        assert callback.state_dict() == before
        assert callback._current_pl_module is previous_module
        assert not list(tmp_path.glob("*.pth"))

    @pytest.mark.parametrize("evaluates_base_model", [True, False])
    def test_ema_tracking_is_independent_of_regular_save_step(self, tmp_path: Path, evaluates_base_model: bool) -> None:
        """Validation before the first optimizer step can still update the independent EMA track."""
        callback = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            evaluates_base_model=evaluates_base_model,
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.2, "val/ema_mAP_50_95": 0.8})
        trainer.global_step = 0

        callback.on_validation_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        assert callback._best_ema == pytest.approx(0.8)
        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    @pytest.mark.parametrize("is_global_zero", [True, False])
    def test_fast_dev_fit_end_preserves_existing_checkpoint(self, tmp_path: Path, is_global_zero: bool) -> None:
        """A dry run must not promote, reload or test an earlier run's checkpoints on any rank."""
        module = _make_test_step_module()
        callback = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        trainer = _make_trainer({}, is_global_zero=is_global_zero)
        trainer.fast_dev_run = True
        checkpoint = tmp_path / "checkpoint_best_total.pth"
        torch.save({"model": {"w": torch.ones(1)}, "args": {}}, checkpoint)
        before = checkpoint.read_bytes()

        callback.on_fit_end(trainer, module)

        assert checkpoint.read_bytes() == before
        assert {p.name for p in tmp_path.glob("*.pth")} == {checkpoint.name}
        module.model.load_state_dict.assert_not_called()
        trainer.test.assert_not_called()
        trainer.strategy.barrier.assert_not_called()
        trainer.strategy.broadcast.assert_not_called()

    @pytest.mark.parametrize(
        ("operation", "fast_dev_run", "evaluates_base_model", "expected"),
        [
            pytest.param("fit", True, True, set(), id="fast-dev-base"),
            pytest.param("fit", True, False, set(), id="fast-dev-ema"),
            pytest.param("validate", False, True, set(), id="validate-base"),
            pytest.param("validate", False, False, set(), id="validate-ema"),
            pytest.param(
                "fit",
                False,
                True,
                {"checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth", "checkpoint_best_regular.pth"},
                id="fit-base",
            ),
            pytest.param(
                "fit",
                False,
                False,
                {"checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth"},
                id="fit-ema",
            ),
        ],
    )
    def test_real_trainer_checkpoint_lifecycle(
        self,
        tmp_path: Path,
        loaders: tuple[DataLoader, DataLoader],
        operation: str,
        fast_dev_run: bool,
        evaluates_base_model: bool,
        expected: set[str],
    ) -> None:
        """Real Lightning hooks suppress dry-run/standalone writes while retaining both fit policies."""
        train_loader, val_loader = loaders
        callback = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            evaluates_base_model=evaluates_base_model,
            run_test=False,
        )
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            fast_dev_run=fast_dev_run,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
            callbacks=[callback, RFDETREMACallback()],
            default_root_dir=str(tmp_path),
        )
        module = _EmaMetricModule(regular_value=0.2, ema_value=0.8)

        loader_args = {
            "fit": {"train_dataloaders": train_loader, "val_dataloaders": val_loader},
            "validate": {"dataloaders": val_loader, "verbose": False},
        }
        getattr(trainer, operation)(module, **loader_args[operation])

        assert {p.name for p in tmp_path.glob("*.pth")} == expected


# ---------------------------------------------------------------------------
# model_name in checkpoint payload (#887)
# ---------------------------------------------------------------------------


class TestCheckpointModelName:
    """Verify model_name is stored in checkpoint payloads."""

    @pytest.mark.parametrize(
        ("cb_kwargs", "trainer_metrics", "model_config_factory", "checkpoint_filename", "expected_model_name"),
        [
            pytest.param(
                {},
                {"val/mAP_50_95": 0.5},
                lambda: MagicMock(model_name="RFDETRLarge"),
                "checkpoint_best_regular.pth",
                "RFDETRLarge",
                id="regular-explicit-model-name",
            ),
            pytest.param(
                {},
                {"val/mAP_50_95": 0.5},
                lambda: RFDETRMediumConfig(model_name=None),
                "checkpoint_best_regular.pth",
                "RFDETRMedium",
                id="regular-infers-model-name-from-config-type",
            ),
            pytest.param(
                {"monitor_ema": "val/ema_mAP_50_95"},
                {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6},
                lambda: MagicMock(model_name="RFDETRMedium"),
                "checkpoint_best_ema.pth",
                "RFDETRMedium",
                id="ema-explicit-model-name",
            ),
        ],
    )
    def test_checkpoint_contains_model_name(
        self,
        tmp_path: Path,
        cb_kwargs: dict,
        trainer_metrics: dict,
        model_config_factory,
        checkpoint_filename: str,
        expected_model_name: str,
    ) -> None:
        """Checkpoint payload stores model_name, explicit or inferred from the model_config type.

        Covers an explicit model_name on a mocked config, inference from a real ModelConfig subclass when unset, and the
        EMA checkpoint path.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), **cb_kwargs)
        trainer = _make_trainer(trainer_metrics)
        pl_module = _make_pl_module()
        pl_module.model_config = model_config_factory()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / checkpoint_filename, weights_only=False)
        assert ckpt["model_name"] == expected_model_name

    def test_regular_checkpoint_model_name_absent_when_not_set(self, tmp_path: Path) -> None:
        """model_name key is absent when model_config has no model_name attribute."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert "model_name" not in ckpt

    def test_deprecated_config_raises_runtime_error(self, tmp_path: Path) -> None:
        """RFDETRLargeDeprecatedConfig raises RuntimeError — deprecated configs are unsupported."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.model_config = RFDETRLargeDeprecatedConfig(model_name=None)

        with pytest.raises(RuntimeError) as exc_info:
            cb.on_validation_end(trainer, pl_module)
        assert "Deprecated model config" in str(exc_info.value)


# ---------------------------------------------------------------------------
# rfdetr_version in checkpoint payload
# ---------------------------------------------------------------------------


class TestCheckpointRfdetrVersion:
    """Verify rfdetr_version is stored in checkpoint payloads."""

    @pytest.mark.parametrize(("cb_kwargs", "metrics", "checkpoint_filename"), _REGULAR_AND_EMA_CASES)
    def test_checkpoint_contains_rfdetr_version(
        self, mocker: MockerFixture, tmp_path: Path, cb_kwargs: dict, metrics: dict, checkpoint_filename: str
    ) -> None:
        """Regular and EMA checkpoints include the rfdetr_version string."""
        mocker.patch("rfdetr.training.callbacks.best_model.get_version", return_value="test-version")
        cb = BestModelCallback(output_dir=str(tmp_path), **cb_kwargs)

        cb.on_validation_end(_make_trainer(metrics), _make_pl_module())

        ckpt = torch.load(tmp_path / checkpoint_filename, weights_only=False)
        assert ckpt["rfdetr_version"] == "test-version"

    def test_best_total_preserves_rfdetr_version_after_strip(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """strip_checkpoint must preserve rfdetr_version in the final checkpoint."""
        mocker.patch("rfdetr.training.callbacks.best_model.get_version", return_value="test-version")
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        data = torch.load(tmp_path / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)
        assert data["rfdetr_version"] == "test-version"

    def test_rfdetr_version_absent_when_get_version_returns_none(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """rfdetr_version must be omitted when get_version() cannot resolve the version."""
        mocker.patch("rfdetr.training.callbacks.best_model.get_version", return_value=None)
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert "rfdetr_version" not in ckpt


class TestCheckpointNotes:
    """Verify user-supplied notes are persisted in .pth checkpoint files."""

    @pytest.mark.parametrize(
        "notes",
        [
            "simple string",
            pytest.param({"date": "2026-01-01", "labeller": "Alice"}, id="dict"),
            pytest.param(["class_a", "class_b"], id="list"),
            42,
            pytest.param("", id="empty-string"),
            pytest.param({}, id="empty-dict"),
            pytest.param([], id="empty-list"),
            0,
            False,
        ],
    )
    def test_notes_stored_in_args_dict(self, tmp_path: Path, notes: object) -> None:
        """Notes from TrainConfig, falsy non-None values included, are stored under checkpoint['args']['notes']."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, notes=notes)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert checkpoint["args"]["notes"] == notes

    def test_notes_absent_when_not_provided(self, tmp_path: Path) -> None:
        """When notes=None (default), no top-level 'notes' key is written to checkpoint."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert "notes" not in checkpoint


# ---------------------------------------------------------------------------
# _serialize_model_config — schema sync from live weights
# ---------------------------------------------------------------------------


def _make_kp_active_mask(schema: list[int]) -> torch.Tensor:
    """Build a bool _kp_active_mask tensor from a keypoints-per-class schema list."""
    max_kp = max(schema) if schema else 0
    mask = torch.zeros(len(schema), max_kp, dtype=torch.bool)
    for cls_idx, n in enumerate(schema):
        mask[cls_idx, :n] = True
    return mask


class TestSerializeModelConfig:
    """Verify _serialize_model_config syncs schema-critical fields from live model weights."""

    @pytest.mark.parametrize(
        ("dumped_config", "weights", "field", "expected"),
        [
            pytest.param(
                {"num_keypoints_per_class": [0, 17], "num_classes": 2},
                {"_kp_active_mask": _make_kp_active_mask([0, 33])},
                "num_keypoints_per_class",
                [0, 33],
                id="keypoint-schema-from-kp-active-mask",
            ),
            pytest.param(
                {"num_keypoints_per_class": [0, 33], "num_classes": 90},
                {"class_embed.weight": torch.zeros(3, 256)},
                "num_classes",
                2,
                id="num-classes-from-class-embed-weight",
            ),
            pytest.param(
                {"num_keypoints_per_class": [0, 17], "num_classes": 2},
                {"w": torch.zeros(1)},
                "num_keypoints_per_class",
                [0, 17],
                id="keypoint-schema-kept-without-kp-active-mask",
            ),
        ],
    )
    def test_schema_field_follows_live_weights(self, dumped_config: dict, weights: dict, field: str, expected) -> None:
        """Stale schema fields in model_config are overridden by the live weights, and left alone without them."""
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_dump.return_value = dumped_config
        pl_module.model.state_dict.return_value = weights

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is not None
        assert result[field] == expected

    def test_no_sync_when_field_absent_from_dumped_config(self) -> None:
        """_kp_active_mask in weights does not insert num_keypoints_per_class when key absent from config."""
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_dump.return_value = {"num_classes": 2}
        pl_module.model.state_dict.return_value = {
            "_kp_active_mask": _make_kp_active_mask([0, 33]),
        }

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is not None
        assert "num_keypoints_per_class" not in result

    def test_returns_none_when_model_config_absent(self) -> None:
        """Returns None when pl_module exposes no model_config."""
        pl_module = _make_pl_module()
        pl_module.model_config = None

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is None

    def test_returns_dict_directly_for_dict_model_config(self) -> None:
        """Returns a dict model_config as-is without any weight-based sync."""
        pl_module = _make_pl_module()
        raw = {"num_classes": 3, "num_keypoints_per_class": [0, 5]}
        pl_module.model_config = raw

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is raw


class TestSaveFinalCheckpoint:
    """Check the final ``.pth`` the fine-tuning cookbooks write after their trainer finishes."""

    def test_saves_the_payload_built_from_the_model_and_both_configs(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """The file holds the checkpoint payload of the live weights, the training config and the model config.

        ``from_checkpoint`` rebuilds the model from this file alone, so each of the three inputs must reach the payload
        builder together with the model name.
        """
        model = torch.nn.Linear(2, 1)
        module = SimpleNamespace(model=model)
        trainer = MagicMock()
        train_config = MagicMock(model_dump=MagicMock(return_value={"epochs": 3}))
        model_config = MagicMock(model_dump=MagicMock(return_value={"resolution": 384}))
        build = mocker.patch.object(BestModelCallback, "_build_checkpoint_payload", return_value={"marker": 1})

        _save_final_checkpoint(module, trainer, train_config, model_config, tmp_path / "final.pth", "RFDETRSmall")

        args, kwargs = build.call_args
        assert args[1:] == ({"epochs": 3}, trainer)
        assert kwargs == {"model_name": "RFDETRSmall", "model_config_dict": {"resolution": 384}}
        assert torch.load(tmp_path / "final.pth") == {"marker": 1}

    def test_unwraps_a_compiled_model_before_taking_its_weights(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """A ``torch.compile`` wrapper prefixes every key with ``_orig_mod.``, so the weights come from inside it."""
        raw = torch.nn.Linear(2, 1)
        module = SimpleNamespace(model=SimpleNamespace(_orig_mod=raw))
        configs = MagicMock()
        build = mocker.patch.object(BestModelCallback, "_build_checkpoint_payload", return_value={})

        _save_final_checkpoint(module, MagicMock(), configs, configs, tmp_path / "final.pth", "RFDETRSmall")

        assert list(build.call_args.args[0]) == list(raw.state_dict())

    def test_creates_the_missing_parent_directories(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """The cookbooks save under a fresh output folder, so the directory is created and the path is returned."""
        module = SimpleNamespace(model=torch.nn.Linear(2, 1))
        configs = MagicMock()
        mocker.patch.object(BestModelCallback, "_build_checkpoint_payload", return_value={})
        output_path = tmp_path / "new" / "folder" / "final.pth"

        result = _save_final_checkpoint(module, MagicMock(), configs, configs, output_path, "RFDETRSmall")

        assert result == output_path
        assert output_path.exists()


# ---------------------------------------------------------------------------
# TestRFDETREarlyStopping
# ---------------------------------------------------------------------------


class TestRFDETREarlyStopping:
    """Verify early stopping logic mirrors legacy EarlyStoppingCallback."""

    def test_patience_not_counted_during_skip_window(self) -> None:
        """Patience wait_count stays 0 and training does not stop during skipped epochs."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001, skip_best_epochs=2)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.9}, current_epoch=0), pl_module)
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=1)
        cb.on_validation_end(trainer, pl_module)
        assert cb.wait_count == 0
        assert trainer.should_stop is False

    @pytest.mark.parametrize(
        ("patience", "metrics", "baseline"),
        [
            pytest.param(1, [0.9, 0.8, 0.7], 0.7, id="patience-1-gentle-decline"),
            pytest.param(2, [0.95, 0.85, 0.40], 0.40, id="patience-2-stronger-skipped-epochs"),
        ],
    )
    def test_first_eligible_epoch_sets_baseline_with_zero_wait(
        self, patience: int, metrics: list[float], baseline: float
    ) -> None:
        """The first eligible epoch becomes the best_score baseline even if skipped epochs scored higher.

        With skip_best_epochs=2 the third epoch is the first eligible one; patience wait_count stays 0.
        """
        cb = RFDETREarlyStopping(patience=patience, min_delta=0.001, skip_best_epochs=2)
        pl_module = _make_pl_module()

        for epoch, value in enumerate(metrics):
            cb.on_validation_end(_make_trainer({"val/mAP_50_95": value}, current_epoch=epoch), pl_module)

        assert cb.best_score.item() == pytest.approx(baseline)
        assert cb.wait_count == 0

    def test_patience_triggers_stop_after_skip_window(self) -> None:
        """Patience counts normally after skip window; triggers stop when exhausted."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001, skip_best_epochs=2)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.9}, current_epoch=0), pl_module)
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.8}, current_epoch=1), pl_module)
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.7}, current_epoch=2), pl_module)
        trainer = _make_trainer({"val/mAP_50_95": 0.7}, current_epoch=3)
        cb.on_validation_end(trainer, pl_module)
        assert trainer.should_stop is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="default"),
            pytest.param({"skip_best_epochs": 0}, id="explicit-zero"),
        ],
    )
    def test_skip_best_epochs_zero_does_not_defer_epoch_zero(self, kwargs: dict) -> None:
        """skip_best_epochs=0 (explicit or default) makes epoch 0 eligible for patience tracking."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001, **kwargs)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        assert cb.best_score is not None
        assert cb.best_score.item() == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "invalid_value, exc_type",
        [
            pytest.param(True, TypeError, id="bool-true"),
            pytest.param(2.5, TypeError, id="float"),
            pytest.param("3", TypeError, id="string"),
            pytest.param(-1, ValueError, id="negative"),
        ],
    )
    def test_skip_best_epochs_invalid_input_raises(self, invalid_value: object, exc_type: type[Exception]) -> None:
        """RFDETREarlyStopping raises TypeError for non-int and ValueError for negative skip_best_epochs."""
        with pytest.raises(exc_type):
            RFDETREarlyStopping(patience=5, skip_best_epochs=invalid_value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("patience", "expect_stop"),
        [(5, False), (3, True)],
    )
    def test_stop_follows_patience_after_stagnation(self, patience: int, expect_stop: bool) -> None:
        """Three epochs without improvement stop training only once ``patience`` is exhausted."""
        cb = RFDETREarlyStopping(patience=patience, min_delta=0.001)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.5}), pl_module)  # baseline

        for _ in range(3):
            trainer = _make_trainer({"val/mAP_50_95": 0.5})
            cb.on_validation_end(trainer, pl_module)

        assert trainer.should_stop is expect_stop
        assert cb.wait_count == 3

    def test_counter_resets_on_improvement(self) -> None:
        """2 stagnant epochs then 1 improvement resets counter to 0."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001)
        pl_module = _make_pl_module()

        # Set baseline
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        # 2 stagnant
        for _ in range(2):
            trainer = _make_trainer({"val/mAP_50_95": 0.5})
            cb.on_validation_end(trainer, pl_module)
        assert cb.wait_count == 2

        # Improvement
        trainer = _make_trainer({"val/mAP_50_95": 0.6})
        cb.on_validation_end(trainer, pl_module)
        assert cb.wait_count == 0

    def test_min_delta_respected(self) -> None:
        """Improvement smaller than min_delta does not reset counter."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.01)
        pl_module = _make_pl_module()

        # Set baseline at 0.5
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        # Improve by only half of min_delta
        trainer = _make_trainer({"val/mAP_50_95": 0.505})
        cb.on_validation_end(trainer, pl_module)
        assert cb.wait_count == 1  # not reset

    @pytest.mark.parametrize(
        ("use_ema", "metrics", "expected_best"),
        [
            pytest.param(
                True, {"val/mAP_50_95": 0.8, "val/ema_mAP_50_95": 0.3}, 0.3, id="use-ema-true-tracks-ema-only"
            ),
            pytest.param(False, {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6}, 0.6, id="use-ema-false-tracks-max"),
        ],
    )
    def test_best_score_follows_use_ema(self, use_ema: bool, metrics: dict, expected_best: float) -> None:
        """With both metrics logged, use_ema=True tracks the EMA value alone and use_ema=False tracks their max."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001, use_ema=use_ema)

        cb.on_validation_end(_make_trainer(metrics), _make_pl_module())

        assert cb.best_score.item() == pytest.approx(expected_best)

    @pytest.mark.parametrize(
        ("metrics", "expected_best"),
        [
            pytest.param({"val/mAP_50_95": 0.45}, 0.45, id="only-regular"),
            pytest.param({"val/ema_mAP_50_95": 0.55}, 0.55, id="only-ema"),
        ],
    )
    def test_single_available_metric_is_tracked(self, metrics: dict, expected_best: float) -> None:
        """When only the regular or only the EMA monitor key is logged, that metric alone sets the baseline.

        Only-EMA is the shape ``eval_ema_only`` produces (``val/mAP_50_95`` never populated, see #1285). Unlike
        ``BestModelCallback.on_validation_end``, where that shape was an outright bug (see
        ``test_ema_checkpoint_saved_when_regular_monitor_absent``), this callback's ``regular_val is None`` branching
        already handled it; this case locks that in.
        """
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001)

        cb.on_validation_end(_make_trainer(metrics), _make_pl_module())

        assert cb.best_score.item() == pytest.approx(expected_best)
        assert cb.wait_count == 0

    @pytest.mark.parametrize("hook", ["on_validation_end", "on_train_epoch_end"])
    def test_hook_without_validation_metrics_is_noop(self, hook: str) -> None:
        """With no validation metric logged, neither hook counts patience or stops training."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001)
        trainer = _make_trainer({})

        getattr(cb, hook)(trainer, _make_pl_module())

        assert cb.wait_count == 0
        assert trainer.should_stop is False

    @pytest.mark.parametrize(
        ("use_ema", "maps", "patience", "min_delta", "expected_stop_epoch"),
        [
            pytest.param(
                False,
                [0.10, 0.20, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
                3,
                0.01,
                5,
                id="use-ema-false-plateau",
            ),
            pytest.param(
                True,
                [0.05, 0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                3,
                0.01,
                5,
                id="use-ema-true-plateau",
            ),
        ],
    )
    def test_trigger_epoch_matches_expected(
        self,
        use_ema: bool,
        maps: list,
        patience: int,
        min_delta: float,
        expected_stop_epoch: int,
    ) -> None:
        """RFDETREarlyStopping stops at the expected epoch for a plateau sequence.

        Drives the callback with an identical mAP sequence and asserts the trigger epoch matches the expected value.
        """
        new_cb = RFDETREarlyStopping(
            patience=patience,
            min_delta=min_delta,
            use_ema=use_ema,
            verbose=False,
        )
        pl_module = _make_pl_module()
        new_stop_epoch: int | None = None
        for epoch, m in enumerate(maps):
            metrics = {"val/mAP_50_95": m}
            if use_ema:
                metrics["val/ema_mAP_50_95"] = m
            trainer = _make_trainer(metrics, current_epoch=epoch)
            new_cb.on_validation_end(trainer, pl_module)
            if trainer.should_stop:
                new_stop_epoch = epoch
                break

        assert new_stop_epoch == expected_stop_epoch
