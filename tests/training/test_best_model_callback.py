# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for :class:`BestModelCallback` and :class:`RFDETREarlyStopping`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning import __version__ as ptl_version
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader, TensorDataset

from rfdetr.config import RFDETRLargeDeprecatedConfig, RFDETRMediumConfig
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(
    metrics: dict[str, float],
    current_epoch: int = 1,
    is_global_zero: bool = True,
    callbacks: list[object] | None = None,
) -> MagicMock:
    """Create a minimal mock Trainer with controllable callback_metrics.

    Sets the attributes required by ModelCheckpoint and EarlyStopping
    skip-guards so that callbacks run normally in unit tests.
    """
    trainer = MagicMock()
    trainer.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
    trainer.current_epoch = current_epoch
    trainer.is_global_zero = is_global_zero
    trainer.callbacks = callbacks or []
    trainer.should_stop = False
    # Required by ModelCheckpoint._should_skip_saving_checkpoint
    trainer.fast_dev_run = False
    trainer.state.fn = TrainerFn.FITTING
    trainer.sanity_checking = False
    trainer.global_step = 1  # int; differs from _last_global_step_saved=0
    # Required by EarlyStopping._log_info (world_size > 1 check)
    trainer.world_size = 1
    # Required by ModelCheckpoint.check_monitor_top_k and EarlyStopping (DDP reduce)
    trainer.strategy.reduce_boolean_decision.side_effect = lambda x, **kwargs: x
    # Prevent MagicMock auto-attribute from triggering class_names enrichment.
    trainer.datamodule.class_names = None
    return trainer


def _make_pl_module() -> MagicMock:
    """Create a minimal mock RFDETRModule with state_dict and train_config."""
    pl_module = MagicMock()
    pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
    # Use a real dict so torch.save can pickle it (MagicMock is not picklable).
    pl_module.train_config = {"lr": 0.001}
    return pl_module


class _ResumeTinyModule(LightningModule):
    """Tiny LightningModule used to validate real ckpt_path resume behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return torch.nn.functional.mse_loss(pred, y)

    def validation_step(self, batch, batch_idx):
        del batch, batch_idx
        self.log("val/mAP_50_95", torch.tensor(0.5), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)


class _EvalIntervalModule(LightningModule):
    """Tiny module that only logs val/mAP_50_95 every ``eval_interval`` epochs.

    Simulates RF-DETR's COCO-eval skip behaviour: validation runs every epoch
    but the metric key is absent on non-eval epochs.
    """

    def __init__(self, eval_interval: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}
        self._eval_interval = eval_interval

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def validation_step(self, batch, batch_idx):
        del batch, batch_idx
        if self.current_epoch % self._eval_interval == 0:
            self.log("val/mAP_50_95", torch.tensor(0.5), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)


class _ResumeProbeCallback(Callback):
    """Capture the first train epoch index for resume assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.first_train_epoch: int | None = None

    def on_train_epoch_start(self, trainer, pl_module):
        del pl_module
        if self.first_train_epoch is None:
            self.first_train_epoch = trainer.current_epoch


# ---------------------------------------------------------------------------
# TestBestModelCallback
# ---------------------------------------------------------------------------


class TestBestModelCallback:
    """Verify best-model checkpoint saving and selection."""

    @pytest.mark.parametrize(
        "monitor_ema, metrics, checkpoint_file",
        [
            pytest.param(None, {"val/mAP_50_95": 0.9}, "checkpoint_best_regular.pth", id="regular"),
            pytest.param(
                "val/ema_mAP_50_95",
                {"val/mAP_50_95": 0.5, "val/ema_mAP_50_95": 0.9},
                "checkpoint_best_ema.pth",
                id="ema",
            ),
        ],
    )
    def test_skip_best_epochs_no_checkpoint_during_skip_window(
        self, tmp_path: Path, monitor_ema: str | None, metrics: dict, checkpoint_file: str
    ) -> None:
        """No checkpoint written for epochs before skip_best_epochs."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema=monitor_ema, skip_best_epochs=2)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer(metrics, current_epoch=0), pl_module)
        cb.on_validation_end(_make_trainer(metrics, current_epoch=1), pl_module)
        assert not (tmp_path / checkpoint_file).exists()

    @pytest.mark.parametrize(
        "monitor_ema, skip_metrics, eligible_metrics, checkpoint_file",
        [
            pytest.param(
                None,
                {"val/mAP_50_95": 0.9},
                {"val/mAP_50_95": 0.7},
                "checkpoint_best_regular.pth",
                id="regular",
            ),
            pytest.param(
                "val/ema_mAP_50_95",
                {"val/mAP_50_95": 0.5, "val/ema_mAP_50_95": 0.9},
                {"val/mAP_50_95": 0.5, "val/ema_mAP_50_95": 0.7},
                "checkpoint_best_ema.pth",
                id="ema",
            ),
        ],
    )
    def test_skip_best_epochs_checkpoint_saved_on_first_eligible_epoch(
        self,
        tmp_path: Path,
        monitor_ema: str | None,
        skip_metrics: dict,
        eligible_metrics: dict,
        checkpoint_file: str,
    ) -> None:
        """Checkpoint written on the first epoch at or after skip_best_epochs."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema=monitor_ema, skip_best_epochs=2)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer(skip_metrics, current_epoch=0), pl_module)
        cb.on_validation_end(_make_trainer(skip_metrics, current_epoch=1), pl_module)
        cb.on_validation_end(_make_trainer(eligible_metrics, current_epoch=2), pl_module)
        assert (tmp_path / checkpoint_file).exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="default"),
            pytest.param({"skip_best_epochs": 0}, id="explicit_zero"),
        ],
    )
    def test_skip_best_epochs_zero_does_not_defer_epoch_zero(self, tmp_path: Path, kwargs: dict) -> None:
        """skip_best_epochs=0 (explicit or default) makes epoch 0 eligible for checkpoint."""
        cb = BestModelCallback(output_dir=str(tmp_path), **kwargs)
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=0)
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_regular.pth").exists()
        assert cb.best_model_score is not None

    def test_skip_best_epochs_exceeds_total_epochs_produces_no_checkpoint(self, tmp_path: Path) -> None:
        """No checkpoint when skip_best_epochs >= total epochs (all epochs skipped)."""
        cb = BestModelCallback(output_dir=str(tmp_path), skip_best_epochs=3)
        pl_module = _make_pl_module()

        for epoch in range(2):
            trainer = _make_trainer({"val/mAP_50_95": 0.9}, current_epoch=epoch)
            cb.on_validation_end(trainer, pl_module)

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()
        assert cb.best_model_score is None

    @pytest.mark.parametrize(
        "invalid_value, exc_type",
        [
            pytest.param(True, TypeError, id="bool_true"),
            pytest.param(2.5, TypeError, id="float"),
            pytest.param("3", TypeError, id="string"),
            pytest.param(-1, ValueError, id="negative"),
        ],
    )
    def test_skip_best_epochs_invalid_input_raises(
        self, tmp_path: Path, invalid_value: object, exc_type: type[Exception]
    ) -> None:
        """BestModelCallback raises TypeError for non-int and ValueError for negative skip_best_epochs."""
        with pytest.raises(exc_type):
            BestModelCallback(output_dir=str(tmp_path), skip_best_epochs=invalid_value)  # type: ignore[arg-type]

    def test_regular_checkpoint_saved_on_improvement(self, tmp_path: Path) -> None:
        """Metric 0.5 > initial 0.0 causes checkpoint_best_regular.pth to be saved."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_regular_checkpoint_not_saved_on_no_improvement(self, tmp_path: Path) -> None:
        """Metric 0.3 after best 0.5 does not create a checkpoint file."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        pl_module = _make_pl_module()

        # First call sets best to 0.5
        trainer1 = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer1, pl_module)

        # Record mtime to verify no overwrite
        path = tmp_path / "checkpoint_best_regular.pth"
        stat_before = path.stat().st_mtime_ns

        # Second call with worse metric (same global_step → ModelCheckpoint skip guard fires)
        trainer2 = _make_trainer({"val/mAP_50_95": 0.3})
        cb.on_validation_end(trainer2, pl_module)

        assert path.stat().st_mtime_ns == stat_before

    def test_ema_checkpoint_saved_on_ema_improvement(self, tmp_path: Path) -> None:
        """When monitor_ema is set and EMA metric improves, EMA checkpoint is saved."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_ema.pth").exists()

    def test_ema_checkpoint_saves_ema_callback_weights(self, tmp_path: Path) -> None:
        """EMA checkpoint must store EMA callback weights, not live model weights."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        ema_state = {"w": torch.ones(1)}
        ema_callback = MagicMock()
        ema_callback.get_ema_model_state_dict.return_value = ema_state
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6},
            callbacks=[ema_callback],
        )
        pl_module = _make_pl_module()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_ema.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == ema_state

    def test_regular_checkpoint_uses_ema_weights_when_ema_enabled(self, tmp_path: Path) -> None:
        """Regular checkpoint must store EMA-evaluated weights when EMA is enabled."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        ema_state = {"w": torch.ones(1)}
        ema_callback = MagicMock()
        ema_callback.get_ema_model_state_dict.return_value = ema_state
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.6},
            callbacks=[ema_callback],
        )
        pl_module = _make_pl_module()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == ema_state

    def test_best_total_regular_wins(self, tmp_path: Path) -> None:
        """Regular model wins when best_regular > best_ema."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            run_test=False,
        )
        pl_module = _make_pl_module()

        # Epoch with regular=0.6, ema=0.5
        trainer = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        assert total.exists()
        data = torch.load(total, map_location="cpu", weights_only=False)
        assert "model" in data
        assert "args" in data

    def test_best_total_ema_wins(self, tmp_path: Path) -> None:
        """EMA wins when best_ema > best_regular (strict >)."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            run_test=False,
        )
        pl_module = _make_pl_module()

        # Give regular a lower value, EMA a higher value
        trainer = _make_trainer({"val/mAP_50_95": 0.5, "val/ema_mAP_50_95": 0.7})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        assert total.exists()
        # The EMA checkpoint should have been the source
        ema_data = torch.load(
            tmp_path / "checkpoint_best_ema.pth",
            map_location="cpu",
            weights_only=False,
        )
        total_data = torch.load(total, map_location="cpu", weights_only=False)
        # total is stripped so only model + args
        assert total_data["model"] == ema_data["model"]

    def test_best_total_ema_equal_uses_regular(self, tmp_path: Path) -> None:
        """When best_ema == best_regular, regular wins (strict > for EMA)."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            run_test=False,
        )
        pl_module = _make_pl_module()

        # Equal metrics
        trainer = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.6})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        assert total.exists()
        # Regular should have been chosen since EMA didn't strictly win
        regular_data = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        total_data = torch.load(total, map_location="cpu", weights_only=False)
        assert total_data["model"] == regular_data["model"]

    def test_best_total_stripped_of_optimizer(self, tmp_path: Path) -> None:
        """checkpoint_best_total.pth must NOT contain optimizer or lr_scheduler keys."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            run_test=False,
        )
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        data = torch.load(total, map_location="cpu", weights_only=False)
        assert "optimizer" not in data
        assert "lr_scheduler" not in data
        # Must contain model and args
        assert "model" in data
        assert "args" in data

    def test_run_test_true_calls_trainer_test(self, tmp_path: Path) -> None:
        """run_test=True causes trainer.test() when module defines test_step()."""
        from pytorch_lightning import LightningModule

        class _ModuleWithTestStep(LightningModule):
            def test_step(self, batch: object, batch_idx: int) -> None: ...

        # Use a real subclass (not MagicMock) so type() inspection sees test_step.
        pl_module = _ModuleWithTestStep()
        pl_module.model = MagicMock()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
        pl_module.train_config = {"lr": 0.001}

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_called_once_with(pl_module, datamodule=trainer.datamodule, verbose=False)

    def test_run_test_true_without_test_step_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=True but no test_step override — trainer.test() is NOT called.

        The guard in BestModelCallback.on_fit_end() skips trainer.test() for
        modules that do not override LightningModule.test_step() to avoid a
        MisconfigurationException from PTL.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        pl_module = _make_pl_module()  # MagicMock — no test_step on its class
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_not_called()

    def test_run_test_true_without_best_checkpoint_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=True must not test final weights when no best checkpoint was produced."""
        from pytorch_lightning import LightningModule

        class _ModuleWithTestStep(LightningModule):
            def test_step(self, batch: object, batch_idx: int) -> None: ...

        pl_module = _ModuleWithTestStep()
        pl_module.model = MagicMock()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
        pl_module.train_config = {"lr": 0.001}

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True, skip_best_epochs=2)
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=0)

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        assert not (tmp_path / "checkpoint_best_total.pth").exists()
        trainer.test.assert_not_called()

    def test_run_test_loads_best_weights_before_test(self, tmp_path: Path) -> None:
        """on_fit_end loads checkpoint_best_total.pth weights before trainer.test().

        Mirrors legacy main.py:602-609 which loads the best checkpoint into the
        model before running test evaluation so the test loop measures the best
        model, not the end-of-training state.
        """
        from pytorch_lightning import LightningModule

        class _ModuleWithTestStep(LightningModule):
            def test_step(self, batch: object, batch_idx: int) -> None: ...

        pl_module = _ModuleWithTestStep()
        pl_module.model = MagicMock()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
        pl_module.train_config = {"lr": 0.001}

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        # Model weights must be loaded from checkpoint_best_total.pth with strict=True
        pl_module.model.load_state_dict.assert_called_once()
        call_kwargs = pl_module.model.load_state_dict.call_args.kwargs
        assert call_kwargs.get("strict") is True, "load_state_dict must be called with strict=True"

    def test_run_test_false_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=False means trainer.test() is never called."""
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_not_called()

    def test_checkpoint_class_names_populated_from_datamodule(self, tmp_path: Path) -> None:
        """Saved checkpoint args.class_names reflects dataset class names.

        Regression test for #509: checkpoints were saved with class_names=None
        when the user did not pass class_names explicitly, causing reloaded-model
        inference to fall through to COCO labels instead of dataset labels.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        custom_names = ["cat", "dog"]
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        trainer.datamodule.class_names = custom_names

        pl_module = _make_pl_module()
        # Real TrainConfig with class_names unset — the bug scenario.
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == custom_names

    def test_ema_checkpoint_class_names_populated_from_datamodule(self, tmp_path: Path) -> None:
        """EMA checkpoint args.class_names also reflects dataset class names.

        Regression test for #509: EMA checkpoint path was not enriched with
        class names, so EMA-selected runs would still return COCO labels after reload.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        custom_names = ["cat", "dog"]
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        trainer.datamodule.class_names = custom_names

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_ema.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == custom_names

    def test_checkpoint_class_names_not_overwritten_when_already_set(self, tmp_path: Path) -> None:
        """Explicitly-set class_names in TrainConfig are preserved in the checkpoint.

        When the user passes class_names=['defect'] to TrainConfig, the saved
        checkpoint must keep that value even if the datamodule reports different names.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        trainer.datamodule.class_names = ["other_class"]  # would overwrite if bug exists

        pl_module = _make_pl_module()
        explicit_names = ["defect"]
        pl_module.train_config = TrainConfig(
            dataset_dir=str(tmp_path / "ds"), tensorboard=False, class_names=explicit_names
        )

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == explicit_names

    def test_checkpoint_explicit_empty_class_names_not_overwritten_by_datamodule(self, tmp_path: Path) -> None:
        """TrainConfig(class_names=[]) is preserved even when datamodule has non-empty names.

        Guard-bypass regression: the truthiness check `not getattr(..., "class_names", None)`
        treated an explicit empty list the same as None (both falsy), silently overwriting
        the user's intent with the datamodule's names. The fix uses `is None` identity.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        trainer.datamodule.class_names = ["cat", "dog"]  # would overwrite if bug exists

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, class_names=[])

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == [], (
            "Explicit class_names=[] in TrainConfig must not be overwritten by datamodule names"
        )

    def test_ema_checkpoint_explicit_empty_class_names_not_overwritten_by_datamodule(self, tmp_path: Path) -> None:
        """EMA path: TrainConfig(class_names=[]) is preserved even when datamodule has non-empty names.

        Mirrors the regular checkpoint guard-bypass regression test for the EMA path.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        trainer.datamodule.class_names = ["cat", "dog"]  # would overwrite if bug exists

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, class_names=[])

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_ema.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == [], (
            "Explicit class_names=[] in TrainConfig must not be overwritten by datamodule names (EMA path)"
        )

    def test_checkpoint_empty_class_names_populated_from_datamodule(self, tmp_path: Path) -> None:
        """Checkpoint preserves explicitly-empty dataset class names.

        Empty list should be treated as a provided value, not as missing.
        """
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        trainer.datamodule.class_names = []

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == []

    def test_ema_checkpoint_empty_class_names_populated_from_datamodule(self, tmp_path: Path) -> None:
        """EMA checkpoint preserves explicitly-empty dataset class names."""
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        trainer.datamodule.class_names = []

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_ema.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["class_names"] == []

    # --- PTL-compatible format tests ---

    def test_regular_checkpoint_args_is_dict(self, tmp_path: Path) -> None:
        """Saved args must be a plain dict (not a Pydantic object) for weights_only=True compat."""
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        # weights_only=True must succeed now that args is a plain dict.
        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=True)
        assert isinstance(ckpt["args"], dict)

    def test_regular_checkpoint_has_ptl_state_dict_key(self, tmp_path: Path) -> None:
        """Saved regular checkpoint must include 'state_dict' with model. prefix."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    def test_regular_checkpoint_has_loops_key(self, tmp_path: Path) -> None:
        """Saved regular checkpoint must include 'loops' with fit_loop epoch counter."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=3)
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert "loops" in ckpt
        ep = ckpt["loops"]["fit_loop"]["epoch_progress"]
        assert ep["current"]["completed"] == 4  # epoch 3 + 1

    def test_regular_checkpoint_has_ptl_version_key(self, tmp_path: Path) -> None:
        """Saved regular checkpoint must include 'pytorch-lightning_version'."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert ckpt.get("pytorch-lightning_version") == ptl_version

    def test_ema_checkpoint_has_ptl_state_dict_key(self, tmp_path: Path) -> None:
        """Saved EMA checkpoint must include 'state_dict' with model. prefix."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_ema.pth", map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    def test_ema_checkpoint_has_loops_key(self, tmp_path: Path) -> None:
        """Saved EMA checkpoint must include 'loops' with fit_loop epoch counter."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6}, current_epoch=5)
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_ema.pth", map_location="cpu", weights_only=False)
        assert "loops" in ckpt
        ep = ckpt["loops"]["fit_loop"]["epoch_progress"]
        assert ep["current"]["completed"] == 6  # epoch 5 + 1

    def test_state_dict_values_match_model_weights(self, tmp_path: Path) -> None:
        """state_dict values must be identical to the original model weights."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        weights = {"w": torch.randn(3, 3)}
        pl_module.model.state_dict.return_value = weights

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert torch.equal(ckpt["state_dict"]["model.w"], weights["w"])

    def test_best_total_preserves_ptl_keys_after_strip(self, tmp_path: Path) -> None:
        """strip_checkpoint must preserve state_dict and loops in the final file."""
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        data = torch.load(total, map_location="cpu", weights_only=False)
        assert "state_dict" in data, "strip_checkpoint must preserve 'state_dict'"
        assert "loops" in data, "strip_checkpoint must preserve 'loops'"
        assert "pytorch-lightning_version" in data, "strip_checkpoint must preserve 'pytorch-lightning_version'"
        assert "optimizer_states" in data, "strip_checkpoint must preserve 'optimizer_states'"
        assert "lr_schedulers" in data, "strip_checkpoint must preserve 'lr_schedulers'"

    def test_not_global_zero_does_not_save(self, tmp_path: Path) -> None:
        """Non-main process (is_global_zero=False) must not write any files."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        pl_module = _make_pl_module()
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.9, "val/ema_mAP_50_95": 0.9},
            is_global_zero=False,
        )

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()
        assert not (tmp_path / "checkpoint_best_ema.pth").exists()
        assert not (tmp_path / "checkpoint_best_total.pth").exists()

    def test_train_epoch_end_ignores_missing_validation_metrics(self, tmp_path: Path) -> None:
        """Train-epoch end must not try to checkpoint when validation metrics were not logged."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({})

        cb.on_train_epoch_end(trainer, _make_pl_module())

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_validation_end_ignores_missing_validation_metrics(self, tmp_path: Path) -> None:
        """on_validation_end must not raise when val/mAP_50_95 was not logged (non-eval epoch)."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({})  # empty metrics — no val/mAP_50_95 key
        trainer.fit_loop.epoch_loop.val_loop._has_run = True

        cb.on_validation_end(trainer, _make_pl_module())  # must not raise

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_eval_interval_does_not_crash(self, tmp_path: Path) -> None:
        """BestModelCallback must not crash over 3 epochs when metrics are only logged every 2nd epoch."""
        torch.manual_seed(0)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer = Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[cb],
            default_root_dir=str(tmp_path),
        )
        trainer.fit(_EvalIntervalModule(eval_interval=2), train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Checkpoint must be written on eval epochs (0 and 2) — at least one must exist.
        assert (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_best_total_checkpoint_resumes_via_trainer_fit_ckpt_path(self, tmp_path: Path) -> None:
        """checkpoint_best_total.pth must restore epoch/step when passed to Trainer.fit(ckpt_path=...)."""
        torch.manual_seed(0)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        save_cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer_first = Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[save_cb],
            default_root_dir=str(tmp_path),
        )
        trainer_first.fit(_ResumeTinyModule(), train_dataloaders=train_loader, val_dataloaders=val_loader)

        ckpt_path = tmp_path / "checkpoint_best_total.pth"
        assert ckpt_path.exists()
        first_phase_global_step = trainer_first.global_step
        assert first_phase_global_step == 2
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt_data["global_step"] == first_phase_global_step

        resume_probe = _ResumeProbeCallback()
        trainer_second = Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[resume_probe],
            default_root_dir=str(tmp_path),
        )
        trainer_second.fit(
            _ResumeTinyModule(),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path),
        )

        # PTL applies loop restoration by the first train epoch start.
        assert resume_probe.first_train_epoch == 1
        # In the stripped-checkpoint resume path, optimizer state is intentionally
        # fresh; this resumed phase contributes exactly one epoch with 2 steps.
        assert trainer_second.current_epoch == 2
        assert trainer_second.global_step == 2


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

    def test_first_eligible_epoch_sets_baseline_with_zero_wait(self) -> None:
        """First eligible epoch becomes best_score baseline; patience wait_count stays 0."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001, skip_best_epochs=2)
        pl_module = _make_pl_module()
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.9}, current_epoch=0), pl_module)
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.8}, current_epoch=1), pl_module)
        cb.on_validation_end(_make_trainer({"val/mAP_50_95": 0.7}, current_epoch=2), pl_module)
        assert cb.best_score.item() == pytest.approx(0.7)
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

    def test_skip_best_epochs_uses_first_eligible_epoch_as_baseline(self) -> None:
        """A stronger skipped epoch must not block the first eligible epoch from becoming best."""
        cb = RFDETREarlyStopping(patience=2, min_delta=0.001, skip_best_epochs=2)
        pl_module = _make_pl_module()

        trainer_epoch0 = _make_trainer({"val/mAP_50_95": 0.95}, current_epoch=0)
        cb.on_validation_end(trainer_epoch0, pl_module)
        trainer_epoch1 = _make_trainer({"val/mAP_50_95": 0.85}, current_epoch=1)
        cb.on_validation_end(trainer_epoch1, pl_module)

        trainer_epoch2 = _make_trainer({"val/mAP_50_95": 0.40}, current_epoch=2)
        cb.on_validation_end(trainer_epoch2, pl_module)

        assert cb.best_score.item() == pytest.approx(0.40)
        assert cb.wait_count == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="default"),
            pytest.param({"skip_best_epochs": 0}, id="explicit_zero"),
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
            pytest.param(True, TypeError, id="bool_true"),
            pytest.param(2.5, TypeError, id="float"),
            pytest.param("3", TypeError, id="string"),
            pytest.param(-1, ValueError, id="negative"),
        ],
    )
    def test_skip_best_epochs_invalid_input_raises(self, invalid_value: object, exc_type: type[Exception]) -> None:
        """RFDETREarlyStopping raises TypeError for non-int and ValueError for negative skip_best_epochs."""
        with pytest.raises(exc_type):
            RFDETREarlyStopping(patience=5, skip_best_epochs=invalid_value)  # type: ignore[arg-type]

    def test_no_stop_within_patience(self) -> None:
        """3 epochs with no improvement, patience=5 -- training continues."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001)
        pl_module = _make_pl_module()

        # Seed best_score with initial improvement
        trainer0 = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer0, pl_module)

        # 3 stagnant epochs
        for _ in range(3):
            trainer = _make_trainer({"val/mAP_50_95": 0.5})
            cb.on_validation_end(trainer, pl_module)

        assert trainer.should_stop is False
        assert cb.wait_count == 3

    def test_stops_after_patience_exceeded(self) -> None:
        """patience=3 with 3 no-improvement epochs triggers stop."""
        cb = RFDETREarlyStopping(patience=3, min_delta=0.001)
        pl_module = _make_pl_module()

        # Set baseline
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        # 3 stagnant epochs
        for _ in range(3):
            trainer = _make_trainer({"val/mAP_50_95": 0.5})
            cb.on_validation_end(trainer, pl_module)

        assert trainer.should_stop is True

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

    def test_use_ema_true_monitors_ema_only(self) -> None:
        """use_ema=True with both metrics available uses EMA value only."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001, use_ema=True)
        pl_module = _make_pl_module()

        # EMA is 0.3 (low), regular is 0.8 (high)
        trainer = _make_trainer({"val/mAP_50_95": 0.8, "val/ema_mAP_50_95": 0.3})
        cb.on_validation_end(trainer, pl_module)

        # best_score should reflect EMA value, not regular
        assert cb.best_score.item() == pytest.approx(0.3)

    def test_use_ema_false_monitors_max(self) -> None:
        """use_ema=False with both metrics uses max(regular, ema)."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001, use_ema=False)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        cb.on_validation_end(trainer, pl_module)

        # max(0.4, 0.6) = 0.6
        assert cb.best_score.item() == pytest.approx(0.6)

    def test_only_regular_available(self) -> None:
        """When EMA key is absent, uses regular metric without error."""
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.45})
        cb.on_validation_end(trainer, pl_module)

        assert cb.best_score.item() == pytest.approx(0.45)
        assert cb.wait_count == 0

    def test_neither_available_is_noop(self) -> None:
        """Neither metric present causes no counter increment and no stop."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001)
        pl_module = _make_pl_module()

        trainer = _make_trainer({})  # no metrics at all
        cb.on_validation_end(trainer, pl_module)

        assert cb.wait_count == 0
        assert trainer.should_stop is False

    def test_train_epoch_end_ignores_missing_validation_metrics(self) -> None:
        """Train-epoch end must not evaluate early stopping when validation did not run."""
        cb = RFDETREarlyStopping(patience=1, min_delta=0.001)
        trainer = _make_trainer({})

        cb.on_train_epoch_end(trainer, _make_pl_module())

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
                id="use_ema_false_plateau",
            ),
            pytest.param(
                True,
                [0.05, 0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                3,
                0.01,
                5,
                id="use_ema_true_plateau",
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

        Drives the callback with an identical mAP sequence and asserts the
        trigger epoch matches the expected value.
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


# ---------------------------------------------------------------------------
# model_name in checkpoint payload (#887)
# ---------------------------------------------------------------------------


class TestCheckpointModelName:
    """Verify model_name is stored in checkpoint payloads."""

    def test_regular_checkpoint_contains_model_name(self, tmp_path: Path) -> None:
        """Regular checkpoint includes model_name from model_config."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_name = "RFDETRLarge"

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert ckpt["model_name"] == "RFDETRLarge"

    def test_regular_checkpoint_model_name_absent_when_not_set(self, tmp_path: Path) -> None:
        """model_name key is absent when model_config has no model_name attribute."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert "model_name" not in ckpt

    def test_regular_checkpoint_infers_model_name_from_model_config_type(self, tmp_path: Path) -> None:
        """When model_name is unset, infer class name from concrete ModelConfig type."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.model_config = RFDETRMediumConfig(model_name=None)

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert ckpt["model_name"] == "RFDETRMedium"

    def test_ema_checkpoint_contains_model_name(self, tmp_path: Path) -> None:
        """EMA checkpoint also includes model_name."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_name = "RFDETRMedium"

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_ema.pth", weights_only=False)
        assert ckpt["model_name"] == "RFDETRMedium"

    def test_deprecated_config_raises_runtime_error(self, tmp_path: Path) -> None:
        """RFDETRLargeDeprecatedConfig raises RuntimeError — deprecated configs are unsupported."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.model_config = RFDETRLargeDeprecatedConfig(model_name=None)

        with pytest.raises(RuntimeError, match="Deprecated model config"):
            cb.on_validation_end(trainer, pl_module)


# ---------------------------------------------------------------------------
# rfdetr_version in checkpoint payload
# ---------------------------------------------------------------------------


class TestCheckpointRfdetrVersion:
    """Verify rfdetr_version is stored in checkpoint payloads."""

    def test_regular_checkpoint_contains_rfdetr_version(self, tmp_path: Path) -> None:
        """Regular checkpoint includes rfdetr_version string."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        expected_version = "test-version"

        with patch(
            "rfdetr.training.callbacks.best_model.get_version",
            return_value=expected_version,
        ):
            cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert "rfdetr_version" in ckpt
        assert ckpt["rfdetr_version"] == expected_version

    def test_ema_checkpoint_contains_rfdetr_version(self, tmp_path: Path) -> None:
        """EMA checkpoint also includes rfdetr_version."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6})
        pl_module = _make_pl_module()
        expected_version = "test-version"

        with patch(
            "rfdetr.training.callbacks.best_model.get_version",
            return_value=expected_version,
        ):
            cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_ema.pth", weights_only=False)
        assert "rfdetr_version" in ckpt
        assert ckpt["rfdetr_version"] == expected_version

    def test_best_total_preserves_rfdetr_version_after_strip(self, tmp_path: Path) -> None:
        """strip_checkpoint must preserve rfdetr_version in the final checkpoint."""
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        expected_version = "test-version"

        with patch(
            "rfdetr.training.callbacks.best_model.get_version",
            return_value=expected_version,
        ):
            cb.on_validation_end(trainer, pl_module)
            cb.on_fit_end(trainer, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        data = torch.load(total, map_location="cpu", weights_only=False)
        assert "rfdetr_version" in data
        assert data["rfdetr_version"] == expected_version

    def test_rfdetr_version_absent_when_get_version_returns_none(self, tmp_path: Path) -> None:
        """rfdetr_version must be omitted when get_version() cannot resolve the version."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        with patch("rfdetr.training.callbacks.best_model.get_version", return_value=None):
            cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=False)
        assert "rfdetr_version" not in ckpt


# ---------------------------------------------------------------------------
# _best_ema state persistence across resume (#969)
# ---------------------------------------------------------------------------


class TestBestEmaStatePersistence:
    """Regression tests for _best_ema not surviving Lightning checkpoint resume.

    Before the fix, BestModelCallback did not override state_dict() /
    load_state_dict(), so _best_ema was never included in the Lightning callback
    state bundle.  On resume via trainer.fit(ckpt_path=...) the callback was
    reconstructed fresh with _best_ema=0.0, causing any positive post-resume EMA
    value to trivially overwrite checkpoint_best_ema.pth with inferior weights.

    Regression tests for GitHub issue #969.
    """

    def test_state_dict_includes_best_ema(self, tmp_path: Path) -> None:
        """state_dict() must include _best_ema so it survives Lightning checkpointing."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()

        # Drive _best_ema to 0.75 via a validation pass.
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75})
        cb.on_validation_end(trainer, pl_module)

        state = cb.state_dict()

        assert "_best_ema" in state, "_best_ema must be present in state_dict() output"
        assert state["_best_ema"] == pytest.approx(0.75)

    def test_load_state_dict_restores_best_ema(self, tmp_path: Path) -> None:
        """load_state_dict() must restore _best_ema from the persisted state."""
        # First callback: train to _best_ema=0.75.
        cb_first = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75})
        cb_first.on_validation_end(trainer, pl_module)
        saved_state = cb_first.state_dict()

        # Second callback: simulate a fresh resume by loading the saved state.
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb_resumed.load_state_dict(saved_state)

        assert cb_resumed._best_ema == pytest.approx(0.75), (
            "load_state_dict() must restore _best_ema; without fix it stays 0.0"
        )

    def test_resume_does_not_clobber_ema_checkpoint_with_inferior_weights(self, tmp_path: Path) -> None:
        """After resume, inferior post-resume EMA must not overwrite checkpoint_best_ema.pth.

        Without the fix: _best_ema resets to 0.0 on resume, so any positive EMA
        metric (0.5) trivially satisfies ema_val > _best_ema and overwrites the
        checkpoint saved pre-resume (0.75).
        """
        # --- Pre-resume phase: establish EMA best of 0.75 ---
        cb_pre = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()
        trainer_pre = _make_trainer(
            {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75},
            current_epoch=5,
        )
        cb_pre.on_validation_end(trainer_pre, pl_module)

        ema_path = tmp_path / "checkpoint_best_ema.pth"
        assert ema_path.exists()
        baseline_epoch = torch.load(ema_path, map_location="cpu", weights_only=False)["epoch"]
        saved_state = cb_pre.state_dict()

        # --- Resume phase: fresh callback loaded from saved state ---
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb_resumed.load_state_dict(saved_state)

        # Post-resume validation reports EMA=0.5 — worse than pre-resume best (0.75).
        trainer_post = _make_trainer(
            {"val/mAP_50_95": 0.45, "val/ema_mAP_50_95": 0.5},
            current_epoch=7,
        )
        cb_resumed.on_validation_end(trainer_post, pl_module)

        assert torch.load(ema_path, map_location="cpu", weights_only=False)["epoch"] == baseline_epoch, (
            "checkpoint_best_ema.pth must not be overwritten by an inferior post-resume EMA value"
        )

    def test_on_fit_end_selects_ema_winner_after_resume(self, tmp_path: Path) -> None:
        """on_fit_end picks EMA winner correctly when _best_ema is properly restored.

        Without the fix: _best_ema=0.0 after resume, so regular (0.6) wins over
        the true EMA best (0.8) — checkpoint_best_total.pth is built from the
        wrong source.  Use epoch number as a distinguisher: pre-resume EMA was
        saved at epoch 3; regular was saved at epoch 1; total epoch must be 3
        (EMA epoch) when EMA correctly wins.
        """
        # Pre-resume epoch 1: regular best=0.6.
        cb_pre = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        trainer_ep1 = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.5}, current_epoch=1)
        cb_pre.on_validation_end(trainer_ep1, pl_module)

        # Pre-resume epoch 3: EMA best=0.8 (better than epoch-1 EMA=0.5).
        trainer_ep3 = _make_trainer({"val/mAP_50_95": 0.55, "val/ema_mAP_50_95": 0.8}, current_epoch=3)
        cb_pre.on_validation_end(trainer_ep3, pl_module)

        saved_state = cb_pre.state_dict()

        # Resume: fresh callback, load state, run fit_end.
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        cb_resumed.load_state_dict(saved_state)

        # fit_end uses _best_ema to decide EMA vs regular winner.
        # trainer_ep3 carries best_model_score=0.6 (regular) and _best_ema=0.8 (EMA wins).
        cb_resumed.on_fit_end(trainer_ep3, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        assert total.exists()
        total_data = torch.load(total, map_location="cpu", weights_only=False)
        # strip_checkpoint preserves the `loops` key.
        # epoch_progress.current.completed == trainer.current_epoch + 1 at save time.
        # EMA checkpoint was saved at epoch 3 → completed=4.
        # Regular checkpoint was saved at epoch 1 → completed=2.
        # If _best_ema was NOT restored (bug), regular wins → completed=2.
        # If _best_ema IS restored (fix), EMA wins → completed=4.
        completed = total_data["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
        assert completed == 4, (
            "on_fit_end must select EMA (epoch 3, best=0.8) over regular (epoch 1, best=0.6); "
            f"got epoch_completed={completed} — _best_ema not restored from state_dict"
        )

    @pytest.mark.parametrize(
        ("mutate_state", "expected_best_ema"),
        [
            pytest.param(
                lambda state: state.pop("_best_ema"),
                0.0,
                id="missing_key",
            ),
            pytest.param(
                lambda state: state.__setitem__("_best_ema", int(1)),
                1.0,
                id="int_coercion",
            ),
            pytest.param(
                lambda state: state.__setitem__("_best_ema", str("0.75")),
                0.75,
                id="string_coercion",
            ),
        ],
    )
    def test_load_state_dict_backward_compat(self, tmp_path: Path, mutate_state, expected_best_ema: float) -> None:
        """load_state_dict() keeps backward-compatible _best_ema restoration behavior."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        state = cb.state_dict()
        mutate_state(state)

        cb.load_state_dict(state)

        assert isinstance(cb._best_ema, float)
        assert cb._best_ema == expected_best_ema

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="neg_inf"),
        ],
    )
    def test_load_state_dict_non_finite_values(self, bad_value) -> None:
        """load_state_dict() resets non-finite persisted _best_ema values to 0.0."""
        cb = BestModelCallback(output_dir=".")
        cb._best_ema = 999.0
        state = cb.state_dict()
        state["_best_ema"] = bad_value

        cb.load_state_dict(state)

        assert cb._best_ema == 0.0

    def test_load_state_dict_does_not_mutate_caller_dict(self, tmp_path: Path) -> None:
        """load_state_dict must not pop or mutate the caller-supplied dict."""
        cb1 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb1._best_ema = 0.75
        original_sd = cb1.state_dict()
        saved = dict(original_sd)

        cb2 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb2.load_state_dict(original_sd)

        assert original_sd["_best_ema"] == 0.75
        assert original_sd == saved

    def test_state_dict_roundtrip_initial_zero(self, tmp_path: Path) -> None:
        """state_dict/load_state_dict round-trips the default _best_ema=0.0 correctly."""
        cb1 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        sd = cb1.state_dict()

        assert "_best_ema" in sd
        assert sd["_best_ema"] == 0.0

        cb2 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb2.load_state_dict(sd)

        assert cb2._best_ema == 0.0


# ---------------------------------------------------------------------------
# TestCheckpointNotes
# ---------------------------------------------------------------------------


class TestCheckpointNotes:
    """Verify user-supplied notes are persisted in .pth checkpoint files."""

    @pytest.mark.parametrize(
        "notes",
        [
            pytest.param("simple string", id="string"),
            pytest.param({"date": "2026-01-01", "labeller": "Alice"}, id="dict"),
            pytest.param(["class_a", "class_b"], id="list"),
            pytest.param(42, id="int"),
        ],
    )
    def test_notes_accessible_via_args_dict(self, tmp_path: Path, notes: object) -> None:
        """Notes supplied via TrainConfig are accessible under checkpoint['args']['notes']."""
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, notes=notes)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["notes"] == notes

    def test_notes_also_preserved_in_args_dict(self, tmp_path: Path) -> None:
        """Notes are also accessible inside checkpoint['args']['notes'] via TrainConfig dump."""
        from rfdetr.config import TrainConfig

        notes = {"project": "ceramics", "batch": 7}
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, notes=notes)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["notes"] == notes

    def test_notes_absent_when_not_provided(self, tmp_path: Path) -> None:
        """When notes=None (default), no top-level 'notes' key is written to checkpoint."""
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert "notes" not in checkpoint

    @pytest.mark.parametrize(
        "notes",
        [
            pytest.param("", id="empty_string"),
            pytest.param({}, id="empty_dict"),
            pytest.param([], id="empty_list"),
            pytest.param(0, id="zero"),
            pytest.param(False, id="false"),
        ],
    )
    def test_falsy_notes_stored_in_args_dict(self, tmp_path: Path, notes: object) -> None:
        """Falsy but non-None notes values are preserved in checkpoint['args']['notes']."""
        from rfdetr.config import TrainConfig

        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, notes=notes)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(
            tmp_path / "checkpoint_best_regular.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["args"]["notes"] == notes
