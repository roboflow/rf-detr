# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for :class:`BestModelCallback` and :class:`RFDETREarlyStopping`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning.trainer.states import TrainerFn

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
    return trainer


def _make_pl_module() -> MagicMock:
    """Create a minimal mock RFDETRModule with state_dict and train_config."""
    pl_module = MagicMock()
    pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
    # Use a real dict so torch.save can pickle it (MagicMock is not picklable).
    pl_module.train_config = {"lr": 0.001}
    return pl_module


# ---------------------------------------------------------------------------
# TestBestModelCallback
# ---------------------------------------------------------------------------


class TestBestModelCallback:
    """Verify best-model checkpoint saving and selection."""

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


# ---------------------------------------------------------------------------
# TestRFDETREarlyStopping
# ---------------------------------------------------------------------------


class TestRFDETREarlyStopping:
    """Verify early stopping logic mirrors legacy EarlyStoppingCallback."""

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

    @pytest.mark.parametrize(
        "use_ema, maps, patience, min_delta, expected_stop_epoch",
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
