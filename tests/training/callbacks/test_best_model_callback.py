# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for :class:`BestModelCallback` best-checkpoint tracking, resume and end-of-fit testing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning import __version__ as ptl_version
from torch import Tensor
from torch.utils.data import DataLoader

from rfdetr.config import TrainConfig
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.training.callbacks.ema import RFDETREMACallback
from tests.training.callbacks.helpers import _REGULAR_AND_EMA_CASES, _EmaMetricModule, _make_test_step_module
from tests.training.callbacks.helpers import _make_best_model_pl_module as _make_pl_module
from tests.training.callbacks.helpers import _make_best_model_trainer as _make_trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cpu_trainer(
    tmp_path: Path, max_epochs: int, callbacks: list[Callback], num_sanity_val_steps: int = 0
) -> Trainer:
    """Build a quiet CPU ``Trainer`` limited to 2 train and 1 validation batches per epoch.

    Examples:
        >>> _make_cpu_trainer(Path("."), 3, []).max_epochs
        3
    """
    return Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=callbacks,
        default_root_dir=str(tmp_path),
    )


@pytest.fixture
def ema_callback() -> MagicMock:
    """EMA callback mock exposing weights ``{"w": ones}`` and a real-dict ``state_dict``.

    torch.save cannot pickle a MagicMock.
    """
    callback = MagicMock()
    callback.get_ema_model_state_dict.return_value = {"w": torch.ones(1)}
    callback.state_dict.return_value = {}
    return callback


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


class _MetricModule(LightningModule):
    """LightningModule that logs a caller-controlled ``val/mAP_50_95`` value each epoch.

    Used to discriminate a genuinely *restored* ``best_model_score``/checkpoint from one a
    fresh callback happened to compute or overwrite on its own: the pre- and post-resume
    phases log different values (the post-resume one always worse), so a restored
    high-water mark and a wrongly-reset one diverge, and a wrongly-reset one would let the
    worse post-resume epoch overwrite the good on-disk checkpoint.
    """

    def __init__(self, metric_value: float) -> None:
        """Initialize with the fixed ``val/mAP_50_95`` value to log every epoch.

        Args:
            metric_value: Value logged as ``val/mAP_50_95`` on every validation epoch.
        """
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}
        self._metric_value = metric_value

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute MSE loss for one training batch.

        Args:
            batch: ``(x, y)`` tensors from the ``TensorDataset`` loader.
            batch_idx: Index of the batch within the current epoch (unused).

        Returns:
            Scalar MSE loss.
        """
        del batch_idx
        x, y = batch
        pred = self.model(x)
        return torch.nn.functional.mse_loss(pred, y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Log the fixed ``val/mAP_50_95`` value, ignoring the actual batch contents.

        Args:
            batch: Unused; only the epoch-level log call matters for these tests.
            batch_idx: Unused.
        """
        del batch, batch_idx
        self.log("val/mAP_50_95", torch.tensor(self._metric_value), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a plain SGD optimizer sufficient to drive ``Trainer.fit()``."""
        return torch.optim.SGD(self.model.parameters(), lr=0.01)


class _EvalIntervalModule(LightningModule):
    """Tiny module that only logs val/mAP_50_95 every ``eval_interval`` epochs.

    Simulates RF-DETR's COCO-eval skip behaviour: validation runs every epoch but the metric key is absent on non-eval
    epochs.
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


class _VaryingMetricModule(LightningModule):
    """LightningModule that logs a different ``val/mAP_50_95`` value on each successive epoch.

    ``values`` is consumed one entry per validation epoch (indexed by ``self.current_epoch``, clamped to the last entry
    once exhausted), letting a single ``Trainer.fit()`` call drive stateful callbacks (e.g. ``RFDETREarlyStopping``'s
    ``wait_count``) through both improving and non-improving epochs.
    """

    def __init__(self, values: list[float]) -> None:
        """Initialize with the per-epoch ``val/mAP_50_95`` sequence.

        Args:
            values: One value per validation epoch; the last entry repeats once exhausted.
        """
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}
        self._values = values

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute MSE loss for one training batch.

        Args:
            batch: ``(x, y)`` tensors from the ``TensorDataset`` loader.
            batch_idx: Index of the batch within the current epoch (unused).

        Returns:
            Scalar MSE loss.
        """
        del batch_idx
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Log ``values[current_epoch]`` (clamped to the last entry) as ``val/mAP_50_95``.

        Args:
            batch: Unused; only the epoch-level log call matters for these tests.
            batch_idx: Unused.
        """
        del batch, batch_idx
        idx = min(self.current_epoch, len(self._values) - 1)
        self.log("val/mAP_50_95", torch.tensor(self._values[idx]), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a plain SGD optimizer sufficient to drive ``Trainer.fit()``."""
        return torch.optim.SGD(self.model.parameters(), lr=0.01)


class _CallbackStateCaptureCallback(Callback):
    """Snapshot other callbacks' resumed state at the first post-resume train-epoch start.

    Runs before any post-resume training step can mutate ``ema_callback``/``early_stop_callback``, so the captured
    values reflect exactly what PTL's ``_call_callbacks_load_state_dict`` restored from the checkpoint.
    """

    def __init__(self, ema_callback: RFDETREMACallback, early_stop_callback: RFDETREarlyStopping) -> None:
        """Store the callbacks to snapshot once training resumes.

        Args:
            ema_callback: The resumed ``RFDETREMACallback`` instance to read state from.
            early_stop_callback: The resumed ``RFDETREarlyStopping`` instance to read state from.
        """
        super().__init__()
        self._ema_callback = ema_callback
        self._early_stop_callback = early_stop_callback
        self.ema_latest_update_step: int | None = None
        self.early_stop_wait_count: int | None = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Capture ``ema_callback``/``early_stop_callback`` state on the first post-resume epoch.

        Args:
            trainer: Unused; required by the ``Callback`` hook signature.
            pl_module: Unused; required by the ``Callback`` hook signature.
        """
        del trainer, pl_module
        if self.ema_latest_update_step is None:
            self.ema_latest_update_step = self._ema_callback._latest_update_step
            self.early_stop_wait_count = self._early_stop_callback.wait_count


class _SanityCheckEndProbe(Callback):
    """Snapshot whether a checkpoint path exists at the exact moment PTL's sanity check ends.

    ``on_sanity_check_end`` fires between the sanity-check validation pass and the first real training epoch, so it
    catches a checkpoint written from sanity-check contamination before a real epoch's own (possibly identical) score
    could produce the same file for a legitimate reason.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """Store the checkpoint path to check for at sanity-check end.

        Args:
            checkpoint_path: File whose existence is snapshotted in ``on_sanity_check_end``.
        """
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self.existed_after_sanity_check: bool | None = None

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record whether ``checkpoint_path`` exists right as the sanity check ends.

        Args:
            trainer: Unused; required by the ``Callback`` hook signature.
            pl_module: Unused; required by the ``Callback`` hook signature.
        """
        del trainer, pl_module
        self.existed_after_sanity_check = self._checkpoint_path.exists()


class TestBestModelCallback:
    """Verify best-model checkpoint saving and selection."""

    def test_legacy_positional_argument_order(self, tmp_path: Path) -> None:
        """Appending the new base-model flag preserves the legacy positional constructor order."""
        callback = BestModelCallback(str(tmp_path), "val/mAP_50_95", None, False, 3, 0.2)

        assert callback._run_test is False
        assert callback._skip_best_epochs == 3
        assert callback._smooth_alpha == pytest.approx(0.2)
        assert callback._evaluates_base_model is True

    def test_checkpoint_payload_includes_model_config_when_provided(self) -> None:
        """Saved ``.pth`` checkpoints should carry the architecture schema needed for reload."""
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        payload = BestModelCallback._build_checkpoint_payload(
            {"w": torch.zeros(1)},
            {"num_classes": 1},
            trainer,
            model_name="RFDETRKeypointPreview",
            model_config_dict={"num_keypoints_per_class": [0, 17], "use_grouppose_keypoints": True},
        )

        assert payload["model_config"] == {"num_keypoints_per_class": [0, 17], "use_grouppose_keypoints": True}

    def test_checkpoint_payload_loops_include_validate_and_test_stubs(self) -> None:
        """Checkpoint loops dict must include validate_loop and test_loop stubs.

        trainer.validate(ckpt_path=...) and trainer.test(ckpt_path=...) call restore_loops() which does
        checkpoint["loops"]["validate_loop"] / checkpoint["loops"]["test_loop"] — KeyError if absent.
        """
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        payload = BestModelCallback._build_checkpoint_payload(
            {"w": torch.zeros(1)},
            {"num_classes": 1},
            trainer,
        )

        loops = payload["loops"]
        assert "validate_loop" in loops, "validate_loop missing — trainer.validate(ckpt_path=...) will KeyError"
        assert "test_loop" in loops, "test_loop missing — trainer.test(ckpt_path=...) will KeyError"
        assert "state_dict" in loops["validate_loop"]
        assert "state_dict" in loops["test_loop"]

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
            pytest.param({"skip_best_epochs": 0}, id="explicit-zero"),
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
            pytest.param(True, TypeError, id="bool-true"),
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

    def test_ema_checkpoint_not_written_during_sanity_check(self, tmp_path: Path) -> None:
        """PTL's pre-training sanity check must not seed checkpoint_best_ema.pth.

        Regression test for #1348: PyTorch Lightning's sanity-check validation pass (which runs before any real training
        epoch) was landing straight into the EMA improvement check below, so any positive sanity-check score — common
        when starting a new training run initialized with pretrain_weights from a checkpoint pretrained on a different
        dataset — got written out as the permanent "best" EMA checkpoint before a single real epoch ran. The regular-
        checkpoint path already skips this via ModelCheckpoint._should_skip_saving_checkpoint's own
        trainer.sanity_checking guard; the EMA block reimplements its own saving logic and never inherited that guard.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.1, "val/ema_mAP_50_95": 0.83},
            current_epoch=0,
            sanity_checking=True,
        )

        cb.on_validation_end(trainer, pl_module)

        assert not (tmp_path / "checkpoint_best_ema.pth").exists()
        assert cb._best_ema == 0.0

    def test_ema_checkpoint_written_on_first_real_epoch_after_sanity_check(self, tmp_path: Path) -> None:
        """A genuine epoch 0 right after the sanity check still saves the EMA checkpoint.

        Confirms the sanity_checking guard is specific to the sanity-check pass, not a blanket "never write at epoch 0"
        — the correct guard is trainer.sanity_checking, not trainer.current_epoch == 0.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()

        sanity_trainer = _make_trainer(
            {"val/mAP_50_95": 0.1, "val/ema_mAP_50_95": 0.83},
            current_epoch=0,
            sanity_checking=True,
        )
        cb.on_validation_end(sanity_trainer, pl_module)

        real_trainer = _make_trainer(
            {"val/mAP_50_95": 0.2, "val/ema_mAP_50_95": 0.35},
            current_epoch=0,
            sanity_checking=False,
        )
        cb.on_validation_end(real_trainer, pl_module)

        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        assert cb._best_ema == pytest.approx(0.35)

    def test_smoothed_regular_not_updated_during_sanity_check(self, tmp_path: Path) -> None:
        """The regular-monitor smoothing accumulator must not warm up from the sanity check.

        Guards the secondary smoothing state (smooth_alpha > 0) alongside the EMA checkpoint above: both are custom
        bookkeeping that sits outside ModelCheckpoint's own sanity-check guard.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.9)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.83}, current_epoch=0, sanity_checking=True)

        cb.on_validation_end(trainer, pl_module)

        assert cb._smoothed_regular == 0.0

    def test_ema_checkpoint_not_written_during_real_ptl_sanity_check(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """End-to-end: an unmodified Trainer with the sanity check enabled must not seed checkpoint_best_ema.pth.

        The three tests above call on_validation_end directly with trainer.sanity_checking mocked True — they pin the
        callback's own guard logic but not the premise it depends on: that PyTorch Lightning actually invokes
        on_validation_end with trainer.sanity_checking=True during its real sanity-check pass
        (Trainer._run_sanity_check, which reuses the same evaluation loop as real validation). This test runs a real
        Trainer with num_sanity_val_steps left enabled to close that gap, using a probe callback's on_sanity_check_end
        hook to pin down that no checkpoint exists at that exact moment — before a same-valued real epoch 0 could
        produce the same file for a legitimate reason.
        """
        train_loader, val_loader = loaders

        checkpoint_path = tmp_path / "checkpoint_best_ema.pth"
        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        probe = _SanityCheckEndProbe(checkpoint_path)
        trainer = _make_cpu_trainer(tmp_path, 1, [save_cb, ema_cb, probe], num_sanity_val_steps=2)
        # Same value on every validation call (sanity check and the real epoch both log unconditionally) —
        # reproduces #1348 exactly. The probe below, not the end-state checkpoint, is what pins the timing down.
        trainer.fit(
            _EmaMetricModule(regular_value=0.2, ema_value=0.83),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        assert probe.existed_after_sanity_check is False, (
            "checkpoint_best_ema.pth existed right after PTL's sanity check ended — the sanity-check validation pass "
            "wrote it before any real epoch ran"
        )
        assert checkpoint_path.exists()
        assert save_cb._best_ema == pytest.approx(0.83)

    def test_ema_checkpoint_saved_when_regular_monitor_absent(self, tmp_path: Path) -> None:
        """Under `eval_ema_only`, val/mAP_50_95 is never logged — only val/ema_mAP_50_95 is.

        Regression test for #1285: the regular-monitor guard in on_validation_end used to `return` before reaching the
        EMA block, so checkpoint_best_ema.pth was never written in this mode even though the EMA metric had real data
        every epoch.
        """
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        # No "val/mAP_50_95" key at all — matches eval_ema_only routing (config.py:1068-1074).
        trainer = _make_trainer({"val/ema_mAP_50_95": 0.6})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_ema_checkpoint_saves_ema_callback_weights(self, tmp_path: Path, ema_callback: MagicMock) -> None:
        """EMA checkpoint must store EMA callback weights, not live model weights."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6}, callbacks=[ema_callback])

        cb.on_validation_end(trainer, _make_pl_module())

        checkpoint = torch.load(tmp_path / "checkpoint_best_ema.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == ema_callback.get_ema_model_state_dict.return_value

    def test_regular_checkpoint_uses_live_weights_when_ema_enabled(
        self, tmp_path: Path, ema_callback: MagicMock
    ) -> None:
        """Regular checkpoint must store live weights even when EMA is tracked separately."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        trainer = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.6}, callbacks=[ema_callback])

        cb.on_validation_end(trainer, _make_pl_module())

        checkpoint = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == {"w": torch.zeros(1)}

    @pytest.mark.parametrize(
        ("regular", "ema", "winner"),
        [
            pytest.param(0.6, 0.5, "regular", id="regular-wins"),
            pytest.param(0.5, 0.7, "ema", id="ema-wins"),
            pytest.param(0.6, 0.6, "regular", id="tie-goes-to-regular"),
        ],
    )
    def test_best_total_comes_from_winning_source(
        self, tmp_path: Path, regular: float, ema: float, winner: str
    ) -> None:
        """checkpoint_best_total.pth is built from the better track; EMA must strictly beat regular to win."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": regular, "val/ema_mAP_50_95": ema})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        total_data = torch.load(tmp_path / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)
        source_data = torch.load(tmp_path / f"checkpoint_best_{winner}.pth", map_location="cpu", weights_only=False)
        # total is stripped but keeps provenance keys (e.g. best_total_source) beyond model + args
        assert "args" in total_data
        assert total_data["model"] == source_data["model"]
        assert total_data["best_total_source"] == winner

    @pytest.mark.parametrize(
        ("monitor_ema", "metrics", "expect_file"),
        [
            pytest.param("val/ema_mAP_50_95", {"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.7}, True, id="ema-enabled"),
            pytest.param(None, {"val/mAP_50_95": 0.6}, False, id="ema-disabled"),
        ],
    )
    def test_last_ema_written_only_when_ema_enabled(
        self, tmp_path: Path, monitor_ema: str | None, metrics: dict, expect_file: bool
    ) -> None:
        """on_fit_end writes last_ema.pth with the final EMA weights only when EMA tracking is enabled."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema=monitor_ema, run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer(metrics)
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        assert (tmp_path / "last_ema.pth").exists() is expect_file

    def test_best_ema_backfilled_when_metric_never_improved(self, tmp_path: Path) -> None:
        """When the EMA metric never beats 0.0, on_fit_end backfills checkpoint_best_ema.pth with final EMA weights."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        # EMA metric stays at 0.0 so on_validation_end never saves checkpoint_best_ema.pth.
        trainer = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.0})
        cb.on_validation_end(trainer, pl_module)
        assert not (tmp_path / "checkpoint_best_ema.pth").exists()

        cb.on_fit_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_ema.pth").exists()

    def test_zero_ema_score_promotes_fresh_weights_in_existing_output_dir(self, tmp_path: Path) -> None:
        """A zero EMA score backfills current weights when an older checkpoint has no restored score."""
        previous = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        previous.on_validation_end(_make_trainer({"val/ema_mAP_50_95": 0.75}), _make_pl_module())
        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        callback = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            evaluates_base_model=False,
            run_test=False,
        )
        trainer = _make_trainer({"val/ema_mAP_50_95": 0.0}, current_epoch=2)
        module = _make_pl_module()
        module.model.state_dict.return_value = {"w": torch.ones(1)}

        callback.on_validation_end(trainer, module)
        callback.on_fit_end(trainer, module)

        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        total = torch.load(tmp_path / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)
        assert total["best_total_source"] == "ema"
        assert torch.equal(total["model"]["w"], torch.ones(1))

    def test_best_ema_not_overwritten_when_metric_improved(self, tmp_path: Path) -> None:
        """on_fit_end must not rewrite checkpoint_best_ema.pth when a true best was saved during training."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.5, "val/ema_mAP_50_95": 0.7})
        cb.on_validation_end(trainer, pl_module)
        assert (tmp_path / "checkpoint_best_ema.pth").exists()

        # Spy that records each destination while delegating to the real writer.
        written: list[str] = []
        real_writer = cb._write_ema_checkpoint

        def _spy(trainer_arg, module_arg, state_dict_arg, dest) -> None:
            written.append(Path(dest).name)
            real_writer(trainer_arg, module_arg, state_dict_arg, dest)

        cb._write_ema_checkpoint = _spy  # type: ignore[method-assign]
        cb.on_fit_end(trainer, pl_module)

        assert "checkpoint_best_ema.pth" not in written
        assert "last_ema.pth" in written

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
        pl_module = _make_test_step_module()

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_called_once_with(pl_module, datamodule=trainer.datamodule, verbose=False)

    def test_run_test_true_enters_trainer_test_on_non_main_rank(self, tmp_path: Path) -> None:
        """A non-main rank must still enter the collective trainer.test() once the main process broadcasts go.

        Regression test: ``on_fit_end`` used to return early on every rank but the main one, so under DDP the main
        process called the collective ``trainer.test()`` alone and hung waiting for the others.
        """
        pl_module = _make_test_step_module()
        torch.save({"model": {"w": torch.zeros(1)}}, tmp_path / "checkpoint_best_total.pth")

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, is_global_zero=False)
        # The main process decides; this rank receives (should_test, chose_ema) from the broadcast.
        trainer.strategy.broadcast.side_effect = lambda obj, src=0: (True, False)

        cb.on_fit_end(trainer, pl_module)

        trainer.strategy.barrier.assert_called_once()
        pl_module.model.load_state_dict.assert_called_once()
        trainer.test.assert_called_once_with(pl_module, datamodule=trainer.datamodule, verbose=False)

    def test_run_test_true_skips_trainer_test_inside_spawned_worker(self, tmp_path: Path) -> None:
        """Inside a spawn-launcher worker, run_test=True must warn and skip instead of launching a second time.

        Regression test: calling ``trainer.test()`` from a ``ddp_spawn`` worker made ``_MultiProcessingLauncher``
        spawn another set of processes and fail with ``DistNetworkError: EADDRINUSE``.
        """
        from pytorch_lightning.strategies.launchers.multiprocessing import _MultiProcessingLauncher

        pl_module = _make_test_step_module()

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        trainer.strategy.launcher = MagicMock(spec=_MultiProcessingLauncher)

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        assert (tmp_path / "checkpoint_best_total.pth").exists()
        trainer.strategy.barrier.assert_not_called()
        trainer.test.assert_not_called()

    def test_run_test_true_without_test_step_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=True but no test_step override — trainer.test() is NOT called.

        The guard in BestModelCallback.on_fit_end() skips trainer.test() for modules that do not override
        LightningModule.test_step() to avoid a MisconfigurationException from PTL.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        pl_module = _make_pl_module()  # MagicMock — no test_step on its class
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_not_called()

    def test_run_test_true_without_best_checkpoint_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=True must not test final weights when no best checkpoint was produced."""
        pl_module = _make_test_step_module()

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True, skip_best_epochs=2)
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=0)

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        assert not (tmp_path / "checkpoint_best_total.pth").exists()
        trainer.test.assert_not_called()

    def test_run_test_skips_total_checkpoint_from_an_earlier_fit(self, tmp_path: Path) -> None:
        """Test evaluation requires a checkpoint selected by the current callback state."""
        total_path = tmp_path / "checkpoint_best_total.pth"
        total_path.touch()
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({})

        cb.on_fit_end(trainer, _make_test_step_module())

        trainer.test.assert_not_called()
        assert total_path.exists()

    def test_run_test_loads_best_weights_before_test(self, tmp_path: Path) -> None:
        """on_fit_end loads checkpoint_best_total.pth weights before trainer.test().

        Mirrors legacy main.py:602-609 which loads the best checkpoint into the model before running test evaluation so
        the test loop measures the best model, not the end-of-training state.
        """
        pl_module = _make_test_step_module()

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        # Weights come from checkpoint_best_total.pth with strict=False plus a manual missing/unexpected-key check.
        pl_module.model.load_state_dict.assert_called_once()
        call_args, call_kwargs = pl_module.model.load_state_dict.call_args
        assert call_kwargs.get("strict") is False, "load_state_dict must be called with strict=False"
        assert call_args[0] == {"w": torch.zeros(1)}, "no _extra_state keys to filter — dict passes through unchanged"

    def test_run_test_strips_extra_state_from_regular_weights_before_load(self, tmp_path: Path) -> None:
        """A live-model checkpoint's Transformer Engine `_extra_state` entries must not reach load_state_dict.

        Under FP8, ``_get_live_model_state_dict`` writes the regular checkpoint straight from the live module's
        ``state_dict()``, which — unlike the EMA export — still carries ``_extra_state`` keys. Transformer Engine
        rejects their pickle round-trip via ``set_extra_state()``, so they must be excluded from the dict handed to
        ``load_state_dict`` rather than merely tolerated afterward via ``strict=False``.
        """
        # The model itself is missing the "w" tensor after reload — a genuine mismatch that must still raise.
        pl_module = _make_test_step_module(
            {"w": torch.zeros(1), "layer._extra_state": b"fp8-scaling-history"}, missing_keys=["w"]
        )

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        with pytest.raises(RuntimeError) as exc_info:
            cb.on_fit_end(trainer, pl_module)

        assert "Error loading best regular weights" in str(exc_info.value)
        assert "'w'" in str(exc_info.value)

        loaded_dict = pl_module.model.load_state_dict.call_args[0][0]
        assert "layer._extra_state" not in loaded_dict, "_extra_state must be filtered out before load_state_dict"
        assert loaded_dict == {"w": torch.zeros(1)}

    def test_run_test_false_skips_trainer_test(self, tmp_path: Path) -> None:
        """run_test=False means trainer.test() is never called."""
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.5})

        cb.on_validation_end(trainer, pl_module)
        cb.on_fit_end(trainer, pl_module)

        trainer.test.assert_not_called()

    @pytest.mark.parametrize(("cb_kwargs", "metrics", "checkpoint_filename"), _REGULAR_AND_EMA_CASES)
    @pytest.mark.parametrize(
        ("datamodule_class_names", "train_config_kwargs", "expected"),
        [
            pytest.param(["cat", "dog"], {}, ["cat", "dog"], id="unset-config-uses-datamodule-names"),
            # The datamodule names would overwrite the explicit ones if the bug existed.
            pytest.param(["other_class"], {"class_names": ["defect"]}, ["defect"], id="explicit-names-kept"),
            # Guard-bypass regression: `not getattr(..., "class_names", None)` treated an explicit empty list like
            # None (both falsy); the fix uses an `is None` identity check.
            pytest.param(["cat", "dog"], {"class_names": []}, [], id="explicit-empty-names-kept"),
            pytest.param([], {}, [], id="unset-config-uses-empty-datamodule-names"),
        ],
    )
    def test_checkpoint_class_names_reflects_datamodule_and_train_config(
        self,
        tmp_path: Path,
        cb_kwargs: dict,
        metrics: dict,
        checkpoint_filename: str,
        datamodule_class_names: list,
        train_config_kwargs: dict,
        expected: list,
    ) -> None:
        """Saved checkpoint args.class_names reflects dataset class names, explicit TrainConfig values winning.

        Regression tests for #509: checkpoints were saved with class_names=None when the user did not pass class_names
        explicitly (fixed by falling back to the datamodule), and a separate guard-bypass let an explicit empty list get
        silently overwritten by the datamodule's names. Covers both the regular and EMA checkpoint paths across unset,
        explicit, and explicit-empty class_names in TrainConfig.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), **cb_kwargs)
        trainer = _make_trainer(metrics)
        trainer.datamodule.class_names = datamodule_class_names

        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False, **train_config_kwargs)

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / checkpoint_filename, map_location="cpu", weights_only=False)
        assert checkpoint["args"]["class_names"] == expected

    # --- PTL-compatible format tests ---

    def test_regular_checkpoint_args_is_dict(self, tmp_path: Path) -> None:
        """Saved args must be a plain dict (not a Pydantic object) for weights_only=True compat."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()
        pl_module.train_config = TrainConfig(dataset_dir=str(tmp_path / "ds"), tensorboard=False)

        cb.on_validation_end(trainer, pl_module)

        # weights_only=True must succeed now that args is a plain dict.
        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=True)
        assert isinstance(ckpt["args"], dict)

    @pytest.mark.parametrize(("cb_kwargs", "metrics", "checkpoint_filename"), _REGULAR_AND_EMA_CASES)
    def test_checkpoint_has_ptl_state_dict_key(
        self, tmp_path: Path, cb_kwargs: dict, metrics: dict, checkpoint_filename: str
    ) -> None:
        """Saved regular and EMA checkpoints include a 'state_dict' whose keys carry the 'model.' prefix."""
        cb = BestModelCallback(output_dir=str(tmp_path), **cb_kwargs)

        cb.on_validation_end(_make_trainer(metrics), _make_pl_module())

        ckpt = torch.load(tmp_path / checkpoint_filename, map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    @pytest.mark.parametrize(("cb_kwargs", "metrics", "checkpoint_filename"), _REGULAR_AND_EMA_CASES)
    @pytest.mark.parametrize("current_epoch", [3, 5])
    def test_checkpoint_has_loops_key(
        self, tmp_path: Path, cb_kwargs: dict, metrics: dict, checkpoint_filename: str, current_epoch: int
    ) -> None:
        """Saved regular and EMA checkpoints include 'loops' whose fit_loop completed-epoch counter is epoch + 1."""
        cb = BestModelCallback(output_dir=str(tmp_path), **cb_kwargs)

        cb.on_validation_end(_make_trainer(metrics, current_epoch=current_epoch), _make_pl_module())

        ckpt = torch.load(tmp_path / checkpoint_filename, map_location="cpu", weights_only=False)
        assert "loops" in ckpt
        assert ckpt["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == current_epoch + 1

    def test_regular_checkpoint_has_ptl_version_key(self, tmp_path: Path) -> None:
        """Saved regular checkpoint must include 'pytorch-lightning_version'."""
        cb = BestModelCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        pl_module = _make_pl_module()

        cb.on_validation_end(trainer, pl_module)

        ckpt = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert ckpt.get("pytorch-lightning_version") == ptl_version

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

    def test_fit_completes_when_metric_missing_on_some_epochs(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """BestModelCallback must not crash over 3 epochs when metrics are only logged every 2nd epoch."""
        train_loader, val_loader = loaders

        cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer = _make_cpu_trainer(tmp_path, 3, [cb])
        trainer.fit(_EvalIntervalModule(eval_interval=2), train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Checkpoint must be written on eval epochs (0 and 2) — at least one must exist.
        assert (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_best_total_checkpoint_resumes_via_trainer_fit_ckpt_path(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """checkpoint_best_total.pth must restore epoch/step when passed to Trainer.fit(ckpt_path=...)."""
        train_loader, val_loader = loaders

        save_cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer_first = _make_cpu_trainer(tmp_path, 1, [save_cb])
        trainer_first.fit(_ResumeTinyModule(), train_dataloaders=train_loader, val_dataloaders=val_loader)

        ckpt_path = tmp_path / "checkpoint_best_total.pth"
        assert ckpt_path.exists()
        first_phase_global_step = trainer_first.global_step
        assert first_phase_global_step == 2
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt_data["global_step"] == first_phase_global_step

        resume_probe = _ResumeProbeCallback()
        trainer_second = _make_cpu_trainer(tmp_path, 2, [resume_probe])
        trainer_second.fit(
            _ResumeTinyModule(),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path),
        )

        # PTL applies loop restoration by the first train epoch start.
        assert resume_probe.first_train_epoch == 1
        # Optimizer state is intentionally fresh on a stripped-checkpoint resume: one epoch of 2 steps.
        assert trainer_second.current_epoch == 2
        assert trainer_second.global_step == 2

    @pytest.mark.parametrize("checkpoint_name", ["checkpoint_best_regular.pth", "checkpoint_best_total.pth"])
    def test_best_model_score_survives_real_trainer_fit_resume(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader], checkpoint_name: str
    ) -> None:
        """``best_model_score`` and the on-disk checkpoint must survive a real ``trainer.fit(ckpt_path=...)`` resume.

        Regression test for the "callbacks" key being absent from ``_build_checkpoint_payload``'s output — the bug this
        PR fixes. Every existing regression test for callback-state persistence (``TestBestEmaStatePersistence``) calls
        ``state_dict()``/``load_state_dict()`` directly in Python, never through PTL's own
        ``_call_callbacks_load_state_dict``, so none of them would catch the payload missing the "callbacks" key.
        Parametrized over ``checkpoint_best_regular.pth`` (built directly by ``_build_checkpoint_payload``) and
        ``checkpoint_best_total.pth`` (built via ``shutil.copy2`` + ``strip_checkpoint``, a separate code path that
        needs ``"callbacks"`` in ``_PTL_COMPAT_KEYS``).
        """
        train_loader, val_loader = loaders

        save_cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer_first = _make_cpu_trainer(tmp_path, 1, [save_cb])
        trainer_first.fit(_MetricModule(0.7), train_dataloaders=train_loader, val_dataloaders=val_loader)
        assert save_cb.best_model_score is not None
        assert save_cb.best_model_score.item() == pytest.approx(0.7)

        ckpt_path = tmp_path / checkpoint_name
        assert ckpt_path.exists()
        first_weights = {
            k: v.clone() for k, v in torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].items()
        }

        # Fresh callback instance, resumed via a real Trainer.fit(ckpt_path=...) — exercises PTL's
        # own `_call_callbacks_load_state_dict`, not a direct load_state_dict() call.
        resumed_cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer_second = _make_cpu_trainer(tmp_path, 2, [resumed_cb])
        # Post-resume epoch logs a *worse* metric (0.3): without the fix, best_model_score resets to
        # None on a fresh callback and 0.3 trivially "improves" on it, overwriting the on-disk
        # checkpoint with worse weights — the exact failure mode from the issue this PR fixes.
        trainer_second.fit(
            _MetricModule(0.3),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path),
        )

        assert resumed_cb.best_model_score is not None
        assert resumed_cb.best_model_score.item() == pytest.approx(0.7), (
            'best_model_score must be restored from the checkpoint\'s "callbacks" key; without the '
            "fix a fresh BestModelCallback starts with best_model_score=None and 0.3 wins by default"
        )
        # Only checked for the regular file: checkpoint_best_total.pth is re-copied by the second fit's on_fit_end, so
        # equality there would not prove the regular checkpoint survived.
        if checkpoint_name == "checkpoint_best_regular.pth":
            reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            for key, value in reloaded["model"].items():
                assert torch.equal(value, first_weights[key]), (
                    f"checkpoint_best_regular.pth weights for {key!r} changed after a worse post-resume "
                    "epoch — the good checkpoint was overwritten"
                )

    def test_ema_and_early_stopping_state_survive_real_trainer_fit_resume(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """RFDETREMACallback and RFDETREarlyStopping state must also survive a real ``ckpt_path`` resume.

        ``test_best_model_score_survives_real_trainer_fit_resume`` only registers ``BestModelCallback`` on the
        trainer, so it exercises restoration of best-score tracking alone. The warning added to ``RFDETR.train()``
        promises "best-score tracking, EMA, early stopping" all resume correctly together — this test registers
        all three callbacks so ``_build_checkpoint_payload``'s generic ``trainer.callbacks`` loop is actually
        exercised for the other two, not just ``BestModelCallback`` itself. Lightweight checkpoints reset the
        optimizer loop's global step, so EMA restores its averaged weights but resets its step guard to permit the
        first new batch update.
        """
        train_loader, val_loader = loaders

        # last_ema.pth (enabled by monitor_ema) is rebuilt in on_fit_end with live callback state, so it can hold a
        # nonzero wait_count; checkpoint_best_regular.pth is only rewritten on improvement, which resets wait_count.
        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        early_stop_cb = RFDETREarlyStopping(patience=5, min_delta=0.001, verbose=False)
        trainer_first = _make_cpu_trainer(tmp_path, 2, [save_cb, ema_cb, early_stop_cb])
        # Epoch 0: 0.7 (best, wait_count 0->0). Epoch 1: 0.5 (worse, wait_count 0->1) — training ends
        # with wait_count=1 and ema_cb mid-averaging, both captured live into last_ema.pth.
        trainer_first.fit(_VaryingMetricModule([0.7, 0.5]), train_dataloaders=train_loader, val_dataloaders=val_loader)

        ckpt_path = tmp_path / "last_ema.pth"
        assert ckpt_path.exists()
        persisted_callbacks = torch.load(ckpt_path, map_location="cpu", weights_only=False)["callbacks"]
        persisted_ema_step = persisted_callbacks[ema_cb.state_key]["latest_update_step"]
        persisted_wait_count = persisted_callbacks[early_stop_cb.state_key]["wait_count"]
        # Both must be genuinely non-default so a fresh, un-restored callback (which starts at 0) cannot
        # accidentally satisfy the post-resume assertions below.
        assert persisted_ema_step > 0
        assert persisted_wait_count > 0

        ema_callback = RFDETREMACallback()
        early_stop_callback = RFDETREarlyStopping(patience=5, min_delta=0.001, verbose=False)
        capture = _CallbackStateCaptureCallback(ema_callback, early_stop_callback)
        # last_ema.pth is built in on_fit_end, after the epoch counter reached 2, so PTL's restored fit_loop counts 3
        # epochs as completed: max_epochs must be 4 (not 3) for one real epoch to run.
        trainer_second = _make_cpu_trainer(
            tmp_path,
            4,
            [
                BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False),
                ema_callback,
                early_stop_callback,
                capture,
            ],
        )
        trainer_second.fit(
            _VaryingMetricModule([0.7, 0.5, 0.9, 0.9]),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path),
        )

        assert capture.ema_latest_update_step == 0, (
            "RFDETREMACallback must reset its absolute step guard when a lightweight checkpoint resets "
            "trainer.global_step, otherwise the resumed phase skips EMA updates"
        )
        assert capture.early_stop_wait_count == persisted_wait_count, (
            "RFDETREarlyStopping.wait_count must be restored from the checkpoint's 'callbacks' key; "
            "without the fix a fresh callback starts at 0"
        )

    def test_ema_checkpoint_weights_survive_real_trainer_fit_resume(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """``checkpoint_best_ema.pth`` weights must survive a real ``ckpt_path`` resume too.

        ``test_best_model_score_survives_real_trainer_fit_resume`` only parametrizes over
        ``checkpoint_best_regular.pth``/``checkpoint_best_total.pth`` and never asserts weight
        preservation for the EMA track. The fix in ``_build_checkpoint_payload`` applies identically to
        ``checkpoint_best_ema.pth`` (same code path, via ``_write_ema_checkpoint``), but without a
        dedicated regression test a broken ``_best_ema`` restoration on this specific file would slip
        through: a worse post-resume EMA epoch would trivially beat a reset ``_best_ema=0.0`` and
        overwrite the good EMA checkpoint on disk, exactly like the regular-checkpoint failure mode this
        PR fixes.
        """
        train_loader, val_loader = loaders

        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        trainer_first = _make_cpu_trainer(tmp_path, 1, [save_cb, ema_cb])
        trainer_first.fit(
            _EmaMetricModule(regular_value=0.4, ema_value=0.75),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        assert save_cb._best_ema == pytest.approx(0.75)

        ckpt_path = tmp_path / "checkpoint_best_ema.pth"
        assert ckpt_path.exists()
        first_ema_weights = {
            k: v.clone() for k, v in torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].items()
        }

        # Fresh callbacks, resumed via a real Trainer.fit(ckpt_path=...) — exercises PTL's own
        # _call_callbacks_load_state_dict for both BestModelCallback._best_ema and the EMA callback's
        # own averaging state, not a direct load_state_dict() call.
        resumed_save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        resumed_ema_cb = RFDETREMACallback()
        trainer_second = _make_cpu_trainer(tmp_path, 2, [resumed_save_cb, resumed_ema_cb])
        # Post-resume epoch logs a *worse* EMA metric (0.3): without the fix, _best_ema resets to 0.0 on
        # a fresh callback and 0.3 trivially "improves" on it, overwriting the on-disk EMA checkpoint.
        trainer_second.fit(
            _EmaMetricModule(regular_value=0.35, ema_value=0.3),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path),
        )

        assert resumed_save_cb._best_ema == pytest.approx(0.75), (
            '_best_ema must be restored from the checkpoint\'s "callbacks" key; without the fix a '
            "fresh BestModelCallback starts with _best_ema=0.0 and the worse post-resume EMA (0.3) "
            "wins by default"
        )
        reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key, value in reloaded["model"].items():
            assert torch.equal(value, first_ema_weights[key]), (
                f"checkpoint_best_ema.pth weights for {key!r} changed after a worse post-resume EMA "
                "epoch — the good EMA checkpoint was overwritten"
            )

    def test_best_model_score_not_restored_when_output_dir_changes(
        self, tmp_path: Path, loaders: tuple[DataLoader, DataLoader]
    ) -> None:
        """A resumed ``output_dir`` different from the checkpoint's own does not restore ``best_model_score``.

        Pins PTL's own ``ModelCheckpoint.load_state_dict()`` gate (installed pytorch-lightning,
        ``model_checkpoint.py``): it only restores ``best_model_score``/``best_k_models`` when the resumed callback's
        ``dirpath`` matches the value stored in the checkpoint's ``"callbacks"`` state exactly; on a mismatch it warns
        and restores only ``best_model_path``. RF-DETR's ``resume`` and ``output_dir`` config fields are independent
        (``config.py``), so pointing ``resume=`` at one directory while writing to another is a config-permitted
        combination, not a corrupted setup — this test locks in the resulting (previously untested) behavior.

        ``max_epochs=1`` on both phases means the checkpoint's restored fit-loop progress (1 epoch already completed)
        leaves zero epochs left to run in phase two: the assertions below capture exactly what ``load_state_dict()``
        restored, with no confound from a subsequent training epoch overwriting ``best_model_score``/``best_model_path``
        again on its own.
        """
        train_loader, val_loader = loaders

        save_cb = BestModelCallback(output_dir=str(tmp_path), run_test=False)
        trainer_first = _make_cpu_trainer(tmp_path, 1, [save_cb])
        trainer_first.fit(_MetricModule(0.7), train_dataloaders=train_loader, val_dataloaders=val_loader)
        assert save_cb.best_model_score.item() == pytest.approx(0.7)

        ckpt_path = tmp_path / "checkpoint_best_regular.pth"
        assert ckpt_path.exists()

        # Resume into a DIFFERENT output_dir than the one the checkpoint was saved under.
        other_output_dir = tmp_path / "other_output_dir"
        other_output_dir.mkdir()
        resumed_cb = BestModelCallback(output_dir=str(other_output_dir), run_test=False)
        trainer_second = _make_cpu_trainer(tmp_path, 1, [resumed_cb])
        with pytest.warns(UserWarning, match="dirpath has changed"):
            trainer_second.fit(
                _MetricModule(0.3),
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=str(ckpt_path),
            )

        # No epoch ran post-resume (fit_loop already shows 1/1 epochs completed), so these values are
        # exactly what load_state_dict() produced, undisturbed by any further training.
        assert trainer_second.global_step == 0
        # best_model_score was NOT carried over — the dirpath mismatch skips it, leaving the fresh
        # callback's own default (None), exactly as if it had never resumed at all.
        assert resumed_cb.best_model_score is None
        # best_model_path IS still carried over regardless of the dirpath mismatch (PTL restores it
        # unconditionally), even though it now points at a path under the OLD output_dir.
        assert resumed_cb.best_model_path == str(ckpt_path)
