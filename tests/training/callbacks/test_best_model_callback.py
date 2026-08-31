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
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from rfdetr.config import RFDETRLargeDeprecatedConfig, RFDETRMediumConfig
from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.training.callbacks.ema import RFDETREMACallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(
    metrics: dict[str, float],
    current_epoch: int = 1,
    is_global_zero: bool = True,
    callbacks: list[object] | None = None,
    sanity_checking: bool = False,
) -> MagicMock:
    """Create a minimal mock Trainer with controllable callback_metrics.

    Sets the attributes required by ModelCheckpoint and EarlyStopping skip-guards so that callbacks run normally in unit
    tests.

    Args:
        metrics: Validation metric names mapped to values, exposed as ``trainer.callback_metrics``.
        current_epoch: Value for ``trainer.current_epoch``.
        is_global_zero: Value for ``trainer.is_global_zero``.
        callbacks: Value for ``trainer.callbacks``; defaults to an empty list.
        sanity_checking: Value for ``trainer.sanity_checking`` — True mimics PyTorch Lightning's pre-training
            sanity-check pass (see #1348).

    Returns:
        A mock standing in for a :class:`~pytorch_lightning.Trainer`.

    Examples:
        >>> trainer = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=2, sanity_checking=True)
        >>> trainer.current_epoch, trainer.sanity_checking
        (2, True)
        >>> trainer.callback_metrics["val/mAP_50_95"].item()
        0.5
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
    trainer.sanity_checking = sanity_checking
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


class _EmaMetricModule(LightningModule):
    """LightningModule that logs independently controlled ``val/mAP_50_95`` and ``val/ema_mAP_50_95`` values.

    Used to drive :class:`BestModelCallback`'s EMA-checkpoint bookkeeping (``checkpoint_best_ema.pth``) with a
    real, trainable model, so :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback` produces genuinely
    different EMA weights per :class:`~pytorch_lightning.Trainer` phase.
    """

    def __init__(self, regular_value: float, ema_value: float) -> None:
        """Initialize with the fixed regular/EMA metric values to log every epoch.

        Args:
            regular_value: Value logged as ``val/mAP_50_95`` on every validation epoch.
            ema_value: Value logged as ``val/ema_mAP_50_95`` on every validation epoch.
        """
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}
        self._regular_value = regular_value
        self._ema_value = ema_value

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
        """Log the fixed regular and EMA metric values, ignoring the actual batch contents.

        Args:
            batch: Unused; only the epoch-level log calls matter for these tests.
            batch_idx: Unused.
        """
        del batch, batch_idx
        self.log("val/mAP_50_95", torch.tensor(self._regular_value), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ema_mAP_50_95", torch.tensor(self._ema_value), on_step=False, on_epoch=True, prog_bar=False)

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


# ---------------------------------------------------------------------------
# TestBestModelCallback
# ---------------------------------------------------------------------------


class TestEvaluatesBaseModel:
    """Regular-track behaviour when validation does not evaluate the base model (``TrainConfig.eval_base_model``)."""

    def test_regular_checkpoint_is_not_written_when_base_model_is_not_evaluated(self, tmp_path: Path) -> None:
        """No base-weights checkpoint may be saved while the primary key carries the EMA score.

        Under the default eval policy COCOEvalCallback mirrors the EMA score onto ``val/mAP_50_95``, so the key is
        present and the parent ModelCheckpoint would happily save the base weights against it — checkpointing one model
        on the other model's score. Only the EMA track is entitled to write this epoch.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_ema_checkpoint_is_still_written_when_base_model_is_not_evaluated(self, tmp_path: Path) -> None:
        """Suppressing the regular track must not also suppress the EMA track.

        The EMA track is the only checkpoint tracking that runs under the default policy (#1285); disabling both would
        leave a run with no best checkpoint at all.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", evaluates_base_model=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.9, "val/ema_mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_ema.pth").exists()

    def test_regular_checkpoint_is_written_when_base_model_is_evaluated(self, tmp_path: Path) -> None:
        """eval_base_model=True (and every use_ema=False run) keeps the regular track working as before.

        The suppression must be scoped to the configuration that actually stops evaluating the base model, not applied
        whenever EMA happens to be enabled.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_smoothing_accumulator_ignores_the_mirrored_ema_score(self, tmp_path: Path) -> None:
        """The regular smoothing accumulator must not absorb EMA values it will never checkpoint against.

        ``_smoothed_regular`` is persisted in the callback state and compared against the raw EMA score in
        ``on_fit_end``; letting the mirrored EMA score feed it would corrupt that comparison on a later resume.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=False, smooth_alpha=0.5)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert cb._smoothed_regular == 0.0


class TestBestModelCallback:
    """Verify best-model checkpoint saving and selection."""

    def test_existing_positional_arguments_keep_their_meaning(self, tmp_path: Path) -> None:
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

    def test_ema_checkpoint_not_written_during_real_ptl_sanity_check(self, tmp_path: Path) -> None:
        """End-to-end: an unmodified Trainer with the sanity check enabled must not seed checkpoint_best_ema.pth.

        The three tests above call on_validation_end directly with trainer.sanity_checking mocked True — they pin the
        callback's own guard logic but not the premise it depends on: that PyTorch Lightning actually invokes
        on_validation_end with trainer.sanity_checking=True during its real sanity-check pass
        (Trainer._run_sanity_check, which reuses the same evaluation loop as real validation). This test runs a real
        Trainer with num_sanity_val_steps left enabled to close that gap, using a probe callback's on_sanity_check_end
        hook to pin down that no checkpoint exists at that exact moment — before a same-valued real epoch 0 could
        produce the same file for a legitimate reason.
        """
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        checkpoint_path = tmp_path / "checkpoint_best_ema.pth"
        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        probe = _SanityCheckEndProbe(checkpoint_path)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=2,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[save_cb, ema_cb, probe],
            default_root_dir=str(tmp_path),
        )
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

    def test_ema_checkpoint_saves_ema_callback_weights(self, tmp_path: Path) -> None:
        """EMA checkpoint must store EMA callback weights, not live model weights."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        ema_state = {"w": torch.ones(1)}
        ema_callback = MagicMock()
        ema_callback.get_ema_model_state_dict.return_value = ema_state
        # Real dict, not MagicMock's auto-generated one: _build_checkpoint_payload now calls
        # callback.state_dict() on every trainer callback to populate the "callbacks" key
        # (#checkpoint-resume-callbacks-fix), and torch.save can't pickle a bare MagicMock.
        ema_callback.state_dict.return_value = {}
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6},
            callbacks=[ema_callback],
        )
        pl_module = _make_pl_module()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_ema.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == ema_state

    def test_regular_checkpoint_uses_live_weights_when_ema_enabled(self, tmp_path: Path) -> None:
        """Regular checkpoint must store live weights even when EMA is tracked separately."""
        cb = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
        )
        ema_state = {"w": torch.ones(1)}
        ema_callback = MagicMock()
        ema_callback.get_ema_model_state_dict.return_value = ema_state
        # Real dict, not MagicMock's auto-generated one: _build_checkpoint_payload now calls
        # callback.state_dict() on every trainer callback to populate the "callbacks" key
        # (#checkpoint-resume-callbacks-fix), and torch.save can't pickle a bare MagicMock.
        ema_callback.state_dict.return_value = {}
        trainer = _make_trainer(
            {"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.6},
            callbacks=[ema_callback],
        )
        pl_module = _make_pl_module()
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}

        cb.on_validation_end(trainer, pl_module)

        checkpoint = torch.load(tmp_path / "checkpoint_best_regular.pth", map_location="cpu", weights_only=False)
        assert checkpoint["model"] == {"w": torch.zeros(1)}

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
        assert data["best_total_source"] == "regular"

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
        # total is stripped but keeps provenance keys (e.g. best_total_source) beyond model + args
        assert total_data["model"] == ema_data["model"]
        assert total_data["best_total_source"] == "ema"

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
        assert total_data["best_total_source"] == "regular"

    def test_last_ema_saved_when_ema_enabled(self, tmp_path: Path) -> None:
        """on_fit_end writes last_ema.pth with the final EMA weights when EMA tracking is enabled."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.7})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        assert (tmp_path / "last_ema.pth").exists()

    def test_last_ema_not_saved_when_ema_disabled(self, tmp_path: Path) -> None:
        """on_fit_end must not write last_ema.pth when EMA tracking is disabled (monitor_ema=None)."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema=None, run_test=False)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.6})
        cb.on_validation_end(trainer, pl_module)

        cb.on_fit_end(trainer, pl_module)

        assert not (tmp_path / "last_ema.pth").exists()

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

    def test_zero_ema_score_promotes_ema_checkpoint_for_ema_only_runs(self, tmp_path: Path) -> None:
        """A valid zero EMA score still produces and promotes the total checkpoint when no base model was evaluated."""
        callback = BestModelCallback(
            output_dir=str(tmp_path),
            monitor_ema="val/ema_mAP_50_95",
            evaluates_base_model=False,
            run_test=False,
        )
        trainer = _make_trainer({"val/ema_mAP_50_95": 0.0})

        callback.on_validation_end(trainer, _make_pl_module())
        callback.on_fit_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_ema.pth").exists()
        total = torch.load(tmp_path / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)
        assert total["best_total_source"] == "ema"

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

        Mirrors legacy main.py:602-609 which loads the best checkpoint into the model before running test evaluation so
        the test loop measures the best model, not the end-of-training state.
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

        Regression test for #509: checkpoints were saved with class_names=None when the user did not pass class_names
        explicitly, causing reloaded-model inference to fall through to COCO labels instead of dataset labels.
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

        Regression test for #509: EMA checkpoint path was not enriched with class names, so EMA-selected runs would
        still return COCO labels after reload.
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

        When the user passes class_names=['defect'] to TrainConfig, the saved checkpoint must keep that value even if
        the datamodule reports different names.
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

        Guard-bypass regression: the truthiness check `not getattr(..., "class_names", None)` treated an explicit empty
        list the same as None (both falsy), silently overwriting the user's intent with the datamodule's names. The fix
        uses `is None` identity.
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
        """Saved regular checkpoint must include 'state_dict' with model.

        prefix.
        """
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
        """Saved EMA checkpoint must include 'state_dict' with model.

        prefix.
        """
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

    @pytest.mark.parametrize("checkpoint_name", ["checkpoint_best_regular.pth", "checkpoint_best_total.pth"])
    def test_best_model_score_survives_real_trainer_fit_resume(self, tmp_path: Path, checkpoint_name: str) -> None:
        """``best_model_score`` and the on-disk checkpoint must survive a real ``trainer.fit(ckpt_path=...)`` resume.

        Regression test for the "callbacks" key being absent from ``_build_checkpoint_payload``'s output — the bug this
        PR fixes. Every existing regression test for callback-state persistence (``TestBestEmaStatePersistence``) calls
        ``state_dict()``/``load_state_dict()`` directly in Python, never through PTL's own
        ``_call_callbacks_load_state_dict``, so none of them would catch the payload missing the "callbacks" key.
        Parametrized over ``checkpoint_best_regular.pth`` (built directly by ``_build_checkpoint_payload``) and
        ``checkpoint_best_total.pth`` (built via ``shutil.copy2`` + ``strip_checkpoint``, a separate code path that
        needs ``"callbacks"`` in ``_PTL_COMPAT_KEYS``).
        """
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
        trainer_second = Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[resumed_cb],
            default_root_dir=str(tmp_path),
        )
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
        # The worse post-resume epoch must not have overwritten checkpoint_best_regular.pth on disk.
        # checkpoint_best_total.pth IS re-copied by trainer_second's own on_fit_end (every trainer.fit()
        # call re-runs on_fit_end, not just the first), so re-asserting equality there would only prove
        # the second copy reproduced the first byte-for-byte, not that the regular checkpoint itself
        # survived untouched — hence this assertion only runs for the regular checkpoint.
        if checkpoint_name == "checkpoint_best_regular.pth":
            reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            for key, value in reloaded["model"].items():
                assert torch.equal(value, first_weights[key]), (
                    f"checkpoint_best_regular.pth weights for {key!r} changed after a worse post-resume "
                    "epoch — the good checkpoint was overwritten"
                )

    def test_ema_and_early_stopping_state_survive_real_trainer_fit_resume(self, tmp_path: Path) -> None:
        """RFDETREMACallback and RFDETREarlyStopping state must also survive a real ``ckpt_path`` resume.

        ``test_best_model_score_survives_real_trainer_fit_resume`` only registers ``BestModelCallback`` on the
        trainer, so it exercises restoration of best-score tracking alone. The warning added to ``RFDETR.train()``
        promises "best-score tracking, EMA, early stopping" all resume correctly together — this test registers
        all three callbacks so ``_build_checkpoint_payload``'s generic ``trainer.callbacks`` loop is actually
        exercised for the other two, not just ``BestModelCallback`` itself. Lightweight checkpoints reset the
        optimizer loop's global step, so EMA restores its averaged weights but resets its step guard to permit the
        first new batch update.
        """
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        # monitor_ema enables last_ema.pth: unlike checkpoint_best_regular.pth (only rewritten on a
        # metric *improvement*, which is exactly the condition that resets RFDETREarlyStopping.wait_count
        # to 0 — so it can never capture a nonzero wait_count), last_ema.pth is rebuilt unconditionally
        # in on_fit_end with each callback's *live* end-of-training state.
        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        early_stop_cb = RFDETREarlyStopping(patience=5, min_delta=0.001, verbose=False)
        trainer_first = Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[save_cb, ema_cb, early_stop_cb],
            default_root_dir=str(tmp_path),
        )
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
        # last_ema.pth is built in on_fit_end, after the epoch counter has already advanced past both
        # trained epochs (0 and 1) to 2 — one further than the on_validation_end-time snapshot
        # _make_fit_loop_state assumes (see its docstring). PTL's restored fit_loop therefore treats 3
        # epochs as already completed, so max_epochs must be 4 (not 3) for one real epoch to run here.
        trainer_second = Trainer(
            max_epochs=4,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[
                BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False),
                ema_callback,
                early_stop_callback,
                capture,
            ],
            default_root_dir=str(tmp_path),
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

    def test_ema_checkpoint_weights_survive_real_trainer_fit_resume(self, tmp_path: Path) -> None:
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
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        save_cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        ema_cb = RFDETREMACallback()
        trainer_first = Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[save_cb, ema_cb],
            default_root_dir=str(tmp_path),
        )
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
        trainer_second = Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[resumed_save_cb, resumed_ema_cb],
            default_root_dir=str(tmp_path),
        )
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

    def test_best_model_score_not_restored_when_output_dir_changes(self, tmp_path: Path) -> None:
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
        trainer_first.fit(_MetricModule(0.7), train_dataloaders=train_loader, val_dataloaders=val_loader)
        assert save_cb.best_model_score.item() == pytest.approx(0.7)

        ckpt_path = tmp_path / "checkpoint_best_regular.pth"
        assert ckpt_path.exists()

        # Resume into a DIFFERENT output_dir than the one the checkpoint was saved under.
        other_output_dir = tmp_path / "other_output_dir"
        other_output_dir.mkdir()
        resumed_cb = BestModelCallback(output_dir=str(other_output_dir), run_test=False)
        trainer_second = Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[resumed_cb],
            default_root_dir=str(tmp_path),
        )
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
        """Patience=3 with 3 no-improvement epochs triggers stop."""
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

    def test_only_ema_available(self) -> None:
        """When the regular monitor key is absent, falls back to EMA without error.

        Mirrors `test_only_regular_available`; this is the exact shape `eval_ema_only` produces
        (`val/mAP_50_95` never populated, see #1285) and was previously untested here, unlike
        `BestModelCallback.on_validation_end`'s sibling guard where the same input shape was an
        outright bug (see `test_ema_checkpoint_saved_when_regular_monitor_absent`). This callback's
        `regular_val is None` branching already handled it correctly; this test locks that in.
        """
        cb = RFDETREarlyStopping(patience=5, min_delta=0.001)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/ema_mAP_50_95": 0.55})
        cb.on_validation_end(trainer, pl_module)

        assert cb.best_score.item() == pytest.approx(0.55)
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

    Before the fix, BestModelCallback did not override state_dict() / load_state_dict(), so _best_ema was never included
    in the Lightning callback state bundle.  On resume via trainer.fit(ckpt_path=...) the callback was reconstructed
    fresh with _best_ema=0.0, causing any positive post-resume EMA value to trivially overwrite checkpoint_best_ema.pth
    with inferior weights.

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

        Without the fix: _best_ema resets to 0.0 on resume, so any positive EMA metric (0.5) trivially satisfies ema_val
        > _best_ema and overwrites the checkpoint saved pre-resume (0.75).
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

        Without the fix: _best_ema=0.0 after resume, so regular (0.6) wins over the true EMA best (0.8) —
        checkpoint_best_total.pth is built from the wrong source.  The payload's ``best_total_source`` field records
        which source won, so EMA selection is asserted directly instead of via a saved-epoch proxy.
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
        # best_total_source records the winning source directly.
        # If _best_ema was NOT restored (bug), regular wins → "regular".
        # If _best_ema IS restored (fix), EMA wins → "ema".
        source = total_data["best_total_source"]
        assert source == "ema", (
            "on_fit_end must select EMA (best=0.8) over regular (best=0.6); "
            f"got best_total_source={source!r} — _best_ema not restored from state_dict"
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

    def test_state_dict_round_trip_preserves_all_three_smooth_fields(self, tmp_path: Path) -> None:
        """state_dict/load_state_dict round-trip preserves _best_ema, _smoothed_regular, and _best_raw_regular.

        Scenario: user trains with smooth_alpha=0.5, two validation epochs advance all three accumulators,
        then a fresh callback is restored from the persisted state — all three fields must survive unchanged.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", smooth_alpha=0.5)
        pl_module = _make_pl_module()

        # Epoch 0: raw=0.6 → smoothed=0.3, EMA=0.5 (saves EMA checkpoint)
        trainer0 = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.5}, current_epoch=0)
        trainer0.global_step = 1
        cb.on_validation_end(trainer0, pl_module)

        # Epoch 1: raw=0.4 → smoothed=0.35, EMA=0.7 > 0.5 (saves new EMA checkpoint); raw at smoothed-best=0.4
        trainer1 = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.7}, current_epoch=1)
        trainer1.global_step = 2
        cb.on_validation_end(trainer1, pl_module)

        state = cb.state_dict()

        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", smooth_alpha=0.5)
        cb_resumed.load_state_dict(state)

        assert cb_resumed._best_ema == pytest.approx(cb._best_ema), "_best_ema must survive round-trip"
        assert cb_resumed._smoothed_regular == pytest.approx(cb._smoothed_regular), (
            "_smoothed_regular must survive round-trip"
        )
        assert cb_resumed._best_raw_regular == pytest.approx(cb._best_raw_regular), (
            "_best_raw_regular must survive round-trip"
        )

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
# TestBestModelSmoothAlpha
# ---------------------------------------------------------------------------


class TestBestModelSmoothAlpha:
    """Verify ``smooth_alpha`` smooths the monitored metric before checkpoint comparison.

    Without smoothing, a single noisy spike (e.g. 0.9 surrounded by 0.3-0.4 values) locks the best checkpoint to that
    spike epoch.  With ``smooth_alpha=0.5`` the EMA of the metric is what the parent ModelCheckpoint compares, so the
    later, steadily-improving epochs can overtake the early spike.
    """

    @pytest.mark.parametrize(
        ("invalid_value", "exc_type"),
        [
            pytest.param(True, TypeError, id="bool_true"),
            pytest.param("0.5", TypeError, id="string"),
            pytest.param(-0.1, ValueError, id="negative"),
            pytest.param(1.0, ValueError, id="exactly_one"),
            pytest.param(1.5, ValueError, id="greater_than_one"),
            pytest.param(float("nan"), ValueError, id="nan"),
            pytest.param(float("inf"), ValueError, id="inf"),
        ],
    )
    def test_invalid_smooth_alpha_raises(
        self, tmp_path: Path, invalid_value: object, exc_type: type[Exception]
    ) -> None:
        """BestModelCallback raises TypeError for non-numeric and ValueError for out-of-range smooth_alpha."""
        with pytest.raises(exc_type):
            BestModelCallback(output_dir=str(tmp_path), smooth_alpha=invalid_value)  # type: ignore[arg-type]

    def test_smoothing_disabled_when_alpha_zero(self, tmp_path: Path) -> None:
        """With smooth_alpha=0.0 (default), the raw metric drives checkpoint selection unchanged."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.0)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.42}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        # best_model_score reflects the raw value because no smoothing was applied.
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.42)
        # The smoothing accumulator must stay at its zero default when smoothing is disabled.
        assert cb._smoothed_regular == 0.0

    def test_smoothing_uses_ema_of_raw_metric(self, tmp_path: Path) -> None:
        """With smooth_alpha=0.5, best_model_score reflects the smoothed EMA, not the raw value."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        # EMA: 0.5 * 0.0 + 0.5 * 0.8 = 0.4 — the smoothed value, not the raw 0.8.
        assert cb._smoothed_regular == pytest.approx(0.4)
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.4)

    def test_smoothing_prefers_steady_improvement_over_early_spike(self, tmp_path: Path) -> None:
        """A late, smoothed run that beats the smoothed early spike wins the best checkpoint."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()

        # Epoch 0: raw=0.8 → smoothed=0.4
        # Epoch 1: raw=0.3 → smoothed=0.35
        # Epoch 2: raw=0.4 → smoothed=0.375
        # Epoch 3: raw=0.6 → smoothed=0.4875  (overtakes the early spike's smoothed 0.4)
        raw_per_epoch = [0.8, 0.3, 0.4, 0.6]
        for epoch, value in enumerate(raw_per_epoch):
            trainer = _make_trainer({"val/mAP_50_95": value}, current_epoch=epoch)
            # Distinct global_step per epoch so ModelCheckpoint does not skip saves on
            # the second-and-later calls due to its same-step guard.
            trainer.global_step = epoch + 1
            cb.on_validation_end(trainer, pl_module)

        # The best smoothed value should be epoch-3's 0.4875, not epoch-0's 0.4.
        assert cb._smoothed_regular == pytest.approx(0.4875)
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.4875)

    def test_callback_metrics_restored_after_super_call(self, tmp_path: Path) -> None:
        """trainer.callback_metrics[monitor] must hold the original raw tensor after on_validation_end returns."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        raw_tensor = trainer.callback_metrics["val/mAP_50_95"]

        cb.on_validation_end(trainer, pl_module)

        # The original tensor (raw value 0.8) must be back in callback_metrics so other
        # callbacks and metrics.csv see the unsmoothed value.
        assert trainer.callback_metrics["val/mAP_50_95"] is raw_tensor
        assert trainer.callback_metrics["val/mAP_50_95"].item() == pytest.approx(0.8)

    def test_state_dict_includes_smoothed_regular(self, tmp_path: Path) -> None:
        """state_dict() must include _smoothed_regular so resumed training continues from the correct EMA value."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        state = cb.state_dict()

        assert "_smoothed_regular" in state
        assert state["_smoothed_regular"] == pytest.approx(0.4)

    def test_load_state_dict_restores_smoothed_regular(self, tmp_path: Path) -> None:
        """load_state_dict() must restore _smoothed_regular so the EMA continues from the persisted value."""
        cb_first = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb_first.on_validation_end(trainer, pl_module)
        saved_state = cb_first.state_dict()

        cb_resumed = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        cb_resumed.load_state_dict(saved_state)

        assert cb_resumed._smoothed_regular == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="neg_inf"),
        ],
    )
    def test_load_state_dict_non_finite_smoothed_regular_resets_to_zero(self, tmp_path: Path, bad_value: float) -> None:
        """load_state_dict() resets non-finite persisted _smoothed_regular values to 0.0."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        state = cb.state_dict()
        state["_smoothed_regular"] = bad_value

        cb.load_state_dict(state)

        assert cb._smoothed_regular == 0.0

    def test_load_state_dict_smoothed_regular_missing_key_defaults_to_zero(self, tmp_path: Path) -> None:
        """load_state_dict() defaults _smoothed_regular to 0.0 when key is absent (backward compat)."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        cb._smoothed_regular = 0.42
        state = cb.state_dict()
        state.pop("_smoothed_regular")

        cb.load_state_dict(state)

        assert cb._smoothed_regular == 0.0

    @pytest.mark.parametrize(
        "skip_epochs",
        [
            pytest.param(2, id="skip_two"),
        ],
    )
    def test_ema_accumulator_warms_up_during_skip_window(self, tmp_path: Path, skip_epochs: int) -> None:
        """_smoothed_regular is pre-warmed by skip-window epochs so epoch 2 starts from a non-zero EMA.

        Scenario: skip_best_epochs=2, smooth_alpha=0.5, metric=0.5 each epoch.
        The EMA accumulator must update on epochs 0 and 1 (skip window) so that by epoch 2 it
        is already non-zero.  No checkpoint file should be written during the skip window.
        """
        cb = BestModelCallback(
            output_dir=str(tmp_path), monitor_regular="val/mAP_50_95", smooth_alpha=0.5, skip_best_epochs=skip_epochs
        )
        pl_module = _make_pl_module()

        # Epochs 0 and 1 are inside the skip window.
        for epoch in range(skip_epochs):
            trainer_skip = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=epoch)
            trainer_skip.global_step = epoch + 1
            cb.on_validation_end(trainer_skip, pl_module)

        # No checkpoint must have been written during the skip window.
        assert not (tmp_path / "checkpoint_best_regular.pth").exists(), (
            "checkpoint must not be written during the skip window (epochs 0 and 1)"
        )

        # Epoch 2 is the first eligible epoch — accumulator must be pre-warmed.
        trainer_eligible = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=skip_epochs)
        trainer_eligible.global_step = skip_epochs + 1
        cb.on_validation_end(trainer_eligible, pl_module)

        assert cb._smoothed_regular > 0.0, (
            "_smoothed_regular must be > 0.0 at epoch 2 because the skip-window epochs warmed the EMA"
        )

    def test_callback_metrics_restored_when_super_raises(self, tmp_path: Path) -> None:
        """trainer.callback_metrics[monitor] is restored in the finally block even when super() raises."""
        from unittest.mock import patch

        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        raw_tensor = trainer.callback_metrics["val/mAP_50_95"]

        with patch(
            "rfdetr.training.callbacks.best_model.ModelCheckpoint.on_validation_end",
            side_effect=RuntimeError("simulated failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                cb.on_validation_end(trainer, pl_module)

        assert trainer.callback_metrics["val/mAP_50_95"] is raw_tensor


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

    def test_syncs_keypoint_schema_from_kp_active_mask(self) -> None:
        """Stale num_keypoints_per_class in model_config is overridden by _kp_active_mask from weights."""
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_dump.return_value = {
            "num_keypoints_per_class": [0, 17],
            "num_classes": 2,
        }
        pl_module.model.state_dict.return_value = {
            "_kp_active_mask": _make_kp_active_mask([0, 33]),
        }

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is not None
        assert result["num_keypoints_per_class"] == [0, 33]

    def test_syncs_num_classes_from_class_embed_weight(self) -> None:
        """Stale num_classes in model_config is overridden by class_embed.weight.shape[0] from weights."""
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_dump.return_value = {
            "num_keypoints_per_class": [0, 33],
            "num_classes": 90,
        }
        pl_module.model.state_dict.return_value = {
            "class_embed.weight": torch.zeros(3, 256),
        }

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is not None
        assert result["num_classes"] == 2

    def test_no_sync_when_kp_mask_absent(self) -> None:
        """num_keypoints_per_class is left unchanged when model has no _kp_active_mask."""
        pl_module = _make_pl_module()
        pl_module.model_config = MagicMock()
        pl_module.model_config.model_dump.return_value = {
            "num_keypoints_per_class": [0, 17],
            "num_classes": 2,
        }
        pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}

        result = BestModelCallback._serialize_model_config(pl_module)

        assert result is not None
        assert result["num_keypoints_per_class"] == [0, 17]

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


# ---------------------------------------------------------------------------
# TestOnFitEndEMASwapSuppression
# ---------------------------------------------------------------------------


class _TestStepLinearModule(LightningModule):
    """Real module with ``test_step`` and a tiny linear model for weight-identity assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.train_config = {"lr": 0.001}

    def test_step(self, batch: object, batch_idx: int) -> None:
        """No-op test step so BestModelCallback.on_fit_end runs trainer.test()."""


def _fill_module_weight(pl_module: LightningModule, value: float) -> None:
    """Set the single linear weight of *pl_module* to *value* in-place."""
    with torch.no_grad():
        pl_module.model.weight.fill_(value)


class TestOnFitEndEMASwapSuppression:
    """The fit-end test run must evaluate the best checkpoint, not the final EMA weights.

    Regression tests: ``BestModelCallback.on_fit_end`` loads ``checkpoint_best_total.pth`` into the module and then
    calls ``trainer.test()`` — but ``RFDETREMACallback.on_test_epoch_start`` used to swap in the final EMA weights,
    silently overwriting the just-loaded best weights for the whole test run.
    """

    @staticmethod
    def _build_fit_end_scenario(
        tmp_path: Path,
    ) -> tuple[BestModelCallback, RFDETREMACallback, LightningModule, MagicMock, list[torch.Tensor]]:
        """Arrange best=3.0 checkpoint, EMA=5.0 average model, live=7.0 weights, and a hook-simulating trainer."""
        from torch.optim.swa_utils import AveragedModel

        pl_module = _TestStepLinearModule()
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        ema_cb = RFDETREMACallback()

        # Best checkpoint saved at weight 3.0.
        _fill_module_weight(pl_module, 3.0)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        # EMA average model captured at weight 5.0.
        _fill_module_weight(pl_module, 5.0)
        ema_cb._average_model = AveragedModel(
            model=pl_module,
            use_buffers=True,
            avg_fn=ema_cb._avg_fn,
        )
        ema_cb._average_model.eval()

        # Final live weights end training at 7.0.
        _fill_module_weight(pl_module, 7.0)

        trainer.callbacks = [ema_cb, cb]
        observed_weights: list[torch.Tensor] = []

        def _fake_test(module: object, datamodule: object = None, verbose: bool = False) -> None:
            # Simulate the PTL test loop invoking the EMA callback's test hooks.
            ema_cb.on_test_epoch_start(trainer, pl_module)
            observed_weights.append(pl_module.model.weight.detach().clone())
            ema_cb.on_test_epoch_end(trainer, pl_module)

        trainer.test = MagicMock(side_effect=_fake_test)
        return cb, ema_cb, pl_module, trainer, observed_weights

    def test_fit_end_test_runs_on_best_weights_not_ema(self, tmp_path: Path) -> None:
        """During the fit-end test run the module must hold the best checkpoint weights (3.0), not EMA (5.0)."""
        cb, _ema_cb, pl_module, trainer, observed_weights = self._build_fit_end_scenario(tmp_path)

        cb.on_fit_end(trainer, pl_module)

        assert observed_weights, "trainer.test() must have been invoked"
        expected = torch.full_like(observed_weights[0], 3.0)
        assert torch.allclose(observed_weights[0], expected), (
            f"test ran with weight {observed_weights[0].item():.1f}; expected best-checkpoint weight 3.0"
        )

    def test_ema_swap_suppression_cleared_after_fit_end(self, tmp_path: Path) -> None:
        """After on_fit_end the EMA callback must swap again for standalone trainer.test() runs."""
        cb, ema_cb, pl_module, trainer, _observed = self._build_fit_end_scenario(tmp_path)

        cb.on_fit_end(trainer, pl_module)

        assert ema_cb.suppress_test_swap is False

    def test_ema_swap_suppression_cleared_when_test_raises(self, tmp_path: Path) -> None:
        """Suppression must be lifted even when trainer.test() raises."""
        cb, ema_cb, pl_module, trainer, _observed = self._build_fit_end_scenario(tmp_path)
        trainer.test = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            cb.on_fit_end(trainer, pl_module)

        assert ema_cb.suppress_test_swap is False
