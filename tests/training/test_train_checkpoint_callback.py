# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for :class:`rfdetr.training.callbacks.train_checkpoint.RFDETRTrainCheckpointCallback`."""

from pathlib import Path
from unittest.mock import MagicMock

from rfdetr.training.callbacks.train_checkpoint import RFDETRTrainCheckpointCallback


def _make_trainer(current_epoch: int, is_global_zero: bool = True) -> MagicMock:
    """Create a minimal trainer mock for checkpoint callback tests."""
    trainer = MagicMock()
    trainer.current_epoch = current_epoch
    trainer.is_global_zero = is_global_zero
    return trainer


class TestRFDETRTrainCheckpointCallback:
    """Verify latest and interval training checkpoint saving behavior."""

    def test_saves_latest_checkpoint_every_epoch(self, tmp_path: Path) -> None:
        """Always writes checkpoint.pth at train epoch end."""
        cb = RFDETRTrainCheckpointCallback(output_dir=str(tmp_path), checkpoint_interval=10)
        trainer = _make_trainer(current_epoch=0)

        cb.on_train_epoch_end(trainer, pl_module=MagicMock())

        trainer.save_checkpoint.assert_called_once_with(str(tmp_path / "checkpoint.pth"), weights_only=False)

    def test_saves_interval_checkpoint_when_epoch_matches_interval(self, tmp_path: Path) -> None:
        """Writes checkpoint_<N>.pth when (current_epoch+1) is divisible by interval."""
        cb = RFDETRTrainCheckpointCallback(output_dir=str(tmp_path), checkpoint_interval=2)
        trainer = _make_trainer(current_epoch=1)  # epoch number = 2

        cb.on_train_epoch_end(trainer, pl_module=MagicMock())

        assert trainer.save_checkpoint.call_count == 2
        trainer.save_checkpoint.assert_any_call(str(tmp_path / "checkpoint.pth"), weights_only=False)
        trainer.save_checkpoint.assert_any_call(str(tmp_path / "checkpoint_2.pth"), weights_only=False)

    def test_does_not_save_interval_checkpoint_when_not_due(self, tmp_path: Path) -> None:
        """Skips checkpoint_<N>.pth when epoch is not an interval boundary."""
        cb = RFDETRTrainCheckpointCallback(output_dir=str(tmp_path), checkpoint_interval=3)
        trainer = _make_trainer(current_epoch=1)  # epoch number = 2

        cb.on_train_epoch_end(trainer, pl_module=MagicMock())

        trainer.save_checkpoint.assert_called_once_with(str(tmp_path / "checkpoint.pth"), weights_only=False)

    def test_non_global_zero_does_not_save_anything(self, tmp_path: Path) -> None:
        """Non-main process does not write latest or interval checkpoints."""
        cb = RFDETRTrainCheckpointCallback(output_dir=str(tmp_path), checkpoint_interval=2)
        trainer = _make_trainer(current_epoch=1, is_global_zero=False)

        cb.on_train_epoch_end(trainer, pl_module=MagicMock())

        trainer.save_checkpoint.assert_not_called()
