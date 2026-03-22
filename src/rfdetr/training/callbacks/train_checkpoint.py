# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Training checkpoint callback for latest and interval `.pth` checkpoints."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class RFDETRTrainCheckpointCallback(Callback):
    """Save latest and periodic training checkpoints in legacy RF-DETR filenames.

    Writes:
    - ``checkpoint.pth`` at the end of every train epoch.
    - ``checkpoint_<N>.pth`` every ``checkpoint_interval`` epochs (1-indexed).

    These checkpoints are full Trainer checkpoints (include optimizer/scheduler
    state) and are intended for resuming training.

    Args:
        output_dir: Directory where checkpoints should be written.
        checkpoint_interval: Save a numbered interval checkpoint every N epochs.
    """

    def __init__(self, output_dir: str, checkpoint_interval: int = 10) -> None:
        super().__init__()
        self._output_dir = Path(output_dir)
        self._checkpoint_interval = max(1, int(checkpoint_interval))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Persist latest and interval checkpoints at train-epoch end.

        Args:
            trainer: Active Lightning trainer.
            pl_module: The Lightning module being trained.
        """
        del pl_module
        if not trainer.is_global_zero:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        latest_path = self._output_dir / "checkpoint.pth"
        trainer.save_checkpoint(str(latest_path), weights_only=False)

        epoch_num = trainer.current_epoch + 1
        if epoch_num % self._checkpoint_interval == 0:
            interval_path = self._output_dir / f"checkpoint_{epoch_num}.pth"
            trainer.save_checkpoint(str(interval_path), weights_only=False)
