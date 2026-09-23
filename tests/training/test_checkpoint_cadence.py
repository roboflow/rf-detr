# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Which epochs write ``last.ckpt`` and ``checkpoint_<epoch>.ckpt`` when ``eval_interval > 1``.

``build_trainer`` registers two plain ``ModelCheckpoint`` callbacks next to ``BestModelCallback``. With
``check_val_every_n_epoch != 1`` Lightning's ``ModelCheckpoint`` saves from ``on_validation_end`` unless told to save on
train epoch end, so without ``save_on_train_epoch_end=True`` both callbacks only fired on the epochs ``eval_interval``
validated: ``last.ckpt`` went stale between validations and interval archives skipped every epoch that was not also an
evaluation epoch. Runs a real ``Trainer.fit()`` with mocked model internals (no dataset, no GPU) and inspects the files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel


class _RecordLastCheckpointEpoch(Callback):
    """At every train epoch start, record the ``epoch`` stored in ``last.ckpt`` (``None`` if absent).

    Checkpoint callbacks run after every other callback's ``on_train_epoch_end``, so the previous epoch's write is only
    observable from the next epoch's start hook.
    """

    def __init__(self, output_dir: Path) -> None:
        self.last_path = output_dir / "last.ckpt"
        self.epochs: list[int | None] = []
        self.current_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Track the active epoch and append the prior checkpoint epoch."""
        self.current_epoch = trainer.current_epoch
        if trainer.current_epoch == 0:
            return
        if not self.last_path.exists():
            self.epochs.append(None)
            return
        self.epochs.append(int(torch.load(self.last_path, map_location="cpu", weights_only=False)["epoch"]))


def _fit(
    tmp_path: Path,
    epochs: int,
    eval_interval: int,
    checkpoint_interval: int,
    improve_validation: bool = False,
) -> tuple[Path, list[int | None]]:
    """Fit ``epochs`` epochs of two batches each and return the output dir plus the recorded ``last.ckpt`` epochs.

    When ``improve_validation`` is true, predictions miss before the final epoch and match exactly in its validation.

    Examples:
        >>> import contextlib, io
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as d:
        ...     with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ...         out, epochs = _fit(Path(d), epochs=2, eval_interval=1, checkpoint_interval=2)
        ...     sorted(p.name for p in out.glob("checkpoint_*.ckpt")), epochs
        (['checkpoint_1.ckpt'], [0])
    """
    mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=3)
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=epochs,
        batch_size=2,
        num_workers=0,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        use_ema=False,
        tensorboard=False,
        run_test=False,
    )
    recorder = _RecordLastCheckpointEpoch(Path(tc.output_dir))
    postprocess = MagicMock(side_effect=_fake_postprocess)
    if improve_validation:
        postprocess = MagicMock(
            side_effect=lambda outputs, orig_sizes: [
                {
                    "boxes": torch.tensor(
                        [[14.4, 14.4, 17.6, 17.6]] if recorder.current_epoch == epochs - 1 else [[5.0, 5.0, 20.0, 20.0]]
                    ),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
                for _ in range(orig_sizes.shape[0])
            ]
        )
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), postprocess),
        ),
        patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            limit_train_batches=2,
            limit_val_batches=2,
            log_every_n_steps=1,
        )
        trainer.callbacks.append(recorder)
        trainer.fit(module, datamodule=datamodule)
    return Path(tc.output_dir), recorder.epochs


class TestCheckpointCadenceWithEvalInterval:
    """``eval_interval`` must not change which epochs write the resume and archive checkpoints."""

    def test_last_ckpt_is_rewritten_every_epoch(self, tmp_path):
        """With eval_interval=2, last.ckpt at the start of epoch N must hold epoch N-1 for every N."""
        _, epochs = _fit(tmp_path, epochs=6, eval_interval=2, checkpoint_interval=3)

        assert epochs == [0, 1, 2, 3, 4]

    def test_interval_archives_land_on_every_interval_epoch(self, tmp_path):
        """With checkpoint_interval=3 and eval_interval=2, epoch 2 (not an eval epoch) must still be archived."""
        out, _ = _fit(tmp_path, epochs=6, eval_interval=2, checkpoint_interval=3)

        assert sorted(p.name for p in out.glob("checkpoint_*.ckpt")) == ["checkpoint_2.ckpt", "checkpoint_5.ckpt"]

    def test_last_ckpt_carries_that_epochs_validation_state(self, tmp_path):
        """Saving on train epoch end must still capture the validation that ran inside the same epoch."""
        out, _ = _fit(tmp_path, epochs=2, eval_interval=1, checkpoint_interval=5, improve_validation=True)

        state = torch.load(out / "last.ckpt", map_location="cpu", weights_only=False)
        best_states = [value for key, value in state["callbacks"].items() if key.startswith("BestModelCallback")]
        assert best_states
        assert state["epoch"] == 1
        torch.testing.assert_close(best_states[0]["best_model_score"], torch.tensor(1.0), rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("eval_interval", [1, 2])
    def test_final_epoch_is_saved(self, tmp_path, eval_interval):
        """The last epoch is always archived, whether or not eval_interval divides the epoch count."""
        out, _ = _fit(tmp_path, epochs=3, eval_interval=eval_interval, checkpoint_interval=3)

        assert (out / "checkpoint_2.ckpt").exists()
        assert int(torch.load(out / "last.ckpt", map_location="cpu", weights_only=False)["epoch"]) == 2
