# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

from pathlib import Path
from unittest.mock import patch

from rfdetr import RFDETRNano


def test_resume_with_completed_epochs_calls_on_train_end_callback(tmp_path: Path) -> None:
    """Old-style on_train_end callbacks are not forwarded to PTL.

    In the legacy engine.py path, callbacks added to ``model.callbacks["on_train_end"]`` were invoked at the end of
    training (including when the loop was skipped). In the PTL path the old-style callback dict on the model instance is
    not consulted; use PTL ``Callback`` objects via ``build_trainer()`` instead.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.callbacks["on_train_end"].append(_callback)

    with (
        patch("rfdetr.training.RFDETRModelModule"),
        patch("rfdetr.training.RFDETRDataModule"),
        patch("rfdetr.training.build_trainer"),
    ):
        model.train(
            dataset_dir=str(tmp_path),
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            output_dir=str(output_dir),
            device="cpu",
        )

    # Old-style callbacks on model.callbacks are no longer invoked in the PTL path.
    assert callback_calls == 0
