# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

import warnings
from pathlib import Path
from unittest.mock import patch

from rfdetr import RFDETRNano


def test_resume_with_completed_epochs_returns_early(tmp_path: Path) -> None:
    """Passing start_epoch emits DeprecationWarning and still reaches trainer.fit().

    In the legacy engine.py path, ``start_epoch=epochs`` caused the training loop
    to be skipped (``range(start_epoch, epochs)`` was empty), which triggered an
    ``UnboundLocalError`` when accessing ``test_stats``.

    In the PTL path, ``start_epoch`` is a deprecated kwarg that is absorbed and
    ignored (PTL resumes automatically via ``ckpt_path``). The shim should emit
    the warning and still call ``trainer.fit(...)`` without raising.

    Args:
        tmp_path: Pytest temporary directory.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")

    with (
        patch("rfdetr.training.RFDETRModule"),
        patch("rfdetr.training.RFDETRDataModule"),
        patch("rfdetr.training.build_trainer") as mock_build_trainer,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        model.train(
            dataset_dir=str(tmp_path),
            epochs=1,
            start_epoch=1,
            batch_size=1,
            grad_accum_steps=1,
            output_dir=str(output_dir),
            device="cpu",
        )

    depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("start_epoch" in str(w.message) for w in depr), "Expected a DeprecationWarning mentioning start_epoch"
    mock_build_trainer.return_value.fit.assert_called_once()


def test_resume_with_completed_epochs_calls_on_train_end_callback(tmp_path: Path) -> None:
    """Old-style on_train_end callbacks are not forwarded to PTL.

    In the legacy engine.py path, callbacks added to ``model.callbacks["on_train_end"]``
    were invoked at the end of training (including when the loop was skipped).
    In the PTL path the old-style callback dict on the model instance is not consulted;
    use PTL ``Callback`` objects via ``build_trainer()`` instead.
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
        patch("rfdetr.training.RFDETRModule"),
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
