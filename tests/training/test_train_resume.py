# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rfdetr import RFDETRNano


def test_resume_with_completed_epochs_returns_early(tmp_path: Path) -> None:
    """Training reaches trainer.fit() when resuming a run that already completed its epochs.

    In the legacy engine.py path, ``start_epoch=epochs`` caused the training loop to be skipped (``range(start_epoch,
    epochs)`` was empty), which triggered an ``UnboundLocalError`` when accessing ``test_stats``. In the PTL path,
    resuming happens automatically via ``ckpt_path`` and does not require a ``start_epoch`` kwarg.

    Args:
        tmp_path: Pytest temporary directory.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")

    with (
        patch("rfdetr.training.RFDETRModelModule"),
        patch("rfdetr.training.RFDETRDataModule"),
        patch("rfdetr.training.build_trainer") as mock_build_trainer,
    ):
        model.train(
            dataset_dir=str(tmp_path),
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            output_dir=str(output_dir),
            device="cpu",
        )

    mock_build_trainer.return_value.fit.assert_called_once()


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


def _train_with_resume(tmp_path: Path, ckpt_path: Path, output_dir: Path) -> None:
    """Run ``model.train(resume=...)`` with the PTL stack mocked out, exercising only config/warning logic.

    Shared by the resume-warning tests below: none of them care about an actual training run, only about
    what ``RFDETR.train()`` logs before ``trainer.fit()`` is reached (which is itself mocked away).

    Args:
        tmp_path: Pytest temporary directory, used as ``dataset_dir``.
        ckpt_path: Path passed as ``resume=``.
        output_dir: Path passed as ``output_dir=``.
    """
    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
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
            resume=str(ckpt_path),
            device="cpu",
        )


def _train_with_resume_capturing_warnings(
    tmp_path: Path, ckpt_path: Path, output_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Run :func:`_train_with_resume` with the "rf-detr" logger's warnings captured by ``caplog``.

    ``get_logger()`` (``rfdetr.utilities.logger``) sets ``propagate = False`` on the "rf-detr" logger
    so it does not double-log through the root logger's own handlers. ``caplog``'s capturing handler
    lives on the root logger, so records never reach it unless propagation is temporarily re-enabled —
    same pattern already used in ``tests/utilities/test_state_dict.py``.

    Args:
        tmp_path: Pytest temporary directory, used as ``dataset_dir``.
        ckpt_path: Path passed as ``resume=``.
        output_dir: Path passed as ``output_dir=``.
        caplog: Pytest's log-capturing fixture.

    Examples:
        >>> _train_with_resume_capturing_warnings  # doctest: +SKIP
        Needs pytest's tmp_path/caplog fixtures, can't run standalone outside a test.
    """
    rf_detr_logger = logging.getLogger("rf-detr")
    prev_propagate = rf_detr_logger.propagate
    rf_detr_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _train_with_resume(tmp_path, ckpt_path, output_dir)
    finally:
        rf_detr_logger.propagate = prev_propagate


def test_resume_from_checkpoint_with_callback_state_warns_it_resumes_correctly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with a non-empty "callbacks" key gets the "will resume correctly" warning.

    Regression test for the RFDETR.train() resume warning: without peeking at the checkpoint's own
    "callbacks" key first, this branch and the one in
    ``test_resume_from_checkpoint_without_callback_state_warns_it_restarts_cold`` below would be
    indistinguishable and the warning would always claim full restoration.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)

    _train_with_resume_capturing_warnings(tmp_path, ckpt_path, output_dir, caplog)

    assert any("will resume correctly" in record.message for record in caplog.records)
    assert not any("predates callback-state persistence" in record.message for record in caplog.records)


def test_resume_from_checkpoint_without_callback_state_warns_it_restarts_cold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with no "callbacks" key (or an empty one) warns everything restarts cold.

    Simulates a checkpoint written before ``BestModelCallback._build_checkpoint_payload`` started persisting per-
    callback state (or any run where every registered callback had empty state): PTL's
    ``_call_callbacks_load_state_dict`` no-ops on a missing/empty "callbacks" key, silently skipping best-
    score/EMA/early-stopping restoration — the warning must not claim otherwise.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}}, ckpt_path)

    _train_with_resume_capturing_warnings(tmp_path, ckpt_path, output_dir, caplog)

    assert any("restart cold" in record.message for record in caplog.records)
    assert not any("will resume correctly" in record.message for record in caplog.records)


def test_resume_dirpath_mismatch_warns_best_score_restarts_fresh(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Resuming with output_dir different from the checkpoint's own directory warns best-score tracking resets.

    Mirrors PTL's own ``ModelCheckpoint.load_state_dict()`` dirpath gate (see
    ``tests/training/callbacks/test_best_model_callback.py::test_best_model_score_not_restored_when_output_dir_changes``
    for the callback-level regression test); this is the config-level warning that surfaces the same
    gap to a user calling ``RFDETR.train(resume=..., output_dir=...)`` directly.
    """
    source_dir = tmp_path / "source_output"
    source_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = source_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)

    other_output_dir = tmp_path / "other_output_dir"
    other_output_dir.mkdir(parents=True, exist_ok=True)

    _train_with_resume_capturing_warnings(tmp_path, ckpt_path, other_output_dir, caplog)

    assert any(
        "best-score tracking" in record.message and "restarts fresh" in record.message for record in caplog.records
    )
