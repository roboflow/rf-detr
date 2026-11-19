# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

import logging
from pathlib import Path

import pytest
import torch

from rfdetr import RFDETRNano


@pytest.fixture
def mock_build_trainer(mocker):
    """Patch the PTL training stack and return the ``build_trainer`` mock, so ``RFDETR.train()`` stops at ``fit()``."""
    mocker.patch("rfdetr.training.RFDETRModelModule")
    mocker.patch("rfdetr.training.RFDETRDataModule")
    return mocker.patch("rfdetr.training.build_trainer")


@pytest.fixture
def rf_detr_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture "rf-detr" logger warnings in ``caplog``.

    ``get_logger()`` (``rfdetr.utilities.logger``) sets ``propagate = False`` on the "rf-detr" logger, while
    ``caplog``'s handler lives on the root logger, so propagation is re-enabled for the test (same pattern as
    ``tests/utilities/test_state_dict.py``).
    """
    rf_detr_logger = logging.getLogger("rf-detr")
    prev_propagate = rf_detr_logger.propagate
    rf_detr_logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="rf-detr"):
        yield caplog
    rf_detr_logger.propagate = prev_propagate


def _train(tmp_path: Path, output_dir: Path, resume: Path | None = None) -> RFDETRNano:
    """Run ``RFDETRNano.train()`` for one epoch on CPU (PTL stack mocked by ``mock_build_trainer``) and return the
    model.

    Args:
        tmp_path: Pytest temporary directory, used as ``dataset_dir``.
        output_dir: Path passed as ``output_dir=``.
        resume: Optional checkpoint path passed as ``resume=``.

    Examples:
        >>> _train  # doctest: +SKIP
        Needs patched training dependencies and a pytest tmp_path fixture.
    """
    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.train(
        dataset_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        resume=None if resume is None else str(resume),
        device="cpu",
    )
    return model


def test_resume_with_completed_epochs_returns_early(tmp_path: Path, mock_build_trainer) -> None:
    """Training reaches trainer.fit() when resuming a run that already completed its epochs.

    In the legacy engine.py path, ``start_epoch=epochs`` caused the training loop to be skipped (``range(start_epoch,
    epochs)`` was empty), which triggered an ``UnboundLocalError`` when accessing ``test_stats``. In the PTL path,
    resuming happens automatically via ``ckpt_path`` and does not require a ``start_epoch`` kwarg.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    _train(tmp_path, output_dir)

    mock_build_trainer.return_value.fit.assert_called_once()


def test_resume_with_completed_epochs_skips_on_train_end_callback(tmp_path: Path, mock_build_trainer) -> None:
    """Old-style ``model.callbacks["on_train_end"]`` callbacks are not forwarded to PTL.

    In the legacy engine.py path they ran at the end of training (even when the loop was skipped). In the PTL path the
    callback dict on the model instance is not consulted; use PTL ``Callback`` objects via ``build_trainer()``.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.callbacks["on_train_end"].append(_callback)
    model.train(
        dataset_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    assert callback_calls == 0


def test_resume_with_callback_state_warns_it_resumes_correctly(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with a non-empty "callbacks" key gets the "will resume correctly" warning.

    Regression test for the RFDETR.train() resume warning: without peeking at the checkpoint's own "callbacks" key
    first, this branch and ``test_resume_without_callback_state_warns_it_restarts_cold`` would be indistinguishable and
    the warning would always claim full restoration.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)

    _train(tmp_path, output_dir, ckpt_path)

    assert any("matching configured callbacks" in record.message for record in rf_detr_caplog.records)
    assert not any("predates callback-state persistence" in record.message for record in rf_detr_caplog.records)


def test_resume_without_callback_state_warns_it_restarts_cold(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with no "callbacks" key (or an empty one) warns everything restarts cold.

    Simulates a checkpoint written before ``BestModelCallback._build_checkpoint_payload`` persisted per-callback state
    (or a run where every registered callback had empty state): PTL's ``_call_callbacks_load_state_dict`` no-ops on a
    missing/empty "callbacks" key, silently skipping best-score/EMA/early-stopping restoration.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}}, ckpt_path)

    _train(tmp_path, output_dir, ckpt_path)

    assert any("restart cold" in record.message for record in rf_detr_caplog.records)
    assert not any("matching configured callbacks" in record.message for record in rf_detr_caplog.records)


def test_resume_dirpath_mismatch_warns_best_score_restarts_fresh(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """Resuming with output_dir different from the checkpoint's own directory warns best-score tracking resets.

    Mirrors PTL's own ``ModelCheckpoint.load_state_dict()`` dirpath gate (callback-level regression test:
    ``tests/training/callbacks/test_best_model_callback.py::test_best_model_score_not_restored_when_output_dir_changes``);
    this is the config-level warning that surfaces the same gap to a user calling ``RFDETR.train(resume=...)``.
    """
    source_dir = tmp_path / "source_output"
    source_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = source_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)
    other_output_dir = tmp_path / "other_output_dir"
    other_output_dir.mkdir(parents=True, exist_ok=True)

    _train(tmp_path, other_output_dir, ckpt_path)

    assert any(
        "best-score tracking" in record.message and "restarts fresh" in record.message
        for record in rf_detr_caplog.records
    )
