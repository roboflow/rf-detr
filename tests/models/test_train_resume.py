# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

from pathlib import Path

from rfdetr import RFDETRNano


def test_resume_with_completed_epochs_returns_early(synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
    """Resuming training when start_epoch >= epochs must not raise UnboundLocalError.

    This is a regression test for a bug where resuming from a checkpoint whose
    epoch equals or exceeds the target number of epochs caused an UnboundLocalError
    because the training loop never executed, leaving ``test_stats`` undefined.

    The test simulates the end-state of checkpoint loading by passing
    ``start_epoch=epochs`` directly, which is equivalent to loading a checkpoint
    with ``checkpoint['epoch'] = epochs - 1`` and ``resume`` set.

    Args:
        synthetic_shape_dataset_dir: Path to a synthetic COCO-style dataset.
        tmp_path: Pytest temporary directory.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")

    # start_epoch=2 with epochs=2 simulates having loaded a checkpoint for epoch 1
    # (checkpoint['epoch'] + 1 == epochs), so the training loop range(2, 2) is empty.
    # In the old code this raised UnboundLocalError on test_stats["results_json"].
    model.train(
        dataset_dir=str(synthetic_shape_dataset_dir),
        epochs=2,
        start_epoch=2,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        device="cpu",
    )


def test_resume_with_completed_epochs_calls_on_train_end_callback(
    synthetic_shape_dataset_dir: Path, tmp_path: Path
) -> None:
    """Resuming with completed epochs should still execute on_train_end callbacks."""
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.callbacks["on_train_end"].append(_callback)

    model.train(
        dataset_dir=str(synthetic_shape_dataset_dir),
        epochs=2,
        start_epoch=2,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    assert callback_calls == 1
