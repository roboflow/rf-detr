# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

import shutil
from pathlib import Path

import pytest

from rfdetr import RFDETRNano
from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset


@pytest.fixture(scope="module")
def tiny_coco_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny COCO-format dataset for testing.

    Args:
        tmp_path_factory: Pytest factory for temporary directories.

    Returns:
        Path to the generated dataset directory.
    """
    dataset_dir = tmp_path_factory.mktemp("tiny_coco")
    generate_coco_dataset(
        output_dir=str(dataset_dir),
        num_images=4,
        img_size=64,
        class_mode="shape",
        min_objects=1,
        max_objects=2,
        split_ratios=DatasetSplitRatios(train=0.5, val=0.5, test=0.0),
    )
    val_dir = dataset_dir / "val"
    valid_dir = dataset_dir / "valid"
    if val_dir.exists() and not valid_dir.exists():
        val_dir.rename(valid_dir)
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "_annotations.coco.json").write_text((valid_dir / "_annotations.coco.json").read_text())
        for item in valid_dir.iterdir():
            if item.is_file() and item.name != "_annotations.coco.json":
                shutil.copy2(item, test_dir / item.name)
    yield dataset_dir
    shutil.rmtree(dataset_dir)


class TestResumeTrainingFromCompletedCheckpoint:
    """Tests for correct behavior when resuming from an already-completed checkpoint."""

    def test_resume_with_completed_epochs_returns_early(self, tiny_coco_dataset: Path, tmp_path: Path) -> None:
        """Resuming training when start_epoch >= epochs must not raise UnboundLocalError.

        This is a regression test for a bug where resuming from a checkpoint whose
        epoch equals or exceeds the target number of epochs caused an UnboundLocalError
        because the training loop never executed, leaving ``test_stats`` undefined.

        The test simulates the end-state of checkpoint loading by passing
        ``start_epoch=epochs`` directly, which is equivalent to loading a checkpoint
        with ``checkpoint['epoch'] = epochs - 1`` and ``resume`` set.

        Args:
            tiny_coco_dataset: Path to a minimal COCO-style dataset.
            tmp_path: Pytest temporary directory.
        """
        output_dir = tmp_path / "train_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")

        # start_epoch=2 with epochs=2 simulates having loaded a checkpoint for epoch 1
        # (checkpoint['epoch'] + 1 == epochs), so the training loop range(2, 2) is empty.
        # In the old code this raised UnboundLocalError on test_stats["results_json"].
        model.train(
            dataset_dir=str(tiny_coco_dataset),
            epochs=2,
            start_epoch=2,
            batch_size=1,
            grad_accum_steps=1,
            output_dir=str(output_dir),
            device="cpu",
        )
