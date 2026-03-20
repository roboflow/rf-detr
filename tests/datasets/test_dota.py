# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for DOTAv1 dataset loader."""

from pathlib import Path

import pytest
from PIL import Image

from rfdetr.datasets.dota import DotaDetection


@pytest.fixture
def dota_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal DOTAv1 dataset directory structure."""
    split = "train"
    images_dir = tmp_path / split / "images"
    labels_dir = tmp_path / split / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create two test images
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(images_dir / f"img_{i:04d}.png")

    # Create label files
    # Image 0: two objects
    (labels_dir / "img_0000.txt").write_text(
        "10 10 50 10 50 40 10 40 plane 0\n"
        "60 60 90 60 90 90 60 90 ship 0\n"
    )

    # Image 1: one object
    (labels_dir / "img_0001.txt").write_text(
        "20 20 80 20 80 80 20 80 plane 0\n"
    )

    # Image 2: no objects (empty label file)
    (labels_dir / "img_0002.txt").write_text("")

    return tmp_path


class TestDotaDetection:
    """Tests for DotaDetection dataset."""

    def test_load_dataset(self, dota_dataset_dir: Path) -> None:
        """Dataset should load with correct number of images."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["plane", "ship"],
            oriented=True,
        )
        assert len(ds) == 3

    def test_oriented_item(self, dota_dataset_dir: Path) -> None:
        """First item should have 2 boxes with corners."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["plane", "ship"],
            oriented=True,
        )
        img, target = ds[0]
        assert target["labels"].shape == (2,)
        assert target["boxes"].shape == (2, 4)
        assert "obb_corners" in target
        assert target["obb_corners"].shape == (2, 8)

    def test_non_oriented_item(self, dota_dataset_dir: Path) -> None:
        """Non-oriented mode should produce 4-dim boxes without corners."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["plane", "ship"],
            oriented=False,
        )
        img, target = ds[0]
        assert target["boxes"].shape == (2, 4)
        assert "obb_corners" not in target

    def test_empty_annotations(self, dota_dataset_dir: Path) -> None:
        """Image with no annotations should produce empty tensors."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["plane", "ship"],
            oriented=True,
        )
        img, target = ds[2]
        assert target["boxes"].shape[0] == 0
        assert target["labels"].shape[0] == 0
        assert target["obb_corners"].shape == (0, 8)

    def test_unknown_class_skipped(self, dota_dataset_dir: Path) -> None:
        """Unknown class names should be skipped."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["plane"],  # "ship" not included
            oriented=False,
        )
        img, target = ds[0]
        # Only "plane" should be included, "ship" skipped
        assert target["labels"].shape == (1,)

    def test_class_indices(self, dota_dataset_dir: Path) -> None:
        """Class indices should match the order in class_names."""
        ds = DotaDetection(
            root=str(dota_dataset_dir),
            split="train",
            class_names=["ship", "plane"],  # Reversed order
            oriented=False,
        )
        img, target = ds[0]
        # "plane" should be index 1, "ship" should be index 0
        assert 1 in target["labels"].tolist()
        assert 0 in target["labels"].tolist()

    def test_missing_images_dir_raises(self, tmp_path: Path) -> None:
        """Missing images directory should raise FileNotFoundError."""
        (tmp_path / "train" / "labels").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Images directory"):
            DotaDetection(
                root=str(tmp_path),
                split="train",
                class_names=["plane"],
            )
