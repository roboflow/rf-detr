# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from rfdetr.datasets.dota_detection import (
    DOTA_V1_CLASSES,
    DotaDetection,
    DotaNormalize,
    corners_list_to_tensor,
    parse_dota_annotation,
)


@pytest.fixture()
def dota_root(tmp_path: Path) -> Path:
    """Create a minimal DOTA directory with one image and annotation."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labelTxt"
    images_dir.mkdir()
    labels_dir.mkdir()

    img = Image.new("RGB", (100, 100), color="red")
    img.save(images_dir / "P0001.png")

    ann_text = "10 10 50 10 50 40 10 40 plane 0\n60 60 90 60 90 90 60 90 ship 0\n20 20 30 20 30 30 20 30 plane 1\n"
    (labels_dir / "P0001.txt").write_text(ann_text)

    return tmp_path


class TestParseDotaAnnotation:
    def test_parses_valid_lines(self, dota_root: Path) -> None:
        ann_path = dota_root / "labelTxt" / "P0001.txt"
        annotations = parse_dota_annotation(ann_path)
        assert len(annotations) == 3

    def test_annotation_fields(self, dota_root: Path) -> None:
        ann_path = dota_root / "labelTxt" / "P0001.txt"
        ann = parse_dota_annotation(ann_path)[0]
        assert ann["category"] == "plane"
        assert ann["difficulty"] == 0
        assert len(ann["corners"]) == 8

    def test_skips_short_lines(self, tmp_path: Path) -> None:
        ann_path = tmp_path / "test.txt"
        ann_path.write_text("10 20 30\n10 10 50 10 50 40 10 40 plane 0\n")
        annotations = parse_dota_annotation(ann_path)
        assert len(annotations) == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        ann_path = tmp_path / "empty.txt"
        ann_path.write_text("")
        assert parse_dota_annotation(ann_path) == []

    def test_difficulty_defaults_to_zero(self, tmp_path: Path) -> None:
        ann_path = tmp_path / "no_diff.txt"
        ann_path.write_text("10 10 50 10 50 40 10 40 plane\n")
        ann = parse_dota_annotation(ann_path)[0]
        assert ann["difficulty"] == 0


class TestCornersListToTensor:
    def test_shape(self) -> None:
        result = corners_list_to_tensor([0, 0, 10, 0, 10, 5, 0, 5])
        assert result.shape == (4, 2)

    def test_values(self) -> None:
        result = corners_list_to_tensor([1, 2, 3, 4, 5, 6, 7, 8])
        expected = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32)
        assert torch.equal(result, expected)


class TestDotaDetection:
    def test_len(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        assert len(dataset) == 1

    def test_getitem_returns_image_and_target(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        image, target = dataset[0]
        assert isinstance(image, Image.Image)
        assert "boxes_obb" in target
        assert "labels" in target
        assert "corners" in target

    def test_filters_difficult_by_default(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert target["labels"].shape[0] == 2

    def test_includes_difficult_when_flag_set(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root, include_difficult=True)
        _, target = dataset[0]
        assert target["labels"].shape[0] == 3

    def test_boxes_obb_shape(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert target["boxes_obb"].shape == (2, 5)

    def test_corners_shape(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert target["corners"].shape == (2, 4, 2)

    def test_labels_are_valid_indices(self, dota_root: Path) -> None:
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert (target["labels"] >= 0).all()
        assert (target["labels"] < len(DOTA_V1_CLASSES)).all()

    def test_skips_unknown_categories(self, dota_root: Path) -> None:
        (dota_root / "labelTxt" / "P0001.txt").write_text("10 10 50 10 50 40 10 40 unknown_category 0\n")
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert target["labels"].shape[0] == 0

    def test_missing_images_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "labelTxt").mkdir()
        with pytest.raises(FileNotFoundError):
            DotaDetection(root=tmp_path)

    def test_missing_annotation_file_returns_empty(self, dota_root: Path) -> None:
        (dota_root / "labelTxt" / "P0001.txt").unlink()
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        assert target["labels"].shape[0] == 0

    def test_axis_aligned_box_angle_near_zero(self, dota_root: Path) -> None:
        (dota_root / "labelTxt" / "P0001.txt").write_text("0 0 10 0 10 5 0 5 plane 0\n")
        dataset = DotaDetection(root=dota_root)
        _, target = dataset[0]
        angle = target["boxes_obb"][0, 4].item()
        assert abs(angle) < 0.01 or abs(angle - math.pi) < 0.01


class TestDotaNormalize:
    def test_normalizes_boxes(self) -> None:
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 200)
        corners = torch.tensor([[[10, 10], [50, 10], [50, 40], [10, 40]]], dtype=torch.float32)
        target = {"corners": corners, "boxes_obb": torch.zeros(1, 5)}

        image_out, target_out = normalize(image, target)
        obb = target_out["boxes_obb"]
        assert obb[0, 0].item() < 1.0
        assert obb[0, 1].item() < 1.0

    def test_none_target_passthrough(self) -> None:
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 100)
        image_out, target_out = normalize(image, None)
        assert target_out is None

    def test_empty_corners(self) -> None:
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 100)
        target = {"corners": torch.zeros(0, 4, 2), "boxes_obb": torch.zeros(0, 5)}
        _, target_out = normalize(image, target)
        assert target_out["boxes_obb"].shape == (0, 5)
