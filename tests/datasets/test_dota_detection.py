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
    OBBGeometricTransform,
    build_dota,
    corners_list_to_tensor,
    make_dota_transforms,
    parse_dota_annotation,
)
from rfdetr.utilities.rotated_box_ops import corners_to_cxcywha


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
        """Centre and size scale by width and height respectively.

        Asserts exact values: a w/h scale swap would keep both components below 1.0,
        so a bounds-only check cannot detect it.
        """
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 200)  # H=100, W=200
        corners = torch.tensor([[[10, 10], [50, 10], [50, 40], [10, 40]]], dtype=torch.float32)
        target = {"corners": corners, "boxes_obb": torch.zeros(1, 5)}

        _, target_out = normalize(image, target)
        # cx=30/200, cy=25/100, w=40/200, h=30/100
        expected = torch.tensor([0.15, 0.25, 0.20, 0.30])
        assert torch.allclose(target_out["boxes_obb"][0, :4], expected, atol=1e-5)

    def test_boxes_alias_matches_boxes_obb(self) -> None:
        """The ``boxes`` alias consumed by the COCO eval callback stays in sync."""
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 200)
        corners = torch.tensor([[[10, 10], [50, 10], [50, 40], [10, 40]]], dtype=torch.float32)
        target = {"corners": corners, "boxes_obb": torch.zeros(1, 5)}

        _, target_out = normalize(image, target)
        assert torch.allclose(target_out["boxes"], target_out["boxes_obb"][..., :4])

    def test_zero_area_box_is_dropped(self) -> None:
        """A box collapsed to zero area by augmentation must not reach the loss.

        _obb_to_gaussian clamps degenerate sizes to a floor, so an unfiltered w=0 box trains silently against a
        meaningless target instead of failing loudly.
        """
        normalize = DotaNormalize()
        image = torch.rand(3, 100, 100)
        corners = torch.tensor(
            [
                [[10, 10], [50, 10], [50, 40], [10, 40]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            dtype=torch.float32,
        )
        target = {"corners": corners, "boxes_obb": torch.zeros(2, 5), "labels": torch.tensor([3, 7])}

        _, target_out = normalize(image, target)
        assert target_out["boxes_obb"].shape == (1, 5)
        assert target_out["labels"].tolist() == [3]
        assert target_out["corners"].shape == (1, 4, 2)

    def test_size_refreshed_to_post_transform_shape(self) -> None:
        """``size`` must track the transformed image, not the original."""
        normalize = DotaNormalize()
        image = torch.rand(3, 64, 128)
        _, target_out = normalize(image, {"size": torch.as_tensor([999, 999])})
        assert target_out["size"].tolist() == [64, 128]

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


class TestOBBGeometricTransform:
    """Corners must survive augmentation as a rigid quadrilateral."""

    def test_horizontal_flip_mirrors_x_only(self) -> None:
        alb = pytest.importorskip("albumentations")
        transform = OBBGeometricTransform(alb.HorizontalFlip(p=1.0))
        image = Image.new("RGB", (100, 50))
        corners = torch.tensor([[[10.0, 5.0], [40.0, 5.0], [40.0, 20.0], [10.0, 20.0]]])

        _, out = transform(image, {"corners": corners, "labels": torch.tensor([0])})

        assert torch.allclose(out["corners"][0, :, 0], torch.tensor([89.0, 59.0, 59.0, 89.0]))
        assert torch.allclose(out["corners"][0, :, 1], corners[0, :, 1])

    def test_out_of_bounds_corners_are_not_clamped(self) -> None:
        """Clipping corners individually would deform the box.

        The four corners of a rotated box are not independent, so clamping each one into the frame produces a different
        quadrilateral rather than a cropped box.
        """
        alb = pytest.importorskip("albumentations")
        transform = OBBGeometricTransform(alb.NoOp())
        image = Image.new("RGB", (100, 100))
        corners = torch.tensor([[[-20.0, 10.0], [60.0, -30.0], [100.0, 20.0], [20.0, 60.0]]])

        _, out = transform(image, {"corners": corners, "labels": torch.tensor([0])})

        assert torch.allclose(out["corners"], corners, atol=1e-4)

    def test_geometry_preserved_for_box_crossing_edge(self) -> None:
        """Width and angle of a partially out-of-frame box survive the transform."""
        alb = pytest.importorskip("albumentations")
        transform = OBBGeometricTransform(alb.NoOp())
        image = Image.new("RGB", (100, 100))
        corners = torch.tensor([[[-20.0, 10.0], [60.0, -30.0], [100.0, 20.0], [20.0, 60.0]]])

        _, out = transform(image, {"corners": corners, "labels": torch.tensor([0])})

        assert torch.allclose(corners_to_cxcywha(out["corners"]), corners_to_cxcywha(corners), atol=1e-4)

    def test_two_boxes_keep_their_own_corners(self) -> None:
        """Instance ids must route each corner back to the box it came from."""
        alb = pytest.importorskip("albumentations")
        transform = OBBGeometricTransform(alb.HorizontalFlip(p=1.0))
        image = Image.new("RGB", (100, 100))
        corners = torch.tensor(
            [
                [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
                [[50.0, 50.0], [80.0, 50.0], [80.0, 70.0], [50.0, 70.0]],
            ]
        )

        _, out = transform(image, {"corners": corners, "labels": torch.tensor([0, 1])})

        assert torch.allclose(out["corners"][0, :, 0], torch.tensor([99.0, 89.0, 89.0, 99.0]))
        assert torch.allclose(out["corners"][1, :, 0], torch.tensor([49.0, 19.0, 19.0, 49.0]))

    def test_empty_corners_passthrough(self) -> None:
        alb = pytest.importorskip("albumentations")
        transform = OBBGeometricTransform(alb.HorizontalFlip(p=1.0))
        image = Image.new("RGB", (32, 32))
        target = {"corners": torch.zeros(0, 4, 2), "labels": torch.zeros(0, dtype=torch.int64)}

        image_out, out = transform(image, target)

        assert image_out.size == (32, 32)
        assert out["corners"].shape == (0, 4, 2)


class TestMakeDotaTransforms:
    def test_train_returns_compose(self) -> None:
        transforms = make_dota_transforms("train", 512)
        assert transforms is not None

    def test_val_returns_compose(self) -> None:
        transforms = make_dota_transforms("val", 512)
        assert transforms is not None


class TestBuildDota:
    def test_builds_dataset(self, dota_root: Path) -> None:
        import types

        args = types.SimpleNamespace(dataset_dir=str(dota_root.parent))
        root_with_split = dota_root.parent / "train"
        root_with_split.mkdir(exist_ok=True)
        (root_with_split / "images").mkdir(exist_ok=True)
        (root_with_split / "labelTxt").mkdir(exist_ok=True)
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(root_with_split / "images" / "test.png")
        (root_with_split / "labelTxt" / "test.txt").write_text("5 5 20 5 20 20 5 20 plane 0\n")
        args.dataset_dir = str(dota_root.parent)
        dataset = build_dota("train", args, 256)
        assert isinstance(dataset, DotaDetection)

    def test_getitem_after_transforms_normalizes_coords(self, tmp_path: Path) -> None:
        """Geometric transforms must update corners; normalized box coords must be in [0, 1]."""
        import types

        root = tmp_path / "val"
        (root / "images").mkdir(parents=True)
        (root / "labelTxt").mkdir()
        img = Image.new("RGB", (100, 100), color="green")
        img.save(root / "images" / "img.png")
        (root / "labelTxt" / "img.txt").write_text("5 5 40 5 40 40 5 40 plane 0\n")

        args = types.SimpleNamespace(dataset_dir=str(tmp_path))
        dataset = build_dota("val", args, 64)

        image_tensor, target = dataset[0]

        assert target["boxes_obb"].shape[0] == 1, "one box should survive transforms"
        obb = target["boxes_obb"][0]
        assert (obb[:4] >= 0).all() and (obb[:4] <= 1).all(), "normalized box coords out of [0, 1]"
        assert 0.0 <= obb[4].item() < math.pi, "angle out of [0, pi)"
