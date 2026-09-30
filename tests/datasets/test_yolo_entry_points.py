# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from rfdetr.datasets import build_dataset, detect_roboflow_format
from rfdetr.datasets.yolo import YoloDetection, _resolve_yolo_split_dirs, is_valid_yolo_dataset
from rfdetr.detr import RFDETR


@pytest.fixture(params=["roboflow", "val", "images-first", "relative-path", "absolute-path"])
def yolo_entry_point_dataset(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Create two tiny splits in a layout already supported by the YOLO resolver.

    Examples:
        >>> # Pytest supplies the temporary directory and layout parameter.
        >>> yolo_entry_point_dataset(tmp_path, request)  # doctest: +SKIP
    """
    layout = request.param
    base = tmp_path / "content" if layout in ("relative-path", "absolute-path") else tmp_path
    config = "names: [person]\n"
    if layout == "relative-path":
        config += "path: content\n"
    elif layout == "absolute-path":
        config += f"path: {base.as_posix()}\n"
    for split, image_count in (("train", 1), ("val", 2)):
        if layout == "roboflow":
            split_dir = "valid" if split == "val" else split
            image_dir = base / split_dir / "images"
            label_dir = base / split_dir / "labels"
        elif layout == "val":
            image_dir = base / split / "images"
            label_dir = base / split / "labels"
        else:
            image_dir = base / "images" / split
            label_dir = base / "labels" / split
            config += f"{split}: images/{split}\n"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for idx in range(image_count):
            Image.new("RGB", (8, 6), color="white").save(image_dir / f"sample{idx}.png")
            (label_dir / f"sample{idx}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(config, encoding="utf-8")
    return tmp_path


@pytest.fixture(params=["data.yaml", "data.yml"])
def yolo_entry_point_root(yolo_entry_point_dataset: Path, request: pytest.FixtureRequest) -> Path:
    """Select either supported YAML filename for the dataset.

    Examples:
        >>> # Pytest supplies the dataset and YAML filename parameter.
        >>> yolo_entry_point_root(yolo_entry_point_dataset, request)  # doctest: +SKIP
    """
    if request.param != "data.yaml":
        (yolo_entry_point_dataset / "data.yaml").rename(yolo_entry_point_dataset / request.param)
    return yolo_entry_point_dataset


class TestYoloDatasetEntryPoints:
    """Format detection and class discovery accept the existing split resolver's layouts."""

    def test_format_detection(self, yolo_entry_point_root: Path) -> None:
        """Automatic format detection recognizes every supported layout."""
        assert detect_roboflow_format(yolo_entry_point_root) == "yolo"

    def test_dataset_validation(self, yolo_entry_point_root: Path) -> None:
        """The class-discovery gate accepts resolvable training and validation splits."""
        assert is_valid_yolo_dataset(str(yolo_entry_point_root))

    def test_class_discovery(self, yolo_entry_point_root: Path) -> None:
        """The facade discovers the same names without constructing a model."""
        assert RFDETR._load_classes(str(yolo_entry_point_root)) == ["person"]

    @pytest.mark.parametrize("dataset_file", ["roboflow", "yolo"])
    def test_dataset_builder(self, yolo_entry_point_root: Path, dataset_file: str) -> None:
        """Automatic and explicit builder routes both load the validation samples."""
        args = SimpleNamespace(
            dataset_dir=str(yolo_entry_point_root),
            dataset_file=dataset_file,
            square_resize_div_64=True,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
            augmentation_backend="torchvision",
        )
        dataset = build_dataset("val", args, resolution=64)
        assert isinstance(dataset, YoloDetection)
        assert len(dataset) == 2
        image, target = dataset[0]
        assert tuple(image.shape) == (3, 64, 64)
        assert target["labels"].tolist() == [0]
        assert target["boxes"].tolist() == [[0.5, 0.5, 0.5, 0.5]]

    def test_coco_detection_precedence(self, yolo_entry_point_root: Path) -> None:
        """A COCO annotation file keeps priority when YOLO is also available."""
        train_dir = yolo_entry_point_root / "train"
        train_dir.mkdir(exist_ok=True)
        (train_dir / "_annotations.coco.json").write_text(
            json.dumps({"categories": [{"id": 0, "name": "coco-person"}], "annotations": []}),
            encoding="utf-8",
        )
        assert detect_roboflow_format(yolo_entry_point_root) == "coco"
        assert RFDETR._load_classes(str(yolo_entry_point_root)) == ["coco-person"]


class TestYoloEntryPointFallbacks:
    """Keep the existing split requirements, fallback behavior, and path guards."""

    def test_format_detection_requires_only_training_images(self, tmp_path: Path) -> None:
        """Format recognition does not require validation or labels in a legacy layout."""
        (tmp_path / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        (tmp_path / "train" / "images").mkdir(parents=True)
        assert detect_roboflow_format(tmp_path) == "yolo"
        assert not is_valid_yolo_dataset(str(tmp_path))

    @pytest.mark.parametrize("config", ["names: [person]\ntrain: missing/images\n", "invalid: ["])
    def test_legacy_fallback(self, tmp_path: Path, config: str) -> None:
        """Unusable YAML paths and malformed YAML retain the legacy filesystem fallback."""
        (tmp_path / "data.yaml").write_text(config, encoding="utf-8")
        for split in ("train", "valid"):
            for subdir in ("images", "labels"):
                (tmp_path / split / subdir).mkdir(parents=True)
        assert detect_roboflow_format(tmp_path) == "yolo"
        assert is_valid_yolo_dataset(str(tmp_path))

    def test_yaml_paths_outside_dataset_are_not_detected(self, tmp_path: Path) -> None:
        """Entrypoints preserve the resolver's rejection of paths outside the dataset root."""
        root = tmp_path / "dataset"
        root.mkdir()
        for subdir in ("images", "labels"):
            (tmp_path / subdir).mkdir()
        data_file = root / "data.yaml"
        data_file.write_text("names: [person]\ntrain: ../images\nval: ../images\n", encoding="utf-8")
        assert _resolve_yolo_split_dirs(root, data_file, "train") == (root / "train/images", root / "train/labels")
        assert not is_valid_yolo_dataset(str(root))
        with pytest.raises(ValueError, match="Could not detect dataset format"):
            detect_roboflow_format(root)

    def test_empty_directory_reports_no_format(self, tmp_path: Path) -> None:
        """Format detection raises a clear error on a directory with no COCO or YOLO markers.

        A plain empty directory (no data.yaml/data.yml, no train images, no COCO annotation file) is the baseline
        failure case; the only existing coverage of this ValueError goes through the path-traversal rejection instead.
        """
        with pytest.raises(ValueError, match="Could not detect dataset format"):
            detect_roboflow_format(tmp_path)

    @pytest.mark.parametrize("layout", ["images-first", "relative-path", "absolute-path"])
    def test_train_only_yaml_layout_is_invalid(self, tmp_path: Path, layout: str) -> None:
        """A train-only YAML-declared layout fails the class-discovery validity gate.

        Only the legacy-layout train-only case was covered before; a YAML-declared train path (images-first, relative-
        path, or absolute-path base) with no val/valid split must be rejected the same way, since building still
        requires a resolvable val split.
        """
        base = tmp_path / "content" if layout in ("relative-path", "absolute-path") else tmp_path
        config = "names: [person]\n"
        if layout == "relative-path":
            config += "path: content\n"
        elif layout == "absolute-path":
            config += f"path: {base.as_posix()}\n"
        image_dir = base / "images" / "train"
        label_dir = base / "labels" / "train"
        config += "train: images/train\n"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (8, 6), color="white").save(image_dir / "sample0.png")
        (label_dir / "sample0.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text(config, encoding="utf-8")
        assert not is_valid_yolo_dataset(str(tmp_path))

    def test_malformed_yaml_class_discovery_raises(self, tmp_path: Path) -> None:
        """Class-name loading surfaces the parse error instead of silently misreading names.

        The legacy filesystem fallback tolerates malformed YAML for format detection and
        validity (test_legacy_fallback), but class discovery still opens and parses the
        same malformed file directly: it must not silently return the wrong names.
        """
        (tmp_path / "data.yaml").write_text("invalid: [", encoding="utf-8")
        for split in ("train", "valid"):
            for subdir in ("images", "labels"):
                (tmp_path / split / subdir).mkdir(parents=True)
        with pytest.raises(yaml.YAMLError):
            RFDETR._load_classes(str(tmp_path))
