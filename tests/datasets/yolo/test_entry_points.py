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
from rfdetr.datasets.yolo import (
    YoloDetection,
    _resolve_yolo_split_dirs,
    _resolve_yolo_split_dirs_with_notes,
    is_valid_yolo_dataset,
)
from rfdetr.detr import RFDETR


@pytest.fixture(params=["roboflow", "val", "images-first", "relative-path", "absolute-path", "ultralytics-path"])
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
    elif layout == "ultralytics-path":
        # Stock Ultralytics YAML shape: ``path:`` repeats the root's own name, so joining it to
        # the root would point at a child that does not exist.
        config += "path: coco8\n"
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

    def test_split_fallback_is_not_reported_by_the_gates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolving a split for a gate records the fallback without logging it.

        Format detection, the validity gate, class discovery and both builder calls all resolve the same splits. The
        resolver logged its own fallback, so one dataset with an unusable declaration warned once per gate per split
        instead of once per split. The first assertion confirms this layout still produces a note at all, so the second
        is not vacuous.
        """
        (tmp_path / "declared").mkdir()
        for split in ("train", "valid"):
            for subdir in ("images", "labels"):
                (tmp_path / split / subdir).mkdir(parents=True)
        (tmp_path / "data.yaml").write_text("names: [person]\ntrain: declared\n", encoding="utf-8")
        recorded: list[object] = []
        monkeypatch.setattr("rfdetr.datasets.yolo.logger.warning", lambda *args: recorded.append(args))
        monkeypatch.setattr("rfdetr.datasets.yolo.logger.log", lambda *args: recorded.append(args))

        assert _resolve_yolo_split_dirs_with_notes(tmp_path, tmp_path / "data.yaml", "train")[1] != ()
        detect_roboflow_format(tmp_path)
        is_valid_yolo_dataset(str(tmp_path))
        RFDETR._load_classes(str(tmp_path))
        assert recorded == []

    def test_symlink_loop_in_declared_split_is_reported_as_undetectable(self, tmp_path: Path) -> None:
        """A symlink loop under a declared split leaves both gates answering, not raising.

        The containment guard caught only ``ValueError`` from ``Path.resolve()``, which also raises ``RuntimeError`` for
        a loop up to Python 3.12, so a looped ``train:`` target propagated out of ``train()`` instead of reporting a
        dataset it cannot read. The outcome is asserted rather than the exception type, because newer interpreters
        resolve a loop without raising and reach the same answer by a different route.
        """
        loop_head = tmp_path / "loopA"
        try:
            loop_head.symlink_to(tmp_path / "loopB", target_is_directory=True)
            (tmp_path / "loopB").symlink_to(loop_head, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlinks in this environment: {exc}")
        (tmp_path / "data.yaml").write_text("names: [person]\ntrain: loopA/images\n", encoding="utf-8")
        assert not is_valid_yolo_dataset(str(tmp_path))
        with pytest.raises(ValueError, match="Could not detect dataset format"):
            detect_roboflow_format(tmp_path)

    def test_relative_dataset_dir_without_path_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative dataset root is joined onto a declared split path exactly once.

        With no ``path:`` key the resolver used the root as its own base and then joined it onto the root again whenever
        it was relative, so ``ds`` became ``ds/ds``. Every declared split then looked missing and an images-first layout
        fell back to the Roboflow convention, which this dataset does not have.
        """
        for split in ("train", "val"):
            (tmp_path / "ds" / "images" / split).mkdir(parents=True)
            (tmp_path / "ds" / "labels" / split).mkdir(parents=True)
        (tmp_path / "ds" / "data.yaml").write_text(
            "names: [person]\ntrain: images/train\nval: images/val\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        assert detect_roboflow_format(Path("ds")) == "yolo"
        assert is_valid_yolo_dataset("ds")

    def test_file_where_split_directory_belongs_is_rejected(self, tmp_path: Path) -> None:
        """A plain file standing in for a split directory does not satisfy the validity gate.

        The gate resolved each split and then only asked whether the path existed, so a stray ``train/labels`` file — a
        truncated download, or an archive that unpacked a file over the directory — counted as a usable split and the
        real failure surfaced later inside the loader.
        """
        (tmp_path / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "labels").write_text("", encoding="utf-8")
        for subdir in ("images", "labels"):
            (tmp_path / "valid" / subdir).mkdir(parents=True)
        assert not is_valid_yolo_dataset(str(tmp_path))

    def test_present_but_unresolved_data_file_is_named_in_the_errors(self, tmp_path: Path) -> None:
        """Both gates name the YOLO data file they found instead of implying none exists.

        A root holding a ``data.yaml`` whose declared splits do not resolve reported "Could not detect dataset format
        ... Expected ... data.yaml or data.yml" and a class-discovery error listing the same filenames it had just read,
        so the message pointed at a missing file as the cause rather than at the unresolved split.
        """
        (tmp_path / "data.yaml").write_text("names: [person]\ntrain: images/train\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Found the YOLO data file"):
            detect_roboflow_format(tmp_path)
        with pytest.raises(FileNotFoundError, match="could not both be resolved"):
            RFDETR._load_classes(str(tmp_path))

    def test_conventional_split_symlinked_outside_root_is_rejected(self, tmp_path: Path) -> None:
        """A conventional split directory symlinked out of the dataset root is refused.

        Containment governed only YAML-declared paths, so identical storage was refused when named by ``data.yaml`` and
        accepted when reached through a symlinked ``train/images``. The guard decided by layout instead of by
        destination; both routes now share one rule.
        """
        root = tmp_path / "dataset"
        outside = tmp_path / "elsewhere" / "images"
        outside.mkdir(parents=True)
        (root / "train").mkdir(parents=True)
        try:
            (root / "train" / "images").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlinks in this environment: {exc}")
        (root / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not detect dataset format"):
            detect_roboflow_format(root)

    def test_dataset_root_reached_through_a_symlink_is_accepted(self, tmp_path: Path) -> None:
        """A dataset whose entire root is a symlink stays detectable.

        Containment resolves the split and the root, so mounting a whole dataset on other storage and linking to it —
        the usual cluster arrangement — keeps working. Only an individual split escaping its own root is refused.
        """
        real_root = tmp_path / "storage" / "dataset"
        (real_root / "train" / "images").mkdir(parents=True)
        (real_root / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        link = tmp_path / "linked-dataset"
        try:
            link.symlink_to(real_root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlinks in this environment: {exc}")
        assert detect_roboflow_format(link) == "yolo"

    def test_declared_training_images_are_detected_without_labels(self, tmp_path: Path) -> None:
        """A declared training image directory identifies YOLO whether or not labels exist.

        Detection resolved the declared split through the labels-aware resolver, so a YAML layout whose ``train:``
        images had no ``labels`` sibling fell back to ``train/images`` and went undetected — while the very same
        unlabelled dataset in the legacy layout was detected. The validity gate still requires labels, so it keeps
        saying no.
        """
        (tmp_path / "data.yaml").write_text("names: [person]\ntrain: images/train\n", encoding="utf-8")
        (tmp_path / "images" / "train").mkdir(parents=True)
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
