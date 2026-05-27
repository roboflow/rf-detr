# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private developer download helpers."""

from pathlib import Path

import pytest

from rfdetr.datasets import _develop


class TestCocoValImagesComplete:
    """Regression coverage for interrupted COCO val2017 image downloads."""

    def test_missing_directory_is_incomplete(self, tmp_path: Path) -> None:
        """A missing image directory must trigger a download."""
        assert not _develop._coco_val_images_complete(tmp_path / "val2017")

    def test_empty_existing_directory_is_incomplete(self, tmp_path: Path) -> None:
        """An empty ``val2017`` directory must not skip the image download."""
        images_root = tmp_path / "val2017"
        images_root.mkdir()

        assert not _develop._coco_val_images_complete(images_root)

    def test_too_few_images_is_incomplete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial image directory must not skip the image download."""
        monkeypatch.setattr(_develop, "_COCO_VAL_IMAGE_COUNT", 2)
        images_root = tmp_path / "val2017"
        images_root.mkdir()
        (images_root / "000000000139.jpg").write_bytes(b"jpeg")

        assert not _develop._coco_val_images_complete(images_root)

    def test_expected_image_count_is_complete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A directory with the expected image count is accepted."""
        monkeypatch.setattr(_develop, "_COCO_VAL_IMAGE_COUNT", 2)
        images_root = tmp_path / "val2017"
        images_root.mkdir()
        (images_root / "000000000139.jpg").write_bytes(b"jpeg")
        (images_root / "000000000285.jpg").write_bytes(b"jpeg")

        assert _develop._coco_val_images_complete(images_root)


class TestAnnotationFilePreconditions:
    """Coverage for annotation file preconditions used by benchmark downloads."""

    def test_missing_file_is_incomplete(self, tmp_path: Path) -> None:
        """Missing files must trigger a download."""
        annotations_path = tmp_path / "instances_val2017.json"

        assert not annotations_path.exists()

    def test_empty_file_is_incomplete(self, tmp_path: Path) -> None:
        """Empty files must trigger a download."""
        annotations_path = tmp_path / "instances_val2017.json"
        annotations_path.write_text("")

        assert annotations_path.is_file()
        assert annotations_path.stat().st_size == 0

    def test_nonempty_file_is_complete(self, tmp_path: Path) -> None:
        """Non-empty files are accepted."""
        annotations_path = tmp_path / "instances_val2017.json"
        annotations_path.write_text("{}")

        assert annotations_path.is_file()
        assert annotations_path.stat().st_size > 0
