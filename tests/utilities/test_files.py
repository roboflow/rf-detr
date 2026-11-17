# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the file listing helper the fine-tuning cookbooks use to find sample images."""

from __future__ import annotations

from pathlib import Path

from rfdetr.utilities.files import _find_images

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class TestFindImages:
    """Check which files the cookbooks pick up as images."""

    def test_returns_only_image_files(self, tmp_path: Path) -> None:
        """A dataset folder also holds annotation files, and only the images are listed."""
        (tmp_path / "a.jpg").touch()
        (tmp_path / "_annotations.coco.json").touch()

        found = _find_images(tmp_path, _IMAGE_EXTENSIONS)

        assert found == [tmp_path / "a.jpg"]

    def test_searches_every_subfolder(self, tmp_path: Path) -> None:
        """Roboflow exports keep images in nested folders, so the search goes all the way down."""
        (tmp_path / "train" / "deep").mkdir(parents=True)
        (tmp_path / "train" / "deep" / "a.png").touch()

        found = _find_images(tmp_path, _IMAGE_EXTENSIONS)

        assert found == [tmp_path / "train" / "deep" / "a.png"]

    def test_matches_the_suffix_in_any_case(self, tmp_path: Path) -> None:
        """Cameras write ``.JPG``, and a case-sensitive match would skip those files."""
        (tmp_path / "a.JPG").touch()

        found = _find_images(tmp_path, _IMAGE_EXTENSIONS)

        assert found == [tmp_path / "a.JPG"]

    def test_sorts_the_paths(self, tmp_path: Path) -> None:
        """The same folder always lists its images in the same order, so a cookbook picks the same sample."""
        for name in ("c.jpg", "a.jpg", "b.jpg"):
            (tmp_path / name).touch()

        found = _find_images(tmp_path, _IMAGE_EXTENSIONS)

        assert found == [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]

    def test_returns_an_empty_list_for_a_missing_directory(self, tmp_path: Path) -> None:
        """The cookbooks try ``test``, then ``valid``, then ``train``, so a split that does not exist is not an
        error."""
        found = _find_images(tmp_path / "missing", _IMAGE_EXTENSIONS)

        assert found == []
