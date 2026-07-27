# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private developer download helpers."""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rfdetr.datasets._develop import (
    _coco_val_images_complete,
    _download_and_extract,
    _download_lock,
    _nonempty_file_exists,
)


class TestCocoValImagesComplete:
    """Regression coverage for interrupted COCO val2017 image downloads."""

    def test_missing_directory_is_incomplete(self, tmp_path: Path) -> None:
        """A missing image directory must trigger a download."""
        assert not _coco_val_images_complete(tmp_path / "val2017")

    def test_empty_existing_directory_is_incomplete(self, tmp_path: Path) -> None:
        """An empty ``val2017`` directory must not skip the image download."""
        images_root = tmp_path / "val2017"
        images_root.mkdir()

        assert not _coco_val_images_complete(images_root)

    @pytest.mark.parametrize(
        "file_count,expected",
        [
            pytest.param(1, False, id="below_threshold_is_incomplete"),
            pytest.param(2, True, id="at_threshold_is_complete"),
            pytest.param(3, True, id="above_threshold_is_complete"),
        ],
    )
    def test_file_count_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_count: int, expected: bool
    ) -> None:
        """Directory completeness reflects the >= threshold semantics."""
        import rfdetr.datasets._develop as _develop_mod

        monkeypatch.setattr(_develop_mod, "_COCO_VAL_IMAGE_COUNT", 2)
        images_root = tmp_path / "val2017"
        images_root.mkdir()
        for i in range(file_count):
            (images_root / f"{i:012d}.jpg").write_bytes(b"jpeg")

        assert _coco_val_images_complete(images_root) is expected


class TestNonemptyFileExists:
    """Regression coverage for annotation file integrity checks in benchmark downloads."""

    def test_missing_file_is_incomplete(self, tmp_path: Path) -> None:
        """A missing annotation file must trigger a download."""
        annotations_path = tmp_path / "instances_val2017.json"

        assert not _nonempty_file_exists(annotations_path)

    def test_empty_file_is_incomplete(self, tmp_path: Path) -> None:
        """An empty annotation file must trigger a re-download."""
        annotations_path = tmp_path / "instances_val2017.json"
        annotations_path.write_bytes(b"")

        assert not _nonempty_file_exists(annotations_path)

    def test_nonempty_file_is_complete(self, tmp_path: Path) -> None:
        """A non-empty annotation file is accepted without re-download."""
        annotations_path = tmp_path / "instances_val2017.json"
        annotations_path.write_bytes(b"{}")

        assert _nonempty_file_exists(annotations_path)


class TestDownloadLock:
    """Coverage for the cross-process file-lock context manager."""

    def test_timeout_raises_when_lock_held(self, tmp_path: Path) -> None:
        """TimeoutError is raised immediately when the lock file already exists and timeout_s=0."""
        lock_path = tmp_path / "test.lock"
        lock_path.touch()

        with pytest.raises(TimeoutError):
            with _download_lock(lock_path, timeout_s=0, poll_s=0):
                pass


class TestDownloadAndExtract:
    """Coverage for the ZIP download-and-extract helper."""

    def _make_zip(self, members: dict) -> bytes:
        """Build an in-memory ZIP archive from a mapping of filename→content."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_path_traversal_raises_runtime_error(self, tmp_path: Path) -> None:
        """A ZIP entry escaping dest_dir must raise RuntimeError (path-traversal guard)."""
        zip_bytes = self._make_zip({"../evil.txt": "malicious"})
        url = "http://example.com/test.zip"

        def fake_urlretrieve(url: str, dest: str) -> None:
            Path(dest).write_bytes(zip_bytes)

        with patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(RuntimeError, match="Unsafe path detected"):
                _download_and_extract(url, tmp_path)
