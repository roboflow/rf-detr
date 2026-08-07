# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private developer download helpers."""

import io
import threading
import time
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
        """Build an in-memory ZIP archive from a mapping of filename→content.

        Example:
            >>> archive = TestDownloadAndExtract()._make_zip({"hello.txt": "world"})
            >>> isinstance(archive, bytes)
            True
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_path_traversal_raises_runtime_error(self, tmp_path: Path) -> None:
        """A ZIP entry escaping dest_dir must raise RuntimeError (path-traversal guard)."""
        zip_bytes = self._make_zip({"../evil.txt": "malicious"})
        url = "http://example.com/test.zip"

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            Path(dest).write_bytes(zip_bytes)
            return dest, {}

        with patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(RuntimeError, match="Unsafe path detected"):
                _download_and_extract(url, tmp_path)


class TestConcurrentDownloadSafety:
    """Regression coverage for concurrent downloads of the same COCO asset.

    Benchmark fixtures used to guard the shared ``data/`` directory with one lock *per fixture* while all of them
    downloaded the same archive. Under ``pytest-xdist`` two workers could therefore write the same zip path at once; the
    second ``open("wb")`` truncated the archive the first was still streaming members out of, surfacing as ``EOFError``
    from ``zipfile._read2``. Serialization now lives in ``_download_and_extract`` itself, keyed on the asset, so every
    caller of a URL contends on one lock regardless of which fixture invoked it.
    """

    def _zip_bytes(self) -> bytes:
        """Build a small in-memory ZIP archive used as a stand-in for a COCO asset.

        Example:
            >>> isinstance(TestConcurrentDownloadSafety()._zip_bytes(), bytes)
            True
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("annotations/instances_val2017.json", "{}")
        return buf.getvalue()

    def test_same_asset_downloads_never_overlap(self, tmp_path: Path) -> None:
        """Two concurrent callers of one URL must not be inside the download window together."""
        url = "http://example.com/annotations_trainval2017.zip"
        zip_bytes = self._zip_bytes()
        counter_lock = threading.Lock()
        active = 0
        max_concurrent = 0

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            nonlocal active, max_concurrent
            with counter_lock:
                active += 1
                max_concurrent = max(max_concurrent, active)
            time.sleep(0.2)
            Path(dest).write_bytes(zip_bytes)
            with counter_lock:
                active -= 1
            return dest, {}

        with patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve):
            threads = [threading.Thread(target=_download_and_extract, args=(url, tmp_path)) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        assert max_concurrent == 1, f"{max_concurrent} callers wrote the same archive concurrently"

    def test_distinct_assets_use_distinct_locks(self, tmp_path: Path) -> None:
        """Serialization is per asset, so unrelated URLs are not forced to queue behind each other."""
        zip_bytes = self._zip_bytes()

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            Path(dest).write_bytes(zip_bytes)
            return dest, {}

        with patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve):
            _download_and_extract("http://example.com/val2017.zip", tmp_path)
            _download_and_extract("http://example.com/train2017.zip", tmp_path)

        lock_names = sorted(path.name for path in tmp_path.glob(".*.lock"))
        assert lock_names == [], f"lock files leaked after completion: {lock_names}"

    def test_completed_asset_is_not_redownloaded(self, tmp_path: Path) -> None:
        """A caller that waited on the lock re-checks completeness and skips a redundant fetch."""
        url = "http://example.com/annotations_trainval2017.zip"
        calls: list[str] = []

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            calls.append(url)
            Path(dest).write_bytes(self._zip_bytes())
            return dest, {}

        with patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve):
            _download_and_extract(url, tmp_path, is_complete=lambda: True)

        assert calls == [], "asset already present on disk was downloaded again"

    def test_eof_error_during_extraction_is_retried(self, tmp_path: Path) -> None:
        """A mid-stream truncation surfaces as EOFError and must be retried, not propagated."""
        url = "http://example.com/annotations_trainval2017.zip"
        zip_bytes = self._zip_bytes()
        attempts: list[str] = []

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            attempts.append(url)
            Path(dest).write_bytes(zip_bytes)
            return dest, {}

        def flaky_extract(zip_path: Path, dest_dir_resolved: Path) -> None:
            if len(attempts) == 1:
                raise EOFError
            (dest_dir_resolved / "extracted.json").write_text("{}")

        with (
            patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve),
            patch("rfdetr.datasets._develop._extract_zip", side_effect=flaky_extract),
            patch("rfdetr.datasets._develop.time.sleep"),
        ):
            _download_and_extract(url, tmp_path)

        assert len(attempts) == 2, f"expected one retry after EOFError, saw {len(attempts)} attempt(s)"
        assert (tmp_path / "extracted.json").exists()
