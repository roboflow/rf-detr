# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private developer download helpers."""

import io
import threading
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

import rfdetr.datasets._develop as _develop_mod
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

    def test_concurrent_callers_for_the_same_url_do_not_race(self, tmp_path: Path) -> None:
        """Two independent callers requesting the same URL/dest_dir must not corrupt each other.

        Regression test for a case where two pytest fixtures (``download_coco_val`` and
        ``download_coco_val_keypoints``) each guarded their own call to
        ``_download_and_extract`` with a *different* lock file, even though both called it
        with the identical URL and ``dest_dir``. Under pytest-xdist, one worker's
        ``urlretrieve`` could start overwriting the shared zip on disk while another worker
        was mid-extraction, surfacing as a bare ``EOFError`` from inside ``zipfile``.

        This reproduces that interleaving directly: the first caller is paused inside
        ``_extract_zip`` (simulating the window where the file is being read) while a
        second caller is launched concurrently for the same URL. If the two calls are not
        serialized on a lock keyed by the shared resource, the second caller's
        ``urlretrieve`` starts writing to ``zip_path`` while the first is still reading it.
        """
        zip_bytes = self._make_zip({"hello.txt": "world"})
        url = "http://example.com/shared.zip"
        events: list[str] = []
        events_lock = threading.Lock()
        first_extract_started = threading.Event()
        second_caller_launched = threading.Event()

        def fake_urlretrieve(url: str, dest: str) -> tuple[str, dict[str, str]]:
            with events_lock:
                events.append("fetch-start")
            Path(dest).write_bytes(zip_bytes)
            return dest, {"Content-Length": str(len(zip_bytes))}

        real_extract_zip = _develop_mod._extract_zip

        def paused_extract_zip(zip_path: Path, dest_dir_resolved: Path) -> None:
            with events_lock:
                events.append("extract-start")
            first_extract_started.set()
            # Give the second caller a chance to run; if the lock is shared it can only
            # block, never actually reach "fetch-start" before this returns.
            second_caller_launched.wait(timeout=5)
            real_extract_zip(zip_path, dest_dir_resolved)
            with events_lock:
                events.append("extract-end")

        def run_second_caller() -> None:
            first_extract_started.wait(timeout=5)
            second_caller_launched.set()
            _download_and_extract(url, tmp_path)
            with events_lock:
                events.append("second-caller-done")

        with (
            patch("rfdetr.datasets._develop.urlretrieve", side_effect=fake_urlretrieve),
            patch("rfdetr.datasets._develop._extract_zip", side_effect=paused_extract_zip),
        ):
            second_thread = threading.Thread(target=run_second_caller)
            second_thread.start()
            _download_and_extract(url, tmp_path)
            second_thread.join(timeout=5)

        assert not second_thread.is_alive()
        # The second caller's fetch must not start until the first caller's extraction
        # (and therefore the whole first call) has finished — that ordering is what a
        # shared, resource-keyed lock guarantees and what a per-fixture lock does not.
        assert events.index("extract-end") < events.index("second-caller-done")
        assert events.count("fetch-start") == 2
        assert events.count("extract-start") == 2
        assert (tmp_path / "hello.txt").read_bytes() == b"world"
