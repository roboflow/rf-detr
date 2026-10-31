# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterable, Iterator, Optional
from unittest.mock import Mock

import pytest
import requests

from rfdetr.utilities.files import _compute_file_md5, _download_file, _mkstemp_default_mode, _validate_file_md5


class _DummyTqdm:
    """Minimal tqdm stand-in for download tests.

    This avoids real progress bars while preserving the context manager and `update` calls used by the downloader.
    """

    def __init__(self, **kwargs: object) -> None:
        """Store initialization kwargs for optional inspection."""
        self.kwargs = kwargs

    def __enter__(self) -> "_DummyTqdm":
        """Return self to satisfy context manager protocol."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Propagate exceptions raised inside the context."""
        return False

    def update(self, size: int) -> None:
        """No-op progress update for compatibility with tqdm."""
        return None


class _FakeResponse:
    """Test double for requests responses used by the downloader.

    Provides headers, iterable content chunks, observable close behavior, and optional HTTP errors.
    """

    def __init__(
        self,
        content_chunks: Iterable[bytes],
        headers: Optional[dict[str, str]] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the fake response with content and metadata."""
        self._content_chunks = list(content_chunks)
        self.headers = headers or {}
        self._raise_error = raise_error
        self.close = Mock()

    def raise_for_status(self) -> None:
        """Raise the configured HTTP error, if any."""
        if self.close.called:
            raise AssertionError("response closed before HTTP status validation")
        if self._raise_error is not None:
            raise self._raise_error

    def iter_content(self, chunk_size: int = 1024) -> Iterator[bytes]:
        """Yield the configured content chunks."""
        for chunk in self._content_chunks:
            if self.close.called:
                raise AssertionError("response closed before streaming completed")
            yield chunk


def _assert_no_download_temp_files(tmp_path: Path, filename: str = "weights.bin") -> None:
    """Assert that randomized temporary download files were cleaned up.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _assert_no_download_temp_files(Path(d))
    """
    assert not list(tmp_path.glob(f"{filename}.*.tmp"))


_CONTENT = "Test content"
_CONTENT_MD5 = hashlib.md5(_CONTENT.encode()).hexdigest()
_URL = "https://example.com/file.bin"


class TestFileMD5Validation:
    """Test MD5 hash computation and validation."""

    @pytest.mark.parametrize(
        ("content", "expected_hash"),
        [
            pytest.param(b"Hello, World!", "65a8e27d8879283831b664bd8b7f0ad4", id="known-text"),
            pytest.param(b"", "d41d8cd98f00b204e9800998ecf8427e", id="empty-file"),
        ],
    )
    def test_compute_file_md5_matches_known_hash(self, tmp_path: Path, content: bytes, expected_hash: str) -> None:
        """MD5 of a file equals the well-known digest of its content, including the empty file."""
        path = tmp_path / "data.bin"
        path.write_bytes(content)

        assert _compute_file_md5(path) == expected_hash

    def test_compute_file_md5_large_file(self, tmp_path: Path) -> None:
        """MD5 of a 1 MiB file is a well-formed 32-char lowercase hex digest, so chunked reading works."""
        path = tmp_path / "large.bin"
        path.write_bytes(b"A" * (1024 * 1024))

        hash_value = _compute_file_md5(path)

        assert len(hash_value) == 32
        assert all(c in "0123456789abcdef" for c in hash_value)

    @pytest.mark.parametrize(
        ("expected_hash", "is_valid"),
        [
            pytest.param(_CONTENT_MD5, True, id="matching-hash"),
            pytest.param(_CONTENT_MD5.upper(), True, id="matching-uppercase-hash"),
            pytest.param("0" * 32, False, id="wrong-hash"),
        ],
    )
    def test_validate_file_md5(self, tmp_path: Path, expected_hash: str, is_valid: bool) -> None:
        """Validation accepts the file's own digest in either letter case and rejects a wrong digest."""
        path = tmp_path / "data.txt"
        path.write_text(_CONTENT)

        assert _validate_file_md5(path, expected_hash) is is_valid

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Validation of a missing file returns False instead of raising."""
        assert _validate_file_md5(tmp_path / "nonexistent_file_xyz.txt", "abc123") is False


@pytest.fixture
def mock_get(mocker: pytest.MonkeyPatch) -> Mock:
    """Replace the downloader's ``tqdm`` and ``requests.get``; return the ``requests.get`` mock."""
    mocker.patch("rfdetr.utilities.files.tqdm", _DummyTqdm)
    return mocker.patch("rfdetr.utilities.files.requests.get")


class TestDownloadFile:
    """Test download helper behavior and failure cleanup."""

    def test_download_file_missing_content_length(self, mock_get: Mock, tmp_path: Path):
        """Download succeeds when content-length is missing."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([b"hello", b"world"], headers={})
        mock_get.return_value = response

        _download_file(_URL, str(target_path))

        assert target_path.read_bytes() == b"helloworld"
        _assert_no_download_temp_files(tmp_path)
        mock_get.assert_called_once_with(_URL, stream=True, timeout=30.0)
        response.close.assert_called_once_with()

    def test_download_file_http_error(self, mock_get: Mock, tmp_path: Path):
        """HTTP errors raise and do not create files."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([], raise_error=requests.HTTPError("bad request"))
        mock_get.return_value = response

        with pytest.raises(requests.HTTPError):
            _download_file(_URL, str(target_path))

        assert not target_path.exists()
        _assert_no_download_temp_files(tmp_path)
        response.close.assert_called_once_with()

    def test_download_file_stream_error_cleans_temp(self, mock_get: Mock, tmp_path: Path):
        """Streaming errors clean up temp files."""
        target_path = tmp_path / "weights.bin"

        class _StreamErrorResponse(_FakeResponse):
            def iter_content(self, chunk_size: int = 1024) -> Iterator[bytes]:
                yield b"partial"
                raise RuntimeError("stream failure")

        response = _StreamErrorResponse([b"partial"], headers={"content-length": "7"})
        mock_get.return_value = response

        with pytest.raises(RuntimeError):
            _download_file(_URL, str(target_path))

        assert not target_path.exists()
        _assert_no_download_temp_files(tmp_path)
        response.close.assert_called_once_with()

    def test_download_file_md5_failure_cleans_temp(self, mock_get: Mock, tmp_path: Path):
        """MD5 failure removes temp file and target is not created."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([b"data"], headers={"content-length": "4"})
        mock_get.return_value = response

        with pytest.raises(ValueError):
            _download_file(_URL, str(target_path), expected_md5="0" * 32)

        assert not target_path.exists()
        _assert_no_download_temp_files(tmp_path)
        response.close.assert_called_once_with()

    def test_download_file_replace_failure_cleans_temp(self, mock_get: Mock, mocker, tmp_path: Path):
        """Replace failure removes temp file and target is not created."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([b"data"], headers={"content-length": "4"})
        mock_get.return_value = response
        mocker.patch("rfdetr.utilities.files.os.replace", side_effect=PermissionError("replace denied"))

        with pytest.raises(PermissionError) as exc_info:
            _download_file(_URL, str(target_path))

        assert "replace denied" in str(exc_info.value)
        assert not target_path.exists()
        _assert_no_download_temp_files(tmp_path)
        response.close.assert_called_once_with()

    def test_download_file_uses_fdopen_without_reopen(self, mock_get: Mock, mocker, tmp_path: Path) -> None:
        """Download writes through mkstemp file descriptor to avoid TOCTOU reopen."""
        target_path = tmp_path / "weights.bin"
        mock_get.return_value = _FakeResponse([b"data"], headers={"content-length": "4"})
        mock_open = mocker.patch(
            "rfdetr.utilities.files.open", side_effect=AssertionError("must not reopen temp filename")
        )

        _download_file(_URL, str(target_path))

        assert target_path.read_bytes() == b"data"
        _assert_no_download_temp_files(tmp_path)
        mock_open.assert_not_called()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("process_umask", [0o022, 0o027], indirect=True, ids=oct)
    def test_download_file_mode_follows_umask(self, mock_get: Mock, process_umask: int, tmp_path: Path) -> None:
        """The downloaded file gets the mode ``open()`` would give it, not ``mkstemp``'s owner-only ``0o600``."""
        target_path = tmp_path / "weights.bin"
        mock_get.return_value = _FakeResponse([b"data"], headers={"content-length": "4"})

        _download_file(_URL, str(target_path))

        assert stat.S_IMODE(target_path.stat().st_mode) == 0o666 & ~process_umask


class TestMkstempDefaultMode:
    """Tests for the temp-file helper behind the atomic writes, whose mode follows the umask."""

    def test_retries_a_colliding_name(self, tmp_path: Path, mocker) -> None:
        """A candidate name that is already taken is skipped in favour of the next one."""
        (tmp_path / f"tmp{'a' * 16}").touch()
        mocker.patch("rfdetr.utilities.files.secrets.token_hex", side_effect=["a" * 16, "b" * 16])

        fd, path = _mkstemp_default_mode(tmp_path)

        os.close(fd)
        assert Path(path).name == f"tmp{'b' * 16}"

    def test_raises_when_every_candidate_name_is_taken(self, tmp_path: Path, mocker) -> None:
        """Running out of ``tempfile.TMP_MAX`` candidates raises instead of looping forever."""
        (tmp_path / f"tmp{'a' * 16}").touch()
        mocker.patch("rfdetr.utilities.files.tempfile.TMP_MAX", 3)
        mocker.patch("rfdetr.utilities.files.secrets.token_hex", return_value="a" * 16)

        with pytest.raises(FileExistsError) as exc_info:
            _mkstemp_default_mode(tmp_path)

        assert "No usable temporary file name" in str(exc_info.value)

    @pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs elevated privileges on Windows")
    def test_never_follows_a_planted_symlink(self, tmp_path: Path, mocker) -> None:
        """A symlink planted at a candidate name counts as taken, so the file it points to is never created."""
        link_target = tmp_path / "link_target"
        (tmp_path / f"tmp{'a' * 16}").symlink_to(link_target)
        mocker.patch("rfdetr.utilities.files.secrets.token_hex", side_effect=["a" * 16, "b" * 16])

        fd, _ = _mkstemp_default_mode(tmp_path)

        os.close(fd)
        assert not link_target.exists()

    def test_creates_the_file_in_the_given_directory(self, tmp_path: Path) -> None:
        """The temp file lands in the destination directory, so ``os.replace`` stays on one filesystem."""
        fd, path = _mkstemp_default_mode(tmp_path)
        os.close(fd)
        assert Path(path).parent.resolve() == tmp_path.resolve()

    def test_file_name_carries_prefix_and_suffix(self, tmp_path: Path) -> None:
        """The random part of the name sits between the caller's prefix and suffix."""
        fd, path = _mkstemp_default_mode(tmp_path, prefix="weights.pth.", suffix=".tmp")
        os.close(fd)
        name = Path(path).name
        assert (name.startswith("weights.pth."), name.endswith(".tmp")) == (True, True)

    def test_returns_an_absolute_path_for_a_relative_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative directory still yields an absolute path, as ``tempfile.mkstemp`` returns."""
        monkeypatch.chdir(tmp_path)
        fd, path = _mkstemp_default_mode(Path("."))
        os.close(fd)
        assert os.path.isabs(path)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("process_umask", [0o022, 0o027], indirect=True, ids=oct)
    def test_mode_follows_umask(self, tmp_path: Path, process_umask: int) -> None:
        """The file is created ``0o666`` minus the umask, as ``open()`` creates one, not ``mkstemp``'s ``0o600``."""
        fd, path = _mkstemp_default_mode(tmp_path)
        os.close(fd)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o666 & ~process_umask
