# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional
from unittest.mock import Mock, patch

import pytest
import requests

from rfdetr.utilities.files import _compute_file_md5, _download_file, _validate_file_md5


class _DummyTqdm:
    """
    Minimal tqdm stand-in for download tests.

    This avoids real progress bars while preserving the context manager and
    `update` calls used by the downloader.
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Store initialization kwargs for optional inspection.
        """
        self.kwargs = kwargs

    def __enter__(self) -> "_DummyTqdm":
        """
        Return self to satisfy context manager protocol.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """
        Propagate exceptions raised inside the context.
        """
        return False

    def update(self, size: int) -> None:
        """
        No-op progress update for compatibility with tqdm.
        """
        return None


class _FakeResponse:
    """
    Test double for requests responses used by the downloader.

    Provides headers, iterable content chunks, and optional HTTP error behavior
    via `raise_for_status`.
    """

    def __init__(
        self,
        content_chunks: Iterable[bytes],
        headers: Optional[dict[str, str]] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the fake response with content and metadata.
        """
        self._content_chunks = list(content_chunks)
        self.headers = headers or {}
        self._raise_error = raise_error

    def raise_for_status(self) -> None:
        """
        Raise the configured HTTP error, if any.
        """
        if self._raise_error is not None:
            raise self._raise_error

    def iter_content(self, chunk_size: int = 1024) -> Iterator[bytes]:
        """
        Yield the configured content chunks.
        """
        for chunk in self._content_chunks:
            yield chunk


class TestFileMD5Validation:
    """Test MD5 hash computation and validation."""

    def test_compute_file_md5(self):
        """Test MD5 hash computation for a simple file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_file = f.name

        try:
            # Known MD5 hash for "Hello, World!"
            expected_hash = "65a8e27d8879283831b664bd8b7f0ad4"
            actual_hash = _compute_file_md5(temp_file)
            assert actual_hash == expected_hash
        finally:
            os.unlink(temp_file)

    def test_validate_file_md5_success(self):
        """Test successful MD5 validation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Compute the actual hash first
            expected_hash = _compute_file_md5(temp_file)

            # Validation should succeed
            assert _validate_file_md5(temp_file, expected_hash) is True
        finally:
            os.unlink(temp_file)

    def test_validate_file_md5_failure(self):
        """Test MD5 validation failure with wrong hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Use a wrong hash
            wrong_hash = "0" * 32

            # Validation should fail
            assert _validate_file_md5(temp_file, wrong_hash) is False
        finally:
            os.unlink(temp_file)

    @pytest.mark.parametrize("hash_case", ["lower", "upper"], ids=["lowercase", "uppercase"])
    def test_validate_file_md5_case_insensitive(self, hash_case: Literal["lower", "upper"]) -> None:
        """Test that MD5 validation is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Get hash in lowercase
            hash_lower = _compute_file_md5(temp_file)
            test_hash = hash_lower.upper() if hash_case == "upper" else hash_lower

            # Each case variant (lower/upper) should validate successfully
            assert _validate_file_md5(temp_file, test_hash) is True
        finally:
            os.unlink(temp_file)

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_file = "/tmp/nonexistent_file_xyz.txt"
        assert _validate_file_md5(nonexistent_file, "abc123") is False

    def test_compute_file_md5_empty_file(self):
        """Test MD5 hash computation for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Create empty file
            temp_file = f.name

        try:
            # Known MD5 hash for empty file
            expected_hash = "d41d8cd98f00b204e9800998ecf8427e"
            actual_hash = _compute_file_md5(temp_file)
            assert actual_hash == expected_hash
        finally:
            os.unlink(temp_file)

    def test_compute_file_md5_large_file(self):
        """Test MD5 computation for larger file (tests chunking)."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # Write 1MB of data
            data = b"A" * (1024 * 1024)
            f.write(data)
            temp_file = f.name

        try:
            # Compute hash (should handle chunking correctly)
            hash_value = _compute_file_md5(temp_file)

            # Verify it's a valid MD5 hash format
            assert len(hash_value) == 32
            assert all(c in "0123456789abcdef" for c in hash_value)
        finally:
            os.unlink(temp_file)


class TestDownloadFile:
    """Test download helper behavior and failure cleanup."""

    @patch("rfdetr.utilities.files.tqdm", _DummyTqdm)
    @patch("rfdetr.utilities.files.requests.get")
    def test_download_file_missing_content_length(self, mock_get: Mock, tmp_path: Path):
        """Download succeeds when content-length is missing."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([b"hello", b"world"], headers={})
        mock_get.return_value = response

        _download_file("https://example.com/file.bin", str(target_path))

        assert target_path.exists()
        assert target_path.read_bytes() == b"helloworld"
        assert not (tmp_path / "weights.bin.tmp").exists()
        mock_get.assert_called_once_with("https://example.com/file.bin", stream=True, timeout=30.0)

    @patch("rfdetr.utilities.files.requests.get")
    def test_download_file_http_error(self, mock_get: Mock, tmp_path: Path):
        """HTTP errors raise and do not create files."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([], raise_error=requests.HTTPError("bad request"))
        mock_get.return_value = response

        with pytest.raises(requests.HTTPError):
            _download_file("https://example.com/file.bin", str(target_path))

        assert not target_path.exists()
        assert not (tmp_path / "weights.bin.tmp").exists()

    @patch("rfdetr.utilities.files.tqdm", _DummyTqdm)
    @patch("rfdetr.utilities.files.requests.get")
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
            _download_file("https://example.com/file.bin", str(target_path))

        assert not target_path.exists()
        assert not (tmp_path / "weights.bin.tmp").exists()

    @patch("rfdetr.utilities.files.tqdm", _DummyTqdm)
    @patch("rfdetr.utilities.files.requests.get")
    def test_download_file_md5_failure_cleans_temp(self, mock_get: Mock, tmp_path: Path):
        """MD5 failure removes temp file and target is not created."""
        target_path = tmp_path / "weights.bin"
        response = _FakeResponse([b"data"], headers={"content-length": "4"})
        mock_get.return_value = response

        with pytest.raises(ValueError):
            _download_file(
                "https://example.com/file.bin",
                str(target_path),
                expected_md5="0" * 32,
            )

        assert not target_path.exists()
        assert not (tmp_path / "weights.bin.tmp").exists()
