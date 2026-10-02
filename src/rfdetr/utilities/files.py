# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""File download and MD5 validation helpers, plus the temp-file helpers behind atomic, umask-honouring writes."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import secrets
import stat
import tempfile

import requests
from tqdm.auto import tqdm

from rfdetr.utilities.logger import get_logger

logger = get_logger()
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0


def _mkstemp_default_mode(
    directory: str | os.PathLike[str], *, prefix: str = "tmp", suffix: str = ""
) -> tuple[int, str]:
    """Create and open a new temporary file whose permissions follow the process umask.

    A :func:`tempfile.mkstemp`-like helper for files that are renamed into place with :func:`os.replace`. ``mkstemp``
    creates its file readable and writable by the owner only (``0o600``), and the rename keeps that mode, so the final
    file would be unreadable to every other user, unlike the same file written with :func:`open`. This creates the file
    with mode ``0o666`` and lets the OS apply the umask, exactly as :func:`open` does when it creates a file.
    :func:`open` on an existing file keeps that file's mode instead, so callers replacing one move the temp file into
    place with :func:`_replace_keeping_mode`.

    It is not a drop-in replacement: ``directory`` is required, ``prefix`` and ``suffix`` are keyword-only, and it
    neither raises the ``tempfile.mkstemp`` audit event nor retries on the ``PermissionError`` Windows reports for a
    name taken by a directory.

    Args:
        directory: Directory to create the file in.
        prefix: Start of the file name.
        suffix: End of the file name.

    Returns:
        An open OS-level file descriptor and the absolute path of the file.

    Raises:
        FileExistsError: If no unused file name was found.
        OSError: If the file cannot be created (e.g. directory missing or not writable).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     fd, path = _mkstemp_default_mode(directory, prefix="weights.pth.", suffix=".tmp")
        ...     os.close(fd)
        ...     os.path.basename(path).startswith("weights.pth."), path.endswith(".tmp")
        (True, True)
    """
    # The flags tempfile.mkstemp uses; O_BINARY keeps Windows from translating newlines.
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    for _ in range(tempfile.TMP_MAX):
        path = os.path.abspath(os.path.join(directory, f"{prefix}{secrets.token_hex(8)}{suffix}"))
        try:
            return os.open(path, flags, 0o666), path
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "No usable temporary file name found", os.fspath(directory))


def _replace_keeping_mode(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Move ``source`` over ``destination`` with :func:`os.replace`, keeping the permission bits ``destination`` had.

    Rewriting an existing file with :func:`open` keeps its mode, but a rename puts the new file's own mode in its place,
    so an owner-only ``0o600`` destination would come out with the temp file's umask-derived mode, e.g. ``0o644``. When
    ``destination`` exists, its mode is copied onto ``source`` first; when it does not, ``source`` keeps the mode it was
    created with. Skipped on Windows, where the permission bits are only a read-only flag and there is no umask.

    Args:
        source: The fully written temporary file, in the same directory as ``destination``.
        destination: Path to replace.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     destination = os.path.join(directory, "config.json")
        ...     with open(destination, "w") as handle:
        ...         _ = handle.write("old")
        ...     os.chmod(destination, 0o600)
        ...     fd, source = _mkstemp_default_mode(directory)
        ...     with os.fdopen(fd, "w") as handle:
        ...         _ = handle.write("new")
        ...     _replace_keeping_mode(source, destination)
        ...     with open(destination) as handle:
        ...         content = handle.read()
        ...     content, os.name == "nt" or stat.S_IMODE(os.stat(destination).st_mode) == 0o600
        ('new', True)
    """
    if os.name != "nt":
        try:
            mode = stat.S_IMODE(os.stat(destination).st_mode)
        except FileNotFoundError:
            pass  # New file: keep the umask-derived mode the temp file was created with.
        else:
            os.chmod(source, mode)
    os.replace(source, destination)


def _compute_file_md5(filepath: str) -> str:
    """Compute MD5 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        MD5 hash as hexadecimal string.
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _validate_file_md5(filepath: str, expected_md5: str) -> bool:
    """Validate that a file's MD5 hash matches the expected hash.

    Args:
        filepath: Path to the file.
        expected_md5: Expected MD5 hash.

    Returns:
        True if hash matches, False otherwise.
    """
    if not os.path.exists(filepath):
        return False

    actual_md5 = _compute_file_md5(filepath)
    return actual_md5.lower() == expected_md5.lower()


def _download_file(
    url: str,
    filename: str,
    expected_md5: str | None = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Download a file from a URL with optional MD5 validation.

    Args:
        url: URL to download from.
        filename: Path to save the file.
        expected_md5: Expected MD5 hash for validation (optional).
        timeout: Timeout in seconds passed to ``requests.get``.

    Raises:
        ValueError: If MD5 validation fails.
    """
    if os.path.exists(filename) and expected_md5:
        if _validate_file_md5(filename, expected_md5):
            logger.info(f"File {filename} already exists with correct MD5 hash. Skipping download.")
            return
        else:
            logger.warning(f"File {filename} exists but MD5 hash mismatch. Re-downloading...")
            os.remove(filename)

    with contextlib.closing(requests.get(url, stream=True, timeout=timeout)) as response:
        response.raise_for_status()
        total_size_header = response.headers.get("content-length")
        try:
            total_size = int(total_size_header) if total_size_header is not None else None
        except (TypeError, ValueError):
            total_size = None

        target_dir = os.path.dirname(filename) or "."
        fd, temp_filename = _mkstemp_default_mode(
            target_dir,
            prefix=f"{os.path.basename(filename)}.",
            suffix=".tmp",
        )
        try:
            with (
                os.fdopen(fd, "wb") as f,
                tqdm(desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        except Exception:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise

    if expected_md5:
        actual_md5 = _compute_file_md5(temp_filename)
        if actual_md5.lower() != expected_md5.lower():
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise ValueError("MD5 mismatch for %s (expected %s, got %s)." % (filename, expected_md5, actual_md5))
        else:
            logger.info(f"MD5 validation successful for {filename}")

    try:
        os.replace(temp_filename, filename)
    except Exception:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise
