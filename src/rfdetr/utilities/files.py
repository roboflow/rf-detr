# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""File download and MD5 validation helpers."""

import hashlib
import os
import shutil

import requests
from tqdm.auto import tqdm

from rfdetr.utilities.logger import get_logger

logger = get_logger()
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0


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

    Uses a per-process temporary file (``filename.<pid>.tmp``) so that
    concurrent calls from multiple processes (e.g. ``torchrun`` workers)
    cannot corrupt each other's downloads.  The final rename is atomic on
    POSIX systems, so even if several processes download the file at the
    same time the result is always a complete, valid file.

    Args:
        url: URL to download from.
        filename: Path to save the file.
        expected_md5: Expected MD5 hash for validation (optional).
        timeout: Timeout in seconds passed to ``requests.get``.

    Raises:
        ValueError: If MD5 validation fails.
    """
    # Check if file exists and has correct hash
    if os.path.exists(filename) and expected_md5:
        if _validate_file_md5(filename, expected_md5):
            logger.info(f"File {filename} already exists with correct MD5 hash. Skipping download.")
            return
        else:
            logger.warning(f"File {filename} exists but MD5 hash mismatch. Re-downloading...")
            os.remove(filename)
    elif os.path.exists(filename):
        return

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_size_header = response.headers.get("content-length")
    try:
        total_size = int(total_size_header) if total_size_header is not None else None
    except (TypeError, ValueError):
        total_size = None

    # Use a per-process unique temp filename to avoid concurrent-write
    # corruption when multiple processes (e.g. torchrun workers) download
    # the same file simultaneously.  Each process writes to its own temp
    # file; the final shutil.move is atomic on POSIX, so whichever process
    # finishes last wins but the content is identical.
    temp_filename = f"{filename}.{os.getpid()}.tmp"
    try:
        with (
            open(temp_filename, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    except Exception:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise

    # Validate MD5 if expected hash is provided.
    if expected_md5:
        actual_md5 = _compute_file_md5(temp_filename)
        if actual_md5.lower() != expected_md5.lower():
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise ValueError("MD5 mismatch for %s (expected %s, got %s)." % (filename, expected_md5, actual_md5))
        else:
            logger.info(f"MD5 validation successful for {filename}")

    # shutil.move handles cross-device moves (e.g. tmpfs → ext4 on Colab).
    # If the destination already exists (another process finished first),
    # shutil.move still replaces it atomically, which is safe because the
    # content is identical across all parallel downloads.
    shutil.move(temp_filename, filename)
