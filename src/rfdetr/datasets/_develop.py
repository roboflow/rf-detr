# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private developer tools for testing and benchmarking RF-DETR.

These utilities are intended for internal use by developers and test suites. They are not part of the public API and may
change without notice.
"""

from __future__ import annotations

import os
import shutil
import time
import zipfile
import zlib
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from rfdetr.utilities.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    import torch

_COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
_COCO_VAL_IMAGE_COUNT: int = 5000


def _coco_val_images_complete(images_dir: Path) -> bool:
    """Check whether the COCO val2017 image directory contains the expected number of JPEG files.

    Returns ``False`` for a missing or empty directory so callers can trigger a
    re-download without inspecting the directory manually.

    Args:
        images_dir: Path to the directory that should contain the val2017 images.

    Returns:
        ``True`` if *images_dir* exists and contains at least ``_COCO_VAL_IMAGE_COUNT``
        ``.jpg`` files, ``False`` otherwise.

    Raises:
        OSError: If *images_dir* exists but cannot be read (e.g. ``PermissionError``
            when the directory is not accessible, or ``FileNotFoundError`` on a
            TOCTOU race between the ``is_dir()`` check and ``iterdir()``).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _coco_val_images_complete(Path(tmpdir) / "val2017")
        False
    """
    if not images_dir.is_dir():
        return False
    return (
        sum(1 for entry in images_dir.iterdir() if entry.is_file() and entry.suffix.lower() == ".jpg")
        >= _COCO_VAL_IMAGE_COUNT
    )


def _nonempty_file_exists(path: Path) -> bool:
    """Check whether a file exists and contains at least one byte.

    Returns ``False`` for a missing or empty file so callers can trigger a
    re-download without inspecting the file manually.

    Args:
        path: Path to the file to check.

    Returns:
        ``True`` if *path* refers to an existing file with ``size > 0``,
        ``False`` otherwise.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _nonempty_file_exists(Path(tmpdir) / "missing.json")
        False
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def get_coco_download_url(asset: Literal["train2017", "val2017", "annotations"]) -> str:
    """Return the official COCO 2017 download URL for the requested asset."""
    return _COCO_URLS[asset]


class _SimpleDataset:
    """Simple synthetic dataset for testing augmentations and training loops.

    Creates synthetic images with varying numbers of bounding boxes to test edge cases in augmentation pipelines,
    particularly the case where num_boxes=2 (which matches orig_size shape [2]).

    Implements the ``__len__`` / ``__getitem__`` protocol expected by ``torch.utils.data.DataLoader`` without inheriting
    from ``torch.utils.data.Dataset``, so importing this class does not pull in torch at module load time.

    Args:
        num_samples: Number of samples in the dataset.
        transforms: Optional transforms to apply (e.g., Compose of AlbumentationsWrapper).

    Examples:
        >>> from albumentations import HorizontalFlip
        >>> from torchvision.transforms.v2 import Compose
        >>> from rfdetr.datasets.transforms import AlbumentationsWrapper
        >>>
        >>> transforms = Compose([
        ...     AlbumentationsWrapper(HorizontalFlip(p=0.5)),
        ... ])
        >>> dataset = _SimpleDataset(num_samples=10, transforms=transforms)
        >>> image, target = dataset[0]
    """

    def __init__(self, num_samples: int = 10, transforms: Any | None = None) -> None:
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        import torch  # lazy: keep torch off the module-load path (see class docstring)

        # Create synthetic image
        image: Image.Image | torch.Tensor = Image.new("RGB", (640, 480))

        # Create synthetic target with varying number of boxes
        # Cycles through 1, 2, and 3 boxes to test different edge cases
        num_boxes = (idx % 3) + 1

        boxes = []
        labels = []
        for i in range(num_boxes):
            x1 = 10 + i * 100
            y1 = 10 + i * 50
            x2 = x1 + 80
            y2 = y1 + 100
            boxes.append([x1, y1, x2, y2])
            labels.append(i + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "orig_size": torch.tensor([480, 640]),
            "size": torch.tensor([480, 640]),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([100.0] * num_boxes),
            "iscrowd": torch.tensor([0] * num_boxes),
        }

        # Apply transforms if any
        if self.transforms:
            image, target = self.transforms(image, target)

        if isinstance(image, torch.Tensor):
            return image, target

        # Convert PIL Image to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target


_DOWNLOAD_MAX_ATTEMPTS: int = 3


def _fetch_zip(url: str, zip_path: Path) -> None:
    """Download *url* to *zip_path*, verifying the transfer is complete.

    The downloaded size is checked against the server's ``Content-Length`` header when it
    is provided. A truncated transfer is the usual cause of a corrupt deflate stream
    (``zlib.error: invalid code lengths set``), so it is rejected here before extraction.

    Args:
        url: URL to a zip archive.
        zip_path: Local path the archive is written to.

    Raises:
        zipfile.BadZipFile: If the downloaded size does not match ``Content-Length``.

    Example:
        >>> _fetch_zip.__name__
        '_fetch_zip'
    """
    _, headers = urlretrieve(url, str(zip_path))
    expected_raw = headers.get("Content-Length")
    actual = zip_path.stat().st_size
    try:
        expected = int(expected_raw) if expected_raw is not None else None
    except (TypeError, ValueError):
        expected = None
    if expected is not None and actual != expected:
        raise zipfile.BadZipFile(f"Truncated download for {url!r}: got {actual} bytes, expected {expected}")


def _extract_zip(zip_path: Path, dest_dir_resolved: Path) -> None:
    """Extract *zip_path* into *dest_dir_resolved*, guarding against path traversal.

    Args:
        zip_path: Path to a zip archive on disk.
        dest_dir_resolved: Resolved destination directory members are written under.

    Raises:
        RuntimeError: If a member would escape *dest_dir_resolved*.

    Example:
        >>> _extract_zip.__name__
        '_extract_zip'
    """
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for member in zf.infolist():
            if not member.filename:
                continue
            target_path = (dest_dir_resolved / member.filename).resolve()
            if not target_path.is_relative_to(dest_dir_resolved):
                raise RuntimeError(f"Unsafe path detected in ZIP file: {member.filename!r}")
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _asset_lock_path(dest_dir: Path, url: str) -> Path:
    """Return the lock file that serializes downloads of *url* into *dest_dir*.

    The lock is keyed on the asset, not on the caller, so every consumer of a given URL
    contends on the same lock no matter which fixture or helper invoked the download.

    Args:
        dest_dir: Directory the archive is downloaded into.
        url: URL to a zip archive.

    Returns:
        Path to the lock file guarding this asset.

    Example:
        >>> _asset_lock_path(Path("/data"), "http://example.com/val2017.zip").name
        '.val2017.zip.lock'
    """
    return dest_dir / f".{url.rsplit('/', 1)[-1]}.lock"


def _download_and_extract(url: str, dest_dir: Path, is_complete: Callable[[], bool] | None = None) -> None:
    """Download a zip archive and safely extract it, retrying on a corrupt transfer.

    Downloads of the same asset are serialized on a per-asset lock, so concurrent callers
    (for example ``pytest-xdist`` workers sharing one data directory) can never write or
    extract the same archive at the same time. Overlapping writes are what produce a
    mid-stream ``EOFError``: a second ``open("wb")`` truncates an archive whose central
    directory another reader has already parsed.

    Truncated downloads and corrupt deflate streams (``zlib.error``/``BadZipFile``/``EOFError``)
    are retried up to ``_DOWNLOAD_MAX_ATTEMPTS`` times. The archive is staged under a
    process-unique name and always removed, both on failure (so a corrupt file never persists
    for the next run) and on success.

    Args:
        url: URL to a zip archive.
        dest_dir: Directory where the archive will be saved and extracted.
        is_complete: Optional predicate re-checked *after* the lock is acquired. When it returns
            ``True`` the download is skipped, so a caller that queued behind another process does
            not redundantly re-fetch an asset that process just finished extracting.

    Raises:
        zipfile.BadZipFile: If every attempt yields a truncated or corrupt archive.
        zlib.error: If the deflate stream is corrupt on the final attempt.
        EOFError: If the archive is truncated mid-stream on the final attempt.
        RuntimeError: If a member would escape *dest_dir* (path traversal — not retried).

    Example:
        >>> _download_and_extract.__name__
        '_download_and_extract'
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dir_resolved = dest_dir.resolve()
    with _download_lock(_asset_lock_path(dest_dir, url)):
        # Double-checked: another process may have finished this asset while we waited.
        if is_complete is not None and is_complete():
            logger.info("Skipping %s; already present in %s.", url, dest_dir)
            return
        # Process-unique staging name so two processes never share one archive path.
        zip_path = dest_dir / f"{url.rsplit('/', 1)[-1]}.{os.getpid()}.part"
        try:
            for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
                try:
                    logger.info("Downloading %s (attempt %d/%d) ...", url, attempt, _DOWNLOAD_MAX_ATTEMPTS)
                    _fetch_zip(url, zip_path)
                    logger.info("Extracting %s ...", zip_path)
                    _extract_zip(zip_path, dest_dir_resolved)
                    break
                except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
                    with suppress(FileNotFoundError):
                        zip_path.unlink()
                    # Fail fast on common non-retryable local filesystem errors.
                    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {13, 28}:
                        raise
                    if attempt == _DOWNLOAD_MAX_ATTEMPTS:
                        raise
                    logger.warning("Download/extract of %s failed (%s); retrying ...", url, exc)
                    time.sleep(2.0 * attempt)
        finally:
            with suppress(FileNotFoundError):
                zip_path.unlink()


@contextmanager
def _download_lock(lock_path: Path, timeout_s: float = 600.0, poll_s: float = 0.5) -> Generator[None, Any, None]:
    """Provide a simple cross-process lock using an exclusive lock file.

    Args:
        lock_path: Path to the lock file used for mutual exclusion.
        timeout_s: Maximum time in seconds to wait for the lock.
        poll_s: Sleep interval in seconds between lock attempts.

    Yields:
        None. The caller runs inside the locked region.

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout.

    Note:
        Lock identity is file existence (``O_CREAT | O_EXCL``), not a held file descriptor.
        A SIGKILL-terminated process will leak the lock file until ``timeout_s`` expires.
        If all worker processes have exited but the lock persists, remove it manually:
        ``rm <lock_path>``.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            # Atomic create; raises FileExistsError if another worker owns the lock.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        # Best-effort cleanup if the lock file was already removed.
        with suppress(FileNotFoundError):
            os.unlink(lock_path)
