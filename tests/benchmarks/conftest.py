# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import os
import shutil
import time
import zipfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Generator
from urllib.request import urlretrieve

import pytest

from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset
from rfdetr.util.utils import seed_all

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"

_COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download a zip file and safely extract it into the destination directory.

    Args:
        url: URL to a zip archive.
        dest_dir: Directory where the archive will be saved and extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / url.rsplit("/", 1)[-1]
    print(f"Downloading {url} ...")
    urlretrieve(url, str(zip_path))
    print(f"Extracting {zip_path} ...")
    dest_dir_resolved = dest_dir.resolve()
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for member in zf.infolist():
            if not member.filename:
                continue
            target_path = (dest_dir_resolved / member.filename).resolve()
            if not target_path.is_relative_to(dest_dir_resolved):
                raise RuntimeError(
                    f"Unsafe path detected in ZIP file: {member.filename!r}"
                )
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
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


@pytest.fixture(scope="session")
def download_coco_val() -> tuple[Path, Path]:
    """Download COCO val2017 images and annotations if not already present.

    Returns:
        Tuple containing the images root directory and annotations file path.
    """
    images_root = _DATA_DIR / "val2017"
    annotations_path = _DATA_DIR / "annotations" / "instances_val2017.json"

    lock_path = _DATA_DIR / ".coco_download.lock"
    with _download_lock(lock_path):
        if not images_root.exists():
            _download_and_extract(_COCO_URLS["val2017"], _DATA_DIR)
        if not annotations_path.exists():
            _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR)

    return images_root, annotations_path


@pytest.fixture(autouse=True)
def seed_everything(request: pytest.FixtureRequest) -> None:
    """Reset random, numpy, torch, and CUDA seeds before each test.

    Defaults to seed 7. Override per-test via indirect parametrize::

        @pytest.mark.parametrize("seed_everything", [42], indirect=True)
        def test_foo(seed_everything): ...

    Args:
        request: Pytest fixture request that may carry an overridden seed.
    """
    seed = request.param if hasattr(request, "param") else 7
    seed_all(seed)


@pytest.fixture(scope="session")
def synthetic_shape_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, Any, None]:
    """Build a synthetic COCO-style dataset on disk and clean it up after tests.

    Args:
        tmp_path_factory: Pytest factory for temporary directories.

    Yields:
        Path to the synthetic dataset directory.
    """
    seed_all()
    dataset_dir = tmp_path_factory.mktemp("synthetic_dataset")
    generate_coco_dataset(
        output_dir=str(dataset_dir),
        num_images=100,
        img_size=224,
        class_mode="shape",
        min_objects=3,
        max_objects=7,
        split_ratios=DatasetSplitRatios(train=0.8, val=0.2, test=0.0),
    )
    val_dir = dataset_dir / "val"
    valid_dir = dataset_dir / "valid"
    if val_dir.exists() and not valid_dir.exists():
        val_dir.rename(valid_dir)
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "_annotations.coco.json").write_text(
            (valid_dir / "_annotations.coco.json").read_text()
        )
        # Ensure test split has corresponding images referenced by the annotations
        for item in valid_dir.iterdir():
            if item.is_file() and item.name != "_annotations.coco.json":
                shutil.copy2(item, test_dir / item.name)
    yield dataset_dir
    shutil.rmtree(dataset_dir)
