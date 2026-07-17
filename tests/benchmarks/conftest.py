# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import socket
from pathlib import Path

import pytest

from rfdetr.datasets._develop import (
    _COCO_URLS,
    _coco_val_images_complete,
    _download_and_extract,
    _download_lock,
    _nonempty_file_exists,
)
from rfdetr.utilities.reproducibility import seed_all

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_COCO_HOST = "images.cocodataset.org"
_COCO_PORT = 80


def _is_online(host: str, port: int, timeout_s: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def download_coco_val() -> tuple[Path, Path]:
    """Download COCO val2017 images and annotations if not already present.

    Returns:
        Tuple containing the images root directory and annotations file path.
    """
    if not _is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping COCO val2017 benchmark tests.")

    images_root = _DATA_DIR / "val2017"
    annotations_path = _DATA_DIR / "annotations" / "instances_val2017.json"

    lock_path = _DATA_DIR / ".coco_download.lock"
    with _download_lock(lock_path):
        if not _coco_val_images_complete(images_root):
            _download_and_extract(_COCO_URLS["val2017"], _DATA_DIR)
        if not _nonempty_file_exists(annotations_path):
            _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR)

    return images_root, annotations_path


@pytest.fixture(scope="session")
def download_coco_val_keypoints() -> tuple[Path, Path]:
    """Prepare COCO val images plus person-keypoint annotations for benchmark tests."""
    if not _is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping COCO keypoint benchmark tests.")

    images_root = _DATA_DIR / "val2017"
    keypoint_annotations = _DATA_DIR / "annotations" / "person_keypoints_val2017.json"

    lock_path = _DATA_DIR / ".coco_keypoint_download.lock"
    with _download_lock(lock_path):
        if not images_root.exists():
            _download_and_extract(_COCO_URLS["val2017"], _DATA_DIR)
        if not keypoint_annotations.exists():
            _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR)

    return images_root, keypoint_annotations


@pytest.fixture(scope="session")
def download_coco_train_val_keypoints() -> Path:
    """Prepare full COCO train/val images plus person-keypoint annotations for release-qualification tests."""
    if not _is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping full COCO keypoint training validation.")

    lock_path = _DATA_DIR / ".coco_keypoint_train_val_download.lock"
    with _download_lock(lock_path):
        if not (_DATA_DIR / "train2017").exists():
            _download_and_extract(_COCO_URLS["train2017"], _DATA_DIR)
        if not (_DATA_DIR / "val2017").exists():
            _download_and_extract(_COCO_URLS["val2017"], _DATA_DIR)
        if (
            not (_DATA_DIR / "annotations" / "person_keypoints_train2017.json").exists()
            or not (_DATA_DIR / "annotations" / "person_keypoints_val2017.json").exists()
        ):
            _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR)

    return _DATA_DIR


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder tests to prioritize long-running training test before xdist distribution.

    This hook runs after collection but before xdist distributes tests to workers. By moving the training test to the
    front, we ensure it gets scheduled early, maximizing parallel resource utilization.
    """
    training_tests = []
    other_tests = []

    for item in items:
        # Prioritize the synthetic training convergence tests (detection + segmentation)
        if "training" in item.nodeid:
            training_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: training tests first, then everything else
    items[:] = training_tests + other_tests
