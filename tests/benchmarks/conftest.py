# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path

import pytest

from rfdetr.datasets._develop import (
    _COCO_URLS,
    _coco_val_images_complete,
    _download_and_extract,
    _nonempty_file_exists,
)
from rfdetr.utilities.reproducibility import seed_all
from tests._online import is_online

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_COCO_HOST = "images.cocodataset.org"
_COCO_PORT = 80


def _ensure_coco_images(images_root: Path, asset: str) -> None:
    """Download a COCO image archive into the shared data directory unless it is already complete.

    Completeness is a file count, not a bare ``exists()`` check: a directory left half-extracted by
    an interrupted or concurrent run must not be mistaken for a finished download. The same
    predicate is handed to ``_download_and_extract`` so it can be re-checked under the asset lock.

    Args:
        images_root: Directory the images are extracted into.
        asset: Key into ``_COCO_URLS`` naming the archive to fetch.

    Example:
        >>> _ensure_coco_images.__name__
        '_ensure_coco_images'
    """

    def complete() -> bool:
        return _coco_val_images_complete(images_root)

    if not complete():
        _download_and_extract(_COCO_URLS[asset], _DATA_DIR, is_complete=complete)


def _ensure_coco_annotations(*annotation_paths: Path) -> None:
    """Download the COCO annotation archive unless every requested annotation file is already present.

    All COCO annotation JSONs ship in one archive, so a single download satisfies any combination of
    requested files. Emptiness is checked rather than mere existence, so a truncated extraction is
    retried instead of silently accepted.

    Args:
        *annotation_paths: Annotation JSON files that must exist and be non-empty.

    Example:
        >>> _ensure_coco_annotations.__name__
        '_ensure_coco_annotations'
    """

    def complete() -> bool:
        return all(_nonempty_file_exists(path) for path in annotation_paths)

    if not complete():
        _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR, is_complete=complete)


@pytest.fixture(scope="session")
def download_coco_val() -> tuple[Path, Path]:
    """Download COCO val2017 images and annotations if not already present.

    Returns:
        Tuple containing the images root directory and annotations file path.

    Example:
        >>> download_coco_val.__name__
        'download_coco_val'
    """
    if not is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping COCO val2017 benchmark tests.")

    images_root = _DATA_DIR / "val2017"
    annotations_path = _DATA_DIR / "annotations" / "instances_val2017.json"

    _ensure_coco_images(images_root, "val2017")
    _ensure_coco_annotations(annotations_path)

    return images_root, annotations_path


@pytest.fixture(scope="session")
def download_coco_val_keypoints() -> tuple[Path, Path]:
    """Prepare COCO val images plus person-keypoint annotations for benchmark tests.

    Example:
        >>> download_coco_val_keypoints.__name__
        'download_coco_val_keypoints'
    """
    if not is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping COCO keypoint benchmark tests.")

    images_root = _DATA_DIR / "val2017"
    keypoint_annotations = _DATA_DIR / "annotations" / "person_keypoints_val2017.json"

    _ensure_coco_images(images_root, "val2017")
    _ensure_coco_annotations(keypoint_annotations)

    return images_root, keypoint_annotations


@pytest.fixture(scope="session")
def download_coco_train_val_keypoints() -> Path:
    """Prepare full COCO train/val images plus person-keypoint annotations for release-qualification tests.

    Example:
        >>> download_coco_train_val_keypoints.__name__
        'download_coco_train_val_keypoints'
    """
    if not is_online(_COCO_HOST, _COCO_PORT):
        pytest.skip("Offline environment, skipping full COCO keypoint training validation.")

    _ensure_coco_images(_DATA_DIR / "train2017", "train2017")
    _ensure_coco_images(_DATA_DIR / "val2017", "val2017")
    _ensure_coco_annotations(
        _DATA_DIR / "annotations" / "person_keypoints_train2017.json",
        _DATA_DIR / "annotations" / "person_keypoints_val2017.json",
    )

    return _DATA_DIR


@pytest.fixture(autouse=True)
def seed_everything(request: pytest.FixtureRequest) -> None:
    """Reset random, numpy, torch, and CUDA seeds before each test.

    Defaults to seed 7. Override per-test via indirect parametrize::

        @pytest.mark.parametrize("seed_everything", [42], indirect=True)
        def test_foo(seed_everything): ...

    Args:
        request: Pytest fixture request that may carry an overridden seed.

    Example:
        >>> seed_everything.__name__
        'seed_everything'
    """
    seed = request.param if hasattr(request, "param") else 7
    seed_all(seed)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder tests to prioritize long-running training test before xdist distribution.

    This hook runs after collection but before xdist distributes tests to workers. By moving the training test to the
    front, we ensure it gets scheduled early, maximizing parallel resource utilization.

    Example:
        >>> pytest_collection_modifyitems.__name__
        'pytest_collection_modifyitems'
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
