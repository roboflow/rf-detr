# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path

import pytest

from rfdetr.datasets._develop import (
    _COCO_URLS,
    _download_and_extract,
    _download_lock,
)
from rfdetr.util.utils import seed_all

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder tests to prioritize long-running training test before xdist distribution.

    This hook runs after collection but before xdist distributes tests to workers.
    By moving the training test to the front, we ensure it gets scheduled early,
    maximizing parallel resource utilization.
    """
    training_tests = []
    other_tests = []

    for item in items:
        # Prioritize the synthetic training convergence test
        if "training" in item.nodeid:
            training_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: training tests first, then everything else
    items[:] = training_tests + other_tests
