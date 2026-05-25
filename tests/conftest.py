# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest

from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset
from rfdetr.utilities.reproducibility import seed_all


@pytest.fixture
def reset_build_namespace_warning_state() -> Generator[None, Any, None]:
    """Reset ``build_namespace`` deprecation call counters before each test.

    ``@deprecated(..., num_warns=1)`` emits only once per process by default.
    This fixture makes warning assertions deterministic regardless of test order.
    """

    from rfdetr._namespace import build_namespace

    state = build_namespace._state
    snapshot = (state.called, state.warned_calls, dict(state.warned_args))

    state.called = 0
    state.warned_calls = 0
    state.warned_args = {}
    try:
        yield
    finally:
        state.called = snapshot[0]
        state.warned_calls = snapshot[1]
        state.warned_args = snapshot[2]


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
        (test_dir / "_annotations.coco.json").write_text((valid_dir / "_annotations.coco.json").read_text())
        # Ensure test split has corresponding images referenced by the annotations
        for item in valid_dir.iterdir():
            if item.is_file() and item.name != "_annotations.coco.json":
                shutil.copy2(item, test_dir / item.name)
    yield dataset_dir
    shutil.rmtree(dataset_dir)


@pytest.fixture(scope="session")
def synthetic_shape_segmentation_dataset_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, Any, None]:
    """Build a synthetic COCO-style dataset with polygon segmentation annotations.

    Same layout as :func:`synthetic_shape_dataset_dir` but every annotation includes a ``segmentation`` polygon field so
    the dataset can be used to train or evaluate segmentation models.

    Args:
        tmp_path_factory: Pytest factory for temporary directories.

    Yields:
        Path to the synthetic segmentation dataset directory.
    """
    seed_all()
    dataset_dir = tmp_path_factory.mktemp("synthetic_seg_dataset")
    generate_coco_dataset(
        output_dir=str(dataset_dir),
        num_images=100,
        img_size=224,
        class_mode="shape",
        min_objects=3,
        max_objects=7,
        split_ratios=DatasetSplitRatios(train=0.8, val=0.2, test=0.0),
        with_segmentation=True,
    )
    val_dir = dataset_dir / "val"
    valid_dir = dataset_dir / "valid"
    if val_dir.exists() and not valid_dir.exists():
        val_dir.rename(valid_dir)
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "_annotations.coco.json").write_text((valid_dir / "_annotations.coco.json").read_text())
        for item in valid_dir.iterdir():
            if item.is_file() and item.name != "_annotations.coco.json":
                shutil.copy2(item, test_dir / item.name)
    yield dataset_dir
    shutil.rmtree(dataset_dir)
