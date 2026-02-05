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


@pytest.fixture(scope="session")
def synthetic_shape_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, Any, None]:
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
    yield dataset_dir
    shutil.rmtree(dataset_dir, ignore_errors=True)
