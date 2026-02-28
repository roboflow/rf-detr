# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for DatasetGridSaver — verifies that annotated grid images are written
without OpenCV layout errors across all supported OpenCV versions."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


class _FakeDataset:
    """Minimal dataset returning a single synthetic image + target."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        # CHW float tensor in ImageNet-normalised range
        image = torch.zeros(3, 224, 224)
        target = {
            "size": torch.tensor([224, 224]),
            "boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.6, 0.6, 0.2, 0.2]]),
            "labels": torch.tensor([0, 1]),
        }
        return image, target


def _collate(batch):
    import rfdetr.util.misc as utils

    images, targets = zip(*batch)
    # NestedTensor expected by DatasetGridSaver
    nested = utils.nested_tensor_from_tensor_list(list(images))
    return nested, list(targets)


def test_save_grid_writes_files(tmp_path: Path) -> None:
    """DatasetGridSaver must write JPEG grid files without raising OpenCV errors."""
    from rfdetr.datasets.save_grids import DatasetGridSaver

    dataset = _FakeDataset(num_samples=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)

    saver = DatasetGridSaver(loader, tmp_path, max_batches=2, dataset_type="train")
    saver.save_grid()

    grids = list(tmp_path.glob("train_batch*_grid.jpg"))
    assert len(grids) == 2, f"Expected 2 grid files, got {len(grids)}"
    for grid_path in grids:
        with Image.open(grid_path) as pil_img:
            img = np.array(pil_img)
        assert img.ndim == 3
        assert img.shape[2] == 3
