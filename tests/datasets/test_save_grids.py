# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DatasetGridSaver — verifies that annotated grid images are written without OpenCV layout errors across all
supported OpenCV versions."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


class _FakeDataset:
    """Minimal dataset returning a single synthetic image + target."""

    def __init__(self, num_samples: int = 4, include_masks: bool = False) -> None:
        self.num_samples = num_samples
        self.include_masks = include_masks

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
        if self.include_masks:
            masks = torch.zeros((2, 224, 224), dtype=torch.bool)
            masks[0, 48:112, 48:112] = True
            masks[1, 120:168, 120:168] = True
            target["masks"] = masks
        return image, target


def _collate(batch):
    from rfdetr.utilities import nested_tensor_from_tensor_list

    images, targets = zip(*batch)
    # NestedTensor expected by DatasetGridSaver
    nested = nested_tensor_from_tensor_list(list(images))
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


def test_save_grid_passes_masks_to_supervision(tmp_path: Path, monkeypatch) -> None:
    """Segmentation targets should be forwarded to supervision as instance masks."""
    from rfdetr.datasets import save_grids
    from rfdetr.datasets.save_grids import DatasetGridSaver

    captured_masks = []

    class _FakeMaskAnnotator:
        def __init__(self, opacity: float) -> None:
            self.opacity = opacity

        def annotate(self, scene, detections):
            captured_masks.append(detections.mask.copy())
            return scene

    monkeypatch.setattr(save_grids, "MaskAnnotator", _FakeMaskAnnotator)

    dataset = _FakeDataset(num_samples=1, include_masks=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=_collate)

    saver = DatasetGridSaver(loader, tmp_path, max_batches=1, dataset_type="train")
    saver.save_grid()

    assert len(captured_masks) == 1
    assert captured_masks[0].shape == (2, 224, 224)
    assert captured_masks[0].dtype == bool
    assert captured_masks[0][0, 48:112, 48:112].all()
