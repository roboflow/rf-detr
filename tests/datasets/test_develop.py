# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for developer dataset helpers."""

import torch
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.datasets._develop import _SimpleDataset


def test_simple_dataset_keeps_tensor_from_transform() -> None:
    """Dataset returns the tensor produced by a torchvision transform pipeline."""
    transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])

    image, target = _SimpleDataset(num_samples=1, transforms=transforms)[0]

    assert image.shape == torch.Size([3, 480, 640])
    assert image.dtype == torch.float32
    assert target["boxes"].shape == torch.Size([1, 4])
