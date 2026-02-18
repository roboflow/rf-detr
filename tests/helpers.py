# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Test helper utilities and classes."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class _SimpleDataset(Dataset):
    """Simple synthetic dataset for testing augmentations and training loops.

    Creates synthetic images with varying numbers of bounding boxes to test
    edge cases in augmentation pipelines, particularly the case where
    num_boxes=2 (which matches orig_size shape [2]).

    Args:
        num_samples: Number of samples in the dataset.
        transforms: Optional transforms to apply (e.g., Compose of AlbumentationsWrapper).

    Examples:
        >>> from rfdetr.datasets.transforms import AlbumentationsWrapper, Compose
        >>> import albumentations as A
        >>>
        >>> transforms = Compose([
        ...     AlbumentationsWrapper(A.HorizontalFlip(p=0.5)),
        ... ])
        >>> dataset = _SimpleDataset(num_samples=10, transforms=transforms)
        >>> image, target = dataset[0]
    """

    def __init__(self, num_samples: int = 10, transforms: Optional[Any] = None) -> None:
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        # Create synthetic image
        image = Image.new('RGB', (640, 480))

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
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'orig_size': torch.tensor([480, 640]),
            'size': torch.tensor([480, 640]),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([100.0] * num_boxes),
            'iscrowd': torch.tensor([0] * num_boxes)
        }

        # Apply transforms if any
        if self.transforms:
            image, target = self.transforms(image, target)

        # Convert PIL Image to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target
