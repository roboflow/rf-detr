# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Transforms shared by training and export/inference code."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torchvision.transforms import Normalize as _TVNormalize

from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh


class Normalize(object):
    """Normalize images and convert absolute xyxy boxes to normalized cxcywh boxes."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the image normalization transform.

        Args:
            mean: Per-channel mean values.
            std: Per-channel standard deviation values.
        """
        self._normalize = _TVNormalize(mean, std)

    def __call__(
        self, image: torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Normalize *image* and normalize target boxes when present.

        Args:
            image: Image tensor in ``[C, H, W]`` format.
            target: Optional target dictionary with absolute xyxy boxes.

        Returns:
            Tuple of normalized image and updated target.
        """
        image = self._normalize(image)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / boxes.new_tensor([w, h, w, h])
            target["boxes"] = boxes
        return image, target
