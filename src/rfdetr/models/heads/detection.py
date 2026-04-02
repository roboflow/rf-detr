# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection head: bounding-box regression + classification projections."""

import math

import torch
import torch.nn as nn

from rfdetr.models.math import MLP


class DetectionHead(nn.Module):
    """Projection head for object detection outputs.

    Wraps the classification linear layer and bounding-box MLP used
    by the LWDETR decoder to produce final detection predictions.

    Args:
        hidden_dim: Feature dimension coming from the transformer decoder.
        num_classes: Number of object classes (excluding background).
        oriented: If ``True``, add an angle prediction head for oriented
            bounding boxes.
    """

    def __init__(self, hidden_dim: int, num_classes: int, oriented: bool = False) -> None:
        super().__init__()
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.oriented = oriented
        self.angle_embed = MLP(hidden_dim, hidden_dim, 1, 3) if oriented else None

    def forward(self, hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project decoder hidden states to class logits and box coordinates.

        Args:
            hs: Decoder output tensor of shape ``(B, N, hidden_dim)``.

        Returns:
            Tuple of ``(outputs_class, outputs_coord)`` where
            ``outputs_class`` has shape ``(B, N, num_classes)`` and
            ``outputs_coord`` has shape ``(B, N, 4)`` or ``(B, N, 5)``
            when oriented. Box format is ``[cx, cy, w, h]`` normalised
            to ``[0, 1]``, with an optional angle in ``[0, pi)``.
        """
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        if self.angle_embed is not None:
            angle = self.angle_embed(hs).sigmoid() * math.pi
            outputs_coord = torch.cat([outputs_coord, angle], dim=-1)
        return outputs_class, outputs_coord
