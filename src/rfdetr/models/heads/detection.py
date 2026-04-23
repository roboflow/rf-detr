# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection head: bounding-box regression + classification projections."""

import torch.nn as nn

from rfdetr.models.math import MLP


class DetectionHead(nn.Module):
    """Projection head for object detection outputs.

    Wraps the classification linear layer and bounding-box MLP used
    by the LWDETR decoder to produce final detection predictions.

    Args:
        hidden_dim: Feature dimension coming from the transformer decoder.
        num_classes: Number of object classes (excluding background).
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, hs):
        """Project decoder hidden states to class logits and box coordinates.

        Args:
            hs: Decoder output tensor of shape ``(B, N, hidden_dim)``.

        Returns:
            Tuple of ``(outputs_class, outputs_coord)`` where
            ``outputs_class`` has shape ``(B, N, num_classes)`` and
            ``outputs_coord`` has shape ``(B, N, 4)`` in ``[cx, cy, w, h]``
            normalised to ``[0, 1]``.
        """
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return outputs_class, outputs_coord
