# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Keypoint detection head for RF-DETR models."""

import torch
import torch.nn as nn

from rfdetr.models.math import MLP


class KeypointHead(nn.Module):
    """MLP-based keypoint head that predicts (x, y, visibility) per keypoint per query.

    Takes decoder query features and regresses keypoint locations and visibility
    scores. Output coordinates are in normalized image space [0, 1].

    Args:
        hidden_dim: Dimension of the decoder query features.
        num_keypoints: Number of keypoints to predict per instance.
        num_layers: Number of layers in the MLP.
    """

    def __init__(self, hidden_dim: int, num_keypoints: int, num_layers: int = 3) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        # Output: num_keypoints * 3 values — (x, y) logits + visibility logit per keypoint
        self.mlp = MLP(hidden_dim, hidden_dim, num_keypoints * 3, num_layers)

    def forward(self, hs: torch.Tensor) -> list[torch.Tensor]:
        """Predict keypoints from decoder query features.

        Args:
            hs: Decoder outputs of shape ``(num_layers, B, N, hidden_dim)``.

        Returns:
            List of keypoint tensors, one per decoder layer, each of shape
            ``(B, N, num_keypoints, 3)`` where the last dim is (x, y, visibility).
            The (x, y) values have sigmoid applied; visibility is a raw logit.
        """
        outputs = []
        for layer_hs in hs:
            # layer_hs: (B, N, hidden_dim)
            raw = self.mlp(layer_hs)  # (B, N, num_keypoints * 3)
            raw = raw.reshape(*raw.shape[:-1], self.num_keypoints, 3)
            # Apply sigmoid to xy coordinates to keep them in [0, 1]
            xy = raw[..., :2].sigmoid()
            vis_logit = raw[..., 2:3]
            kpts = torch.cat([xy, vis_logit], dim=-1)  # (B, N, num_keypoints, 3)
            outputs.append(kpts)
        return outputs
