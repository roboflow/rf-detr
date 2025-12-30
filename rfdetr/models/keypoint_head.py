# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Keypoint Head for RF-DETR Pose Estimation

Outputs (x, y, visibility) for each keypoint per detection query,
following YOLOv11's approach for pose estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class KeypointHead(nn.Module):
    """
    Predicts keypoints for each detection query.

    For each query, outputs num_keypoints * 3 values:
    - x coordinate (normalized 0-1)
    - y coordinate (normalized 0-1)
    - visibility logit (converted to confidence via sigmoid)

    Args:
        hidden_dim: Transformer embedding dimension
        num_keypoints: Number of keypoints to predict (default: 17 for COCO)
        num_layers: Number of MLP layers for each head
    """

    def __init__(
        self,
        hidden_dim: int,
        num_keypoints: int = 17,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim

        # MLP for keypoint coordinate regression
        # Output: num_keypoints * 2 (x, y for each keypoint)
        self.coord_head = MLP(hidden_dim, hidden_dim, num_keypoints * 2, num_layers)

        # Separate head for visibility prediction
        # Output: num_keypoints (visibility logit for each keypoint)
        self.visibility_head = MLP(hidden_dim, hidden_dim, num_keypoints, num_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights for better convergence."""
        # Initialize coordinate head to predict center initially
        nn.init.zeros_(self.coord_head.layers[-1].weight)
        nn.init.constant_(self.coord_head.layers[-1].bias, 0.5)

        # Initialize visibility head with slight negative bias (most keypoints start hidden)
        nn.init.zeros_(self.visibility_head.layers[-1].weight)
        nn.init.zeros_(self.visibility_head.layers[-1].bias)

    def forward(
        self,
        query_features: List[torch.Tensor],
        reference_boxes: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            query_features: List of query features from decoder layers
                           Each tensor shape: [B, num_queries, hidden_dim]
            reference_boxes: Optional reference boxes [B, num_queries, 4]
                            in cxcywh format for relative keypoint prediction

        Returns:
            List of keypoint predictions, one per decoder layer
            Each tensor shape: [B, num_queries, num_keypoints, 3]
            where 3 = (x, y, visibility_logit)
        """
        keypoint_outputs = []

        for qf in query_features:
            B, N, _ = qf.shape

            # Predict coordinates
            coords = self.coord_head(qf)  # [B, N, num_keypoints * 2]
            coords = coords.view(B, N, self.num_keypoints, 2)
            coords = coords.sigmoid()  # Normalize to [0, 1]

            # If reference boxes provided, make coordinates relative to box center
            # This allows the model to predict offsets from the detection box
            if reference_boxes is not None:
                cx, cy, w, h = reference_boxes.unbind(-1)
                # Convert keypoints from relative [0,1] within box to absolute [0,1]
                # Keypoint prediction of 0.5,0.5 means center of box
                kpt_x = cx.unsqueeze(-1) + (coords[..., 0] - 0.5) * w.unsqueeze(-1)
                kpt_y = cy.unsqueeze(-1) + (coords[..., 1] - 0.5) * h.unsqueeze(-1)
                coords = torch.stack([kpt_x, kpt_y], dim=-1)
                # Clamp to valid range
                coords = coords.clamp(0, 1)

            # Predict visibility (as logits, will be converted via sigmoid at inference)
            vis = self.visibility_head(qf)  # [B, N, num_keypoints]
            vis = vis.unsqueeze(-1)  # [B, N, num_keypoints, 1]

            # Combine: [B, N, num_keypoints, 3] where 3 = (x, y, visibility_logit)
            keypoints = torch.cat([coords, vis], dim=-1)
            keypoint_outputs.append(keypoints)

        return keypoint_outputs


# COCO keypoint constants for reference
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# Skeleton connections for COCO (pairs of keypoint indices)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
    [5, 11], [6, 12],  # torso to hips
    [5, 6],  # shoulders
    [5, 7], [6, 8], [7, 9], [8, 10],  # arms
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]  # face
]

# COCO keypoint sigmas for OKS calculation
# These control how strict the matching is for each keypoint
COCO_KEYPOINT_SIGMAS = [
    0.026,  # nose
    0.025, 0.025,  # eyes
    0.035, 0.035,  # ears
    0.079, 0.079,  # shoulders
    0.072, 0.072,  # elbows
    0.062, 0.062,  # wrists
    0.107, 0.107,  # hips
    0.087, 0.087,  # knees
    0.089, 0.089   # ankles
]

# Flip pairs for horizontal augmentation (left <-> right)
COCO_KEYPOINT_FLIP_PAIRS = [
    (1, 2),   # left_eye <-> right_eye
    (3, 4),   # left_ear <-> right_ear
    (5, 6),   # left_shoulder <-> right_shoulder
    (7, 8),   # left_elbow <-> right_elbow
    (9, 10),  # left_wrist <-> right_wrist
    (11, 12), # left_hip <-> right_hip
    (13, 14), # left_knee <-> right_knee
    (15, 16)  # left_ankle <-> right_ankle
]
