# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests proving LWDETR forward contract survives joiner return-shape changes."""

from unittest.mock import MagicMock

import torch

from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def test_lwdetr_default_detection_forward_after_backbone_change() -> None:
    """LWDETR should accept the updated 3-tuple backbone output in non-keypoint mode."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 4
    num_classes = 7

    features = [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]
    poss = [torch.zeros(batch_size, 4, 4, dtype=torch.bool)]

    backbone = MagicMock()
    backbone.return_value = (features, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer_out = (
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, hidden_dim),
    )
    transformer.return_value = transformer_out

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    )

    outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
