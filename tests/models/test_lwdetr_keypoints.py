# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for GroupPose keypoint output wiring in LWDETR."""

from unittest.mock import MagicMock

import torch

from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def _build_feature_batch(batch_size: int, hidden_dim: int) -> list[NestedTensor]:
    return [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]


def test_lwdetr_keypoint_forward_outputs() -> None:
    """GroupPose mode should expose keypoint tensors in model outputs."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 8
    num_classes = 6

    features = _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim)
    poss = [torch.zeros(batch_size, hidden_dim, 4, 4)]

    backbone = MagicMock()
    backbone.return_value = (features, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(2, batch_size, num_queries, hidden_dim),  # hs
        torch.zeros(2, batch_size, num_queries, 4),  # ref_unsigmoid
        torch.zeros(batch_size, num_queries, hidden_dim),  # hs_enc
        torch.zeros(batch_size, num_queries, 4),  # ref_enc
        torch.zeros(2, batch_size, num_queries, 17, hidden_dim),  # keypoint_hs
        torch.zeros(batch_size, num_queries, 17, 8),  # enc_kp_predictions
        torch.zeros(batch_size, num_queries, 17, hidden_dim),  # enc_kp_hidden
    )

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=True,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )

    outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
    assert outputs["pred_keypoints"].shape == (batch_size, num_queries, 17, 8)
    assert outputs["keypoint_hidden_states"].shape == (batch_size, num_queries, 17, hidden_dim)
    assert "pred_keypoints" in outputs["aux_outputs"][0]
    assert "keypoint_hidden_states" in outputs["aux_outputs"][0]


def test_lwdetr_default_detection_contract_unchanged() -> None:
    """Default detection mode should not expose keypoint outputs."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 8
    num_classes = 6

    features = _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim)
    poss = [torch.zeros(batch_size, hidden_dim, 4, 4)]

    backbone = MagicMock()
    backbone.return_value = (features, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(1, batch_size, num_queries, 4),
        torch.zeros(batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, 4),
    )

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
        use_grouppose_keypoints=False,
        num_keypoints_per_class=[],
        grouppose_keypoint_dim_downscale=1,
    )

    outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
    assert "pred_keypoints" not in outputs
    assert "keypoint_hidden_states" not in outputs
