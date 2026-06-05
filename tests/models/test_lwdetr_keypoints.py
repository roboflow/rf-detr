# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for GroupPose keypoint output wiring in LWDETR."""

from unittest.mock import MagicMock

import torch
from torch import nn

from rfdetr.models.heads import ConditionalQueryInitializer
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def _build_feature_batch(batch_size: int, hidden_dim: int) -> list[NestedTensor]:
    return [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]


class _DummyKeypointDecoder(nn.Module):
    """Minimal decoder surface needed for keypoint schema resizing."""

    def __init__(self, hidden_dim: int, num_keypoints_per_class: list[int]) -> None:
        super().__init__()
        self.num_keypoints_per_class = num_keypoints_per_class
        self.keypoint_pos_embed = nn.Parameter(torch.randn(sum(num_keypoints_per_class), hidden_dim))
        self.register_buffer(
            "keypoint_class_mask",
            torch.zeros(1 + sum(num_keypoints_per_class), 1 + sum(num_keypoints_per_class), dtype=torch.bool),
        )


class _DummyKeypointTransformer(nn.Module):
    """Minimal transformer surface needed for LWDETR construction and keypoint schema resizing."""

    def __init__(self, hidden_dim: int, num_keypoints_per_class: list[int]) -> None:
        super().__init__()
        self.d_model = hidden_dim
        self.num_keypoints_per_class = num_keypoints_per_class
        self.decoder = _DummyKeypointDecoder(hidden_dim, num_keypoints_per_class)
        self.keypoint_query_initializer = ConditionalQueryInitializer(hidden_dim, sum(num_keypoints_per_class))
        self.keypoint_query_initializer_enc = ConditionalQueryInitializer(hidden_dim, sum(num_keypoints_per_class))


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
        torch.zeros(batch_size, num_queries, 17, hidden_dim),  # unused keypoint encoder hidden state
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
    assert "keypoint_hidden_states" not in outputs
    assert "pred_keypoints" in outputs["aux_outputs"][0]
    assert "keypoint_hidden_states" not in outputs["aux_outputs"][0]


def test_lwdetr_reinitialize_keypoint_head_updates_schema_dependent_state() -> None:
    """Keypoint schema reinit should resize masks and learned keypoint query embeddings."""
    hidden_dim = 8
    transformer = _DummyKeypointTransformer(hidden_dim=hidden_dim, num_keypoints_per_class=[17])
    model = LWDETR(
        backbone=MagicMock(),
        transformer=transformer,
        segmentation_head=None,
        num_classes=3,
        num_queries=2,
        aux_loss=False,
        group_detr=1,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )

    model.reinitialize_keypoint_head([2, 1])

    assert model.num_keypoints_per_class == [2, 1]
    assert model.get_num_keypoints_per_class() == [2, 1]
    assert model._kp_active_mask.shape == (2, 2)
    assert model._kp_active_mask.tolist() == [[True, True], [True, False]]
    assert transformer.num_keypoints_per_class == [2, 1]
    assert transformer.decoder.num_keypoints_per_class == [2, 1]
    assert transformer.decoder.keypoint_pos_embed.shape == (3, hidden_dim)
    assert transformer.decoder.keypoint_class_mask.shape == (4, 4)
    assert transformer.keypoint_query_initializer.queries.shape == (3, hidden_dim)
    assert transformer.keypoint_query_initializer_enc.queries.shape == (3, hidden_dim)


def test_lwdetr_reset_keypoint_gaussian_parameters_preserves_non_gaussian_rows() -> None:
    """Gaussian reset should only zero precision-Cholesky output rows on decoder and encoder keypoint heads."""
    hidden_dim = 8
    transformer = _DummyKeypointTransformer(hidden_dim=hidden_dim, num_keypoints_per_class=[17])
    model = LWDETR(
        backbone=MagicMock(),
        transformer=transformer,
        segmentation_head=None,
        num_classes=3,
        num_queries=2,
        aux_loss=False,
        group_detr=1,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )
    with torch.no_grad():
        model.keypoint_embed.layers[-1].weight.fill_(3.0)
        model.keypoint_embed.layers[-1].bias.fill_(4.0)
        model.transformer.enc_out_keypoint_embed[0].layers[-1].weight.fill_(5.0)
        model.transformer.enc_out_keypoint_embed[0].layers[-1].bias.fill_(6.0)

    model.reset_keypoint_gaussian_parameters()

    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[:4], torch.full((4, hidden_dim), 3.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[4:7], torch.zeros(3, hidden_dim))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[7:], torch.full((1, hidden_dim), 3.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[:4], torch.full((4,), 4.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[4:7], torch.zeros(3))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[7:], torch.full((1,), 4.0))
    torch.testing.assert_close(
        model.transformer.enc_out_keypoint_embed[0].layers[-1].weight[4:7], torch.zeros(3, hidden_dim)
    )
    torch.testing.assert_close(model.transformer.enc_out_keypoint_embed[0].layers[-1].bias[4:7], torch.zeros(3))


def test_lwdetr_get_num_keypoints_per_class_from_checkpoint() -> None:
    """Checkpoint keypoint schema should be recoverable from `_kp_active_mask`."""
    state_dict = {"_kp_active_mask": torch.tensor([[True, True], [True, False]])}

    assert LWDETR.get_num_keypoints_per_class_from_checkpoint(state_dict) == [2, 1]


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
