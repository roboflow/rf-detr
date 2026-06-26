# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for keypoint config defaults and namespace forwarding."""

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    KeypointTrainConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    SegmentationTrainConfig,
)


def test_keypoint_config_defaults() -> None:
    """Default model/train keypoint configuration values should match the preview contract."""
    model = RFDETRKeypointPreviewConfig()
    train = KeypointTrainConfig(dataset_dir="/tmp")

    assert model.use_grouppose_keypoints is True
    assert model.dual_projector is True
    assert model.dual_projector_kp_only is True
    assert model.num_keypoints_per_class == [17]
    assert model.positional_encoding_size == 576 // 12

    assert train.keypoint_l1_loss_coef == pytest.approx(1.0)
    assert train.keypoint_findable_loss_coef == pytest.approx(1.0)
    assert train.keypoint_visible_loss_coef == pytest.approx(1.0)
    assert train.keypoint_nll_loss_coef == pytest.approx(1.0)
    assert train.cls_loss_coef == pytest.approx(2.0)


def test_keypoint_preview_config_person_schema() -> None:
    """Person-keypoint preview config must expose a person-only schema."""
    model = RFDETRKeypointPreviewConfig()

    assert model.num_keypoints_per_class == [17]
    assert sum(model.num_keypoints_per_class) == 17
    assert model.out_feature_indexes == [3, 6, 9, 12]
    assert model.num_windows == 2
    assert model.dec_layers == 4
    assert model.patch_size == 12
    assert model.resolution == 576
    assert model.pretrain_weights == "rf-detr-keypoint-preview-xlarge.pth"


def test_keypoint_fields_propagate_to_namespace(tmp_path) -> None:
    """All keypoint config fields are forwarded through _namespace_from_configs."""
    model = RFDETRKeypointPreviewConfig()
    train = KeypointTrainConfig(
        dataset_dir=str(tmp_path),
        keypoint_flip_pairs=[0, 1, 2, 3],
        keypoint_l1_loss_coef=1.5,
        keypoint_findable_loss_coef=2.5,
        keypoint_visible_loss_coef=3.5,
        keypoint_nll_loss_coef=4.5,
    )

    namespace = _namespace_from_configs(model, train)

    assert namespace.use_grouppose_keypoints is True
    assert namespace.keypoint_cross_attn is True
    assert namespace.inter_instance_kp_attn is False
    assert namespace.grouppose_keypoint_dim_downscale == 1
    assert namespace.dual_projector is True
    assert namespace.dual_projector_kp_only is True
    assert namespace.num_keypoints_per_class == [17]
    assert namespace.keypoint_flip_pairs == [0, 1, 2, 3]
    assert namespace.keypoint_l1_loss_coef == pytest.approx(1.5)
    assert namespace.keypoint_findable_loss_coef == pytest.approx(2.5)
    assert namespace.keypoint_visible_loss_coef == pytest.approx(3.5)
    assert namespace.keypoint_nll_loss_coef == pytest.approx(4.5)


def test_keypoint_nll_loss_coef_default_restored_to_1_0() -> None:
    """keypoint_nll_loss_coef must default to 1.0 after the 0.5 revert.

    The 0.5 default was introduced to dampen OKS@75 oscillation.  It was later reverted to 1.0 to align with all other
    keypoint loss terms (l1, findable, visible).  This test guards against silent regressions.
    """
    train = KeypointTrainConfig(dataset_dir="/tmp")
    assert train.keypoint_nll_loss_coef == pytest.approx(1.0)


def test_segmentation_train_config_cls_loss_coef_default() -> None:
    """SegmentationTrainConfig.cls_loss_coef must default to 1.0, not the erroneous 5.0.

    The 5.0 value was always present in SegmentationTrainConfig but was dead code pre-v1.7 (namespace builder read from
    ModelConfig=1.0).  The v1.7 TrainConfig ownership migration silently activated it.  This test guards against re-
    introducing that regression.
    """
    tc = SegmentationTrainConfig(dataset_dir="/tmp")
    assert tc.cls_loss_coef == pytest.approx(1.0)


def test_unknown_keypoint_fields_are_not_public_config_fields() -> None:
    """Private keypoint implementation fields are not accepted as public model config."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        RFDETRBaseConfig(num_classes=1, keypoint_private_hidden_dim=256)

    # KeypointTrainConfig (a TrainConfig subclass) uses extra="ignore" for Lightning
    # compatibility, so unknown kwargs are silently dropped rather than raising.
    kc = KeypointTrainConfig(
        dataset_dir="/tmp",
        keypoint_private_hidden_dim=256,
        keypoint_private_loss_coef=1.0,
    )
    assert not hasattr(kc, "keypoint_private_hidden_dim")
    assert not hasattr(kc, "keypoint_private_loss_coef")
