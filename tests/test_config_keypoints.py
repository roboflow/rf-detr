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
)


def test_keypoint_config_defaults() -> None:
    """Default model/train keypoint configuration values should match the preview contract."""
    model = RFDETRKeypointPreviewConfig()
    train = KeypointTrainConfig(dataset_dir="/tmp")

    assert model.use_grouppose_keypoints is True
    assert model.dual_projector is True
    assert model.dual_projector_kp_only is True
    assert model.num_keypoints_per_class == [0, 17]
    assert model.positional_encoding_size == 576 // 12

    assert train.rle is True
    assert train.rle_conditional is True
    assert train.rle_loss_coef == pytest.approx(1.0)
    assert train.keypoint_l1_loss_coef == pytest.approx(1.0)
    assert train.keypoint_findable_loss_coef == pytest.approx(1.0)
    assert train.keypoint_visible_loss_coef == pytest.approx(1.0)
    assert train.keypoint_nll_loss_coef == pytest.approx(1.0)
    assert train.cls_loss_coef == pytest.approx(2.0)


def test_keypoint_preview_config_person_schema() -> None:
    """Person-keypoint preview config must expose a person-only schema."""
    model = RFDETRKeypointPreviewConfig()

    assert model.num_keypoints_per_class == [0, 17]
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
        rle=False,
        rle_conditional=False,
        rle_loss_coef=5.5,
    )

    namespace = _namespace_from_configs(model, train)

    assert namespace.use_grouppose_keypoints is True
    assert namespace.keypoint_cross_attn is True
    assert namespace.inter_instance_kp_attn is False
    assert namespace.grouppose_keypoint_dim_downscale == 1
    assert namespace.dual_projector is True
    assert namespace.dual_projector_kp_only is True
    assert namespace.num_keypoints_per_class == [0, 17]
    assert namespace.keypoint_flip_pairs == [0, 1, 2, 3]
    assert namespace.keypoint_l1_loss_coef == pytest.approx(1.5)
    assert namespace.keypoint_findable_loss_coef == pytest.approx(2.5)
    assert namespace.keypoint_visible_loss_coef == pytest.approx(3.5)
    assert namespace.keypoint_nll_loss_coef == pytest.approx(4.5)
    assert namespace.rle is False
    assert namespace.rle_conditional is False
    assert namespace.rle_loss_coef == pytest.approx(5.5)


def test_no_public_rle_hidden_dim_field() -> None:
    """rle_hidden_dim is intentionally not a public config field."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        RFDETRBaseConfig(num_classes=1, rle_hidden_dim=256)

    with pytest.raises(ValueError, match="Unknown parameter"):
        KeypointTrainConfig(dataset_dir="/tmp", rle_hidden_dim=256)
