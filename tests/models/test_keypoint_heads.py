# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.flows import RealNVP
from rfdetr.models.heads import ConditionalQueryInitializer
from rfdetr.models.heads.keypoints import compute_keypoint_matching_cost, compute_l1_keypoint_loss


def test_conditional_query_initializer_shape() -> None:
    """Initializer output should have expected batch/query/out dimensions."""
    torch.manual_seed(0)
    initializer = ConditionalQueryInitializer(dim=32, num_queries=11, out_dim=16)
    query_features = torch.randn(3, 32)
    queries = initializer(query_features)

    assert queries.shape == (3, 11, 16)


def test_conditional_query_initializer_zero_adaln_identity() -> None:
    """A zeroed AdaLN gate should make initializer return the unmodified learned queries."""
    torch.manual_seed(0)
    initializer = ConditionalQueryInitializer(dim=16, num_queries=5, out_dim=16)
    query_features = torch.randn(4, 16)
    output = initializer(query_features)
    expected = initializer.queries.unsqueeze(0).expand_as(output)

    assert torch.equal(output, expected)


def test_compute_l1_keypoint_loss_smoke() -> None:
    """Loss helper should emit five finite vectors with matching target batch shape."""
    pred_keypoints = torch.randn(3, 17, 7)
    target_keypoints = torch.rand(3, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    target_classes = torch.tensor([0, 0, 0], dtype=torch.int64)
    target_areas = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=target_classes,
        target_areas=target_areas,
        num_keypoints_per_class=[17],
    )

    assert len(losses) == 5
    for loss in losses:
        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()


def test_compute_l1_keypoint_loss_skips_visible_zero_area_rle_residuals() -> None:
    """Visible keypoints on zero-area targets should not send invalid residuals into RLE flow."""
    pred_keypoints = torch.zeros(1, 17, 7)
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([0.0], dtype=torch.float32),
        num_keypoints_per_class=[17],
        flow=RealNVP(),
    )

    for loss in losses:
        assert torch.isfinite(loss).all()


def test_compute_l1_keypoint_loss_rejects_missing_schema() -> None:
    """Missing keypoint schema should fail before producing zero supervision."""
    pred_keypoints = torch.randn(1, 17, 7)
    target_keypoints = torch.rand(1, 17, 3)

    with pytest.raises(ValueError, match="num_keypoints_per_class must be non-empty"):
        compute_l1_keypoint_loss(
            all_pred_keypoints=pred_keypoints,
            target_keypoints=target_keypoints,
            target_classes=torch.tensor([0], dtype=torch.int64),
            target_areas=torch.tensor([1.0], dtype=torch.float32),
            num_keypoints_per_class=[],
        )


def test_compute_keypoint_matching_cost_smoke() -> None:
    """Matching-cost helper should return a five-term cost tensor for each target."""
    all_pred_keypoints = torch.randn(2, 4, 17, 7)
    target_keypoints = torch.rand(2, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    target_classes = torch.tensor([0, 0], dtype=torch.int64)
    target_areas = torch.tensor([1.0, 2.0], dtype=torch.float32)
    cost_l1, cost_findable, cost_visible, cost_nll, cost_rle = compute_keypoint_matching_cost(
        all_pred_keypoints=all_pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=target_classes,
        target_areas=target_areas,
        num_keypoints_per_class=[17],
        flow=None,
    )

    assert cost_l1.shape == (2, 4, 2)
    assert cost_findable.shape == (2, 4, 2)
    assert cost_visible.shape == (2, 4, 2)
    assert cost_nll.shape == (2, 4, 2)
    assert cost_rle.shape == (2, 4, 2)
    assert torch.isfinite(cost_l1).all()
    assert torch.isfinite(cost_findable).all()
    assert torch.isfinite(cost_visible).all()
    assert torch.isfinite(cost_nll).all()
    assert torch.isfinite(cost_rle).all()


def test_compute_keypoint_matching_cost_skips_zero_area_rle_residuals() -> None:
    """Zero-area targets should not produce non-finite keypoint matching costs."""
    all_pred_keypoints = torch.zeros(1, 2, 17, 7)
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    costs = compute_keypoint_matching_cost(
        all_pred_keypoints=all_pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([0.0], dtype=torch.float32),
        num_keypoints_per_class=[17],
        flow=RealNVP(),
    )

    for cost in costs:
        assert torch.isfinite(cost).all()


def test_compute_keypoint_matching_cost_rejects_missing_schema() -> None:
    """Missing keypoint schema should fail before matcher costs become keypoint no-ops."""
    all_pred_keypoints = torch.randn(1, 2, 17, 7)
    target_keypoints = torch.rand(1, 17, 3)

    with pytest.raises(ValueError, match="num_keypoints_per_class must be non-empty"):
        compute_keypoint_matching_cost(
            all_pred_keypoints=all_pred_keypoints,
            target_keypoints=target_keypoints,
            target_classes=torch.tensor([0], dtype=torch.int64),
            target_areas=torch.tensor([1.0], dtype=torch.float32),
            num_keypoints_per_class=[],
            flow=None,
        )
