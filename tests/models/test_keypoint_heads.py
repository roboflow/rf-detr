# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.heads import ConditionalQueryInitializer
from rfdetr.models.heads.keypoints import (
    KEYPOINT_LOG_CHOL_MAX,
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
)


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
    """Loss helper should emit four finite vectors with matching target batch shape."""
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

    assert len(losses) == 4
    for loss in losses:
        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()


def test_compute_l1_keypoint_loss_skips_visible_zero_area_nll_residuals() -> None:
    """Visible keypoints on zero-area targets should not produce non-finite Gaussian NLL."""
    pred_keypoints = torch.zeros(1, 17, 7)
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([0.0], dtype=torch.float32),
        num_keypoints_per_class=[17],
    )

    for loss in losses:
        assert torch.isfinite(loss).all()


def test_compute_l1_keypoint_loss_shifts_nll_floor_to_zero() -> None:
    """Perfect keypoints at max clamped precision should have zero shifted Gaussian NLL."""
    pred_keypoints = torch.zeros(1, 1, 7)
    pred_keypoints[:, :, 4] = KEYPOINT_LOG_CHOL_MAX
    pred_keypoints[:, :, 6] = KEYPOINT_LOG_CHOL_MAX
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)

    _, _, _, nll = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[1],
    )

    torch.testing.assert_close(nll, torch.zeros_like(nll), rtol=1e-4, atol=1e-6)


def test_compute_l1_keypoint_loss_nll_shift_preserves_gradients() -> None:
    """The constant NLL floor shift should not change gradients against the raw r-flow NLL."""
    pred_keypoints = torch.tensor([[[0.2, -0.1, 0.0, 0.0, 0.3, 0.1, -0.2]]], requires_grad=True)
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)
    target_areas = torch.tensor([1.0], dtype=torch.float32)
    _, _, _, shifted_nll = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=target_areas,
        num_keypoints_per_class=[1],
    )
    shifted_nll.sum().backward()
    shifted_grad = pred_keypoints.grad.detach().clone()

    raw_pred_keypoints = pred_keypoints.detach().clone().requires_grad_(True)
    dx = raw_pred_keypoints[:, :, 0] - target_keypoints[:, :, 0]
    dy = raw_pred_keypoints[:, :, 1] - target_keypoints[:, :, 1]
    log_l11 = raw_pred_keypoints[:, :, 4]
    l21 = raw_pred_keypoints[:, :, 5]
    log_l22 = raw_pred_keypoints[:, :, 6]
    u0 = log_l11.exp() * dx + l21 * dy
    u1 = log_l22.exp() * dy
    raw_nll = 0.5 * (u0 * u0 + u1 * u1) / target_areas.unsqueeze(1) - (log_l11 + log_l22)
    raw_nll.sum().backward()

    torch.testing.assert_close(shifted_grad, raw_pred_keypoints.grad, rtol=1e-4, atol=1e-6)


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
    """Matching-cost helper should return a four-term cost tensor for each target."""
    all_pred_keypoints = torch.randn(2, 4, 17, 7)
    target_keypoints = torch.rand(2, 17, 3)
    target_keypoints[:, :, 2] = 2.0
    target_classes = torch.tensor([0, 0], dtype=torch.int64)
    target_areas = torch.tensor([1.0, 2.0], dtype=torch.float32)
    cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
        all_pred_keypoints=all_pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=target_classes,
        target_areas=target_areas,
        num_keypoints_per_class=[17],
    )

    assert cost_l1.shape == (2, 4, 2)
    assert cost_findable.shape == (2, 4, 2)
    assert cost_visible.shape == (2, 4, 2)
    assert cost_nll.shape == (2, 4, 2)
    assert torch.isfinite(cost_l1).all()
    assert torch.isfinite(cost_findable).all()
    assert torch.isfinite(cost_visible).all()
    assert torch.isfinite(cost_nll).all()


def test_compute_keypoint_matching_cost_skips_zero_area_nll_residuals() -> None:
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
        )
