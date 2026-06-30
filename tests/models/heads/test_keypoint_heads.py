# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.heads import ConditionalQueryInitializer
from rfdetr.models.heads.keypoints import (
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
)


def test_conditional_query_initializer_shape() -> None:
    """Initializer output should have expected batch/query/out dimensions."""
    initializer = ConditionalQueryInitializer(dim=32, num_queries=11, out_dim=16)
    query_features = torch.randn(3, 32)
    queries = initializer(query_features)

    assert queries.shape == (3, 11, 16)


def test_conditional_query_initializer_zero_adaln_identity() -> None:
    """A zeroed AdaLN gate should make initializer return the unmodified learned queries."""
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


def test_compute_l1_keypoint_loss_uses_raw_rflow_gaussian_nll() -> None:
    """Perfect keypoints should use raw r-flow NLL without a floor shift."""
    pred_keypoints = torch.zeros(1, 1, 7)
    pred_keypoints[:, :, 4] = 0.3
    pred_keypoints[:, :, 6] = -0.2
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)

    _, _, _, nll = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[1],
    )

    torch.testing.assert_close(nll, torch.tensor([-0.1]), rtol=1e-4, atol=1e-6)


def test_compute_l1_keypoint_loss_does_not_clamp_log_cholesky_nll() -> None:
    """Large precision log-diagonals should remain raw to match r-flow."""
    pred_keypoints = torch.zeros(1, 1, 7)
    pred_keypoints[:, :, 4] = 25.0
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)

    _, _, _, nll = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[1],
    )

    torch.testing.assert_close(nll, torch.tensor([-25.0]), rtol=1e-4, atol=1e-6)


def test_compute_l1_keypoint_loss_raw_nll_gradients_match_reference_formula() -> None:
    """The implemented NLL gradients should match the raw r-flow Gaussian formula."""
    pred_keypoints = torch.tensor([[[0.2, -0.1, 0.0, 0.0, 0.3, 0.1, -0.2]]], requires_grad=True)
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)
    target_areas = torch.tensor([1.0], dtype=torch.float32)
    _, _, _, nll = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=target_areas,
        num_keypoints_per_class=[1],
    )
    nll.sum().backward()
    grad = pred_keypoints.grad.detach().clone()

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

    torch.testing.assert_close(nll.detach(), raw_nll.detach().reshape(-1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad, raw_pred_keypoints.grad, rtol=1e-4, atol=1e-6)


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


def test_compute_keypoint_matching_cost_does_not_clamp_log_cholesky_nll() -> None:
    """Matching NLL should use raw precision log-diagonals to match r-flow."""
    all_pred_keypoints = torch.zeros(1, 1, 1, 7)
    all_pred_keypoints[:, :, :, 4] = 25.0
    target_keypoints = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)

    _, _, _, cost_nll = compute_keypoint_matching_cost(
        all_pred_keypoints=all_pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[1],
    )

    torch.testing.assert_close(cost_nll, torch.tensor([[[-25.0]]]), rtol=1e-4, atol=1e-6)


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


class TestComputeKeypointMatchingCostSmoke:
    """Group: compute_keypoint_matching_cost — shape and boundary checks."""

    def test_n_targets_zero_returns_four_empty_cost_tensors(self) -> None:
        """Empty target set should return four finite (B, Q, 0) cost tensors immediately."""
        b, q = 2, 4
        all_pred_keypoints = torch.randn(b, q, 17, 7)

        cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
            all_pred_keypoints=all_pred_keypoints,
            target_keypoints=torch.empty(0, 17, 3),
            target_classes=torch.empty(0, dtype=torch.int64),
            target_areas=torch.empty(0),
            num_keypoints_per_class=[17],
        )

        for cost, name in (
            (cost_l1, "cost_l1"),
            (cost_findable, "cost_findable"),
            (cost_visible, "cost_visible"),
            (cost_nll, "cost_nll"),
        ):
            assert cost.shape == (b, q, 0), f"{name}: expected shape ({b}, {q}, 0), got {cost.shape}"
            assert torch.isfinite(cost).all(), f"{name}: expected all-finite tensor, got non-finite values"


class TestComputeL1KeypointLossOobClass:
    """Group: compute_l1_keypoint_loss — out-of-range class index handling."""

    def test_class_index_out_of_range_returns_zero_losses_without_raising(self) -> None:
        """Out-of-range class index should emit a warning and return zeros, not raise."""
        pred_keypoints = torch.randn(1, 17, 7)
        target_keypoints = torch.rand(1, 17, 3)
        target_keypoints[:, :, 2] = 2.0
        # class index 2 is out of range for num_keypoints_per_class=[17] (only class 0 defined)
        result = compute_l1_keypoint_loss(
            all_pred_keypoints=pred_keypoints,
            target_keypoints=target_keypoints,
            target_classes=torch.tensor([2], dtype=torch.int64),
            target_areas=torch.tensor([1.0], dtype=torch.float32),
            num_keypoints_per_class=[17],
        )

        assert len(result) == 4, f"Expected 4-tuple, got {len(result)} elements"
        for i, loss in enumerate(result):
            assert loss.shape == (1,), f"Loss[{i}]: expected shape (1,), got {loss.shape}"
            torch.testing.assert_close(
                loss,
                torch.zeros(1),
                msg=f"Loss[{i}]: expected all zeros for out-of-range class, got {loss}",
            )
