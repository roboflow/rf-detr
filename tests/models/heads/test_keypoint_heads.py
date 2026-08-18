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


@pytest.mark.parametrize(
    ("channel", "loss_index"),
    [
        pytest.param(0, 0, id="x_coord_corrupts_location_loss"),
        pytest.param(2, 1, id="findable_logit_corrupts_findable_loss"),
        pytest.param(3, 2, id="visible_logit_corrupts_visible_loss"),
    ],
)
def test_compute_l1_keypoint_loss_ignores_single_nonfinite_prediction_channel(channel: int, loss_index: int) -> None:
    """A single non-finite prediction channel for one of several keypoints must not poison its loss term.

    Regression: location_loss/findable_loss/visible_loss each excluded non-finite positions by
    multiplying the per-keypoint loss by a float mask (0.0/1.0). ``0.0 * nan == nan`` under IEEE 754, so
    a masked-out NaN still propagated through the following ``.sum(-1)`` and poisoned the whole target's
    loss, defeating the ``isfinite`` check that built the mask in the first place. Fixed by swapping in
    a safe placeholder before the loss op (``torch.where``) as well as ``masked_fill`` on the result --
    see ``test_compute_l1_keypoint_loss_gradient_ignores_single_nonfinite_prediction_channel`` for why
    ``masked_fill`` on the output alone is not enough for the *gradient*.
    """
    torch.manual_seed(0)
    pred_keypoints = torch.randn(1, 17, 7)
    pred_keypoints[0, 5, channel] = float("nan")  # one of 17 keypoints goes non-finite on this channel
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0

    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[17],
    )

    assert torch.isfinite(losses[loss_index]).all(), (
        f"a single non-finite prediction on channel {channel} for 1 of 17 keypoints must not poison "
        f"losses[{loss_index}]"
    )


@pytest.mark.parametrize(
    "channel",
    [
        pytest.param(0, id="x_coord"),
        pytest.param(1, id="y_coord"),
        pytest.param(2, id="findable_logit"),
        pytest.param(3, id="visible_logit"),
        pytest.param(4, id="log_l11"),
        pytest.param(5, id="l21"),
        pytest.param(6, id="log_l22"),
    ],
)
def test_compute_l1_keypoint_loss_gradient_ignores_single_nonfinite_prediction_channel(channel: int) -> None:
    """A single non-finite prediction must not poison the *gradient* of any loss term, not just its value.

    Regression: masking a non-finite position by ``masked_fill``-ing the per-keypoint loss (the fix for
    ``test_compute_l1_keypoint_loss_ignores_single_nonfinite_prediction_channel`` above) makes the
    forward *value* finite, but not the gradient. Every loss op's own backward formula still evaluates
    its *original* input locally -- ``binary_cross_entropy_with_logits``'s gradient is
    ``sigmoid(x) - y``, so a non-finite ``x`` gives a non-finite local gradient there regardless of the
    output mask, and ``0.0`` (the correctly zeroed upstream gradient) ``* nan == nan`` propagates it into
    ``x.grad`` anyway -- the same IEEE 754 trap one level down. This also affects the Gaussian NLL term
    (through ``dx``/``dy``/the Cholesky params, several ops before its own ``nan_to_num`` call, which
    only cleans the exact node it is applied to). Fixed by swapping in a safe placeholder value
    (``torch.where``) before each loss op, so its local gradient formula never sees the non-finite input.
    """
    torch.manual_seed(0)
    pred_keypoints = torch.randn(1, 17, 7, requires_grad=True)
    with torch.no_grad():
        pred_keypoints[0, 5, channel] = float("nan")  # one of 17 keypoints goes non-finite on this channel
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0

    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[17],
    )
    sum(loss.sum() for loss in losses).backward()

    grad = pred_keypoints.grad
    assert grad is not None
    assert torch.isfinite(grad).all(), (
        f"a single non-finite prediction on channel {channel} for 1 of 17 keypoints must not poison "
        f"the gradient of any loss term; got grad={grad}"
    )


def test_compute_l1_keypoint_loss_gradient_ignores_log_precision_exp_overflow() -> None:
    """A finite log-precision input whose ``exp()`` overflows must not poison the NLL gradient.

    Regression: ``finite_uncertainty`` only checks the *raw* (pre-``exp``) Cholesky log-precision
    inputs, so a large-but-finite value like ``100.0`` passes it -- ``exp(100)`` then overflows to
    ``inf`` internally, several ops before the existing ``isfinite(u0)``/``isfinite(maha2)`` check that
    catches it and excludes it from the *forward* sum. By the time that check runs, ``u0``/``maha2``
    have already been computed from the overflowed ``l11``, so their own backward formula (e.g.
    ``d(u0**2)/d(u0) == 2*u0 == inf``) still poisons x/y/Cholesky gradients via ``0.0 * inf == nan``,
    the same IEEE 754 trap as the other three loss terms, just one exp() later. Fixed by refining the
    mask with this later-stage overflow and redoing the ``u0``/``u1``/``maha2`` chain with
    ``torch.where``-sanitized inputs before ``exp``/multiply/square see them.
    """
    pred_keypoints = torch.zeros(1, 1, 7, requires_grad=True)
    with torch.no_grad():
        pred_keypoints[0, 0, 4] = 100.0  # raw_log_l11 is finite; exp(100) overflows to inf
    target_keypoints = torch.tensor([[[0.3, -0.2, 2.0]]], dtype=torch.float32)

    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([1.0], dtype=torch.float32),
        num_keypoints_per_class=[1],
    )
    for loss in losses:
        assert torch.isfinite(loss).all()
    sum(loss.sum() for loss in losses).backward()

    grad = pred_keypoints.grad
    assert grad is not None
    assert torch.isfinite(grad).all(), f"exp() overflow of a finite log-precision input must not poison grad={grad}"


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


def test_compute_l1_keypoint_loss_skips_nonfinite_target_area() -> None:
    """A non-finite target area must not poison ``location_loss``/``nll_loss`` despite ``valid_area``.

    Regression: ``valid_area`` correctly excludes a non-finite area's keypoints from
    ``location_loss_mask``, and ``masked_fill`` correctly zeroes their numerator -- but
    ``location_loss``/``nll_raw`` still divide that (already-zeroed) numerator by
    ``area.clamp_min(area_eps)`` directly. ``torch.clamp_min(nan, eps)`` leaves ``nan`` unchanged
    (comparisons against ``nan`` are always false), so the division is ``0.0 / nan == nan``
    regardless of the mask. Fixed by sanitizing ``area`` itself (``torch.where`` on ``valid_area``)
    before either division.
    """
    pred_keypoints = torch.zeros(1, 17, 7, requires_grad=True)
    target_keypoints = torch.rand(1, 17, 3)
    target_keypoints[:, :, 2] = 2.0

    losses = compute_l1_keypoint_loss(
        all_pred_keypoints=pred_keypoints,
        target_keypoints=target_keypoints,
        target_classes=torch.tensor([0], dtype=torch.int64),
        target_areas=torch.tensor([float("nan")], dtype=torch.float32),
        num_keypoints_per_class=[17],
    )
    for loss in losses:
        assert torch.isfinite(loss).all(), f"a non-finite target area must not poison loss={loss}"
    sum(loss.sum() for loss in losses).backward()

    grad = pred_keypoints.grad
    assert grad is not None
    assert torch.isfinite(grad).all(), f"a non-finite target area must not poison grad={grad}"


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

    def test_class_index_out_of_range_zeros_stay_connected_to_graph(self) -> None:
        """Out-of-range guard must return graph-connected zeros for DDP correctness.

        Under DistributedDataParallel, a detached zero would leave the keypoint-head parameters without a gradient path
        on this batch, desyncing the gradient reducer across ranks. The returned zeros must therefore still be a
        function of the head output (grad present and numerically zero), not fresh leaves.
        """
        pred_keypoints = torch.randn(1, 17, 7, requires_grad=True)
        target_keypoints = torch.rand(1, 17, 3)
        target_keypoints[:, :, 2] = 2.0
        result = compute_l1_keypoint_loss(
            all_pred_keypoints=pred_keypoints,
            target_keypoints=target_keypoints,
            target_classes=torch.tensor([2], dtype=torch.int64),
            target_areas=torch.tensor([1.0], dtype=torch.float32),
            num_keypoints_per_class=[17],
        )

        for i, loss in enumerate(result):
            assert loss.requires_grad, f"Loss[{i}] must stay connected to the autograd graph"

        # Gradient must flow back to the head output (a fresh-leaf zero would raise here),
        # and it must be numerically zero so training is unaffected.
        total = sum(loss.sum() for loss in result)
        total.backward()
        assert pred_keypoints.grad is not None, "keypoint-head output received no gradient path"
        torch.testing.assert_close(pred_keypoints.grad, torch.zeros_like(pred_keypoints.grad))

    @pytest.mark.parametrize(
        "fill_value, dtype",
        [
            pytest.param(float("nan"), torch.float32, id="nan"),
            pytest.param(float("inf"), torch.float32, id="inf"),
            pytest.param(1000.0, torch.float16, id="fp16-overflow"),
        ],
    )
    def test_class_index_out_of_range_zeros_finite_under_nonfinite_predictions(
        self, fill_value: float, dtype: torch.dtype
    ) -> None:
        """Out-of-range guard zeros must stay finite even when predictions are NaN/Inf or overflow.

        ``all_pred_keypoints.sum() * 0.0`` yields NaN for non-finite predictions (``nan * 0 == nan``) or when a finite
        fp16 tensor overflows in ``.sum()`` (``inf * 0 == nan``). The value-independent empty reduction must instead
        return exactly-zero, finite losses while staying graph-connected.
        """
        pred_keypoints = torch.full((1, 17, 7), fill_value, dtype=dtype, requires_grad=True)
        target_keypoints = torch.rand(1, 17, 3)
        target_keypoints[:, :, 2] = 2.0
        result = compute_l1_keypoint_loss(
            all_pred_keypoints=pred_keypoints,
            target_keypoints=target_keypoints,
            target_classes=torch.tensor([2], dtype=torch.int64),
            target_areas=torch.tensor([1.0], dtype=torch.float32),
            num_keypoints_per_class=[17],
        )

        for i, loss in enumerate(result):
            assert loss.requires_grad, f"Loss[{i}] must stay connected to the autograd graph"
            assert torch.isfinite(loss).all(), f"Loss[{i}] must be finite, got {loss}"
            torch.testing.assert_close(loss, torch.zeros_like(loss))
