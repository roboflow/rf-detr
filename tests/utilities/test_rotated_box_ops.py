# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import torch

from rfdetr.utilities.rotated_box_ops import (
    box_cxcywha_to_corners,
    corners_to_cxcywha,
    gwd_loss,
    gwd_pairwise,
    kld_loss,
    normalize_angle,
    probiou,
)


class TestNormalizeAngle:
    def test_already_in_range(self) -> None:
        angles = torch.tensor([0.0, 0.5, 1.0, math.pi - 0.01])
        result = normalize_angle(angles)
        assert torch.allclose(result, angles, atol=1e-6)

    def test_negative_angles(self) -> None:
        result = normalize_angle(torch.tensor([-0.5]))
        assert 0 <= result.item() < math.pi

    def test_angles_beyond_pi(self) -> None:
        result = normalize_angle(torch.tensor([math.pi + 0.5]))
        expected = torch.tensor([0.5])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_pi_periodicity(self) -> None:
        angle = torch.tensor([0.3])
        shifted = torch.tensor([0.3 + math.pi])
        assert torch.allclose(normalize_angle(angle), normalize_angle(shifted), atol=1e-6)

    def test_two_pi(self) -> None:
        result = normalize_angle(torch.tensor([2 * math.pi]))
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)


class TestBoxCxcywhaToCorners:
    def test_axis_aligned_box(self) -> None:
        box = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.0]])
        corners = box_cxcywha_to_corners(box)
        assert corners.shape == (1, 4, 2)
        expected = torch.tensor([[[7.0, 18.0], [13.0, 18.0], [13.0, 22.0], [7.0, 22.0]]])
        assert torch.allclose(corners, expected, atol=1e-5)

    def test_90_degree_rotation(self) -> None:
        box = torch.tensor([[0.0, 0.0, 4.0, 2.0, math.pi / 2]])
        corners = box_cxcywha_to_corners(box)
        assert corners.shape == (1, 4, 2)
        xs = corners[0, :, 0]
        ys = corners[0, :, 1]
        assert torch.allclose(xs.min(), torch.tensor(-1.0), atol=1e-5)
        assert torch.allclose(xs.max(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(ys.min(), torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(ys.max(), torch.tensor(2.0), atol=1e-5)

    def test_batch_shape(self) -> None:
        boxes = torch.rand(5, 3, 5)
        corners = box_cxcywha_to_corners(boxes)
        assert corners.shape == (5, 3, 4, 2)

    def test_center_is_mean_of_corners(self) -> None:
        box = torch.tensor([[5.0, 10.0, 8.0, 3.0, 0.7]])
        corners = box_cxcywha_to_corners(box)
        center = corners[0].mean(dim=0)
        assert torch.allclose(center, torch.tensor([5.0, 10.0]), atol=1e-5)


class TestCornersToBoxCxcywha:
    def test_roundtrip(self) -> None:
        original = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        corners = box_cxcywha_to_corners(original)
        recovered = corners_to_cxcywha(corners)
        assert torch.allclose(recovered, original, atol=1e-4)

    def test_roundtrip_batch(self) -> None:
        original = torch.tensor(
            [
                [5.0, 5.0, 10.0, 4.0, 0.0],
                [15.0, 25.0, 8.0, 6.0, 1.0],
                [0.0, 0.0, 3.0, 7.0, 2.5],
            ]
        )
        corners = box_cxcywha_to_corners(original)
        recovered = corners_to_cxcywha(corners)
        assert torch.allclose(recovered, original, atol=1e-4)

    def test_roundtrip_axis_aligned(self) -> None:
        original = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]])
        corners = box_cxcywha_to_corners(original)
        recovered = corners_to_cxcywha(corners)
        assert torch.allclose(recovered, original, atol=1e-4)


class TestGwdLoss:
    def test_identical_boxes_near_zero(self) -> None:
        boxes = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        loss = gwd_loss(boxes, boxes)
        assert loss.shape == (1,)
        assert loss.item() < 0.01

    def test_different_boxes_positive(self) -> None:
        pred = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        target = torch.tensor([[15.0, 25.0, 8.0, 6.0, 1.0]])
        loss = gwd_loss(pred, target)
        assert loss.item() > 0

    def test_angle_boundary_symmetry(self) -> None:
        box_a = torch.tensor([[10.0, 10.0, 6.0, 4.0, 0.01]])
        box_b = torch.tensor([[10.0, 10.0, 6.0, 4.0, math.pi - 0.01]])
        loss = gwd_loss(box_a, box_b)
        assert loss.item() < 0.05

    def test_batch(self) -> None:
        pred = torch.rand(10, 5) * 10
        pred[..., 4] = pred[..., 4] % math.pi
        target = torch.rand(10, 5) * 10
        target[..., 4] = target[..., 4] % math.pi
        loss = gwd_loss(pred, target)
        assert loss.shape == (10,)

    def test_gradients_flow(self) -> None:
        pred = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]], requires_grad=True)
        target = torch.tensor([[15.0, 25.0, 8.0, 6.0, 1.0]])
        loss = gwd_loss(pred, target).sum()
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestKldLoss:
    def test_identical_boxes_zero(self) -> None:
        boxes = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        loss = kld_loss(boxes, boxes)
        assert loss.shape == (1,)
        assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-5)

    def test_different_boxes_positive(self) -> None:
        pred = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        target = torch.tensor([[15.0, 25.0, 8.0, 6.0, 1.0]])
        loss = kld_loss(pred, target)
        assert loss.item() > 0

    def test_angle_sensitivity_scales_with_aspect_ratio(self) -> None:
        thin_box = torch.tensor([[10.0, 10.0, 20.0, 2.0, 0.0]])
        thin_box_rotated = torch.tensor([[10.0, 10.0, 20.0, 2.0, 0.3]])
        square_box = torch.tensor([[10.0, 10.0, 5.0, 5.0, 0.0]])
        square_box_rotated = torch.tensor([[10.0, 10.0, 5.0, 5.0, 0.3]])

        loss_thin = kld_loss(thin_box, thin_box_rotated)
        loss_square = kld_loss(square_box, square_box_rotated)

        assert loss_thin.item() > loss_square.item()

    def test_gradients_flow(self) -> None:
        pred = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]], requires_grad=True)
        target = torch.tensor([[15.0, 25.0, 8.0, 6.0, 1.0]])
        loss = kld_loss(pred, target).sum()
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestProbiou:
    def test_identical_boxes_one(self) -> None:
        boxes = torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]])
        score = probiou(boxes, boxes)
        assert score.shape == (1,)
        assert torch.allclose(score, torch.tensor([1.0]), atol=1e-4)

    def test_far_apart_boxes_near_zero(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        target = torch.tensor([[1000.0, 1000.0, 2.0, 2.0, 0.0]])
        score = probiou(pred, target)
        assert score.item() < 0.01

    def test_range_zero_to_one(self) -> None:
        pred = torch.rand(20, 5) * 10 + 1
        pred[..., 4] = pred[..., 4] % math.pi
        target = torch.rand(20, 5) * 10 + 1
        target[..., 4] = target[..., 4] % math.pi
        scores = probiou(pred, target)
        assert (scores >= -0.01).all()
        assert (scores <= 1.01).all()

    def test_batch(self) -> None:
        pred = torch.rand(8, 5) * 10 + 1
        target = torch.rand(8, 5) * 10 + 1
        scores = probiou(pred, target)
        assert scores.shape == (8,)


class TestGwdPairwise:
    def test_output_shape(self) -> None:
        boxes1 = torch.rand(5, 5) * 10 + 1
        boxes2 = torch.rand(3, 5) * 10 + 1
        boxes1[..., 4] = boxes1[..., 4] % math.pi
        boxes2[..., 4] = boxes2[..., 4] % math.pi
        cost = gwd_pairwise(boxes1, boxes2)
        assert cost.shape == (5, 3)

    def test_diagonal_matches_paired(self) -> None:
        boxes = torch.tensor(
            [
                [10.0, 20.0, 6.0, 4.0, 0.5],
                [5.0, 5.0, 3.0, 7.0, 1.2],
            ]
        )
        cost_matrix = gwd_pairwise(boxes, boxes)
        paired = gwd_loss(boxes, boxes)
        assert torch.allclose(torch.diag(cost_matrix), paired, atol=1e-5)

    def test_self_cost_near_zero_on_diagonal(self) -> None:
        boxes = torch.tensor(
            [
                [10.0, 20.0, 6.0, 4.0, 0.5],
                [5.0, 5.0, 3.0, 7.0, 1.2],
            ]
        )
        cost = gwd_pairwise(boxes, boxes)
        assert torch.allclose(torch.diag(cost), torch.zeros(2), atol=0.01)


class TestEdgeCases:
    def test_zero_size_box_gwd_no_crash(self) -> None:
        pred = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.5]])
        target = torch.tensor([[5.0, 5.0, 3.0, 2.0, 0.5]])
        loss = gwd_loss(pred, target)
        assert torch.isfinite(loss).all()

    def test_zero_size_box_kld_no_crash(self) -> None:
        pred = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.5]])
        target = torch.tensor([[5.0, 5.0, 3.0, 2.0, 0.5]])
        loss = kld_loss(pred, target)
        assert torch.isfinite(loss).all()

    def test_zero_size_box_probiou_no_crash(self) -> None:
        pred = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.5]])
        target = torch.tensor([[5.0, 5.0, 3.0, 2.0, 0.5]])
        score = probiou(pred, target)
        assert torch.isfinite(score).all()

    def test_very_large_boxes(self) -> None:
        pred = torch.tensor([[500.0, 500.0, 1000.0, 800.0, 0.5]])
        target = torch.tensor([[500.0, 500.0, 1000.0, 800.0, 0.5]])
        assert gwd_loss(pred, target).item() < 0.01
        assert kld_loss(pred, target).item() < 0.01
        assert probiou(pred, target).item() > 0.99

    def test_single_element_tensors(self) -> None:
        pred = torch.tensor([5.0, 5.0, 3.0, 2.0, 0.5]).unsqueeze(0)
        target = torch.tensor([5.0, 5.0, 3.0, 2.0, 0.5]).unsqueeze(0)
        assert gwd_loss(pred, target).shape == (1,)
        assert kld_loss(pred, target).shape == (1,)
        assert probiou(pred, target).shape == (1,)
