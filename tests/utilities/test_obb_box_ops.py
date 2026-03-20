# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for oriented bounding box operations."""

import math

import torch

from rfdetr.utilities.box_ops import (
    circular_angle_loss,
    corners_to_obb,
    obb_to_corners,
)


class TestCornersToObb:
    """Tests for corners_to_obb conversion."""

    def test_axis_aligned_box(self) -> None:
        """An axis-aligned rectangle should have angle=0."""
        # Rectangle with corners at (0,0), (4,0), (4,3), (0,3)
        corners = torch.tensor([[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]])
        obb = corners_to_obb(corners)
        assert obb.shape == (1, 5)
        torch.testing.assert_close(obb[0, 0], torch.tensor(2.0))  # cx
        torch.testing.assert_close(obb[0, 1], torch.tensor(1.5))  # cy
        torch.testing.assert_close(obb[0, 2], torch.tensor(4.0))  # w
        torch.testing.assert_close(obb[0, 3], torch.tensor(3.0))  # h
        # angle should be 0 (or very close)
        assert obb[0, 4].item() < 0.01

    def test_rotated_box(self) -> None:
        """A box rotated by 45 degrees."""
        s = math.sqrt(2) / 2
        # Square of side 2 rotated 45 degrees, centered at origin
        corners = torch.tensor([[0.0, -s * 2, s * 2, 0.0, 0.0, s * 2, -s * 2, 0.0]])
        obb = corners_to_obb(corners)
        assert abs(obb[0, 4].item() - math.pi / 4) < 0.01

    def test_batch_conversion(self) -> None:
        """Batch of corners should produce batch of OBBs."""
        corners = torch.rand(5, 8)
        obb = corners_to_obb(corners)
        assert obb.shape == (5, 5)


class TestObbToCorners:
    """Tests for obb_to_corners conversion."""

    def test_axis_aligned_box(self) -> None:
        """An OBB with angle=0 should produce axis-aligned corners."""
        obb = torch.tensor([[2.0, 1.5, 4.0, 3.0, 0.0]])
        corners = obb_to_corners(obb)
        assert corners.shape == (1, 8)
        # Check that corners form a valid rectangle
        pts = corners.reshape(1, 4, 2)
        # Width should be 4
        edge_w = (pts[0, 1] - pts[0, 0]).norm()
        torch.testing.assert_close(edge_w, torch.tensor(4.0))
        # Height should be 3
        edge_h = (pts[0, 2] - pts[0, 1]).norm()
        torch.testing.assert_close(edge_h, torch.tensor(3.0))

    def test_roundtrip(self) -> None:
        """Converting OBB -> corners -> OBB should be identity."""
        obb_orig = torch.tensor([[10.0, 20.0, 8.0, 5.0, 0.5]])
        corners = obb_to_corners(obb_orig)
        obb_recovered = corners_to_obb(corners)
        torch.testing.assert_close(obb_orig, obb_recovered, atol=1e-5, rtol=1e-5)

    def test_roundtrip_batch(self) -> None:
        """Roundtrip should work for batches."""
        obb_orig = torch.tensor(
            [
                [10.0, 20.0, 8.0, 5.0, 0.0],
                [5.0, 5.0, 3.0, 2.0, math.pi / 4],
                [0.0, 0.0, 1.0, 1.0, math.pi / 2],
            ]
        )
        corners = obb_to_corners(obb_orig)
        obb_recovered = corners_to_obb(corners)
        torch.testing.assert_close(obb_orig, obb_recovered, atol=1e-5, rtol=1e-5)


class TestCircularAngleLoss:
    """Tests for circular angle loss."""

    def test_zero_difference(self) -> None:
        """Same angles should give zero loss."""
        pred = torch.tensor([0.0, 0.5, 0.9])
        target = torch.tensor([0.0, 0.5, 0.9])
        loss = circular_angle_loss(pred, target)
        torch.testing.assert_close(loss, torch.zeros(3))

    def test_wrapping(self) -> None:
        """Angles near 0 and 1 should have small loss (wrapping)."""
        pred = torch.tensor([0.01])
        target = torch.tensor([0.99])
        loss = circular_angle_loss(pred, target)
        # Distance should be 0.02, not 0.98
        torch.testing.assert_close(loss, torch.tensor([0.02]))

    def test_max_distance(self) -> None:
        """Maximum distance should be 0.5 (90 degrees apart)."""
        pred = torch.tensor([0.0])
        target = torch.tensor([0.5])
        loss = circular_angle_loss(pred, target)
        torch.testing.assert_close(loss, torch.tensor([0.5]))

    def test_symmetry(self) -> None:
        """Loss should be symmetric."""
        pred = torch.tensor([0.3])
        target = torch.tensor([0.7])
        loss_a = circular_angle_loss(pred, target)
        loss_b = circular_angle_loss(target, pred)
        torch.testing.assert_close(loss_a, loss_b)
