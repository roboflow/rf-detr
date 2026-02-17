# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher


def _make_criterion(depth=True, focal_length=800.0, ball_ids=None):
    if ball_ids is None:
        ball_ids = [0]
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_depth": 5.0,
        "loss_pinhole": 1.0,
    }
    losses = ["labels", "boxes", "cardinality"]
    if depth:
        losses.extend(["depth", "pinhole"])
    return SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=losses,
        group_detr=1,
        focal_length_px=focal_length if depth else None,
        ball_diameter_m=0.22,
        ball_class_ids=ball_ids if depth else [],
        resolution=640,
    )


class TestDepthDistillLoss:
    def test_depth_distill_loss_nonzero(self):
        """loss_depth should compute SmoothL1 between pred_depth and target depth."""
        criterion = _make_criterion()
        B, Q = 1, 10
        outputs = {
            "pred_logits": torch.randn(B, Q, 3),
            "pred_boxes": torch.rand(B, Q, 4).sigmoid(),
            "pred_depth": torch.ones(B, Q, 1) * 50.0,
        }
        targets = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "depth": torch.tensor([[30.0]]),
            }
        ]
        loss_dict = criterion(outputs, targets)
        assert "loss_depth" in loss_dict
        assert loss_dict["loss_depth"].item() > 0  # 50 vs 30 = non-zero loss


class TestPinholeLoss:
    def test_pinhole_loss_zero_when_no_balls(self):
        """Pinhole loss should be zero when no ball classes are matched."""
        criterion = _make_criterion(ball_ids=[99])  # non-existent class
        B, Q = 1, 10
        outputs = {
            "pred_logits": torch.randn(B, Q, 3),
            "pred_boxes": torch.rand(B, Q, 4).sigmoid(),
            "pred_depth": torch.ones(B, Q, 1) * 50.0,
        }
        targets = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "depth": torch.tensor([[30.0]]),
            }
        ]
        loss_dict = criterion(outputs, targets)
        assert "loss_pinhole" in loss_dict
        assert loss_dict["loss_pinhole"].item() == 0.0


class TestNoDepthLosses:
    def test_no_depth_losses_when_disabled(self):
        """SetCriterion without depth losses should work normally."""
        criterion = _make_criterion(depth=False)
        B, Q = 1, 10
        outputs = {
            "pred_logits": torch.randn(B, Q, 3),
            "pred_boxes": torch.rand(B, Q, 4).sigmoid(),
        }
        targets = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            }
        ]
        loss_dict = criterion(outputs, targets)
        assert "loss_depth" not in loss_dict
        assert "loss_pinhole" not in loss_dict
        assert "loss_ce" in loss_dict
