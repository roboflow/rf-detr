# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher


def _make_oriented_matcher() -> HungarianMatcher:
    return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, oriented=True)


def _make_oriented_criterion() -> SetCriterion:
    matcher = _make_oriented_matcher()
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    return SetCriterion(
        num_classes=16,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=False,
    )


class TestOrientedMatcher:
    def test_returns_valid_indices(self) -> None:
        matcher = _make_oriented_matcher()
        outputs = {
            "pred_logits": torch.randn(1, 10, 16),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        targets = [
            {
                "labels": torch.tensor([0, 3]),
                "boxes_obb": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.1, 0.3],
                        [0.3, 0.7, 0.15, 0.1, 1.0],
                    ]
                ),
            }
        ]
        indices = matcher(outputs, targets)
        assert len(indices) == 1
        src_idx, tgt_idx = indices[0]
        assert len(src_idx) == 2
        assert len(tgt_idx) == 2

    def test_oriented_flag_stored(self) -> None:
        matcher = _make_oriented_matcher()
        assert matcher.oriented is True

    def test_non_oriented_default(self) -> None:
        matcher = HungarianMatcher()
        assert matcher.oriented is False


class TestOrientedCriterion:
    def test_loss_boxes_returns_kld(self) -> None:
        criterion = _make_oriented_criterion()
        outputs = {
            "pred_logits": torch.randn(1, 10, 16),
            "pred_boxes": torch.rand(1, 10, 5) * 0.5 + 0.1,
        }
        outputs["pred_boxes"][..., 4] = outputs["pred_boxes"][..., 4] * math.pi
        targets = [
            {
                "labels": torch.tensor([0, 3]),
                "boxes_obb": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.1, 0.3],
                        [0.3, 0.7, 0.15, 0.1, 1.0],
                    ]
                ),
            }
        ]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=2)
        assert "loss_bbox" in losses
        assert "loss_giou" in losses
        assert losses["loss_bbox"].item() >= 0
        assert losses["loss_giou"].item() >= 0

    def test_oriented_flag_propagated(self) -> None:
        criterion = _make_oriented_criterion()
        assert criterion.oriented is True

    def test_non_oriented_criterion(self) -> None:
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=91,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            focal_alpha=0.25,
            losses=["labels", "boxes"],
        )
        assert criterion.oriented is False
