# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for keypoint matching costs in HungarianMatcher."""

import torch

from rfdetr.models.matcher import HungarianMatcher


def _base_outputs(num_queries: int = 2) -> dict[str, torch.Tensor]:
    """Build minimal detection outputs used across matcher keypoint tests."""
    pred_logits = torch.full((1, num_queries, 1), 5.0, dtype=torch.float32)
    pred_boxes = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32).view(1, 1, 4).repeat(1, num_queries, 1)
    return {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
    }


def test_matcher_keypoint_cost_list_of_dicts_targets() -> None:
    """Keypoint matching costs should work with public list-of-dicts targets."""
    matcher = HungarianMatcher(
        cost_class=0.0,
        cost_bbox=1.0,
        cost_giou=0.0,
        num_keypoints_per_class=[1],
        keypoint_l1_loss_coef=10.0,
        keypoint_findable_loss_coef=0.0,
        keypoint_visible_loss_coef=0.0,
        keypoint_nll_loss_coef=0.0,
    )
    outputs = _base_outputs()
    outputs["pred_keypoints"] = torch.zeros((1, 2, 1, 8), dtype=torch.float32)
    outputs["pred_keypoints"][0, 0, 0, :2] = torch.tensor([0.5, 0.5], dtype=torch.float32)
    outputs["pred_keypoints"][0, 1, 0, :2] = torch.tensor([0.0, 0.0], dtype=torch.float32)
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "keypoints": torch.tensor([[[0.5, 0.5, 2.0]]], dtype=torch.float32),
        }
    ]

    matched_queries, matched_targets = matcher(outputs, targets)[0]

    assert matched_queries.tolist() == [0]
    assert matched_targets.tolist() == [0]


def test_matcher_keypoint_cost_coefficients_off() -> None:
    """Zero keypoint coefficients should preserve non-keypoint matching behavior."""
    base_matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)
    keypoint_matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        num_keypoints_per_class=[1],
        keypoint_l1_loss_coef=0.0,
        keypoint_findable_loss_coef=0.0,
        keypoint_visible_loss_coef=0.0,
        keypoint_nll_loss_coef=0.0,
    )
    outputs = _base_outputs()
    outputs["pred_logits"][0, 0, 0] = 10.0
    outputs["pred_logits"][0, 1, 0] = -10.0
    outputs["pred_boxes"][0, 1, :] = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "keypoints": torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32),
        }
    ]
    outputs_with_keypoints = dict(outputs)
    outputs_with_keypoints["pred_keypoints"] = torch.zeros((1, 2, 1, 8), dtype=torch.float32)

    base_indices = base_matcher(outputs, targets)[0]
    keypoint_indices = keypoint_matcher(outputs_with_keypoints, targets)[0]

    assert base_indices[0].tolist() == keypoint_indices[0].tolist()
    assert base_indices[1].tolist() == keypoint_indices[1].tolist()


def test_matcher_keypoint_empty_targets() -> None:
    """Empty keypoint targets should return valid empty match results."""
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        num_keypoints_per_class=[1],
        keypoint_l1_loss_coef=1.0,
        keypoint_findable_loss_coef=1.0,
        keypoint_visible_loss_coef=1.0,
        keypoint_nll_loss_coef=1.0,
    )
    outputs = _base_outputs(num_queries=3)
    outputs["pred_keypoints"] = torch.zeros((1, 3, 1, 8), dtype=torch.float32)
    targets = [
        {
            "labels": torch.zeros((0,), dtype=torch.int64),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "keypoints": torch.zeros((0, 1, 3), dtype=torch.float32),
        }
    ]

    matched_queries, matched_targets = matcher(outputs, targets)[0]

    assert matched_queries.numel() == 0
    assert matched_targets.numel() == 0
