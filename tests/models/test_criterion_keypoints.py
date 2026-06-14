# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for keypoint losses in SetCriterion."""

import torch

from rfdetr.models.criterion import SetCriterion


class _MatcherStub:
    """Matcher stub used to avoid depending on Hungarian matching internals."""

    def __call__(self, outputs, targets, group_detr=1):
        indices = []
        for target in targets:
            num_targets = int(target["labels"].shape[0])
            idx = torch.arange(num_targets, dtype=torch.int64)
            indices.append((idx, idx))
        return indices


def _make_outputs(
    batch_size: int,
    num_queries: int,
    num_keypoints: int,
) -> dict[str, torch.Tensor]:
    return {
        "pred_logits": torch.zeros(batch_size, num_queries, 2),
        "pred_boxes": torch.rand(batch_size, num_queries, 4).clamp(0.05, 0.95),
        "pred_keypoints": torch.randn(batch_size, num_queries, num_keypoints, 8),
    }


def test_loss_keypoints_list_of_dicts_targets() -> None:
    """Keypoint loss should consume list-of-dicts targets used by public training."""
    criterion = SetCriterion(
        num_classes=2,
        matcher=_MatcherStub(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints_per_class=[17],
    )
    outputs = _make_outputs(batch_size=1, num_queries=1, num_keypoints=17)
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
            "keypoints": torch.cat(
                [
                    torch.rand(1, 17, 2),
                    torch.full((1, 17, 1), 2.0),
                ],
                dim=-1,
            ),
        }
    ]

    losses = criterion(outputs, targets)

    assert "loss_keypoints_l1" in losses
    assert "loss_keypoints_findable" in losses
    assert "loss_keypoints_visible" in losses
    assert "loss_keypoints_nll" in losses
    assert all(torch.isfinite(value) for value in losses.values())


def test_loss_keypoints_empty_targets() -> None:
    """Empty target batches should produce finite zero-valued keypoint losses."""
    criterion = SetCriterion(
        num_classes=2,
        matcher=_MatcherStub(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints_per_class=[17],
    )
    outputs = _make_outputs(batch_size=1, num_queries=1, num_keypoints=17)
    targets = [
        {
            "labels": torch.zeros((0,), dtype=torch.int64),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "keypoints": torch.zeros((0, 17, 3), dtype=torch.float32),
        }
    ]

    losses = criterion(outputs, targets)

    assert losses["loss_keypoints_l1"].item() == 0.0
    assert losses["loss_keypoints_findable"].item() == 0.0
    assert losses["loss_keypoints_visible"].item() == 0.0
    assert losses["loss_keypoints_nll"].item() == 0.0


def test_loss_keypoints_person_schema_shape() -> None:
    """Person-only schema `[17]` should be consumed without shape mismatches."""
    criterion = SetCriterion(
        num_classes=2,
        matcher=_MatcherStub(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints_per_class=[17],
    )
    outputs = _make_outputs(batch_size=2, num_queries=2, num_keypoints=17)
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
            "keypoints": torch.rand(1, 17, 3),
        },
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.4, 0.6, 0.3, 0.5]], dtype=torch.float32),
            "keypoints": torch.rand(1, 17, 3),
        },
    ]

    losses = criterion(outputs, targets)

    assert losses["loss_keypoints_l1"].ndim == 0
    assert losses["loss_keypoints_findable"].ndim == 0
    assert losses["loss_keypoints_visible"].ndim == 0
    assert losses["loss_keypoints_nll"].ndim == 0


def test_loss_keypoints_multiclass_schema_kmax_targets() -> None:
    """Heterogeneous keypoint classes should consume Kmax-padded targets."""
    criterion = SetCriterion(
        num_classes=3,
        matcher=_MatcherStub(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints_per_class=[2, 1],
    )
    outputs = _make_outputs(batch_size=1, num_queries=2, num_keypoints=4)
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.4, 0.6, 0.3, 0.5]], dtype=torch.float32),
            "keypoints": torch.tensor(
                [
                    [[0.2, 0.3, 2.0], [0.4, 0.5, 2.0]],
                    [[0.6, 0.7, 2.0], [0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        }
    ]

    losses = criterion(outputs, targets)

    assert losses["loss_keypoints_l1"].ndim == 0
    assert losses["loss_keypoints_findable"].ndim == 0
    assert losses["loss_keypoints_visible"].ndim == 0
    assert losses["loss_keypoints_nll"].ndim == 0
    assert all(torch.isfinite(value) for value in losses.values())
