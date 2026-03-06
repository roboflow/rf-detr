# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.matcher import HungarianMatcher


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_matcher_replaces_non_finite_costs_before_assignment(invalid_value: float) -> None:
    """Matcher should sanitize non-finite costs so assignment still succeeds."""
    matcher = HungarianMatcher()
    outputs = {
        "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor(
            [
                [
                    [invalid_value, 0.5, 0.2, 0.2],
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ],
            dtype=torch.float32,
        ),
    }
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        }
    ]

    matched_queries, matched_targets = matcher(outputs, targets)[0]

    assert matched_queries.tolist() == [1]
    assert matched_targets.tolist() == [0]
