# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.matcher import HungarianMatcher


@pytest.fixture()
def matcher() -> HungarianMatcher:
    """Shared HungarianMatcher instance."""
    return HungarianMatcher()


@pytest.fixture()
def standard_target() -> dict[str, torch.Tensor]:
    """Single-class target with one box at (0.5, 0.5, 0.2, 0.2)."""
    return {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
    }


class TestHungarianMatcherNonFiniteCosts:
    """Tests for non-finite cost matrix sanitization in the Hungarian matcher."""

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
        ],
    )
    def test_replaces_non_finite_costs_before_assignment(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
        invalid_value: float,
    ) -> None:
        """Matcher should sanitize non-finite costs so assignment still succeeds."""
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

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        assert matched_queries.tolist() == [1]
        assert matched_targets.tolist() == [0]

    def test_all_nonfinite_returns_one_match(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """When ALL costs are non-finite, the fallback sentinel (1.0) should allow
        ``linear_sum_assignment`` to complete with a valid assignment.

        This exercises the ``else: replacement_cost = C.new_tensor(1.0)`` branch.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[nan], [nan]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [nan, nan, nan, nan],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        assert len(matched_queries) == len(matched_targets) == 1

    def test_all_nonfinite_query_index_in_range(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """When ALL costs are non-finite, the matched query index must be valid
        (within [0, num_queries)).
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[nan], [nan]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [nan, nan, nan, nan],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, _ = matcher(outputs, [standard_target])[0]

        assert 0 <= matched_queries.item() < 2

    def test_all_nonfinite_target_index_is_zero(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """When ALL costs are non-finite, the single target (index 0) must be
        assigned.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[nan], [nan]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [nan, nan, nan, nan],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        _, matched_targets = matcher(outputs, [standard_target])[0]

        assert matched_targets.item() == 0

    def test_negative_costs_with_nan_selects_valid_query(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """Regression test: when all finite costs are negative and one query
        produces NaN, the matcher must select the valid query, not the NaN one.

        This guards against the bug where ``max_cost * 2`` (the old replacement
        formula) could be smaller than ``max_cost`` when all costs are negative,
        causing the NaN query to appear cheaper than valid queries.
        """
        nan = float("nan")
        # Query 0: NaN box coordinates -> produces non-finite costs
        # Query 1: valid box, low logit -> all-negative but finite costs
        outputs = {
            "pred_logits": torch.tensor([[[0.0], [-10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        # The valid query (index 1) must be matched, not the NaN query.
        assert matched_queries.tolist() == [1]
        assert matched_targets.tolist() == [0]

    @pytest.mark.parametrize(
        "image_idx, expected_query_idx",
        [
            pytest.param(0, 1, id="image0"),
            pytest.param(1, 0, id="image1"),
        ],
    )
    def test_batch_size_greater_than_one(
        self,
        matcher: HungarianMatcher,
        image_idx: int,
        expected_query_idx: int,
    ) -> None:
        """Exercises the ``C.split(sizes, -1)`` loop with batch_size > 1.

        Each image has 2 queries and 1 target. One query per image has NaN
        coordinates; the matcher must select the valid query in each case.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor(
                [
                    [[0.0], [10.0]],  # image 0: query 1 is valid
                    [[10.0], [0.0]],  # image 1: query 0 is valid
                ],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, 0.5, 0.2, 0.2],  # image 0, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # image 0, query 1: valid
                    ],
                    [
                        [0.5, 0.5, 0.2, 0.2],  # image 1, query 0: valid
                        [nan, 0.5, 0.2, 0.2],  # image 1, query 1: NaN
                    ],
                ],
                dtype=torch.float32,
            ),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
        ]

        results = matcher(outputs, targets)

        assert len(results) == 2

        matched_queries, matched_targets = results[image_idx]
        assert matched_queries.tolist() == [expected_query_idx]
        assert matched_targets.tolist() == [0]
