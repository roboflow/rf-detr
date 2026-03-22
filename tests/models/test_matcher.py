# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models import matcher as matcher_module
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

    def test_all_nonfinite_produces_valid_assignment(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """When ALL costs are non-finite, the fallback sentinel (``dtype_info.max``)
        should allow ``linear_sum_assignment`` to complete with a valid 1-to-1
        assignment: exactly one match, query index in [0, num_queries), target
        index 0.

        This exercises the ``else: replacement_cost = C.new_tensor(dtype_info.max)``
        branch.
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
        assert 0 <= matched_queries.item() < 2
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

    def test_group_detr_with_nonfinite_costs(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """Sanitization runs on the full cost matrix before splitting by group, so
        non-finite entries must be handled correctly when ``group_detr > 1``.

        4 queries, 2 groups of 2. Query 0 has a NaN box; query 2 (the best valid
        match in group 1) must be selected across groups.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor(
                [[[0.0], [10.0], [0.0], [10.0]]],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],  # group 0, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # group 0, query 1: valid
                        [nan, nan, nan, nan],  # group 1, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # group 1, query 1: valid
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        results = matcher(outputs, [standard_target], group_detr=2)

        assert len(results) == 1
        matched_queries, matched_targets = results[0]
        # Each group contributes one match; both must map to target 0
        assert matched_targets.tolist() == [0, 0]
        # The valid query in each group (indices 1 and 3) must be selected
        assert set(matched_queries.tolist()) == {1, 3}

    def test_warns_once_per_matcher_instance(
        self, standard_target: dict[str, torch.Tensor], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-finite-cost warning should be emitted once per matcher instance."""
        expected_warning = (
            "Non-finite values detected in matcher cost matrix; "
            "replacing with finite sentinel. "
            "Check for numerical instability."
        )
        warning_messages: list[str] = []

        def record_warning(msg: str, *args: object, **kwargs: object) -> None:
            warning_messages.append(msg)

        monkeypatch.setattr(matcher_module.logger, "warning", record_warning)

        outputs = {
            "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [float("nan"), 0.5, 0.2, 0.2],
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        first_matcher = HungarianMatcher()
        second_matcher = HungarianMatcher()

        first_matcher(outputs, [standard_target])
        first_matcher(outputs, [standard_target])
        second_matcher(outputs, [standard_target])

        assert warning_messages == [expected_warning, expected_warning]


class TestHungarianMatcherSanitization:
    """Unit tests for the private matcher cost sanitization helper."""

    def test_sanitize_cost_matrix_replaces_non_finite_entries(self) -> None:
        """Non-finite entries should be replaced with a larger finite sentinel."""
        cost_matrix = torch.tensor(
            [
                [1.0, float("nan")],
                [float("inf"), -2.0],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert sanitized[0, 1] == 4.0
        assert sanitized[1, 0] == 4.0
        assert sanitized[0, 0] == 1.0
        assert sanitized[1, 1] == -2.0

    def test_sanitize_cost_matrix_all_non_finite_fallback(self) -> None:
        """All-non-finite matrices should fall back to the dtype maximum."""
        cost_matrix = torch.tensor(
            [
                [float("nan"), float("inf")],
                [float("-inf"), float("nan")],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert torch.all(sanitized == torch.finfo(cost_matrix.dtype).max)

    def test_sanitize_cost_matrix_clamps_overflowing_replacement_cost(self) -> None:
        """Overflow in the computed replacement cost should clamp to dtype max."""
        dtype_max = torch.finfo(torch.float32).max
        cost_matrix = torch.tensor(
            [
                [dtype_max, float("nan")],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert sanitized[0, 1] == dtype_max
