# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Callable

import numpy as np
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
        assignment: exactly one match, query index in [0, num_queries), target index 0.

        This exercises the ``else: replacement_cost = C.new_tensor(dtype_info.max)`` branch.
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
        """Regression test: when all finite costs are negative and one query produces NaN, the matcher must select the
        valid query, not the NaN one.

        This guards against the bug where ``max_cost * 2`` (the old replacement formula) could be smaller than
        ``max_cost`` when all costs are negative, causing the NaN query to appear cheaper than valid queries.
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

        Each image has 2 queries and 1 target. One query per image has NaN coordinates; the matcher must select the
        valid query in each case.
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
        """Sanitization runs on the full cost matrix before splitting by group, so non-finite entries must be handled
        correctly when ``group_detr > 1``.

        4 queries, 2 groups of 2. Query 0 has a NaN box; query 2 (the best valid match in group 1) must be selected
        across groups.
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


class TestHungarianMatcherFocalAlpha:
    """The configured ``focal_alpha`` must drive the classification matching cost."""

    def test_focal_alpha_changes_assignment(self) -> None:
        """Two matchers differing only in ``focal_alpha`` must be able to produce different assignments.

        ``focal_alpha`` is accepted, documented as "used in the classification cost", and stored on the matcher, so it
        must actually influence matching. This input is chosen so the optimal query->target pairing flips between
        ``focal_alpha=0.25`` and ``focal_alpha=0.90``; if the cost ignores the configured alpha, both assignments
        collapse to the same result.
        """
        outputs = {
            "pred_logits": torch.tensor(
                [[[2.3936, -1.4217], [2.3731, -2.1974]]],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [[[0.3898, 0.4340, 0.5331, 0.1901], [0.4256, 0.1002, 0.6955, 0.7815]]],
                dtype=torch.float32,
            ),
        }
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.2111, 0.6630, 0.7569, 0.8855], [0.7750, 0.4393, 0.8838, 0.8792]],
                    dtype=torch.float32,
                ),
            }
        ]

        def assignment(focal_alpha: float) -> list[int]:
            matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=focal_alpha)
            matched_queries, matched_targets = matcher(outputs, targets)[0]
            # Queries ordered by the target index they are matched to.
            return matched_queries[matched_targets.argsort()].tolist()

        assert assignment(0.25) != assignment(0.90)
        # Pin the exact expected mappings so a misapplied-alpha refactor is caught even when
        # the two values remain different for unrelated reasons.
        assert assignment(0.25) == [0, 1]
        assert assignment(0.90) == [1, 0]

    @pytest.mark.parametrize(
        "focal_alpha, expected",
        [
            pytest.param(0.0, [0, 1], id="alpha_zero_pos_cost_zeroed"),
            pytest.param(1.0, [1, 0], id="alpha_one_neg_cost_zeroed"),
        ],
    )
    def test_focal_alpha_boundary_values_no_nan(self, focal_alpha: float, expected: list[int]) -> None:
        """Degenerate focal_alpha values (0.0 and 1.0) must not produce NaN and must yield a valid assignment.

        focal_alpha=0.0 zeroes ``pos_cost_class``; focal_alpha=1.0 zeroes ``neg_cost_class``. Neither path touches
        ``log(prob)`` directly (formula uses logsigmoid of logits), so no division-by-zero or NaN can occur.
        """
        outputs = {
            "pred_logits": torch.tensor(
                [[[2.3936, -1.4217], [2.3731, -2.1974]]],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [[[0.3898, 0.4340, 0.5331, 0.1901], [0.4256, 0.1002, 0.6955, 0.7815]]],
                dtype=torch.float32,
            ),
        }
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.2111, 0.6630, 0.7569, 0.8855], [0.7750, 0.4393, 0.8838, 0.8792]],
                    dtype=torch.float32,
                ),
            }
        ]

        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=focal_alpha)
        matched_queries, matched_targets = matcher(outputs, targets)[0]

        assert not matcher._warned_non_finite_costs, "boundary focal_alpha produced non-finite costs"
        result = matched_queries[matched_targets.argsort()].tolist()
        assert result == expected


def _reference_indices_full_class_materialization(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reference matching that materializes the focal class cost over ALL classes before slicing.

    Examples:
        >>> matcher = HungarianMatcher()
        >>> outputs = {
        ...     "pred_logits": torch.zeros(1, 2, 2),
        ...     "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]),
        ... }
        >>> targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        >>> indices = _reference_indices_full_class_materialization(matcher, outputs, targets)
        >>> [(q.tolist(), t.tolist()) for q, t in indices]
        [([0], [0])]
    """
    from scipy.optimize import linear_sum_assignment

    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

    bs, num_queries = outputs["pred_logits"].shape[:2]
    logits = outputs["pred_logits"].flatten(0, 1)
    prob = logits.sigmoid()
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([t["labels"] for t in targets])
    tgt_bbox = torch.cat([t["boxes"] for t in targets])
    alpha = matcher.focal_alpha
    gamma = matcher_module._FOCAL_LOSS_GAMMA
    neg_cost_class = (1 - alpha) * (prob**gamma) * (-torch.nn.functional.logsigmoid(-logits))
    pos_cost_class = alpha * ((1 - prob) ** gamma) * (-torch.nn.functional.logsigmoid(logits))
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    cost = matcher.cost_bbox * cost_bbox + matcher.cost_class * cost_class + matcher.cost_giou * cost_giou
    cost = cost.view(bs, num_queries, -1).float().cpu()
    sizes = [len(t["boxes"]) for t in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TestClassCostGatherFirst:
    """Class cost computed on gathered target columns must reproduce the full-materialization matching."""

    def test_forward_matches_full_class_materialization_reference(self, matcher: HungarianMatcher) -> None:
        """Random batch: matcher assignment equals the reference that builds [bs*nq, num_classes] first."""
        torch.manual_seed(7)
        bs, num_queries, num_classes = 2, 8, 11
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {
                "labels": torch.tensor([1, 3, 3], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1], [0.5, 0.4, 0.3, 0.2]], dtype=torch.float32
                ),
            },
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
            },
        ]

        actual = matcher(outputs, targets)

        expected = _reference_indices_full_class_materialization(matcher, outputs, targets)
        for (act_q, act_t), (exp_q, exp_t) in zip(actual, expected):
            assert torch.equal(act_q, exp_q)
            assert torch.equal(act_t, exp_t)

    def test_forward_matches_reference_when_one_batch_element_has_zero_targets(self, matcher: HungarianMatcher) -> None:
        """A zero-GT batch element must gather-first-match the reference (empty tgt_ids column selection).

        The gather-first refactor indexes ``flat_pred_logits[:, tgt_ids]`` where ``tgt_ids`` is the concatenation of
        every batch element's labels; an empty-labels element degenerates that slice to a ``[N, 0]`` selection for its
        own queries. This boundary was previously unexercised — every existing test target has >=1 GT box.
        """
        torch.manual_seed(11)
        bs, num_queries, num_classes = 2, 6, 7
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {
                "labels": torch.tensor([2, 4], dtype=torch.int64),
                "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]], dtype=torch.float32),
            },
            {
                "labels": torch.zeros(0, dtype=torch.int64),
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
            },
        ]

        actual = matcher(outputs, targets)

        expected = _reference_indices_full_class_materialization(matcher, outputs, targets)
        for (act_q, act_t), (exp_q, exp_t) in zip(actual, expected):
            assert torch.equal(act_q, exp_q)
            assert torch.equal(act_t, exp_t)
        assert actual[1][0].shape == (0,)
        assert actual[1][1].shape == (0,)


def _reference_indices_pre_diagonal_extraction(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    group_detr: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reference matching using the pre-PR extraction: materialize the full ``[bs, num_queries, total_targets]`` cost
    matrix, then slice per image with ``cost_matrix.split(sizes, -1)`` and ``c[i]`` (the code in ``matcher.py`` before
    the diagonal-block candidate).

    Reuses the matcher's current gather-first class cost so this isolates only the extraction-step change under test,
    not the unrelated class-cost refactor already covered by ``_reference_indices_full_class_materialization``.

    One image, two queries, one target: query 0's box exactly equals the target box (cost_bbox=0,
    cost_giou=-1, the minimum possible), query 1's box is far away. Both queries share identical
    logits, so cost_class is equal for both — bbox/giou alone decide the winner.

    >>> import torch
    >>> from rfdetr.models.matcher import HungarianMatcher
    >>> matcher = HungarianMatcher()
    >>> target_box = [0.5, 0.5, 0.2, 0.2]
    >>> outputs = {
    ...     "pred_logits": torch.zeros(1, 2, 3),
    ...     "pred_boxes": torch.tensor([[target_box, [0.05, 0.05, 0.05, 0.05]]]),
    ... }
    >>> targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([target_box])}]
    >>> indices = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)
    >>> [(q.tolist(), t.tolist()) for q, t in indices]
    [([0], [0])]
    """
    from scipy.optimize import linear_sum_assignment

    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

    bs, num_queries = outputs["pred_logits"].shape[:2]
    flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([t["labels"] for t in targets])
    tgt_bbox = torch.cat([t["boxes"] for t in targets])
    alpha = matcher.focal_alpha
    gamma = matcher_module._FOCAL_LOSS_GAMMA
    tgt_logits = flat_pred_logits[:, tgt_ids]
    tgt_prob = tgt_logits.sigmoid()
    neg_cost_class = (1 - alpha) * (tgt_prob**gamma) * (-torch.nn.functional.logsigmoid(-tgt_logits))
    pos_cost_class = alpha * ((1 - tgt_prob) ** gamma) * (-torch.nn.functional.logsigmoid(tgt_logits))
    cost_class = pos_cost_class - neg_cost_class
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    cost_matrix = matcher.cost_bbox * cost_bbox + matcher.cost_class * cost_class + matcher.cost_giou * cost_giou
    cost_matrix = cost_matrix.view(bs, num_queries, -1).float().cpu()

    sizes = [len(t["boxes"]) for t in targets]
    if num_queries % group_detr != 0:
        raise ValueError(f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})")
    g_num_queries = num_queries // group_detr
    cost_matrix_list = cost_matrix.split(g_num_queries, dim=1)
    indices = []
    for g_i in range(group_detr):
        grouped_cost_matrix = cost_matrix_list[g_i]
        indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(grouped_cost_matrix.split(sizes, -1))]
        if g_i == 0:
            indices = indices_g
        else:
            indices = [
                (
                    np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                    np.concatenate([indice1[1], indice2[1]]),
                )
                for indice1, indice2 in zip(indices, indices_g)
            ]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TestDiagonalBlockExtraction:
    """The ``target_offsets`` diagonal-block extraction in ``matcher.py`` must reproduce the pre-PR
    ``cost_matrix.split(sizes, -1)`` + ``c[i]`` extraction for every batch element, across heterogeneous target counts
    and zero-target elements in any position."""

    def test_heterogeneous_sizes_with_zero_in_the_middle(self, matcher: HungarianMatcher) -> None:
        """Batch of 4 images with sizes [2, 0, 3, 1]: the zero-target element sits between two non-zero elements, so an
        off-by-one in the cumulative ``target_offsets`` would leak columns from a neighboring image into the wrong
        diagonal block."""
        torch.manual_seed(23)
        bs, num_queries, num_classes = 4, 6, 5
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {
                "labels": torch.tensor([1, 3], dtype=torch.int64),
                "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]], dtype=torch.float32),
            },
            {
                "labels": torch.zeros(0, dtype=torch.int64),
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
            },
            {
                "labels": torch.tensor([0, 2, 4], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.4, 0.4, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3], [0.6, 0.3, 0.1, 0.2]], dtype=torch.float32
                ),
            },
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
        ]

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert len(actual) == bs
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"
        assert actual[1][0].shape == (0,)
        assert actual[1][1].shape == (0,)

    def test_heterogeneous_sizes_with_group_detr(self, matcher: HungarianMatcher) -> None:
        """With ``group_detr > 1`` the diagonal block is additionally sliced by ``group_start:group_start +
        g_num_queries`` before being sliced by target offsets; a bug in either slice would corrupt the group-combination
        step that concatenates indices across groups."""
        torch.manual_seed(29)
        bs, num_queries, num_classes, group_detr = 3, 8, 6, 2
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {
                "labels": torch.tensor([2, 5, 0], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1], [0.4, 0.5, 0.2, 0.3]], dtype=torch.float32
                ),
            },
            {
                "labels": torch.zeros(0, dtype=torch.int64),
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
            },
            {
                "labels": torch.tensor([1, 4], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.7, 0.1, 0.1]], dtype=torch.float32),
            },
        ]

        actual = matcher(outputs, targets, group_detr=group_detr)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets, group_detr=group_detr)

        assert len(actual) == bs
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"
        # Each group matches min(g_num_queries, size) queries per image (g_num_queries=4):
        # image 0 has 3 targets -> 3 per group x 2 groups; image 2 has 2 targets -> 2 per group x 2 groups.
        assert actual[0][0].shape == (6,)
        assert actual[2][0].shape == (4,)
        assert actual[1][0].shape == (0,)

    def test_all_batch_elements_have_zero_targets(self, matcher: HungarianMatcher) -> None:
        """Degenerate case: every image in the batch has zero targets, so ``target_offsets`` collapses to all-zero and
        every diagonal block is empty.

        Must not raise and must return an empty assignment for every image, and must agree with the pre-PR extraction on
        that (both return the same trivially-empty indices here, but the comparison is kept so this test actually
        exercises ``_reference_indices_pre_diagonal_extraction`` like its two siblings, instead of only asserting the
        shape of the new code's own output).
        """
        torch.manual_seed(31)
        bs, num_queries, num_classes = 3, 4, 5
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {"labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4, dtype=torch.float32)}
            for _ in range(bs)
        ]

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert len(actual) == bs
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"
        for matched_queries, matched_targets in actual:
            assert matched_queries.shape == (0,)
            assert matched_targets.shape == (0,)


def _new_extraction_indices(
    cost_matrix: torch.Tensor, sizes: list[int], group_detr: int = 1
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Mirrors the post-PR extraction in ``matcher.py`` (``target_offsets`` + ``torch.cat``), decoupled from cost
    computation so it can be property-tested against arbitrary cost-matrix content — including whatever values the
    mask/keypoint cost terms (``matcher.py:266-277``, untouched by this PR) would fold in, without duplicating those
    formulas here.

    Two images, two queries each, one target each: image 0's own column (0) is cheapest at query 1
    (cost 1.0); image 1's own column (1) is cheapest at query 0 (cost 2.0). The other column in each
    row belongs to the other image and must be ignored.

    >>> import torch
    >>> cost_matrix = torch.tensor([
    ...     [[5.0, 100.0], [1.0, 100.0]],
    ...     [[50.0, 2.0], [60.0, 9.0]],
    ... ])
    >>> indices = _new_extraction_indices(cost_matrix, sizes=[1, 1])
    >>> [(q.tolist(), t.tolist()) for q, t in indices]
    [([1], [0]), ([0], [0])]
    """
    from scipy.optimize import linear_sum_assignment

    bs, num_queries = cost_matrix.shape[:2]
    target_offsets = [0]
    for size in sizes:
        target_offsets.append(target_offsets[-1] + size)
    diagonal_cost_matrix = torch.cat(
        [cost_matrix[i, :, target_offsets[i] : target_offsets[i + 1]] for i in range(bs)], dim=-1
    )
    g_num_queries = num_queries // group_detr
    indices = []
    for g_i in range(group_detr):
        group_start = g_i * g_num_queries
        grouped_cost_matrix = diagonal_cost_matrix[group_start : group_start + g_num_queries]
        indices_g = [
            linear_sum_assignment(grouped_cost_matrix[:, target_offsets[i] : target_offsets[i + 1]]) for i in range(bs)
        ]
        if g_i == 0:
            indices = indices_g
        else:
            indices = [
                (np.concatenate([i1[0], i2[0] + g_num_queries * g_i]), np.concatenate([i1[1], i2[1]]))
                for i1, i2 in zip(indices, indices_g)
            ]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def _old_extraction_indices(
    cost_matrix: torch.Tensor, sizes: list[int], group_detr: int = 1
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Mirrors the pre-PR extraction (``cost_matrix.split(sizes, -1)`` + ``c[i]``).

    Same hand-computed case as ``_new_extraction_indices`` — output must match it exactly.

    >>> import torch
    >>> cost_matrix = torch.tensor([
    ...     [[5.0, 100.0], [1.0, 100.0]],
    ...     [[50.0, 2.0], [60.0, 9.0]],
    ... ])
    >>> indices = _old_extraction_indices(cost_matrix, sizes=[1, 1])
    >>> [(q.tolist(), t.tolist()) for q, t in indices]
    [([1], [0]), ([0], [0])]
    """
    from scipy.optimize import linear_sum_assignment

    bs, num_queries = cost_matrix.shape[:2]
    g_num_queries = num_queries // group_detr
    cost_matrix_list = cost_matrix.split(g_num_queries, dim=1)
    indices = []
    for g_i in range(group_detr):
        grouped_cost_matrix = cost_matrix_list[g_i]
        indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(grouped_cost_matrix.split(sizes, -1))]
        if g_i == 0:
            indices = indices_g
        else:
            indices = [
                (np.concatenate([i1[0], i2[0] + g_num_queries * g_i]), np.concatenate([i1[1], i2[1]]))
                for i1, i2 in zip(indices, indices_g)
            ]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TestDiagonalExtractionContentAgnostic:
    """``TestDiagonalBlockExtraction`` above only feeds detection-shaped costs (bbox+class+giou) through the real
    ``matcher.forward``.

    But the extraction step it exercises has no notion of
    where the cost values came from: it slices ``cost_matrix`` by ``sizes``/``target_offsets``
    after the mask (``cost_mask_ce``/``cost_mask_dice``) and keypoint
    (``cost_l1``/``cost_findable``/``cost_visible``/``cost_nll``) terms are already summed into it
    (``matcher.py:266-278``, all untouched by this PR). Property-testing the old and new extraction
    directly on arbitrary cost tensors therefore also covers segmentation and keypoint batches,
    without re-deriving their cost formulas in this test file.
    """

    @pytest.mark.parametrize("group_detr", [1, 2, 3])
    @pytest.mark.parametrize("seed", [41, 42, 43, 44, 45])
    def test_matches_old_extraction_for_arbitrary_cost_content(self, seed: int, group_detr: int) -> None:
        torch.manual_seed(seed)
        sizes = [3, 0, 5, 2]
        bs = len(sizes)
        num_queries = 12
        total_targets = sum(sizes)
        # Wide range and an offset so the sentinel-adjacent negative-cost edge case (see
        # TestHungarianMatcherNonFiniteCosts.test_negative_costs_with_nan_selects_valid_query)
        # is also exercised by some seeds, not just small positive costs.
        cost_matrix = torch.randn(bs, num_queries, total_targets, dtype=torch.float32) * 10 - 3

        actual = _new_extraction_indices(cost_matrix, sizes, group_detr=group_detr)
        expected = _old_extraction_indices(cost_matrix, sizes, group_detr=group_detr)

        assert len(actual) == bs
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"


def _spy_on_compact_path(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``_compute_compact_detection_cost_matrix`` to record how many times it runs, without changing its behavior —
    lets a test assert which branch ``forward`` actually took.

    Examples:
        >>> _spy_on_compact_path(...)  # doctest: +SKIP
        # Needs a real pytest.MonkeyPatch fixture instance, not constructible standalone.
    """
    calls: list[int] = []
    original = HungarianMatcher._compute_compact_detection_cost_matrix

    def spy(
        self: HungarianMatcher, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        calls.append(1)
        return original(self, outputs, targets)

    monkeypatch.setattr(HungarianMatcher, "_compute_compact_detection_cost_matrix", spy)
    return calls


def _random_detection_batch(
    seed: int, sizes: list[int], num_queries: int = 6, num_classes: int = 5
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Random detection-only outputs/targets with the given per-image target counts.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[2, 0])
        >>> outputs["pred_logits"].shape, outputs["pred_boxes"].shape
        (torch.Size([2, 6, 5]), torch.Size([2, 6, 4]))
        >>> [len(target["labels"]) for target in targets]
        [2, 0]
    """
    torch.manual_seed(seed)
    bs = len(sizes)
    outputs = {
        "pred_logits": torch.randn(bs, num_queries, num_classes),
        "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
    }
    targets = [
        {
            "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
            "boxes": torch.rand(size, 4) * 0.4 + 0.3,
        }
        for size in sizes
    ]
    return outputs, targets


def _spy_on_full_path(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record one entry per full-path cost build, by tagging the 2-D ``torch.cdist`` call only that path makes (the
    compact path's ``cdist`` operands are 3-D) — lets a test tell "stayed on the compact path" apart from "built the
    compact matrix, found it non-finite, and fell through to the full path".

    Examples:
        with pytest.MonkeyPatch.context() as patched:
            full_calls = _spy_on_full_path(patched)
            matcher(outputs, targets)
            assert full_calls == []
    """
    calls: list[int] = []
    original = torch.cdist

    def spy(x1: torch.Tensor, x2: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if x1.dim() == 2:
            calls.append(1)
        return original(x1, x2, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", spy)
    return calls


def _detection_batch_with_labels(
    seed: int, labels_per_image: list[list[int]], num_queries: int = 4, num_classes: int = 5
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Detection-only outputs/targets with exactly the given per-image class ids, so a test can place a non-finite logit
    in a class column that is consumed by its own image, consumed only by another image, or consumed by nobody.

    Examples:
        >>> outputs, targets = _detection_batch_with_labels(seed=1, labels_per_image=[[0, 1], [2, 3]])
        >>> outputs["pred_logits"].shape
        torch.Size([2, 4, 5])
        >>> [target["labels"].tolist() for target in targets]
        [[0, 1], [2, 3]]
    """
    torch.manual_seed(seed)
    batch_size = len(labels_per_image)
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4) * 0.4 + 0.3,
    }
    targets = [
        {
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.rand(len(labels), 4) * 0.4 + 0.3,
        }
        for labels in labels_per_image
    ]
    return outputs, targets


def _full_path_indices(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Assignment the full path produces for these exact inputs, obtained by forcing the eligibility gate to ``False``
    inside a self-undoing monkeypatch context — the reference for every compact-vs-full equivalence assertion.

    Examples:
        >>> matcher = HungarianMatcher()
        >>> outputs, targets = _detection_batch_with_labels(seed=2, labels_per_image=[[0], [1]])
        >>> [(q.tolist(), t.tolist()) for q, t in _full_path_indices(matcher, outputs, targets)]
        [([1], [0]), ([3], [0])]
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t: False))
        return matcher(outputs, targets)


def _assert_same_indices(
    actual: list[tuple[torch.Tensor, torch.Tensor]], expected: list[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """Assert two per-image ``(query_indices, target_indices)`` assignments are element-for-element equal.

    Examples:
        >>> pair = [(torch.tensor([0]), torch.tensor([0]))]
        >>> _assert_same_indices(pair, [(torch.tensor([0]), torch.tensor([0]))])
    """
    assert len(actual) == len(expected)
    for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
        assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
        assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"


class TestCompactPathRouting:
    """The padded-compact cost path (``_compute_compact_detection_cost_matrix``) must run only for detection-only
    batches with ``batch_size > 1`` and finite, bounded inputs — every other case must fall back to the diagonal-block
    path from the first PR unchanged."""

    def test_batch_size_one_uses_fallback_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """``bs == 1`` must never take the compact path, even for safe detection-only inputs — the first variant without
        this gate regressed A100 batch-1 latency by 18.1%."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=101, sizes=[3])

        matcher(outputs, targets)

        assert calls == []

    def test_batch_size_greater_than_one_detection_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A safe, detection-only, ``bs > 1`` batch must take the compact path."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=102, sizes=[2, 3])

        matcher(outputs, targets)

        assert calls == [1]

    def test_masks_present_uses_fallback_path(self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher) -> None:
        """A segmentation batch (``masks`` in targets) must skip the compact path entirely, even with ``bs > 1`` and
        otherwise-safe inputs — mask costs are not supported by it."""
        calls = _spy_on_compact_path(monkeypatch)
        torch.manual_seed(103)
        bs, num_queries, num_classes, mask_size = 2, 4, 3, 8
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
            "pred_masks": torch.randn(bs, num_queries, mask_size, mask_size),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "masks": torch.rand(1, mask_size, mask_size),
            },
            {
                "labels": torch.tensor([1, 2], dtype=torch.int64),
                "boxes": torch.rand(2, 4) * 0.4 + 0.3,
                "masks": torch.rand(2, mask_size, mask_size),
            },
        ]

        results = matcher(outputs, targets)

        assert calls == []
        assert len(results) == bs

    def test_keypoints_present_uses_fallback_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A keypoint batch (``pred_keypoints`` in outputs and ``keypoints`` in targets) must skip the compact path
        entirely, even with ``bs > 1`` and otherwise-safe inputs."""
        calls = _spy_on_compact_path(monkeypatch)
        torch.manual_seed(104)
        bs, num_queries, num_classes, num_keypoints, pred_dim = 2, 4, 1, 3, 7
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
            "pred_keypoints": torch.randn(bs, num_queries, num_keypoints, pred_dim),
        }
        keypoint_matcher = HungarianMatcher(num_keypoints_per_class=[num_keypoints])
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "keypoints": torch.rand(1, num_keypoints, 3),
            },
            {
                "labels": torch.tensor([0, 0], dtype=torch.int64),
                "boxes": torch.rand(2, 4) * 0.4 + 0.3,
                "keypoints": torch.rand(2, num_keypoints, 3),
            },
        ]

        results = keypoint_matcher(outputs, targets)

        assert calls == []
        assert len(results) == bs

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(lambda o, t: o["pred_boxes"].__setitem__((1, 0, 2), float("nan")), id="pred_box_nan"),
            pytest.param(lambda o, t: o["pred_boxes"].__setitem__((1, 0, 2), float("inf")), id="pred_box_inf"),
            pytest.param(lambda o, t: t[0]["boxes"].__setitem__((0, 1), float("nan")), id="target_box_nan"),
            pytest.param(lambda o, t: t[0]["boxes"].__setitem__((0, 1), float("inf")), id="target_box_inf"),
            pytest.param(lambda o, t: o["pred_boxes"].__setitem__((1, 0, 2), 1e30), id="pred_box_extreme"),
        ],
    )
    def test_unsafe_inputs_use_fallback_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        matcher: HungarianMatcher,
        corrupt: Callable[[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]], None],
    ) -> None:
        """Each of the 7 unsafe-input cases verified in the exploration script (NaN/Inf on predicted logits, predicted
        boxes, or target boxes, plus one coordinate large enough to risk overflow in ``cdist``/GIoU area terms) must
        route to the fallback path instead of the compact one, for an otherwise compact-eligible (``bs > 1``, detection-
        only) batch."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=105, sizes=[2, 3])
        corrupt(outputs, targets)

        results = matcher(outputs, targets)

        assert calls == []
        assert len(results) == len(targets)

    @pytest.mark.parametrize("seed", [201, 202, 203, 204, 205])
    def test_compact_path_matches_pre_pr1_reference_across_seeds(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher, seed: int
    ) -> None:
        """The compact path's assignment must agree with the pre-PR1 reference (full materialization + ``split(sizes,
        -1)`` + ``c[i]``) across several random seeds — same contract as ``TestDiagonalBlockExtraction``, now exercised
        through the compact route."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=seed, sizes=[2, 4, 1])

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert calls == [1]
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"

    def test_heterogeneous_and_empty_targets_use_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A zero-target image mixed with unequal non-zero counts must still route to the compact path and match the
        pre-PR1 reference — the padded matrix's ``max(T_i)`` column count still has to resolve to the right per-image
        slice after the padding is dropped."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=206, sizes=[0, 3, 1])

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert calls == [1]
        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"
        assert actual[0][0].shape == (0,)

    def test_all_batch_elements_empty_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """Every image with zero targets (``max(T_i) == 0``) is the degenerate case for ``pad_sequence``: the padded
        target dimension collapses to size 0.

        Must still route to the compact path, not raise, and return an empty assignment for every image.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=207, sizes=[0, 0, 0])

        actual = matcher(outputs, targets)

        assert calls == [1]
        assert len(actual) == 3
        for matched_queries, matched_targets in actual:
            assert matched_queries.shape == (0,)
            assert matched_targets.shape == (0,)

    def test_overflowing_cost_weight_falls_through_to_fallback_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_detection_inputs_are_safe`` only bounds ``pred_logits``/box magnitudes, not the matcher's own cost
        coefficients — an extreme coefficient (never a data input, only ever a fixed constructor argument) can still
        push the compact path's weighted cost to overflow.

        Sanitizing that overflow inside the compact matrix used to disagree with the full-cartesian fallback, because
        ``_sanitize_cost_matrix``'s replacement sentinel is computed from each matrix's own finite values, and the
        compact matrix's finite-value statistics differ from the full one's (confirmed independently with
        ``cost_class=3e38``, ``seed=58``, ``sizes=[6, 6]``: same non-finite verdict on both matrices, different
        sentinel, different assignment). ``forward`` must instead fall through to the untouched fallback path whenever
        the compact-weighted cost is not finite, so the two never diverge on this branch — verified here by forcing the
        fallback path directly (via ``_detection_inputs_are_safe``) and comparing.
        """
        calls = _spy_on_compact_path(monkeypatch)
        extreme_matcher = HungarianMatcher(cost_class=3e38, cost_bbox=1, cost_giou=1)
        outputs, targets = _random_detection_batch(seed=58, sizes=[6, 6])

        actual = extreme_matcher(outputs, targets)
        assert calls == [1], "compact path must still be attempted once before falling through"
        assert extreme_matcher._warned_non_finite_costs, "overflow must be detected and warned about once"

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t: False))
        expected = extreme_matcher(outputs, targets)

        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"


class TestNonFiniteLogitsWithoutGateSweep:
    """``_detection_inputs_are_safe`` no longer sweeps ``pred_logits``, so the post-hoc finiteness check on the built
    compact matrix carries the whole burden of matching the full path's assignment.

    Every non-finite logit is either consumed by its own image's diagonal block — where it must reach
    ``compact_cost_matrix`` and force fall-through — or lands somewhere neither path's extracted diagonal blocks read,
    where the compact path must keep running and still return the full path's indices. The batch below pins that
    partition explicitly: classes ``0``/``1`` belong to image 0, classes ``2``/``3`` to image 1, and class ``4`` to
    nobody.
    """

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="neg_inf"),
            pytest.param(float("nan"), id="nan"),
        ],
    )
    def test_consumed_non_finite_logit_falls_through_to_full_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher, bad_value: float
    ) -> None:
        """A non-finite logit in a class column image 0 itself labels must survive the focal formula and the weighted
        sum into the compact matrix, so ``forward`` falls through to the full path and returns its exact indices."""
        compact_calls = _spy_on_compact_path(monkeypatch)
        full_calls = _spy_on_full_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=301, labels_per_image=[[0, 1], [2, 3]])
        outputs["pred_logits"][0, 0, 0] = bad_value

        actual = matcher(outputs, targets)

        assert compact_calls == [1], "the compact matrix must still be attempted before falling through"
        assert full_calls == [1], "a consumed non-finite logit must force the full path to run"
        _assert_same_indices(actual, _full_path_indices(matcher, outputs, targets))

    def test_cross_image_only_non_finite_logit_stays_on_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A non-finite logit in a class only *another* image labels lands in a cross-image block that both paths
        discard, so the compact path must keep running and still agree with the full path.

        The full path materializes that block and would warn; the compact path never builds it. Losing that
        ``logger.warning`` is the disclosed cost of dropping the gate's ``pred_logits`` sweep, asserted here so the
        trade-off is pinned rather than assumed.
        """
        compact_calls = _spy_on_compact_path(monkeypatch)
        full_calls = _spy_on_full_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=302, labels_per_image=[[0, 1], [2, 3]])
        outputs["pred_logits"][0, 0, 2] = float("inf")

        actual = matcher(outputs, targets)

        assert compact_calls == [1]
        assert full_calls == [], "a cross-image-only non-finite logit must not force the full path"
        assert not matcher._warned_non_finite_costs, "the compact path never sees the cross-image block, so never warns"
        _assert_same_indices(actual, _full_path_indices(matcher, outputs, targets))

    def test_never_consumed_non_finite_logit_stays_on_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A non-finite logit in a class column no image labels is read by neither path, so the compact path must keep
        running and still agree with the full path — the old gate rejected this batch for nothing."""
        compact_calls = _spy_on_compact_path(monkeypatch)
        full_calls = _spy_on_full_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=303, labels_per_image=[[0, 1], [2, 3]])
        outputs["pred_logits"][0, 0, 4] = float("nan")

        actual = matcher(outputs, targets)

        assert compact_calls == [1]
        assert full_calls == [], "a never-labelled class column must not force the full path"
        _assert_same_indices(actual, _full_path_indices(matcher, outputs, targets))

    def test_zero_cost_class_still_falls_through_on_consumed_non_finite_logit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``cost_class=0`` is the one arithmetic path that could plausibly annihilate a non-finite class cost, and the
        constructor permits it as long as another coefficient is non-zero.

        It does not annihilate: ``0 * inf`` and ``0 * nan`` are both NaN, so the weighted sum stays non-finite and the
        consumed-logit batch still falls through to the full path.
        """
        compact_calls = _spy_on_compact_path(monkeypatch)
        full_calls = _spy_on_full_path(monkeypatch)
        zero_class_matcher = HungarianMatcher(cost_class=0.0, cost_bbox=1.0, cost_giou=1.0)
        outputs, targets = _detection_batch_with_labels(seed=304, labels_per_image=[[0, 1], [2, 3]])
        outputs["pred_logits"][0, 0, 0] = float("inf")

        actual = zero_class_matcher(outputs, targets)

        assert compact_calls == [1]
        assert full_calls == [1], "0 * inf is NaN, so the compact matrix must still be rejected"
        _assert_same_indices(actual, _full_path_indices(zero_class_matcher, outputs, targets))


class TestGateRoutesInputsThatDivergeBetweenPaths:
    """Inputs the two paths would treat differently must be routed to the full path, so the compact path never becomes
    the reason a batch behaves differently.

    ``pad_sequence`` allocates from the first sequence and silently casts the rest, and ``torch.gather`` rejects class
    indices the full path's ``flat_pred_logits[:, tgt_ids]`` accepts. None of these inputs is reachable from a shipped
    data path; the gate keeps them on the path whose behavior is already established.
    """

    def test_mixed_dtype_target_boxes_use_full_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A float64 ``boxes`` tensor anywhere but index 0 would be silently downcast into the padded tensor, making the
        achieved precision depend on batch ordering — the gate must route it to the full path, whose ``torch.cat``
        promotes to float64 and then fails loudly against float32 predictions."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=305, labels_per_image=[[0, 1], [2, 3]])
        targets[1]["boxes"] = targets[1]["boxes"].double()

        with pytest.raises(RuntimeError, match="expected scalar type Float but found Double"):
            matcher(outputs, targets)

        assert calls == []

    def test_negative_label_uses_full_path_and_keeps_wrap_semantics(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """``torch.gather`` rejects a negative class index, while the full path's ``flat_pred_logits[:, tgt_ids]`` wraps
        it Python-style onto the last class.

        The gate routes such a batch to the full path, so a label of ``-1`` keeps scoring against the last class rather
        than becoming a new hard error. Preserving that wrap is a deliberate back-compat choice, not an endorsement of
        it: the compact path's rejection is arguably the more correct behavior.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=306, labels_per_image=[[0, 1], [2, 3]])
        wrapped_outputs, wrapped_targets = _detection_batch_with_labels(seed=306, labels_per_image=[[0, 1], [2, 3]])
        targets[0]["labels"][0] = -1
        wrapped_targets[0]["labels"][0] = 4  # num_classes - 1, the class -1 wraps onto

        actual = matcher(outputs, targets)

        assert calls == []
        _assert_same_indices(actual, _full_path_indices(matcher, wrapped_outputs, wrapped_targets))

    def test_out_of_range_label_uses_full_path_and_keeps_index_error(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A label at or above ``num_classes`` raised ``IndexError`` before the compact path existed; ``torch.gather``
        would raise ``RuntimeError`` instead, breaking any caller narrowing on ``IndexError``.

        The gate routes the batch to the full path so the original exception type survives.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=307, labels_per_image=[[0, 1], [2, 3]])
        targets[0]["labels"][0] = 99

        with pytest.raises(IndexError):
            matcher(outputs, targets)

        assert calls == []

    def test_in_range_labels_at_both_bounds_still_use_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """The label-range check must not over-reject: class ``0`` and class ``num_classes - 1`` are both valid, and a
        batch using only those two must still take the compact path and match the full path's assignment."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _detection_batch_with_labels(seed=308, labels_per_image=[[0, 4], [4, 0]])

        actual = matcher(outputs, targets)

        assert calls == [1]
        _assert_same_indices(actual, _full_path_indices(matcher, outputs, targets))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCompactPathOnCUDA:
    """The compact path's tensor ops (``pad_sequence``, gather, ``cdist``, ``torch.vmap(generalized_box_iou)``) must
    actually run under real CUDA kernels and agree with the fallback path there — CPU-only tests cannot exercise CUDA-
    specific numerics or catch a CUDA-only failure in these ops, and the CI GPU workflow only selects tests marked
    ``gpu`` (``ci-tests-gpu.yml`` runs ``-m gpu``), so without this class the compact path had zero coverage under the
    device it optimizes for.

    bf16 (the dtype the A100 benchmarks in this PR's body use) is not exercised here: this machine's PyTorch/CUDA build
    does not implement ``cdist`` for bf16 (``cdist_cuda not implemented for BFloat16``, verified directly against
    ``torch.cdist`` before writing this test) — a pre-existing PyTorch/CUDA-build limitation unrelated to this PR, not
    something a test on this machine can respect or route around. The A100 bf16 numbers remain inherited from the
    exploration behind this PR, not re-run here; this class verifies float32 CUDA execution and CPU/CUDA/fallback
    agreement, which is what this machine can actually run and check.
    """

    def test_compact_path_matches_fallback_on_cuda_float32(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        outputs, targets = _random_detection_batch(seed=301, sizes=[2, 4, 1])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        calls = _spy_on_compact_path(monkeypatch)
        actual = matcher(outputs, targets)
        assert calls == [1]
        assert all(query.device.type == "cpu" for query, _ in actual), "assignment indices must return on CPU"

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t: False))
        expected = matcher(outputs, targets)

        for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
            assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"

    def test_compact_path_matches_cpu_reference_on_cuda(self, matcher: HungarianMatcher) -> None:
        """Same inputs, CPU vs CUDA: the compact path must reach the identical assignment regardless of the device the
        model happens to run on."""
        outputs, targets = _random_detection_batch(seed=302, sizes=[2, 3, 1])

        cpu_result = matcher(outputs, targets)

        cuda_outputs = {key: value.cuda() for key, value in outputs.items()}
        cuda_targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        cuda_result = matcher(cuda_outputs, cuda_targets)

        for image_idx, ((cpu_q, cpu_t), (cuda_q, cuda_t)) in enumerate(zip(cpu_result, cuda_result)):
            assert torch.equal(cpu_q, cuda_q), f"query indices diverged for image {image_idx}"
            assert torch.equal(cpu_t, cuda_t), f"target indices diverged for image {image_idx}"


class TestCompactPathCriterionEquivalence:
    """The exploration behind this PR claims the 17 criterion losses (main + 2 aux decoder layers + encoder, each with
    ``labels``/``boxes``/``cardinality``) and their gradients are byte-identical between the compact and fallback paths,
    but that claim lived only in an ad hoc script under ``state/``, not as a persistent test in this repo — a real gap
    found on further review of this diff.

    This backs it with an actual ``SetCriterion`` + ``HungarianMatcher`` pair (not a reimplementation of either), a
    heterogeneous batch with one empty image, real ``aux_outputs``/``enc_outputs`` so all 4 matcher invocations
    ``forward`` makes per training step are exercised (not just the last-layer call), and both losses and gradients
    checked, not just losses.
    """

    def test_losses_and_gradients_match_between_compact_and_fallback_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rfdetr.models.criterion import SetCriterion

        torch.manual_seed(401)
        bs, num_queries, num_classes = 3, 8, 5
        sizes = [2, 0, 3]

        def make_layer_outputs() -> dict[str, torch.Tensor]:
            return {
                "pred_logits": torch.randn(bs, num_queries, num_classes, requires_grad=True),
                "pred_boxes": (torch.rand(bs, num_queries, 4) * 0.4 + 0.3).clone().requires_grad_(True),
            }

        main_outputs = make_layer_outputs()
        aux_outputs = [make_layer_outputs(), make_layer_outputs()]
        enc_outputs = make_layer_outputs()
        outputs = {**main_outputs, "aux_outputs": aux_outputs, "enc_outputs": enc_outputs}
        all_layer_outputs = [main_outputs, *aux_outputs, enc_outputs]

        targets = [
            {
                "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
                "boxes": torch.rand(size, 4) * 0.4 + 0.3,
            }
            for size in sizes
        ]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
        )

        calls = _spy_on_compact_path(monkeypatch)
        compact_losses = criterion(outputs, targets, num_boxes=1.0)
        assert calls == [1, 1, 1, 1], "main + 2 aux layers + enc must each take the compact path once"
        assert len(compact_losses) == 17, "main + 2 aux + enc, each with cardinality/class_error/bbox/giou"
        sum(compact_losses.values()).backward()
        compact_grads = [
            (layer["pred_logits"].grad.clone(), layer["pred_boxes"].grad.clone()) for layer in all_layer_outputs
        ]

        for layer in all_layer_outputs:
            layer["pred_logits"].grad = None
            layer["pred_boxes"].grad = None
        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t: False))
        fallback_losses = criterion(outputs, targets, num_boxes=1.0)
        sum(fallback_losses.values()).backward()

        assert compact_losses.keys() == fallback_losses.keys()
        for key in compact_losses:
            assert torch.equal(compact_losses[key], fallback_losses[key]), f"{key} diverged"
        for layer, (compact_grad_logits, compact_grad_boxes) in zip(all_layer_outputs, compact_grads):
            assert torch.equal(compact_grad_logits, layer["pred_logits"].grad)
            assert torch.equal(compact_grad_boxes, layer["pred_boxes"].grad)
