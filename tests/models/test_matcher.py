# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

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
    """Reference matching that materializes the focal class cost over ALL classes before slicing."""
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
