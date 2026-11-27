# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``HungarianMatcher`` cost matrix: non-finite sanitization, focal alpha, class-cost gather, stacked cost
construction and the projected-feature mask cost."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from rfdetr.models import matcher as matcher_module
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.matcher import HungarianMatcher
from tests.models._matcher_support import (
    _assert_same_indices,
    _detection_batch_with_labels,
    _full_path_indices,
    _random_detection_batch,
    _spy_on_compact_path,
)


@pytest.fixture()
def standard_target() -> dict[str, torch.Tensor]:
    """Single-class target with one box at (0.5, 0.5, 0.2, 0.2)."""
    return {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
    }


class TestHungarianMatcherNonFiniteCosts:
    """Tests for non-finite cost matrix sanitization in the Hungarian matcher."""

    @pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
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

    def test_replaces_non_finite_entries(self) -> None:
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

    def test_all_non_finite_falls_back_to_dtype_max(self) -> None:
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

    def test_clamps_overflowing_replacement_cost(self) -> None:
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
        _assert_same_indices(actual, expected)

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
        _assert_same_indices(actual, expected)
        assert actual[1][0].shape == (0,)
        assert actual[1][1].shape == (0,)


def _spy_on_full_path(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record one entry per full-path cost build, by tagging the 2-D ``pairwise_box_l1_cost`` call only that path makes
    (the compact path's operands are 3-D) — lets a test tell "stayed on the compact path" apart from "built the compact
    matrix, found it non-finite, and fell through to the full path".

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_full_path(patched)
        ...     _ = HungarianMatcher()(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = matcher_module.pairwise_box_l1_cost

    def spy(x1: torch.Tensor, x2: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if x1.dim() == 2:
            calls.append(1)
        return original(x1, x2, *args, **kwargs)

    monkeypatch.setattr(matcher_module, "pairwise_box_l1_cost", spy)
    return calls


class TestDetectionInputsAreSafeDirectly:
    """Direct, isolated coverage of ``HungarianMatcher._detection_inputs_are_safe``.

    ``TestCompactPathRouting`` in ``test_matcher_compact_path.py`` already exercises this method indirectly, through the
    routing behaviour it controls. These tests call it directly instead, so a future change to *how* it computes its
    answer (e.g. vectorizing the per-image loops into a single concatenated check) has a fast, precise regression guard
    that pins down the exact boolean it must keep returning for each case, independent of the compact-path machinery
    around it.
    """

    @pytest.mark.parametrize(
        "sizes",
        [
            pytest.param([2, 3, 1], id="every_image_has_targets"),
            pytest.param([0, 4, 0, 2], id="some_images_empty"),
            pytest.param([0, 0, 0], id="every_image_empty"),
        ],
    )
    def test_safe_batches_return_true(self, sizes: list[int]) -> None:
        """A batch with finite, in-range boxes and labels is safe, whether every image carries targets, only some do
        (mixed with zero-target images), or none do — the last case is vacuously safe, since there is nothing to
        check."""
        outputs, targets = _random_detection_batch(seed=301, sizes=sizes)

        assert HungarianMatcher._detection_inputs_are_safe(outputs, targets) is True

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(lambda o, t: o["pred_boxes"].__setitem__((1, 0, 2), float("nan")), id="nan_in_pred_boxes"),
            pytest.param(
                lambda o, t: t[2]["boxes"].__setitem__((0, 1), float("inf")),
                id="inf_in_a_middle_image_target_boxes",
            ),
            pytest.param(lambda o, t: o["pred_boxes"].__setitem__((0, 0, 0), 1e30), id="extreme_coordinate"),
            pytest.param(lambda o, t: t[1]["labels"].__setitem__(0, -1), id="negative_label_in_a_middle_image"),
            pytest.param(
                lambda o, t: t[-1]["labels"].__setitem__(0, 5),  # num_classes=5, valid range is [0, 5)
                id="out_of_range_label_in_the_last_image",
            ),
            pytest.param(
                lambda o, t: t[1].__setitem__("boxes", t[1]["boxes"].double()), id="target_box_dtype_mismatch"
            ),
            pytest.param(
                lambda o, t: t[1].__setitem__("labels", t[1]["labels"].to("meta")),
                id="target_label_device_mismatch",
            ),
            pytest.param(
                lambda o, t: t[1].__setitem__("labels", t[1]["labels"].to(torch.int32)),
                id="target_label_dtype_mismatch",
            ),
        ],
    )
    def test_unsafe_inputs_return_false(
        self, corrupt: Callable[[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]], None]
    ) -> None:
        """Each unsafe case must be caught: a non-finite value or out-of-range coordinate/label anywhere in the batch —
        including a MIDDLE image's target boxes/labels and the LAST image's labels, not just the first, which guards
        against a vectorized check that only looks at one image's tensor instead of the concatenation of all of them —
        plus a target whose box dtype, label device, or label dtype disagrees with the predictions/batch (metadata
        prechecks, no kernel launch).

        The device case uses ``torch.device('meta')`` rather than requiring real CUDA hardware, since the device-
        mismatch branch is a plain attribute comparison that returns before any value-touching op runs, so ``meta``
        (shape-only, no real storage) exercises the same code path without a GPU.
        """
        outputs, targets = _random_detection_batch(seed=305, sizes=[2, 3, 2, 4])
        corrupt(outputs, targets)

        assert HungarianMatcher._detection_inputs_are_safe(outputs, targets) is False


class TestNonFiniteLogitsWithoutGateSweep:
    """``_detection_inputs_are_safe`` no longer sweeps ``pred_logits``, so the post-hoc finiteness check on the built
    compact matrix carries the whole burden of matching the full path's assignment.

    Every non-finite logit is either consumed by its own image's diagonal block — where it must reach
    ``compact_cost_matrix`` and force fall-through — or lands somewhere neither path's extracted diagonal blocks read,
    where the compact path must keep running and still return the full path's indices. The batch below pins that
    partition explicitly: classes ``0``/``1`` belong to image 0, classes ``2``/``3`` to image 1, and class ``4`` to
    nobody.
    """

    @pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
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


def _spy_on_assign(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Wrap the static ``_assign_compact_cost_matrix`` to record each cost matrix it receives, without changing its
    behavior — lets a test compare the matrix's numeric content between two ``forward()`` calls.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     captured = _spy_on_assign(patched)
        ...     _ = HungarianMatcher()(outputs, targets)
        >>> len(captured)
        1
    """
    captured: list[torch.Tensor] = []
    original = HungarianMatcher._assign_compact_cost_matrix

    def spy(cost_matrix: torch.Tensor, sizes: list[int], group_detr: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        captured.append(cost_matrix.clone())
        return original(cost_matrix, sizes, group_detr)

    monkeypatch.setattr(HungarianMatcher, "_assign_compact_cost_matrix", staticmethod(spy))
    return captured


class TestMatcherDictMaskCostUsesProjectedFeatures:
    """The dict form of ``pred_masks`` (the ``else`` branch around matcher.py:409) point-samples
    ``outputs["pred_masks"]["spatial_features"]`` straight off the dict for the mask cost — it applies no projection of
    its own, trusting whatever produced the dict.

    Only ``SegmentationHead``-level tests existed before this; none ran a real head's dict output through the matcher.
    This builds the dict with a real ``SegmentationHead`` (``sparse_forward(skip_blocks=True)``, non-identity
    ``spatial_features_proj``) and checks the resulting cost matrix is sensitive to whether that projection was applied.
    """

    def test_cost_matrix_differs_between_projected_and_raw_spatial_features(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real head's projected dict output yields a different mask cost matrix than the raw spatial features."""
        torch.manual_seed(10)
        hidden, mask_size, num_queries = 4, 8, 4
        head = SegmentationHead(in_dim=hidden, num_blocks=1, bottleneck_ratio=1, downsample_ratio=1)
        sizes = [1, 2]
        outputs, targets = _random_detection_batch(seed=11, sizes=sizes, num_queries=num_queries)
        bs = len(targets)
        spatial_features = torch.randn(bs, hidden, mask_size, mask_size)
        query_features = torch.randn(bs, num_queries, hidden)
        for target, size in zip(targets, sizes):
            target["masks"] = torch.rand(size, mask_size, mask_size)

        real_dict = head.sparse_forward(spatial_features, [query_features], (mask_size, mask_size), skip_blocks=True)[0]
        matcher = HungarianMatcher()
        captured = _spy_on_assign(monkeypatch)

        outputs["pred_masks"] = real_dict
        torch.manual_seed(20)  # controls the mask point_coords draw, shared across both calls below
        matcher(outputs, targets)
        real_cost = captured[-1]

        with torch.no_grad():
            # Raw, unprojected spatial_features at the same resolution sparse_forward interpolates to —
            # exactly what the pre-fix skip_blocks=True branch used to hand the matcher.
            raw_resized = torch.nn.functional.interpolate(
                spatial_features, size=(mask_size, mask_size), mode="bilinear", align_corners=False
            )
        corrupted_dict = {
            "spatial_features": raw_resized,
            "query_features": real_dict["query_features"].detach(),
            "bias": real_dict["bias"].detach(),
        }
        outputs["pred_masks"] = corrupted_dict
        torch.manual_seed(20)  # same point_coords draw as the real-dict call above, for a fair comparison
        matcher(outputs, targets)
        corrupted_cost = captured[-1]

        assert not torch.allclose(real_cost, corrupted_cost)


def _spy_on_box_l1_cost(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``pairwise_box_l1_cost`` to record one entry per call (any rank), without changing its result.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_box_l1_cost(patched)
        ...     _ = HungarianMatcher()(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = matcher_module.pairwise_box_l1_cost

    def spy(x1: torch.Tensor, x2: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return original(x1, x2, *args, **kwargs)

    monkeypatch.setattr(matcher_module, "pairwise_box_l1_cost", spy)
    return calls


class TestStackedCostConstruction:
    """``_match_many`` folds compatible layers into one stacked cost-construction pass when the padded per-layer matrix
    is small enough; the stacked pass must match the per-layer loop numerically, and oversized batches must keep the
    per-layer loop (stacking regresses compute-bound dense-crowd shapes — plan M2, L4)."""

    def test_stacked_matrices_match_per_layer_numerically(self) -> None:
        """The stacked pass reproduces every layer's compact cost matrix within floating-point tolerance.

        Stacking changes the leading batch extent of the padded gather/cdist/GIoU operations, so PyTorch may produce
        last-bit differences even though the resulting costs are numerically equivalent.
        """
        matcher = HungarianMatcher()
        layers = []
        for seed in (601, 602, 603):
            layer, _ = _random_detection_batch(seed=seed, sizes=[2, 0, 3])
            layers.append(layer)
        _, targets = _random_detection_batch(seed=601, sizes=[2, 0, 3])

        stacked = matcher._compute_stacked_compact_cost_matrices(layers, targets)
        expected = [
            matcher._compute_compact_detection_cost_matrix(layer, targets, clamp_target_labels=True) for layer in layers
        ]

        assert len(stacked) == len(expected)
        for stacked_matrix, expected_matrix in zip(stacked, expected):
            torch.testing.assert_close(stacked_matrix, expected_matrix, rtol=1e-4, atol=1e-6)

    def test_match_many_results_unchanged_by_stacking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_match_many`` returns identical assignments whether the stacked pass is enabled or disabled.

        Forces both routes on the same inputs by toggling the element limit, so a routing bug cannot hide behind the
        shared solver.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=611, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=612, sizes=[2, 3])
        layers = [outputs, layer2]

        stacked_result = matcher._match_many(layers, targets)
        monkeypatch.setattr(matcher_module, "_STACKED_COST_ELEMENT_LIMIT", 0)
        loop_result = matcher._match_many(layers, targets)

        assert stacked_result is not None and loop_result is not None
        for stacked_indices, loop_indices in zip(stacked_result, loop_result):
            _assert_same_indices(stacked_indices, loop_indices)

    @pytest.mark.parametrize(
        ("limit", "expected_cost_builds"),
        [
            pytest.param(10_000_000, 1, id="under-limit-stacks-once"),
            pytest.param(0, 2, id="over-limit-per-layer-loop"),
        ],
    )
    def test_element_limit_routes_between_stacked_and_per_layer(
        self, monkeypatch: pytest.MonkeyPatch, limit: int, expected_cost_builds: int
    ) -> None:
        """The element limit decides between one stacked ``pairwise_box_l1_cost`` call and one call per layer.

        Counts 3-D ``pairwise_box_l1_cost`` calls inside ``_match_many``: the stacked pass issues exactly one for all
        layers, the per-layer loop one per layer. Oversized batches must take the loop because stacking measured
        0.79-1.03x on dense compute-bound shapes (plan M2).
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=613, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=614, sizes=[2, 3])
        monkeypatch.setattr(matcher_module, "_STACKED_COST_ELEMENT_LIMIT", limit)
        calls = _spy_on_box_l1_cost(monkeypatch)

        result = matcher._match_many([outputs, layer2], targets)

        assert result is not None
        assert len(calls) == expected_cost_builds

    @pytest.mark.parametrize(
        ("layer_count", "expected_cost_builds"),
        [
            pytest.param(5, 1, id="total-at-calibrated-limit-stacks-once"),
            pytest.param(6, 6, id="total-over-calibrated-limit-loops-per-layer"),
        ],
    )
    def test_element_limit_includes_layer_count(
        self, monkeypatch: pytest.MonkeyPatch, layer_count: int, expected_cost_builds: int
    ) -> None:
        """The stacked gate measures the full layer-folded padded matrix, not one layer.

        Each layer contributes ``2 * 10 * 10 = 200`` padded elements. A 1,000-element calibration
        therefore permits five layers but must decline six, which would allocate 1,200 elements in
        the stacked pass. The production 350,000-element limit uses the same calculation.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=617, sizes=[10, 10], num_queries=10)
        layers = [outputs]
        for seed in range(618, 617 + layer_count):
            layer, _ = _random_detection_batch(seed=seed, sizes=[10, 10], num_queries=10)
            layers.append(layer)
        monkeypatch.setattr(matcher_module, "_STACKED_COST_ELEMENT_LIMIT", 1_000)
        calls = _spy_on_box_l1_cost(monkeypatch)

        result = matcher._match_many(layers, targets)

        assert result is not None
        assert len(calls) == expected_cost_builds

    def test_mismatched_query_counts_fall_back_to_per_layer(self) -> None:
        """Layers with different query counts cannot stack and must keep the per-layer loop.

        ``_match_many`` never required equal query counts across layers, so the stacked pass must not introduce that
        requirement as a crash or a wrong-shaped matrix.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=615, sizes=[2, 3], num_queries=6)
        layer2, _ = _random_detection_batch(seed=616, sizes=[2, 3], num_queries=12)
        layers = [outputs, layer2]

        actual = matcher._match_many(layers, targets)
        expected = [matcher(layer, targets) for layer in layers]

        assert actual is not None
        for actual_indices, expected_indices in zip(actual, expected):
            _assert_same_indices(actual_indices, expected_indices)
