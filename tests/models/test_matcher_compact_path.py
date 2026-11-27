# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``HungarianMatcher`` padded-compact detection path: diagonal block extraction, routing and its safety
gate, CUDA parity, criterion equivalence and target-side safety caching."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.models import matcher as matcher_module
from rfdetr.models.matcher import HungarianMatcher, _TargetSideSafety
from tests._markers import _requires_cpu_inductor, _requires_cuda
from tests.models._matcher_support import (
    _assert_assignment_lengths,
    _assert_same_indices,
    _detection_batch_with_labels,
    _full_path_indices,
    _random_detection_batch,
    _spy_on_compact_path,
)


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
        _assert_same_indices(actual, expected)
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
        _assert_same_indices(actual, expected)
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
        _assert_same_indices(actual, expected)
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
        """The new diagonal-block extraction returns the same indices as the old one on arbitrary cost tensors."""
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
        _assert_same_indices(actual, expected)


def _total_assignment_cost(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Total cost an assignment achieves, scored on a single CPU cost matrix built from ``outputs``/``targets``.

    Lets a test compare two assignments produced on different devices without demanding identical index tensors: the
    Hungarian solve is optimal on each per-image block, so any assignment that is not itself optimal scores strictly
    worse, while a genuine tie scores the same. Both sides must be scored through this helper against the *same*
    ``outputs``/``targets`` — scoring each against its own device's matrix would re-import the 1-ULP cross-device
    divergence the comparison exists to tolerate.

    Calls ``_compute_compact_detection_cost_matrix``, so it bumps ``_spy_on_compact_path``; call it only after any
    routing assertion.

    Examples:
        >>> matcher = HungarianMatcher()
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[2, 1])
        >>> round(_total_assignment_cost(matcher, outputs, targets, matcher(outputs, targets)), 6)
        -1.739075
    """
    cpu_outputs = {key: value.cpu() for key, value in outputs.items()}
    cpu_targets = [{key: value.cpu() for key, value in target.items()} for target in targets]
    cost_matrix = matcher._compute_compact_detection_cost_matrix(cpu_outputs, cpu_targets).float()
    target_offset = 0
    total = 0.0
    for (query_indices, target_indices), target in zip(indices, cpu_targets):
        total += float(cost_matrix[query_indices, target_indices + target_offset].sum())
        target_offset += len(target["boxes"])
    return total


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

    def test_batch_size_greater_than_one_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A safe, detection-only, ``bs > 1`` batch must take the compact path."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=102, sizes=[2, 3])

        matcher(outputs, targets)

        assert calls == [1]

    def test_masks_present_uses_fallback_path(self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher) -> None:
        """A small segmentation batch (``masks`` in targets, below the masks-hybrid worth-it threshold) must skip the
        compact path entirely, even with ``bs > 1`` and otherwise-safe inputs.

        See ``TestMasksPresentCompactHybrid`` for the larger-batch case, where a compact route now handles this path's
        class/bbox/GIoU terms.
        """
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
            pytest.param(lambda o, t: t[1]["boxes"].__setitem__((0, 1), 1e30), id="target_box_extreme"),
            pytest.param(
                lambda o, t: t[1].__setitem__("labels", t[1]["labels"].to(torch.int32)),
                id="target_label_dtype_mismatch",
            ),
        ],
    )
    def test_unsafe_inputs_use_fallback_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        matcher: HungarianMatcher,
        corrupt: Callable[[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]], None],
    ) -> None:
        """Each unsafe-input case verified in the exploration script (NaN/Inf on predicted logits, predicted boxes, or
        target boxes, plus one coordinate large enough to risk overflow in ``cdist``/GIoU area terms) must route to the
        fallback path instead of the compact one, for an otherwise compact-eligible (``bs > 1``, detection- only)
        batch."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=105, sizes=[2, 3])
        corrupt(outputs, targets)

        results = matcher(outputs, targets)

        assert calls == []
        assert len(results) == len(targets)

    @pytest.mark.parametrize("seed", [201, 202, 203, 204, 205])
    @pytest.mark.parametrize(
        "costs",
        [
            pytest.param((1, 1, 1), id="unit"),
            pytest.param((2.0, 5.0, 2.0), id="shipped"),
        ],
    )
    def test_matches_pre_pr1_reference_across_seeds(
        self, monkeypatch: pytest.MonkeyPatch, seed: int, costs: tuple[float, float, float]
    ) -> None:
        """The compact path's assignment must agree with the pre-PR1 reference (full materialization + ``split(sizes,
        -1)`` + ``c[i]``) across several random seeds — same contract as ``TestDiagonalBlockExtraction``, now exercised
        through the compact route.

        The coefficient triple is parametrized because the weighted sum
        ``cost_bbox * bbox + cost_class * class + cost_giou * giou`` is invariant to any permutation of its
        coefficients when all three are ``1``, which is what the ``matcher`` fixture supplies: at ``1, 1, 1`` a mutant
        that swaps two of them is a literal no-op and no assertion here can see it. ``2.0, 5.0, 2.0`` is what
        ``_defaults.py`` actually ships, and it detects the ``bbox``/``giou`` and ``bbox``/``class`` swaps. It does not
        detect the ``class``/``giou`` swap: those two coefficients are both ``2.0``, so that permutation is still a
        no-op — this closes two of the three permutation mutants, not all three.
        """
        cost_class, cost_bbox, cost_giou = costs
        weighted_matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=seed, sizes=[2, 4, 1])

        actual = weighted_matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(weighted_matcher, outputs, targets)

        assert calls == [1]
        _assert_same_indices(actual, expected)

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
        _assert_same_indices(actual, expected)
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

    @pytest.mark.parametrize("group_detr", [2, 3])
    def test_group_detr_greater_than_one_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher, group_detr: int
    ) -> None:
        """No other ``TestCompactPathRouting`` case passes ``group_detr > 1``, so nothing pinned that a grouped batch
        actually takes the compact route: ``TestDiagonalBlockExtraction.test_heterogeneous_sizes_with_group_detr``
        already asserts correctness at ``group_detr=2`` but carries no spy, so a regression re-routing grouped batches
        to the full path would leave it green. This closes that routing-assertion gap, not a correctness gap.

        ``group_detr=3`` against the fixture's ``num_queries=6`` leaves 2 queries per group, so image 1's 4 targets
        exceed the group's query count and exercise the ``min(group_num_queries, size)`` truncation inside the group
        loop, which ``group_detr=2`` at ``num_queries=8`` never reaches.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=208, sizes=[2, 4, 1])

        actual = matcher(outputs, targets, group_detr=group_detr)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets, group_detr=group_detr)

        assert calls == [1]
        _assert_same_indices(actual, expected)

    def test_num_queries_indivisible_by_group_detr_raises(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """``_assign_compact_cost_matrix`` rejects a ``group_detr`` that does not divide ``num_queries``, since the
        group slice width ``num_queries // group_detr`` would silently drop the remainder queries.

        Every other ``group_detr`` in this file divides evenly, so the guard itself was the one added line with no test
        exercising it. The compact matrix is still built first, so the error is raised from the compact route.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=209, sizes=[2, 3])

        with pytest.raises(ValueError) as exc_info:
            matcher(outputs, targets, group_detr=4)

        assert "divisible" in str(exc_info.value)
        assert calls == [1]

    @pytest.mark.parametrize("seed", [201, 202, 203])
    def test_float64_inputs_match_pre_pr1_reference(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher, seed: int
    ) -> None:
        """Every other tensor in this file is float32, so ``pad_sequence``/``gather``/``cdist``/``vmap`` had no non-
        float32 coverage at all, and ``torch.finfo(boxes.dtype)`` in the gate was only ever evaluated for one dtype (the
        float64 ``coordinate_limit`` is ``8.4e152``, not ``1.2e18``).

        Predictions *and* target boxes are cast together: the gate's metadata precheck routes any batch whose target
        boxes disagree in dtype with ``pred_boxes`` to the full path, so casting only the targets would silently test
        the full path and assert nothing about the compact one.

        Note this does not test float64 assignment precision — ``forward`` calls ``.float()`` on the compact matrix and
        the reference does the same, so both sides are compared after an fp32 downcast. What it covers is that the
        compact path's tensor ops accept and agree under float64 inputs.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=seed, sizes=[2, 4, 1])
        outputs = {key: value.double() for key, value in outputs.items()}
        targets = [{**target, "boxes": target["boxes"].double()} for target in targets]

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert calls == [1]
        _assert_same_indices(actual, expected)

    @pytest.mark.parametrize(
        ("coordinate", "expected_calls"),
        [
            pytest.param(1.0e18, [1], id="just_below_limit"),
            pytest.param(1.2e18, [], id="just_above_limit"),
        ],
    )
    def test_coordinate_limit_boundary_routes_by_magnitude(
        self,
        monkeypatch: pytest.MonkeyPatch,
        matcher: HungarianMatcher,
        coordinate: float,
        expected_calls: list[int],
    ) -> None:
        """The gate's ``coordinate_limit`` is ``torch.finfo(dtype).max ** 0.5 / 16``, which is ``1.1529e18`` for
        float32. The committed unsafe case uses ``1e30``, twelve orders of magnitude above it, so neither the ``<=``
        comparison nor the ``/16`` headroom divisor was pinned by anything: widening the divisor or flipping the
        comparison to ``<`` changed no observable behavior.

        Bracketing the limit from both sides is what constrains where it sits — a single case above it would still leave
        the divisor free to move.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=210, sizes=[2, 3])
        outputs["pred_boxes"][1, 0, 2] = coordinate

        results = matcher(outputs, targets)

        assert calls == expected_calls
        assert len(results) == len(targets)

    def test_more_targets_than_queries_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """Every committed size tuple keeps ``max(T_i)`` at or below ``num_queries``, so a padded target dimension wider
        than the query dimension never flowed through ``cdist``/``gather``/assignment — the rectangular direction the
        compact matrix is least like the fallback's.

        With ``sizes=[9, 7]`` against ``num_queries=6`` each image can only match 6 of its targets, so the assignment is
        target-truncated rather than query-truncated.
        """
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_detection_batch(seed=211, sizes=[9, 7])

        actual = matcher(outputs, targets)
        expected = _reference_indices_pre_diagonal_extraction(matcher, outputs, targets)

        assert calls == [1]
        _assert_same_indices(actual, expected)
        _assert_assignment_lengths(actual, num_queries=6, sizes=[9, 7])

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

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        expected = extreme_matcher(outputs, targets)

        _assert_same_indices(actual, expected)


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

        with pytest.raises(RuntimeError) as exc_info:
            matcher(outputs, targets)

        assert "expected scalar type Float but found Double" in str(exc_info.value)
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
@_requires_cuda
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

    def test_matches_fallback_on_cuda_float32(self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher) -> None:
        """Under real CUDA kernels the compact path must reach an assignment as good as the fallback's on the same
        device.

        Compared by achieved total cost rather than by index identity: the two paths build matrices of different numel
        (``B*Q*max(T_i)`` vs ``B*Q*sum(T_i)``), which lands them on different kernel tail dispatches and makes their
        costs differ at 1 ULP, so identical indices are not something either path guarantees. Cost equality is the
        property that actually matters and it still catches a real break, because the Hungarian solve is optimal on each
        per-image block: any incorrect assignment scores strictly worse except on an exact tie, which is precisely the
        case worth accepting. Both sides are scored on one common matrix built from the same inputs, since scoring each
        against its own matrix would reintroduce the 1-ULP divergence being tolerated.
        """
        outputs, targets = _random_detection_batch(seed=301, sizes=[2, 4, 1])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        calls = _spy_on_compact_path(monkeypatch)
        actual = matcher(outputs, targets)
        assert calls == [1]
        assert all(query.device.type == "cpu" for query, _ in actual), "assignment indices must return on CPU"
        _assert_assignment_lengths(actual, num_queries=6, sizes=[2, 4, 1])

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        expected = matcher(outputs, targets)

        assert _total_assignment_cost(matcher, outputs, targets, actual) == pytest.approx(
            _total_assignment_cost(matcher, outputs, targets, expected)
        )

    def test_matches_cpu_reference_on_cuda(self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher) -> None:
        """Same inputs, CPU vs CUDA: the compact path must reach an assignment of the same cost regardless of the device
        the model happens to run on.

        Cross-device exact index equality is the more fragile of the two comparisons in this class — a different GPU,
        driver, or PyTorch build changes fused-kernel rounding — so this compares achieved total cost on one common CPU
        matrix instead. Index equality was also the only thing here that would have noticed the compact path silently
        not running, and the fallback reaches an optimal cost too, so the spy assertion below replaces that signal
        rather than dropping it; the length assertion rules out a degenerate empty assignment, which would score ``0.0``
        and pass a cost-only check.
        """
        outputs, targets = _random_detection_batch(seed=302, sizes=[2, 3, 1])
        cpu_result = matcher(outputs, targets)
        cuda_outputs = {key: value.cuda() for key, value in outputs.items()}
        cuda_targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        calls = _spy_on_compact_path(monkeypatch)

        cuda_result = matcher(cuda_outputs, cuda_targets)

        assert calls == [1], "the CUDA batch must take the compact path"
        _assert_assignment_lengths(cuda_result, num_queries=6, sizes=[2, 3, 1])
        assert _total_assignment_cost(matcher, outputs, targets, cuda_result) == pytest.approx(
            _total_assignment_cost(matcher, outputs, targets, cpu_result)
        )

    def test_precomputed_target_side_safety_is_reused_on_cuda(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """The target-side safety precompute must run under real CUDA kernels and its result must be reused by a
        subsequent ``forward`` on the same batch, without recomputing the sweep.

        Everything about the cache is otherwise checked on CPU only, yet the device sync it exists to avoid is a CUDA
        cost that does not exist on CPU: the sweep's ``torch.cat``/``isfinite``/comparison/``torch.stack`` chain and the
        ``bool()`` that ends the gate are what this pins as actually working on device. Reuse is asserted through the
        sweep spy rather than through the returned indices, since a recompute would return the same verdict and be
        invisible in the assignment.
        """
        outputs, targets = _random_detection_batch(seed=309, sizes=[2, 3, 1])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        safety = matcher._precompute_target_side_safety(outputs, targets)

        assert safety is not None, "a safe multi-image detection batch must be eligible for the compact path"
        assert safety.pred_boxes_device.type == "cuda", "the precompute must record the device it ran on"
        assert bool(safety.safe) is True, "this batch is safe, so the CUDA sweep must say so"

        sweep_calls = _spy_on_target_side_precheck(monkeypatch)
        compact_calls = _spy_on_compact_path(monkeypatch)
        indices = matcher(outputs, targets, target_side_safety=safety)

        assert sweep_calls == [], "a CUDA precompute matching this call must be reused, not recomputed"
        assert compact_calls == [1], "the batch is safe, so it must still take the compact path"
        _assert_assignment_lengths(indices, num_queries=6, sizes=[2, 3, 1])

    def test_device_mismatched_target_side_safety_falls_back_on_cuda(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A precomputed safety whose recorded device disagrees with the current CUDA batch must be discarded and the
        sweep recomputed on device, returning the actually-correct verdict.

        This is the device arm of the cache-validity guard against real ``cuda`` vs ``cpu`` device objects, where CPU-
        only runs can compare only a synthesized ``torch.device("cuda")``. Mixed CPU/CUDA is exactly how a stale reuse
        would show up in practice -- a value carried over from a host-side batch -- and trusting it here would route a
        batch with an out-of-range label into the compact path, where ``torch.gather`` raises a device-side assert
        instead of the full path's documented ``IndexError``.
        """
        outputs, targets = _random_detection_batch(seed=310, sizes=[2, 3])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        safety = matcher._precompute_target_side_safety(outputs, targets)
        assert bool(safety.safe) is True, "the batch is safe until corrupted below"

        num_classes = outputs["pred_logits"].shape[-1]
        targets[1]["labels"][0] = num_classes + 5  # actually unsafe: out-of-range label
        stale_safety = safety._replace(pred_boxes_device=torch.device("cpu"))

        sweep_calls = _spy_on_target_side_precheck(monkeypatch)
        result = HungarianMatcher._detection_inputs_are_safe(outputs, targets, stale_safety)

        assert result is False, "a cpu-recorded safety must not answer for a cuda batch"
        assert sweep_calls == [1], "the device mismatch must fall back to recomputing the sweep on CUDA"


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

    @pytest.mark.parametrize("compiled", [False, pytest.param(True, marks=_requires_cpu_inductor)])
    @pytest.mark.parametrize("group_detr", [1, 2])
    def test_losses_and_gradients_match_between_compact_and_fallback_paths(
        self, monkeypatch: pytest.MonkeyPatch, compiled: bool, group_detr: int
    ) -> None:
        """Real criterion losses and gradients agree with the eager full-cartesian reference."""
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
        if compiled:
            matcher.enable_compiled_l1_cost()
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
            group_detr=group_detr,
        )

        calls = _spy_on_compact_path(monkeypatch)
        compact_losses = criterion(outputs, targets, num_boxes=1.0)
        # This small batch is far under _STACKED_COST_ELEMENT_LIMIT, so _match_many serves main +
        # 2 aux layers + enc from one stacked compact-cost pass instead of four per-layer ones.
        assert calls == [1], "one stacked compact pass must serve main + 2 aux layers + enc"
        assert len(compact_losses) == 17, "main + 2 aux + enc, each with cardinality/class_error/bbox/giou"
        sum(compact_losses.values()).backward()
        compact_grads = [
            (layer["pred_logits"].grad.clone(), layer["pred_boxes"].grad.clone()) for layer in all_layer_outputs
        ]

        for layer in all_layer_outputs:
            layer["pred_logits"].grad = None
            layer["pred_boxes"].grad = None
        # The eager reference goes through a matcher that never had compilation enabled.
        criterion.matcher = HungarianMatcher()
        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        fallback_losses = criterion(outputs, targets, num_boxes=1.0)
        sum(fallback_losses.values()).backward()

        assert compact_losses.keys() == fallback_losses.keys()
        for key in compact_losses:
            assert torch.equal(compact_losses[key], fallback_losses[key]), f"{key} diverged"
        for layer, (compact_grad_logits, compact_grad_boxes) in zip(all_layer_outputs, compact_grads):
            assert torch.equal(compact_grad_logits, layer["pred_logits"].grad)
            assert torch.equal(compact_grad_boxes, layer["pred_boxes"].grad)


def _spy_on_target_side_precheck(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``_target_side_precheck`` to record how many times it actually recomputes the target-side compact-path
    safety sweep, without changing its behavior — lets a test assert ``SetCriterion.forward`` reuses one precomputed
    result across its several ``matcher()`` calls instead of recomputing it from scratch on each one.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_target_side_precheck(patched)
        ...     _ = HungarianMatcher()._precompute_target_side_safety(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = HungarianMatcher._target_side_precheck

    def spy(
        pred_boxes_dtype: torch.dtype,
        pred_boxes_device: torch.device,
        num_classes: int,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        calls.append(1)
        return original(pred_boxes_dtype, pred_boxes_device, num_classes, targets)

    monkeypatch.setattr(HungarianMatcher, "_target_side_precheck", staticmethod(spy))
    return calls


def _spy_on_precompute_target_side_safety(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``_precompute_target_side_safety`` to record how many times ``SetCriterion.forward`` calls it at all,
    without changing its behavior — one level above ``_spy_on_target_side_precheck``, which records only the sweep
    behind it, so a test can tell "the criterion decided not to precompute" apart from "it precomputed and the matcher's
    own eligibility rule made that cost no sweep".

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_precompute_target_side_safety(patched)
        ...     _ = HungarianMatcher()._precompute_target_side_safety(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = HungarianMatcher._precompute_target_side_safety

    def spy(self: HungarianMatcher, outputs: dict[str, Any], targets: list[dict[str, Any]]) -> Any:
        calls.append(1)
        return original(self, outputs, targets)

    monkeypatch.setattr(HungarianMatcher, "_precompute_target_side_safety", spy)
    return calls


class TestTargetSideSafetyCaching:
    """``HungarianMatcher.forward()``'s compact-path safety gate has a target-side half (dtype/device consistency, label
    range, target-box finiteness/bounds) that depends only on ``targets`` plus ``pred_boxes`` dtype/device and
    ``num_classes`` -- all identical across the up to ``len(aux_outputs)+2`` ``matcher()`` calls
    ``SetCriterion.forward`` makes with the same ``targets`` in one training step.

    ``SetCriterion.forward`` precomputes it once via ``HungarianMatcher._precompute_target_side_safety`` and reuses it,
    instead of every call recomputing it from scratch.
    """

    def _step_outputs_and_targets(
        self, bs: int = 3, num_queries: int = 8, num_classes: int = 5, sizes: list[int] | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build a main+2aux+enc outputs dict and matching targets, for a criterion.forward() step that makes 4
        matcher() calls with the same targets.

        Examples:
            >>> outputs, targets = TestTargetSideSafetyCaching()._step_outputs_and_targets(bs=2, sizes=[2, 3])
            >>> outputs["pred_logits"].shape, len(outputs["aux_outputs"]), "enc_outputs" in outputs
            (torch.Size([2, 8, 5]), 2, True)
            >>> [len(target["labels"]) for target in targets]
            [2, 3]
        """
        sizes = sizes if sizes is not None else [2, 0, 3]
        torch.manual_seed(402)

        def make_layer_outputs() -> dict[str, torch.Tensor]:
            return {
                "pred_logits": torch.randn(bs, num_queries, num_classes),
                "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
            }

        main_outputs = make_layer_outputs()
        aux_outputs = [make_layer_outputs(), make_layer_outputs()]
        enc_outputs = make_layer_outputs()
        outputs = {**main_outputs, "aux_outputs": aux_outputs, "enc_outputs": enc_outputs}
        targets = [
            {
                "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
                "boxes": torch.rand(size, 4) * 0.4 + 0.3,
            }
            for size in sizes
        ]
        return outputs, targets

    def _criterion(self, num_classes: int, num_keypoints_per_class: list[int] | None = None) -> Any:
        """Build a SetCriterion wired to a fresh HungarianMatcher with the labels/boxes/cardinality losses this
        test class exercises.

        ``num_keypoints_per_class`` configures only the matcher, which is what needs it to build a keypoint cost
        matrix; the criterion's own losses stay detection-only, so a keypoint step here exercises the matcher's
        routing without also pulling in the keypoint losses.

        Examples:
            >>> criterion = TestTargetSideSafetyCaching()._criterion(num_classes=5)
            >>> criterion.num_classes, sorted(criterion.losses)
            (5, ['boxes', 'cardinality', 'labels'])
            >>> TestTargetSideSafetyCaching()._criterion(1, num_keypoints_per_class=[3]).matcher.num_keypoints_per_class
            [3]
        """
        from rfdetr.models.criterion import SetCriterion

        return SetCriterion(
            num_classes=num_classes,
            matcher=HungarianMatcher(num_keypoints_per_class=num_keypoints_per_class),
            weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
        )

    def test_target_side_precheck_computed_once_per_criterion_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: prior to precomputing/reusing this, ``_target_side_precheck`` (the expensive
        target-side sweep) recomputed on every one of the 4 matcher() calls a main+2aux+enc step
        makes -- this pins it to exactly 1."""
        outputs, targets = self._step_outputs_and_targets()
        criterion = self._criterion(num_classes=5)

        calls = _spy_on_target_side_precheck(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [1], (
            f"expected _target_side_precheck to run exactly once for the whole step (precomputed and "
            f"reused across all 4 matcher() calls), got {len(calls)} calls"
        )

    def test_target_side_sweep_skipped_for_masks_below_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No wasted work: the target-side sweep must not run for a masks-present step below the hybrid threshold.

        ``_precompute_target_side_safety`` itself is still called -- it owns the compact-path eligibility rule and
        returns None here -- so what this pins is that it retains the worth-it gate rather than sweeping targets for
        segmentation steps whose hybrid route will not run. Device support is forced true to isolate that predicate.

        Uses ``bs`` matching its two targets and a real ``pred_masks`` tensor on every layer (main/aux/enc) so the full
        ``criterion()`` call actually completes end to end -- a prior version of this test used the default ``bs=3``
        with only 2 targets and no ``pred_masks``, silently swallowing the resulting ``KeyError`` with a bare ``except
        Exception: pass``, so it would have kept passing even if the step crashed before ever reaching the precompute
        short-circuit this test means to exercise.
        """
        bs, num_queries, num_classes, mask_size = 2, 8, 5, 4
        outputs, targets = self._step_outputs_and_targets(
            bs=bs, num_queries=num_queries, num_classes=num_classes, sizes=[2, 3]
        )
        for layer_outputs in (outputs, *outputs["aux_outputs"], outputs["enc_outputs"]):
            layer_outputs["pred_masks"] = torch.rand(bs, num_queries, mask_size, mask_size)
        for target in targets:
            target["masks"] = torch.zeros(len(target["labels"]), mask_size, mask_size, dtype=torch.bool)
        criterion = self._criterion(num_classes=num_classes)
        monkeypatch.setattr(HungarianMatcher, "_mask_compact_device_supported", staticmethod(lambda o: True))
        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 10**9)

        calls = _spy_on_target_side_precheck(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [], "the target-side sweep must not run when the mask hybrid is below threshold"

    @pytest.mark.gpu
    @_requires_cuda
    def test_target_side_precheck_computed_once_for_eligible_masks_on_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An eligible CUDA segmentation step must precompute once and reuse the result across main/aux/enc matching."""
        bs, num_queries, num_classes, mask_size = 2, 8, 5, 4
        outputs, targets = self._step_outputs_and_targets(
            bs=bs, num_queries=num_queries, num_classes=num_classes, sizes=[2, 3]
        )
        for layer_outputs in (outputs, *outputs["aux_outputs"], outputs["enc_outputs"]):
            layer_outputs["pred_logits"] = layer_outputs["pred_logits"].cuda()
            layer_outputs["pred_boxes"] = layer_outputs["pred_boxes"].cuda()
            layer_outputs["pred_masks"] = torch.rand(bs, num_queries, mask_size, mask_size, device="cuda")
        targets = [
            {
                **{key: value.cuda() for key, value in target.items()},
                "masks": torch.zeros(len(target["labels"]), mask_size, mask_size, dtype=torch.bool, device="cuda"),
            }
            for target in targets
        ]
        criterion = self._criterion(num_classes=num_classes)
        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 1)

        calls = _spy_on_target_side_precheck(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [1], "the target-side sweep must be precomputed once and reused across all four matcher calls"

    def test_target_side_sweep_skipped_when_keypoints_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No wasted work: the target-side sweep must not run for a keypoint step either, whose compact path can never
        apply regardless (``pred_keypoints`` in outputs and ``keypoints`` in targets), so HungarianMatcher.forward()
        would never reach the safety gate for it.

        The mask hybrid's below-threshold skip is pinned by the test above; this pins the separate keypoint exclusion
        (``"pred_keypoints" in outputs and "keypoints" in targets[0]``), which otherwise has no matcher-level cache
        test -- ``tests/models/test_criterion_keypoints.py`` drives keypoint losses through a matcher stub and never
        reaches ``_precompute_target_side_safety`` at all.

        Like the mask case, ``_precompute_target_side_safety`` itself still runs and returns None; what this pins is
        that it costs no actual sweep. Real ``pred_keypoints`` on every layer (main/aux/enc) plus a matcher configured
        with this batch's keypoint schema keep the full ``criterion()`` call completing end to end, so the test cannot
        pass by crashing before the short-circuit it means to exercise.
        """
        bs, num_queries, num_classes, num_keypoints, pred_dim = 2, 8, 1, 3, 7
        outputs, targets = self._step_outputs_and_targets(
            bs=bs, num_queries=num_queries, num_classes=num_classes, sizes=[2, 3]
        )
        for layer_outputs in (outputs, *outputs["aux_outputs"], outputs["enc_outputs"]):
            layer_outputs["pred_keypoints"] = torch.randn(bs, num_queries, num_keypoints, pred_dim)
        for target in targets:
            target["keypoints"] = torch.rand(len(target["labels"]), num_keypoints, 3)
        criterion = self._criterion(num_classes=num_classes, num_keypoints_per_class=[num_keypoints])

        calls = _spy_on_target_side_precheck(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [], "the target-side sweep must not run when keypoints are present"

    def test_target_side_sweep_skipped_when_batch_size_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No wasted work: the target-side sweep must not run for bs==1, since the compact path requires bs>1 regardless
        of safety.

        As above, ``_precompute_target_side_safety`` still runs and returns None -- the sweep behind it is what must
        not.
        """
        outputs, targets = self._step_outputs_and_targets(bs=1, sizes=[2])
        criterion = self._criterion(num_classes=5)

        calls = _spy_on_target_side_precheck(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [], "the target-side sweep must not run for bs==1"

    def test_precompute_skipped_when_only_one_matcher_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No wasted work: _precompute_target_side_safety must not run for a step with no aux_outputs and no enc_outputs
        (e.g. aux_loss=False with two_stage=False), since that step makes exactly one matcher() call -- precomputing
        there adds an extra device sync with no repeated call to amortize it over."""
        torch.manual_seed(402)
        bs, num_queries, num_classes = 3, 8, 5
        outputs = {
            "pred_logits": torch.randn(bs, num_queries, num_classes),
            "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {
                "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
                "boxes": torch.rand(size, 4) * 0.4 + 0.3,
            }
            for size in [2, 0, 3]
        ]
        criterion = self._criterion(num_classes=num_classes)

        calls = _spy_on_precompute_target_side_safety(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [], "_precompute_target_side_safety must not run when the step makes only one matcher() call"

    def test_precompute_runs_for_an_aux_only_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mirror of the single-call case, on the first arm of the "more than one matcher() call" condition: an
        ``aux_loss=True, two_stage=False`` step has aux_outputs but no enc_outputs, makes 1 + len(aux_outputs) matcher()
        calls with the same targets, and must therefore precompute.

        Only both-arms-present (every other test in this class) and neither-arm-present (the test above) were pinned
        before, so an ``aux_outputs``-only config -- the common one, since two_stage is off by default -- would have
        kept passing if the condition ever narrowed to require enc_outputs.
        """
        outputs, targets = self._step_outputs_and_targets()
        del outputs["enc_outputs"]
        criterion = self._criterion(num_classes=5)

        calls = _spy_on_precompute_target_side_safety(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [1], (
            "a step with aux_outputs and no enc_outputs still makes several matcher() calls to amortize"
        )

    def test_precompute_runs_for_an_enc_only_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other arm: an ``aux_loss=False, two_stage=True`` step has enc_outputs but no aux_outputs, makes 2
        matcher() calls with the same targets, and must therefore precompute too.

        ``outputs.get("aux_outputs")`` is falsy here (absent key), so this arm rests entirely on the ``"enc_outputs" in
        outputs`` half of the condition -- the half nothing exercised in isolation.
        """
        outputs, targets = self._step_outputs_and_targets()
        del outputs["aux_outputs"]
        criterion = self._criterion(num_classes=5)

        calls = _spy_on_precompute_target_side_safety(monkeypatch)
        criterion(outputs, targets, num_boxes=1.0)

        assert calls == [1], "a step with enc_outputs and no aux_outputs makes two matcher() calls to amortize"

    def test_stale_target_side_safety_falls_back_to_a_correct_fresh_check(self) -> None:
        """A _TargetSideSafety computed against a different pred_boxes dtype/device/num_classes must
        never be trusted blindly: _detection_inputs_are_safe must detect the mismatch and recompute
        fresh, returning the actually-correct answer -- not the stale cached one -- even when the
        stale cached value is wrong in the direction that would silently break safety (claims safe
        when the real check is unsafe)."""
        outputs, targets = _random_detection_batch(seed=303, sizes=[2, 3])
        targets[1]["labels"][0] = -1  # actually unsafe: negative label

        stale_safety = _TargetSideSafety(
            safe=torch.tensor(True),  # wrong on purpose: a real precompute for this batch would be False
            targets=targets,
            pred_boxes_dtype=torch.float64,  # deliberately mismatched (real dtype is float32)
            pred_boxes_device=outputs["pred_boxes"].device,
            num_classes=outputs["pred_logits"].shape[-1],
        )

        assert HungarianMatcher._detection_inputs_are_safe(outputs, targets, stale_safety) is False

    @pytest.mark.parametrize(
        ("mismatched_field", "mismatched_value"),
        [
            pytest.param("pred_boxes_device", torch.device("cuda"), id="device_only"),
            pytest.param("num_classes", 99, id="num_classes_only"),
        ],
    )
    def test_single_field_mismatch_falls_back_to_a_fresh_check(
        self, monkeypatch: pytest.MonkeyPatch, mismatched_field: str, mismatched_value: Any
    ) -> None:
        """Each field of the cache-validity guard must be able to trigger the fallback on its own.

        The test above mismatches the dtype; the device and num_classes fields were only ever asserted *equal*, never
        forced unequal, so a guard that dropped either of them would have kept every existing test green. Here identity
        and dtype are left matching and exactly one field is corrupted, which isolates that field's clause of the
        ``targets is targets and dtype == ... and device == ... and num_classes == ...`` chain: the stale cached
        ``safe=True`` must be discarded, the sweep recomputed once, and the actually-correct ``False`` returned.

        Constructing ``torch.device("cuda")`` allocates nothing and needs no CUDA runtime, so the device arm runs on a
        CPU-only machine as a genuine device mismatch rather than as a skip.
        """
        outputs, targets = _random_detection_batch(seed=307, sizes=[2, 3])
        num_classes = outputs["pred_logits"].shape[-1]
        targets[1]["labels"][0] = num_classes + 5  # actually unsafe: out-of-range label

        stale_safety = _TargetSideSafety(
            safe=torch.tensor(True),  # wrong on purpose: a real precompute for this batch would be False
            targets=targets,
            pred_boxes_dtype=outputs["pred_boxes"].dtype,
            pred_boxes_device=outputs["pred_boxes"].device,
            num_classes=num_classes,
        )._replace(**{mismatched_field: mismatched_value})

        calls = _spy_on_target_side_precheck(monkeypatch)
        result = HungarianMatcher._detection_inputs_are_safe(outputs, targets, stale_safety)

        assert result is False, f"a {mismatched_field} mismatch must not be answered from the stale cached value"
        assert calls == [1], f"a {mismatched_field} mismatch must fall back to recomputing the target-side sweep"

    def test_caching_does_not_change_losses_or_gradients_versus_recomputing_every_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The equivalence this PR actually depends on -- reusing one precomputed _TargetSideSafety across a step's
        several matcher() calls returns the same safety verdict, and therefore the same indices/losses/gradients, as
        recomputing the target-side sweep fresh on every call (the pre-caching behavior) -- was previously untested: the
        existing TestCompactPathCriterionEquivalence test forces _detection_inputs_are_safe to always return False,
        which checks compact-vs-full-path equivalence (a pre-existing invariant), not cache-vs-fresh-recompute
        equivalence for the compact path this PR actually changed.

        Runs a real main+2aux+enc step once with caching active, and once with
        HungarianMatcher._precompute_target_side_safety forced to return None -- which makes
        _detection_inputs_are_safe's ``target_side_safety is not None`` check fail on every one of the 4 matcher()
        calls, so each independently recomputes _target_side_precheck from scratch, exactly the behavior before this PR
        introduced caching. Both must take the compact path (this batch is safe) and must produce identical losses and
        gradients.
        """
        torch.manual_seed(403)
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
        criterion = self._criterion(num_classes=num_classes)

        cached_calls = _spy_on_target_side_precheck(monkeypatch)
        cached_losses = criterion(outputs, targets, num_boxes=1.0)
        assert cached_calls == [1], "caching must recompute the target-side sweep exactly once for the whole step"
        sum(cached_losses.values()).backward()
        cached_grads = [
            (layer["pred_logits"].grad.clone(), layer["pred_boxes"].grad.clone()) for layer in all_layer_outputs
        ]
        for layer in all_layer_outputs:
            layer["pred_logits"].grad = None
            layer["pred_boxes"].grad = None

        monkeypatch.undo()
        monkeypatch.setattr(
            HungarianMatcher,
            "_precompute_target_side_safety",
            lambda self, outputs, targets: None,
        )
        # Isolate the pre-existing sequential matcher path: PR3's batched matcher intentionally
        # computes this shared predicate once even without a precomputed cache.
        monkeypatch.setattr(HungarianMatcher, "_match_many", lambda self, outputs_list, targets, **kwargs: None)
        fresh_calls = _spy_on_target_side_precheck(monkeypatch)
        fresh_losses = criterion(outputs, targets, num_boxes=1.0)
        assert fresh_calls == [1, 1, 1, 1], "without caching, all 4 matcher() calls must recompute independently"
        sum(fresh_losses.values()).backward()

        assert cached_losses.keys() == fresh_losses.keys()
        for key in cached_losses:
            assert torch.equal(cached_losses[key], fresh_losses[key]), f"{key} diverged"
        for layer, (cached_logits_grad, cached_boxes_grad) in zip(all_layer_outputs, cached_grads):
            assert torch.equal(cached_logits_grad, layer["pred_logits"].grad)
            assert torch.equal(cached_boxes_grad, layer["pred_boxes"].grad)

    def test_target_side_safety_from_a_different_batch_falls_back_to_a_correct_fresh_check(self) -> None:
        """Regression for a real cross-batch staleness bug: dtype/device/num_classes are essentially constant for an
        entire training run (same model, same precision, same class count), so a _TargetSideSafety legitimately
        precomputed for one batch and then reused for a *different* batch would pass the dtype/device/num_classes check
        even though its cached `safe` no longer reflects the current targets at all.

        Precompute a safety value against a genuinely safe batch A, then call _detection_inputs_are_safe for an
        unrelated batch B that shares dtype/device/num_classes but has an out-of-range label (actually unsafe). Reusing
        A's cached `safe=True` for B would route an unsafe batch into the compact path via forward(), which crashes with
        an out-of-bounds RuntimeError instead of the full path's documented IndexError -- confirmed by reverting the
        identity check and re-running this test, which then fails on this method's own assertion (the gate wrongly
        reports the batch safe).
        """
        matcher = HungarianMatcher()
        outputs_a, targets_a = _random_detection_batch(seed=305, sizes=[2, 3])
        safety_from_batch_a = matcher._precompute_target_side_safety(outputs_a, targets_a)
        assert bool(safety_from_batch_a.safe) is True

        outputs_b, targets_b = _random_detection_batch(seed=306, sizes=[2, 3])
        assert outputs_b["pred_boxes"].dtype == safety_from_batch_a.pred_boxes_dtype
        assert outputs_b["pred_boxes"].device == safety_from_batch_a.pred_boxes_device
        assert outputs_b["pred_logits"].shape[-1] == safety_from_batch_a.num_classes
        targets_b[1]["labels"][0] = outputs_b["pred_logits"].shape[-1] + 5  # actually unsafe: out-of-range label

        assert HungarianMatcher._detection_inputs_are_safe(outputs_b, targets_b, safety_from_batch_a) is False

    def test_matching_target_side_safety_is_reused_without_recomputing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mirror case: when the precomputed _TargetSideSafety's dtype/device/num_classes DO match the current call,
        _detection_inputs_are_safe must reuse it (skip _target_side_precheck) rather than recomputing -- otherwise the
        dtype/device/num_classes match check would be pointless overhead with no actual caching benefit."""
        outputs, targets = _random_detection_batch(seed=304, sizes=[2, 3])
        matcher = HungarianMatcher()
        safety = matcher._precompute_target_side_safety(outputs, targets)

        calls = _spy_on_target_side_precheck(monkeypatch)
        result = HungarianMatcher._detection_inputs_are_safe(outputs, targets, safety)

        assert result is True
        assert calls == [], "a matching _TargetSideSafety must be reused, not recomputed"
