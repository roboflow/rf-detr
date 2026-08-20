# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from rfdetr.evaluation.matching import (
    _compute_mask_iou,
    _match_single_class_segm,
    _match_sorted_iou_matrix,
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)

# ---------------------------------------------------------------------------
# _compute_mask_iou
# ---------------------------------------------------------------------------


class TestComputeMaskIou:
    """Unit tests for the private _compute_mask_iou helper."""

    @staticmethod
    def _bool_mask(h: int, w: int, rows: slice, cols: slice) -> torch.Tensor:
        """Return a [1, h, w] boolean mask with the specified region set to True."""
        m = torch.zeros(h, w, dtype=torch.bool)
        m[rows, cols] = True
        return m.unsqueeze(0)

    def test_identical_masks_give_iou_one(self) -> None:
        """Masks that are identical should produce IoU of exactly 1.0."""
        mask = self._bool_mask(4, 4, slice(0, 2), slice(0, 2))  # [1, 4, 4]
        result = _compute_mask_iou(mask, mask)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) == pytest.approx(1.0)

    def test_disjoint_masks_give_iou_zero(self) -> None:
        """Non-overlapping masks should produce IoU of 0.0."""
        pred = self._bool_mask(4, 4, slice(0, 2), slice(0, 2))
        gt = self._bool_mask(4, 4, slice(2, 4), slice(2, 4))
        result = _compute_mask_iou(pred, gt)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_known_partial_overlap(self) -> None:
        """50% row overlap on a 4x4 grid: inter=4, union=12, IoU=1/3."""
        pred = torch.zeros(1, 4, 4, dtype=torch.bool)
        pred[0, :2, :] = True  # rows 0-1: 8 px
        gt = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt[0, 1:3, :] = True  # rows 1-2: 8 px — 4 px of overlap at row 1
        result = _compute_mask_iou(pred, gt)
        assert float(result[0, 0]) == pytest.approx(4.0 / 12.0)

    def test_empty_masks_return_zero_without_error(self) -> None:
        """All-zero masks must yield IoU 0.0 (no divide-by-zero)."""
        pred = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt = torch.zeros(1, 4, 4, dtype=torch.bool)
        result = _compute_mask_iou(pred, gt)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_output_shape_is_n_by_m(self) -> None:
        """Output shape must be [N, M] for N predictions and M ground truths."""
        pred = torch.zeros(3, 4, 4, dtype=torch.bool)
        gt = torch.zeros(5, 4, 4, dtype=torch.bool)
        result = _compute_mask_iou(pred, gt)
        assert result.shape == (3, 5)


# ---------------------------------------------------------------------------
# _match_sorted_iou_matrix
# ---------------------------------------------------------------------------


class TestMatchSortedIouMatrix:
    """Direct unit tests for the private _match_sorted_iou_matrix helper.

    Every other test in this file drives this function indirectly, through ``_match_single_class_segm``,
    ``_accumulate_bbox_class``, or ``build_matching_data``. These cases call it directly with small hand-built numpy IoU
    matrices, so a regression here is localized to this one function instead of surfacing only as a mismatch several
    call frames up.
    """

    def test_disjoint_and_matched_gts_split_into_tp_and_fp(self) -> None:
        """One detection with sufficient IoU is a TP; one with insufficient IoU is a FP.

        Two detections, each with its own GT (no competition between them), isolates the
        threshold comparison from the greedy claiming logic: row 0's IoU (0.9) clears the 0.5
        threshold against its own GT, row 1's IoU (0.1) does not clear it against its own GT.
        """
        iou_matrix_sorted = np.array([[0.9, 0.0], [0.0, 0.1]], dtype=np.float32)
        gt_crowd_np = np.zeros(2, dtype=np.bool_)

        matches, ignore, total_gt = _match_sorted_iou_matrix(iou_matrix_sorted, gt_crowd_np, iou_threshold=0.5)

        np.testing.assert_array_equal(matches, [1, 0])
        np.testing.assert_array_equal(ignore, [False, False])
        assert total_gt == 2

    def test_detection_matched_only_to_crowd_gt_is_ignored_not_fp(self) -> None:
        """A detection whose only sufficient-IoU GT is a crowd instance is ignored, not a FP.

        One detection, one crowd GT, IoU above threshold: the crowd-ignore branch must fire instead of the false-
        positive branch, and the crowd GT must not inflate total_gt.
        """
        iou_matrix_sorted = np.array([[0.9]], dtype=np.float32)
        gt_crowd_np = np.array([True], dtype=np.bool_)

        matches, ignore, total_gt = _match_sorted_iou_matrix(iou_matrix_sorted, gt_crowd_np, iou_threshold=0.5)

        np.testing.assert_array_equal(matches, [0])
        np.testing.assert_array_equal(ignore, [True])
        assert total_gt == 0

    def test_zero_gt_columns_raises_value_error(self) -> None:
        """An IoU matrix with no GT columns (m=0) raises ValueError, not a silent empty result.

        ``np.argmax`` on the per-detection IoU row is unconditional, so a matrix with zero columns fails inside the loop
        instead of returning early. Both call sites (``_accumulate_bbox_class``, ``_match_single_class_segm``) guard
        against calling this function when a class has no GT, so this test documents/locks the current contract rather
        than exercising a path either caller actually reaches.
        """
        iou_matrix_sorted = np.zeros((2, 0), dtype=np.float32)
        gt_crowd_np = np.zeros(0, dtype=np.bool_)

        with pytest.raises(ValueError, match="argmax of an empty sequence"):
            _match_sorted_iou_matrix(iou_matrix_sorted, gt_crowd_np, iou_threshold=0.5)

    def test_zero_pred_rows_returns_empty_arrays_without_error(self) -> None:
        """An IoU matrix with no prediction rows (n=0) returns empty arrays, unlike the m=0 case.

        The greedy loop iterates ``range(len(iou_matrix_sorted))``, which is 0 when there are no predictions, so the
        loop body — and the ``np.argmax`` call that raises for m=0 — never executes. ``total_gt`` is still computed from
        ``gt_crowd_np`` alone, so it is asserted against a non-trivial mix of crowd and non-crowd GTs rather than an
        all-non-crowd matrix.
        """
        iou_matrix_sorted = np.zeros((0, 2), dtype=np.float32)
        gt_crowd_np = np.array([False, True], dtype=np.bool_)

        matches, ignore, total_gt = _match_sorted_iou_matrix(iou_matrix_sorted, gt_crowd_np, iou_threshold=0.5)

        assert matches.shape == (0,)
        assert ignore.shape == (0,)
        assert total_gt == 1


# ---------------------------------------------------------------------------
# Mask rasterisation shared by the segm matcher tests and build_matching_data
# ---------------------------------------------------------------------------


def _masks_from_boxes(boxes: list[list[float]], height: int, width: int) -> torch.Tensor:
    """Rasterise integer-aligned xyxy boxes into a boolean mask stack.

    Lets the ``iou_type="segm"`` path be driven by the same geometry as the ``"bbox"`` path: for
    integer-aligned boxes the pixel-count mask IoU equals the box-area IoU exactly, so both paths
    see the same IoU matrix and any difference in the result comes from the ranking alone.

    Args:
        boxes: Rows of ``[x1, y1, x2, y2]`` in pixel coordinates.
        height: Height of each rasterised mask.
        width: Width of each rasterised mask.

    Returns:
        Boolean tensor of shape ``[len(boxes), height, width]``.

    Examples:
        >>> _masks_from_boxes([[0, 0, 2, 1]], height=1, width=4).int().tolist()
        [[[1, 1, 0, 0]]]
    """
    masks = torch.zeros(len(boxes), height, width, dtype=torch.bool)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        masks[index, int(y1) : int(y2), int(x1) : int(x2)] = True
    return masks


def _reference_greedy_match(
    order: list[int],
    iou_matrix: torch.Tensor,
    gt_crowd: torch.Tensor,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Naive, independently-written greedy matcher used as a ground-truth oracle.

    Reimplements the COCO greedy-matching contract directly from spec with plain Python loops and
    no shared code with ``_match_single_class_segm``/``_match_sorted_iou_matrix`` (no numpy
    vectorization, no hoisted crowd mask) — a divergence in the optimized/numpy-ported
    implementation (e.g. a tie-break or crowd-handling regression) will disagree with this
    reference instead of silently agreeing with itself. Module-level because it is shared by
    ``TestMatchSingleClassSegm`` (segm path, one class at a time) and
    ``TestBuildMatchingDataBboxDifferential`` (bbox path, one class-slice at a time).

    Args:
        order: Detection indices into ``iou_matrix``'s rows, in descending-score order.
        iou_matrix: Pairwise IoU, ``[n_preds, n_gt]``, rows in original (unsorted) order.
        gt_crowd: Bool tensor ``[n_gt]``, ``True`` for crowd instances.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple ``(matches, ignore, total_gt)`` aligned to ``order``.

    Examples:
        >>> order = [0, 1]
        >>> iou_matrix = torch.tensor([[0.9, 0.0], [0.0, 0.9]])
        >>> gt_crowd = torch.zeros(2, dtype=torch.bool)
        >>> matches, ignore, total_gt = _reference_greedy_match(order, iou_matrix, gt_crowd, 0.5)
        >>> matches.tolist()
        [1, 1]
        >>> total_gt
        2
    """
    n_gt = iou_matrix.shape[1]
    gt_matched = [False] * n_gt
    matches = np.zeros(len(order), dtype=np.int64)
    ignore = np.zeros(len(order), dtype=np.bool_)
    for out_i, orig_i in enumerate(order):
        best_iou, best_gt = -1.0, -1
        for j in range(n_gt):
            if gt_crowd[j] or gt_matched[j]:
                continue
            iou = float(iou_matrix[orig_i, j])
            if iou > best_iou:
                best_iou, best_gt = iou, j
        if best_gt != -1 and best_iou >= iou_threshold:
            matches[out_i] = 1
            gt_matched[best_gt] = True
            continue
        for j in range(n_gt):
            if gt_crowd[j] and float(iou_matrix[orig_i, j]) >= iou_threshold:
                ignore[out_i] = True
                break
    total_gt = int((~gt_crowd.numpy()).sum())
    return matches, ignore, total_gt


# ---------------------------------------------------------------------------
# _match_single_class_segm
# ---------------------------------------------------------------------------


class TestMatchSingleClassSegm:
    """Unit tests for the private _match_single_class_segm helper.

    Cases are written as xyxy boxes and rasterised with ``_masks_from_boxes``, because for integer-aligned boxes the
    pixel-count mask IoU equals the box-area IoU exactly — the expected IoU of each case can be read straight off the
    coordinates. ``test_uses_mask_overlap_not_bounding_box`` is the deliberate exception, and builds masks that no box
    could describe.
    """

    #: Side of the square canvas every box below is rasterised onto — larger than any coordinate used.
    _CANVAS = 64

    @classmethod
    def _mask(cls, *coords: float) -> torch.Tensor:
        """Return a [1, H, W] boolean mask covering the single xyxy box *coords*."""
        return _masks_from_boxes([list(coords)], cls._CANVAS, cls._CANVAS)

    @classmethod
    def _masks(cls, *rows: list[float]) -> torch.Tensor:
        """Return an [N, H, W] boolean mask stack from a sequence of [x1,y1,x2,y2] rows."""
        return _masks_from_boxes(list(rows), cls._CANVAS, cls._CANVAS)

    @staticmethod
    def _random_boxes(count: int, canvas: int) -> torch.Tensor:
        """Return *count* random integer-aligned xyxy boxes that fit inside a *canvas*-square image."""
        top_left = torch.randint(0, canvas // 2, (count, 2))
        extent = torch.randint(1, canvas // 2 + 1, (count, 2))
        return torch.cat([top_left, top_left + extent], dim=1).float()

    def _run(
        self,
        pred_scores: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_crowd: torch.Tensor | None = None,
        iou_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Run the matcher, defaulting every GT to non-crowd."""
        if gt_crowd is None:
            gt_crowd = torch.zeros(len(gt_masks), dtype=torch.bool)
        return _match_single_class_segm(pred_scores, pred_masks, gt_masks, gt_crowd, iou_threshold)

    def test_perfect_overlap_is_tp(self) -> None:
        """A prediction whose mask is identical to the GT mask is a true positive."""
        scores = torch.tensor([0.9])
        mask = self._mask(0, 0, 10, 10)
        _, matches, ignore, total_gt = self._run(scores, mask, mask)
        assert matches[0] == 1
        assert not ignore[0]
        assert total_gt == 1

    def test_disjoint_mask_is_fp(self) -> None:
        """A prediction with no overlap with the GT mask is a false positive."""
        scores = torch.tensor([0.9])
        pred = self._mask(0, 0, 10, 10)
        gt = self._mask(50, 50, 60, 60)
        _, matches, ignore, total_gt = self._run(scores, pred, gt)
        assert matches[0] == 0
        assert not ignore[0]
        assert total_gt == 1

    def test_iou_below_threshold_is_fp(self) -> None:
        """A detection with IoU < threshold must be marked as FP."""
        scores = torch.tensor([0.9])
        pred = self._mask(0, 0, 5, 10)  # area = 50
        gt = self._mask(6, 0, 10, 10)  # area = 40 — no overlap
        _, matches, _, _ = self._run(scores, pred, gt, iou_threshold=0.5)
        assert matches[0] == 0

    def test_greedy_matching_higher_score_wins(self) -> None:
        """When two predictions compete for one GT, the higher-score pred wins."""
        # Sorted descending: [0.9, 0.5] -> first gets TP, second gets FP.
        scores = torch.tensor([0.5, 0.9])
        preds = self._masks([0, 0, 10, 10], [0, 0, 10, 10])
        gt = self._mask(0, 0, 10, 10)
        scores_out, matches, _, _ = self._run(scores, preds, gt)
        assert list(scores_out) == pytest.approx([0.9, 0.5])
        assert matches[0] == 1  # highest score -> TP
        assert matches[1] == 0  # lower score -> FP

    def test_crowd_gt_match_is_ignored_not_fp(self) -> None:
        """A detection matched to a crowd GT is ignored, not a false positive."""
        scores = torch.tensor([0.9])
        mask = self._mask(0, 0, 10, 10)
        gt_crowd = torch.tensor([True])
        _, matches, ignore, total_gt = self._run(scores, mask, mask, gt_crowd=gt_crowd)
        assert matches[0] == 0  # not TP
        assert ignore[0]  # ignored -> not counted as FP
        assert total_gt == 0  # crowd GT excluded from denominator

    def test_non_crowd_gt_counts_in_total_gt(self) -> None:
        """Non-crowd GTs are counted in total_gt."""
        scores = torch.tensor([0.9])
        mask = self._mask(0, 0, 10, 10)
        gt_crowd = torch.tensor([False])
        _, _, _, total_gt = self._run(scores, mask, mask, gt_crowd=gt_crowd)
        assert total_gt == 1

    def test_mixed_crowd_only_non_crowd_in_total_gt(self) -> None:
        """Only non-crowd instances contribute to total_gt."""
        scores = torch.tensor([0.9])
        pred = self._mask(0, 0, 5, 5)  # overlaps neither GT significantly
        gt_masks = self._masks([0, 0, 10, 10], [20, 20, 30, 30])
        gt_crowd = torch.tensor([False, True])  # second GT is crowd
        _, _, _, total_gt = self._run(scores, pred, gt_masks, gt_crowd=gt_crowd)
        assert total_gt == 1

    def test_scores_returned_in_descending_order(self) -> None:
        """Output scores must be sorted in descending order."""
        scores = torch.tensor([0.3, 0.9, 0.6])
        preds = self._masks([0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50])
        gt = self._mask(20, 20, 30, 30)
        scores_out, _, _, _ = self._run(scores, preds, gt)
        assert list(scores_out) == pytest.approx([0.9, 0.6, 0.3])

    def test_uses_mask_overlap_not_bounding_box(self) -> None:
        """Matching is driven by pixel overlap, not by the region the masks span.

        Two interleaved comb masks share no pixel at all (mask IoU 0.0), yet the boxes bounding them overlap at exactly
        the 0.5 threshold — geometry that a box-area proxy would score as a TP and true mask overlap scores as a FP.
        This is the one case in the class whose masks are built directly rather than rasterised from boxes, because no
        box can describe them.
        """
        pred = torch.zeros(1, 4, 4, dtype=torch.bool)
        pred[0, :, ::2] = True  # columns 0 and 2 -> bounding box (0, 0, 3, 4)
        gt = ~pred  # columns 1 and 3 -> bounding box (1, 0, 4, 4), box IoU 8/16 = 0.5
        scores = torch.tensor([0.9])
        _, matches, ignore, total_gt = self._run(scores, pred, gt)
        assert matches[0] == 0
        assert not ignore[0]
        assert total_gt == 1

    def test_greedy_loop_does_not_sync_a_tensor_per_detection(self) -> None:
        """Regression test for #416: the per-detection greedy loop must not force one device-to-host tensor sync
        (``Tensor.__bool__``) per detection, and the numpy-ported matching output must agree with an independent
        reference implementation of the same algorithm at scale (matches/ignore/total_gt) — the sync-count assert alone
        cannot catch a logic regression in the torch->numpy port (e.g. a tie-break or crowd-handling change).

        The geometry is random but integer-aligned, so ``box_iou`` over the source boxes reproduces the mask IoU the
        matcher computes exactly, and the oracle stays independent of ``_compute_mask_iou``.
        """
        n, m, canvas = 50, 10, 32
        scores = torch.rand(n)
        pred_boxes = self._random_boxes(n, canvas)
        gt_boxes = self._random_boxes(m, canvas)
        pred_masks = _masks_from_boxes(pred_boxes.tolist(), canvas, canvas)
        gt_masks = _masks_from_boxes(gt_boxes.tolist(), canvas, canvas)
        gt_crowd = torch.zeros(m, dtype=torch.bool)
        gt_crowd[:3] = True  # exercise the crowd/ignore branch at scale, not just n<=3 cases

        call_count = 0
        orig_bool = torch.Tensor.__bool__

        def counting_bool(self: torch.Tensor) -> bool:
            nonlocal call_count
            call_count += 1
            return orig_bool(self)

        with patch.object(torch.Tensor, "__bool__", counting_bool):
            scores_out, matches, ignore, total_gt = self._run(scores, pred_masks, gt_masks, gt_crowd=gt_crowd)

        assert call_count < n, (
            f"_match_single_class_segm triggered {call_count} tensor->bool syncs for n={n} "
            "detections; expected O(1) syncs, not O(n) — the greedy loop should operate "
            "on host data, not per-iteration device tensor comparisons"
        )

        order = torch.argsort(scores, descending=True).tolist()
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        ref_matches, ref_ignore, ref_total_gt = _reference_greedy_match(order, iou_matrix, gt_crowd, 0.5)
        assert list(scores_out) == pytest.approx(scores[order].tolist())
        assert np.array_equal(matches, ref_matches)
        assert np.array_equal(ignore, ref_ignore)
        assert total_gt == ref_total_gt


# ---------------------------------------------------------------------------
# build_matching_data
# ---------------------------------------------------------------------------

# Geometry whose greedy matching outcome depends on the order the two detections are ranked in.
# The two GTs overlap; the narrow detection can only reach the first, the wide one can reach either:
#
#     IoU(narrow, GT-A) = 0.700    IoU(narrow, GT-B) = 0.133
#     IoU(wide,   GT-A) = 0.667    IoU(wide,   GT-B) = 0.538
#
# Ranking the narrow detection first yields two TPs (it claims GT-A, the wide one falls back to
# GT-B); ranking the wide one first yields one (it claims GT-A, and GT-B is out of the narrow
# detection's reach).
_ORDER_SENSITIVE_GT_BOXES = [[0, 0, 100, 10], [50, 0, 150, 10]]
_NARROW_PRED_BOX = [0, 0, 70, 10]
_WIDE_PRED_BOX = [20, 0, 120, 10]
# Padding that overlaps neither GT, so it cannot change the outcome, and pushes the detection count
# past the ~32-element cutoff above which torch's default (unstable) sort permutes tied scores.
_FILLER_PRED_BOX = [200, 0, 210, 10]
_NUM_FILLER_PREDS = 38
_ORDER_SENSITIVE_MASK_SIZE = (10, 210)


class TestBuildMatchingData:
    """Unit tests for build_matching_data()."""

    @staticmethod
    def _make_pred(
        boxes: list,
        scores: list,
        labels: list,
        masks: torch.Tensor | None = None,
        scores_dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=scores_dtype),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if masks is not None:
            d["masks"] = masks
        return d

    @staticmethod
    def _make_target(
        boxes: list,
        labels: list,
        iscrowd: list | None = None,
        masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if iscrowd is not None:
            d["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
        if masks is not None:
            d["masks"] = masks
        return d

    def test_output_has_required_keys(self) -> None:
        """Every class entry must contain scores, matches, ignore, total_gt."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10]], [0])
        result = build_matching_data([pred], [target])
        assert 0 in result
        assert set(result[0].keys()) == {"scores", "matches", "ignore", "total_gt"}

    def test_perfect_detection_is_tp(self) -> None:
        """A pred box identical to the GT box must be a TP."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10]], [0])
        result = build_matching_data([pred], [target])
        assert result[0]["matches"][0] == 1
        assert result[0]["total_gt"] == 1

    def test_disjoint_box_is_fp(self) -> None:
        """A pred box with no overlap against any GT must be a FP."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[50, 50, 60, 60]], [0])
        result = build_matching_data([pred], [target])
        assert result[0]["matches"][0] == 0
        assert result[0]["total_gt"] == 1

    def test_no_predictions_records_total_gt_only(self) -> None:
        """With no preds for a class, total_gt is recorded but scores list is empty."""
        pred = self._make_pred([], [], [])
        target = self._make_target([[0, 0, 10, 10]], [0])
        result = build_matching_data([pred], [target])
        assert result[0]["total_gt"] == 1
        assert len(result[0]["scores"]) == 0

    def test_no_gts_all_predictions_are_fp(self) -> None:
        """With no GTs for a class, all predictions are FP and total_gt is 0."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([], [])
        result = build_matching_data([pred], [target])
        assert result[0]["matches"][0] == 0
        assert result[0]["total_gt"] == 0

    def test_multi_class_results_are_separated(self) -> None:
        """Two classes in the same image must be tracked independently."""
        pred = self._make_pred([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 1])
        target = self._make_target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])
        result = build_matching_data([pred], [target])
        assert result[0]["matches"][0] == 1
        assert result[1]["matches"][0] == 1
        assert result[0]["total_gt"] == 1
        assert result[1]["total_gt"] == 1

    def test_bbox_iou_is_computed_once_per_image(self) -> None:
        """BBox matching shares one image-wide IoU matrix while preserving class-local matches."""
        pred = self._make_pred(
            [[0, 0, 10, 10], [0, 0, 10, 10], [40, 40, 50, 50]],
            [0.8, 0.95, 0.9],
            [0, 1, 0],
        )
        target = self._make_target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])

        with patch("rfdetr.evaluation.matching.box_iou", wraps=box_iou) as box_iou_spy:
            result = build_matching_data([pred], [target])

        box_iou_spy.assert_called_once_with(pred["boxes"], target["boxes"])
        np.testing.assert_allclose(result[0]["scores"], [0.9, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(result[0]["matches"], [0, 1])
        np.testing.assert_array_equal(result[1]["matches"], [0])
        assert result[0]["total_gt"] == 1
        assert result[1]["total_gt"] == 1

    @pytest.mark.parametrize("iou_type", [pytest.param("bbox", id="bbox"), pytest.param("segm", id="segm")])
    def test_tied_scores_are_ranked_in_input_order_on_both_iou_types(self, iou_type: str) -> None:
        """Detections tied on score are ranked in input order on the bbox path and the segm path alike.

        The two paths ran through one matcher before the per-image IoU matrix was introduced, so
        they broke ties identically; they must still agree afterwards. This is the scenario that
        makes a divergence visible: 40 same-class detections, the two order-sensitive ones tied at
        0.5 and the padding scoring above them, which places the tie group where ``torch.argsort``'s
        default unstable sort reverses it and ``np.argsort(kind="stable")`` does not. Ranked in
        input order the narrow detection claims GT-A and the wide one falls back to GT-B; reversed,
        the wide detection takes GT-A and the narrow one is left with a GT it cannot reach.
        """
        boxes = [_NARROW_PRED_BOX, _WIDE_PRED_BOX, *([_FILLER_PRED_BOX] * _NUM_FILLER_PREDS)]
        scores = [0.5, 0.5, *([0.9] * _NUM_FILLER_PREDS)]
        pred = self._make_pred(
            boxes,
            scores,
            [0] * len(scores),
            masks=_masks_from_boxes(boxes, *_ORDER_SENSITIVE_MASK_SIZE),
        )
        target = self._make_target(
            _ORDER_SENSITIVE_GT_BOXES,
            [0, 0],
            masks=_masks_from_boxes(_ORDER_SENSITIVE_GT_BOXES, *_ORDER_SENSITIVE_MASK_SIZE),
        )

        result = build_matching_data([pred], [target], iou_type=iou_type)

        # Output is in descending-score order, so the higher-scoring padding comes first.
        np.testing.assert_array_equal(result[0]["matches"], [0] * _NUM_FILLER_PREDS + [1, 1])
        assert result[0]["total_gt"] == 2

    def test_near_tied_float64_scores_are_ranked_at_full_precision(self) -> None:
        """Float64 scores that collapse onto one float32 value still rank by their full precision.

        ``0.50000001`` and ``0.50000002`` are distinct in float64 and the same number in float32, so
        casting to float32 before the ranking hands greedy priority to whichever detection came
        first in the input instead of to the higher-scoring one. Here the wide detection is first in
        the input but scores lower: ranked at full precision the narrow detection claims GT-A and
        the wide one falls back to GT-B, while a collapsed ranking gives GT-A to the wide detection
        and leaves the narrow one nothing to match. The returned scores stay float32 either way —
        only the ordering is allowed to see the caller's dtype.
        """
        boxes = [_WIDE_PRED_BOX, _NARROW_PRED_BOX, *([_FILLER_PRED_BOX] * _NUM_FILLER_PREDS)]
        scores = [0.50000001, 0.50000002, *([0.5] * _NUM_FILLER_PREDS)]
        pred = self._make_pred(boxes, scores, [0] * len(scores), scores_dtype=torch.float64)
        target = self._make_target(_ORDER_SENSITIVE_GT_BOXES, [0, 0])

        result = build_matching_data([pred], [target])

        np.testing.assert_array_equal(result[0]["matches"], [1, 1, *([0] * _NUM_FILLER_PREDS)])
        assert result[0]["scores"].dtype == np.float32

    def test_multi_image_batch_accumulates(self) -> None:
        """Two-image batch must concatenate scores and sum total_gt."""
        pred1 = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target1 = self._make_target([[0, 0, 10, 10]], [0])
        pred2 = self._make_pred([[50, 50, 60, 60]], [0.8], [0])
        target2 = self._make_target([[50, 50, 60, 60]], [0])
        result = build_matching_data([pred1, pred2], [target1, target2])
        assert len(result[0]["scores"]) == 2
        assert result[0]["total_gt"] == 2

    def test_crowd_gt_excluded_from_total_and_detection_ignored(self) -> None:
        """A pred matched to a crowd GT must be ignored; crowd GT not counted."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10]], [0], iscrowd=[1])
        result = build_matching_data([pred], [target])
        assert result[0]["total_gt"] == 0
        assert result[0]["ignore"][0]
        assert result[0]["matches"][0] == 0

    def test_mixed_crowd_non_crowd_gts(self) -> None:
        """Pred matched to non-crowd GT is TP; crowd GT not counted in total_gt."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0], iscrowd=[0, 1])
        result = build_matching_data([pred], [target])
        assert result[0]["total_gt"] == 1
        assert result[0]["matches"][0] == 1
        assert not result[0]["ignore"][0]

    def test_segmentation_iou_type_identical_masks(self) -> None:
        """iou_type='segm' path with identical masks must yield a TP."""
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        pred = {
            "boxes": torch.zeros(1, 4),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "masks": mask,
        }
        target = {
            "boxes": torch.zeros(1, 4),
            "labels": torch.tensor([0]),
            "masks": mask,
        }
        result = build_matching_data([pred], [target], iou_type="segm")
        assert result[0]["matches"][0] == 1
        assert result[0]["total_gt"] == 1

    def test_segmentation_missing_masks_raises_value_error(self) -> None:
        """iou_type='segm' without masks must raise ValueError."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10]], [0])
        with pytest.raises(ValueError, match="masks"):
            build_matching_data([pred], [target], iou_type="segm")

    def test_class_only_in_predictions_is_tracked_as_fp(self) -> None:
        """A class seen only in predictions (no GT) must appear in output as FP."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [99])
        target = self._make_target([[0, 0, 10, 10]], [0])
        result = build_matching_data([pred], [target])
        assert 99 in result
        assert result[99]["total_gt"] == 0
        assert result[99]["matches"][0] == 0

    def test_crowd_gt_without_predictions_is_not_counted(self) -> None:
        """Crowd GTs of a class with no predictions must stay out of total_gt."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target(
            [[0, 0, 10, 10], [50, 50, 60, 60], [70, 70, 80, 80]],
            [0, 7, 7],
            iscrowd=[0, 1, 0],
        )
        result = build_matching_data([pred], [target])
        assert result[7]["total_gt"] == 1
        assert len(result[7]["scores"]) == 0
        assert result[0]["total_gt"] == 1

    def test_iscrowd_length_mismatch_raises(self) -> None:
        """An ``iscrowd`` that does not line up with ``labels`` must be rejected, not counted."""
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 7], iscrowd=[0])
        with pytest.raises(ValueError, match="one entry per GT label"):
            build_matching_data([pred], [target])

    def test_iscrowd_length_mismatch_raises_on_image_without_classes(self) -> None:
        """The mismatch is rejected even on an image with no labels and no predictions to loop over."""
        pred = self._make_pred([], [], [])
        target = self._make_target([], [], iscrowd=[1])
        with pytest.raises(ValueError, match="one entry per GT label"):
            build_matching_data([pred], [target])

    @pytest.mark.parametrize(
        "iscrowd",
        [
            pytest.param(torch.zeros(2, 1, dtype=torch.int64), id="column-vector"),
            pytest.param(torch.tensor(0, dtype=torch.int64), id="scalar"),
        ],
    )
    def test_iscrowd_with_wrong_rank_raises(self, iscrowd: torch.Tensor) -> None:
        """An ``iscrowd`` of the right length but the wrong rank must be rejected, not silently applied.

        A ``[M, 1]`` tensor passes a length-only check, and then every row is a non-empty list and therefore truthy, so
        each GT of a class without predictions would drop out of ``total_gt``.
        """
        pred = self._make_pred([[0, 0, 10, 10]], [0.9], [0])
        target = self._make_target([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 7])
        target["iscrowd"] = iscrowd
        with pytest.raises(ValueError, match="one entry per GT label"):
            build_matching_data([pred], [target])

    @pytest.mark.parametrize("num_classes", [pytest.param(40, id="40-classes"), pytest.param(80, id="80-classes")])
    @pytest.mark.parametrize(
        ("with_preds", "with_gts", "expected_matches", "expected_total_gt"),
        [
            pytest.param(True, True, [1], 1, id="preds-and-gts"),
            pytest.param(False, True, [], 1, id="gt-only-classes"),
            pytest.param(True, False, [0], 0, id="pred-only-classes"),
        ],
    )
    def test_does_not_sync_a_tensor_per_class(
        self,
        num_classes: int,
        with_preds: bool,
        with_gts: bool,
        expected_matches: list[int],
        expected_total_gt: int,
    ) -> None:
        """Regression test: the per-class loop must not force a scalar device-to-host read per class per image — that
        turns an O(1) per-image cost into O(num_classes), which dominates wall time on datasets with hundreds of classes
        (e.g. COCO's 80). The branch cases cover the three exits of the loop, each of which used to sync: the matcher
        path, the ``n_pred == 0`` path (which synced a third time on the non-crowd GT count), and ``n_gt == 0``.

        Two properties are asserted, both exact rather than bounds and both at two class counts, so
        that neither a per-class sync nor a fraction of one can creep back in:

        1. every scalar read out of a tensor (``item``/``__bool__``/``__int__``/``__float__``/
           ``__index__``) is gone, not just ``item()``;
        2. the bulk reads that remain (``tolist``) are exactly three per image — the pred labels, the
           GT labels and ``iscrowd`` — and do not grow with the class count.

        Not covered: ``_match_single_class_segm`` still moves its own scores and IoUs to host with
        ``.cpu()/.numpy()`` once per class that has detections. That is pre-existing and untouched
        here; this test fixes the cost of the classes that never reach the matcher.
        """
        # One pred and one GT per class, on a per-image diagonal grid so each parametrized case
        # drives every class down the same branch.
        boxes = [[i * 20, i * 20, i * 20 + 10, i * 20 + 10] for i in range(num_classes)]
        labels = list(range(num_classes))
        pred = self._make_pred(boxes, [0.9] * num_classes, labels) if with_preds else self._make_pred([], [], [])
        target = self._make_target(boxes, labels) if with_gts else self._make_target([], [])

        scalar_reads = ["item", "__bool__", "__int__", "__float__", "__index__"]
        counts: dict[str, int] = dict.fromkeys([*scalar_reads, "tolist"], 0)
        originals = {name: getattr(torch.Tensor, name) for name in counts}

        def make_counter(name: str) -> Callable[..., object]:
            original = originals[name]

            def counting(self: torch.Tensor, *args: object, **kwargs: object) -> object:
                counts[name] += 1
                return original(self, *args, **kwargs)

            return counting

        with ExitStack() as stack:
            for name in counts:
                stack.enter_context(patch.object(torch.Tensor, name, make_counter(name)))
            result = build_matching_data([pred], [target])

        synced = {name: counts[name] for name in scalar_reads if counts[name]}
        assert not synced, (
            f"build_matching_data triggered scalar tensor->host reads {synced} for "
            f"num_classes={num_classes}; the per-class counts come from the host-side label lists, "
            "so no branch of the loop should read a scalar out of a tensor at all"
        )
        assert counts["tolist"] == 3, (
            f"build_matching_data called Tensor.tolist() {counts['tolist']} times for "
            f"num_classes={num_classes}; it must be exactly three per image (pred labels, GT labels, "
            "iscrowd), independent of how many classes the image contains"
        )
        assert len(result) == num_classes
        for class_id in range(num_classes):
            assert result[class_id]["matches"].tolist() == expected_matches
            assert result[class_id]["total_gt"] == expected_total_gt


# ---------------------------------------------------------------------------
# Randomized differential oracle for build_matching_data(iou_type="bbox")
# ---------------------------------------------------------------------------


def _random_bbox_batch(
    rng: np.random.Generator,
    num_shared_preds: int,
    num_shared_gts: int,
    num_shared_classes: int,
    canvas: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build one random single-image bbox pred/target pair for the differential oracle.

    *num_shared_classes* random detections and GTs are drawn from a shared label pool, so a class
    can end up with predictions and GTs, only predictions, only GTs, or neither (if unlucky) — the
    common case ``_accumulate_bbox_class`` must handle via ``np.ix_`` slicing of the shared
    image-wide IoU matrix. On top of that pool, one extra prediction is always given a
    class id of its own (``num_shared_classes``) with no matching GT, and one extra GT is always
    given a class id of its own (``num_shared_classes + 1``) with no matching prediction — this
    guarantees the two empty-class edges (`gt_indices.size == 0`, and a GT-only class) are
    exercised on every trial rather than only probabilistically.

    Args:
        rng: Seeded NumPy random generator, local to the caller (not the global RNG state).
        num_shared_preds: Number of predictions drawn from the shared class pool.
        num_shared_gts: Number of ground truths drawn from the shared class pool.
        num_shared_classes: Size of the shared class-id pool.
        canvas: Boxes are drawn with coordinates inside a ``canvas``-square area.

    Returns:
        A ``(pred, target)`` pair shaped like one ``build_matching_data()`` batch element, with a
        random ``iscrowd`` flag on every GT.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> pred, target = _random_bbox_batch(
        ...     rng, num_shared_preds=3, num_shared_gts=2, num_shared_classes=2, canvas=16
        ... )
        >>> pred["boxes"].shape, pred["scores"].shape, pred["labels"].shape
        (torch.Size([4, 4]), torch.Size([4]), torch.Size([4]))
        >>> target["boxes"].shape, target["labels"].shape, target["iscrowd"].shape
        (torch.Size([3, 4]), torch.Size([3]), torch.Size([3]))
    """

    def _random_boxes(count: int) -> np.ndarray:
        top_left = rng.uniform(0, canvas / 2, size=(count, 2))
        extent = rng.uniform(1, canvas / 2, size=(count, 2))
        return np.concatenate([top_left, top_left + extent], axis=1)

    pred_only_class = num_shared_classes
    gt_only_class = num_shared_classes + 1

    pred_boxes = np.concatenate([_random_boxes(num_shared_preds), _random_boxes(1)], axis=0)
    pred_labels = np.append(rng.integers(0, num_shared_classes, size=num_shared_preds), pred_only_class)
    pred_scores = rng.uniform(0.01, 1.0, size=num_shared_preds + 1)

    gt_boxes = np.concatenate([_random_boxes(num_shared_gts), _random_boxes(1)], axis=0)
    gt_labels = np.append(rng.integers(0, num_shared_classes, size=num_shared_gts), gt_only_class)
    gt_crowd = rng.integers(0, 2, size=num_shared_gts + 1)

    pred = {
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
        "scores": torch.tensor(pred_scores, dtype=torch.float32),
        "labels": torch.tensor(pred_labels, dtype=torch.int64),
    }
    target = {
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.tensor(gt_labels, dtype=torch.int64),
        "iscrowd": torch.tensor(gt_crowd, dtype=torch.int64),
    }
    return pred, target


def _reference_bbox_matching(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    iou_threshold: float,
) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    """Independently compute build_matching_data's bbox-path output, per class, as an oracle.

    Mirrors the sharing strategy ``_accumulate_bbox_class`` uses (one image-wide ``box_iou``
    matrix, sliced per class) so that a per-class slicing bug — e.g. a wrong ``np.ix_`` index —
    disagrees with this independent per-class computation instead of silently agreeing with
    itself. The actual greedy matching for each class is delegated to
    ``_reference_greedy_match``, which reimplements the matching contract from spec rather than
    reusing any of ``rfdetr.evaluation.matching``.

    Args:
        pred: One image's predictions with ``boxes``, ``scores``, ``labels``.
        target: One image's ground truths with ``boxes``, ``labels``, and optional ``iscrowd``.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Dict mapping ``class_id`` to ``(matches, ignore, total_gt)``, with ``matches``/``ignore``
        in the same per-class, descending-score order ``build_matching_data()`` returns.

    Examples:
        >>> pred = {
        ...     "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        ...     "scores": torch.tensor([0.9]),
        ...     "labels": torch.tensor([0]),
        ... }
        >>> target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}
        >>> reference = _reference_bbox_matching(pred, target, iou_threshold=0.5)
        >>> reference[0][0].tolist(), reference[0][2]
        ([1], 1)
    """
    pred_labels_np = pred["labels"].numpy()
    gt_labels_np = target["labels"].numpy()
    scores_np = pred["scores"].numpy()
    gt_crowd = target.get("iscrowd", torch.zeros(len(gt_labels_np), dtype=torch.int64)).bool()
    iou_matrix = box_iou(pred["boxes"], target["boxes"])

    reference: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for class_id in sorted(set(pred_labels_np.tolist()) | set(gt_labels_np.tolist())):
        pred_idx = np.flatnonzero(pred_labels_np == class_id)
        gt_idx = np.flatnonzero(gt_labels_np == class_id)

        if pred_idx.size == 0:
            reference[class_id] = (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.bool_),
                int((~gt_crowd[gt_idx]).sum()),
            )
            continue
        if gt_idx.size == 0:
            reference[class_id] = (
                np.zeros(pred_idx.size, dtype=np.int64),
                np.zeros(pred_idx.size, dtype=np.bool_),
                0,
            )
            continue

        order = np.argsort(-scores_np[pred_idx], kind="stable")
        sub_iou = iou_matrix[np.ix_(pred_idx[order], gt_idx)]
        matches, ignore, total_gt = _reference_greedy_match(
            list(range(len(order))), sub_iou, gt_crowd[gt_idx], iou_threshold
        )
        reference[class_id] = (matches, ignore, total_gt)
    return reference


def _assert_matching_equals_reference(
    result: dict[int, dict[str, np.ndarray | int]],
    reference: dict[int, tuple[np.ndarray, np.ndarray, int]],
) -> None:
    """Assert build_matching_data's per-class output equals a (matches, ignore, total_gt) reference.

    Both ``build_matching_data()`` and ``_reference_bbox_matching()`` key on ``class_id`` and keep
    detections in per-image descending-score order, so comparing arrays directly is valid without
    re-sorting either side.

    Args:
        result: Output of ``build_matching_data()``.
        reference: ``class_id -> (matches, ignore, total_gt)`` from ``_reference_bbox_matching()``.

    Examples:
        >>> result = {0: {"matches": np.array([1]), "ignore": np.array([False]), "total_gt": 1}}
        >>> reference = {0: (np.array([1]), np.array([False]), 1)}
        >>> _assert_matching_equals_reference(result, reference)
    """
    assert set(result) == set(reference)
    for class_id, (ref_matches, ref_ignore, ref_total_gt) in reference.items():
        np.testing.assert_array_equal(result[class_id]["matches"], ref_matches)
        np.testing.assert_array_equal(result[class_id]["ignore"], ref_ignore)
        assert result[class_id]["total_gt"] == ref_total_gt


class TestBuildMatchingDataBboxDifferential:
    """Randomized differential-oracle coverage for build_matching_data(iou_type="bbox").

    ``test_greedy_loop_does_not_sync_a_tensor_per_detection`` (see ``TestMatchSingleClassSegm``) is
    this suite's only other randomized differential oracle, and it exercises the segm path only.
    The bbox path shares one image-wide ``box_iou`` matrix across classes and slices it per class
    via ``np.ix_`` (``_accumulate_bbox_class``); every other bbox-path test in
    ``TestBuildMatchingData`` hand-writes 2-3 detections, none of them drives this class-slicing
    with more than one populated class at a time.
    """

    @pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(50)])
    def test_matches_independent_reference_across_random_batches(self, seed: int) -> None:
        """build_matching_data(iou_type="bbox") agrees with an independent per-class oracle.

        Each of the 50 independently-seeded trials builds one random single-image batch with several shared classes (so
        the ``np.ix_`` per-class slice of the one image-wide ``box_iou`` matrix is exercised for more than one class per
        trial), a random ``iscrowd`` flag on every GT, one prediction-only class, and one GT-only class (see
        ``_random_bbox_batch``). Many small, cheap trials are used instead of one large trial: each trial covers only a
        handful of classes/detections, so the seed sweep covers far more of the class/crowd/empty combinations across 50
        trials than a single larger draw would, for a comparable total detection count.
        """
        rng = np.random.default_rng(seed)
        pred, target = _random_bbox_batch(rng, num_shared_preds=8, num_shared_gts=8, num_shared_classes=4, canvas=32)

        result = build_matching_data([pred], [target], iou_type="bbox")

        reference = _reference_bbox_matching(pred, target, iou_threshold=0.5)
        _assert_matching_equals_reference(result, reference)


# ---------------------------------------------------------------------------
# Helper shared by TestMergeMatchingData and TestDistributedMergeMatchingData
# (used by multiple classes, so module-level rather than a staticmethod)
# ---------------------------------------------------------------------------


def _make_matching_entry(
    scores: list,
    matches: list,
    ignore: list,
    total_gt: int,
) -> dict:
    """Return a compact matching dict as produced by ``build_matching_data()``.

    Examples:
        >>> entry = _make_matching_entry([0.9, 0.5], [1, -1], [False, False], 2)
        >>> entry["total_gt"]
        2
        >>> [round(float(x), 3) for x in entry["scores"]]
        [0.9, 0.5]
    """
    return {
        "scores": np.array(scores, dtype=np.float32),
        "matches": np.array(matches, dtype=np.int64),
        "ignore": np.array(ignore, dtype=bool),
        "total_gt": total_gt,
    }


class TestInitMatchingAccumulator:
    """init_matching_accumulator() returns a correct empty accumulator."""

    def test_returns_empty_dict(self) -> None:
        """Returns an empty dict."""
        assert init_matching_accumulator() == {}

    def test_returned_dict_is_mutable_via_merge(self) -> None:
        """The returned dict can be populated by merge_matching_data."""
        acc = init_matching_accumulator()
        merge_matching_data(acc, {0: _make_matching_entry([0.9], [1], [False], 1)})
        assert 0 in acc


class TestMergeMatchingData:
    """merge_matching_data() correctly accumulates per-class matching dicts."""

    def test_empty_accumulator_copies_new_data(self) -> None:
        """First merge populates the accumulator with the batch data."""
        data = _make_matching_entry([0.9, 0.8], [1, 0], [False, False], 1)
        acc = merge_matching_data({}, {0: data})
        np.testing.assert_allclose(acc[0]["scores"], [0.9, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(acc[0]["matches"], [1, 0])
        assert acc[0]["total_gt"] == 1

    def test_second_merge_concatenates_arrays_and_sums_total_gt(self) -> None:
        """Merging a second batch appends scores/matches/ignore and sums total_gt."""
        acc: dict = {}
        merge_matching_data(acc, {0: _make_matching_entry([0.9], [1], [False], 2)})
        merge_matching_data(acc, {0: _make_matching_entry([0.7], [0], [False], 3)})
        np.testing.assert_allclose(acc[0]["scores"], [0.9, 0.7], rtol=1e-6)
        np.testing.assert_array_equal(acc[0]["matches"], [1, 0])
        assert acc[0]["total_gt"] == 5

    def test_new_class_added_independently(self) -> None:
        """A class not yet in the accumulator is added without touching others."""
        acc = {0: _make_matching_entry([0.9], [1], [False], 1)}
        merge_matching_data(acc, {1: _make_matching_entry([0.5], [0], [False], 2)})
        assert acc[0]["total_gt"] == 1
        assert acc[1]["total_gt"] == 2

    def test_returns_same_accumulator_object(self) -> None:
        """merge_matching_data returns the same dict it was given (in-place)."""
        acc: dict = {}
        result = merge_matching_data(acc, {})
        assert result is acc

    def test_no_op_when_new_data_is_empty(self) -> None:
        """Merging an empty batch leaves the accumulator unchanged."""
        acc = {0: _make_matching_entry([0.9], [1], [False], 1)}
        merge_matching_data(acc, {})
        assert len(acc) == 1
        assert acc[0]["total_gt"] == 1

    def test_copied_arrays_are_independent_of_source(self) -> None:
        """Mutating the source entry after the first merge must not corrupt acc."""
        data = _make_matching_entry([0.9], [1], [False], 1)
        acc: dict = {}
        merge_matching_data(acc, {0: data})
        data["scores"][0] = 0.0
        assert acc[0]["scores"][0] == pytest.approx(0.9)

    def test_multiple_classes_in_single_batch_all_added(self) -> None:
        """All classes present in a single batch are merged into the accumulator."""
        batch = {
            0: _make_matching_entry([0.9], [1], [False], 1),
            1: _make_matching_entry([0.8], [0], [False], 2),
        }
        acc = merge_matching_data({}, batch)
        assert set(acc.keys()) == {0, 1}
        assert acc[0]["total_gt"] == 1
        assert acc[1]["total_gt"] == 2


class TestDistributedMergeMatchingData:
    """distributed_merge_matching_data() gathers and merges across DDP ranks."""

    def test_single_rank_returns_same_content(self) -> None:
        """In single-process mode (world_size=1), data passes through unchanged."""
        local_data = {0: _make_matching_entry([0.9], [1], [False], 1)}
        result = distributed_merge_matching_data(local_data)
        np.testing.assert_allclose(result[0]["scores"], [0.9], rtol=1e-6)
        assert result[0]["total_gt"] == 1

    def test_two_ranks_disjoint_classes(self) -> None:
        """Two ranks with disjoint classes -> merged result contains both."""
        rank0 = {0: _make_matching_entry([0.9], [1], [False], 1)}
        rank1 = {1: _make_matching_entry([0.7], [0], [False], 2)}
        with patch("rfdetr.evaluation.matching.all_gather", return_value=[rank0, rank1]):
            result = distributed_merge_matching_data(rank0)
        assert set(result.keys()) == {0, 1}
        assert result[0]["total_gt"] == 1
        assert result[1]["total_gt"] == 2

    def test_two_ranks_overlapping_class_concatenates(self) -> None:
        """Two ranks sharing class 0 -> arrays concatenated, total_gt summed."""
        rank0 = {0: _make_matching_entry([0.9], [1], [False], 2)}
        rank1 = {0: _make_matching_entry([0.7, 0.5], [0, 1], [False, False], 3)}
        with patch("rfdetr.evaluation.matching.all_gather", return_value=[rank0, rank1]):
            result = distributed_merge_matching_data(rank0)
        np.testing.assert_allclose(result[0]["scores"], [0.9, 0.7, 0.5], rtol=1e-6)
        assert result[0]["total_gt"] == 5

    def test_returns_new_dict_not_input(self) -> None:
        """Result is a new dict, not a reference to the local input."""
        local_data = {0: _make_matching_entry([0.9], [1], [False], 1)}
        result = distributed_merge_matching_data(local_data)
        assert result is not local_data
