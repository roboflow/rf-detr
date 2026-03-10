# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from unittest.mock import patch

import numpy as np
import pytest
import torch

from rfdetr.evaluation.matching import (
    _compute_mask_iou,
    _match_single_class,
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
# _match_single_class
# ---------------------------------------------------------------------------


class TestMatchSingleClass:
    """Unit tests for the private _match_single_class helper."""

    @staticmethod
    def _box(*coords: float) -> torch.Tensor:
        """Return a [1, 4] float32 box tensor from (x1, y1, x2, y2)."""
        return torch.tensor([list(coords)], dtype=torch.float32)

    @staticmethod
    def _boxes(*rows: list[float]) -> torch.Tensor:
        """Return an [N, 4] float32 tensor from a sequence of [x1,y1,x2,y2] rows."""
        return torch.tensor(list(rows), dtype=torch.float32)

    def _run(
        self,
        pred_scores: torch.Tensor,
        pred_items: torch.Tensor,
        gt_items: torch.Tensor,
        gt_crowd: torch.Tensor | None = None,
        iou_threshold: float = 0.5,
        iou_type: str = "bbox",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if gt_crowd is None:
            gt_crowd = torch.zeros(len(gt_items), dtype=torch.bool)
        return _match_single_class(pred_scores, pred_items, gt_items, gt_crowd, iou_threshold, iou_type)

    def test_perfect_overlap_is_tp(self) -> None:
        """A prediction that perfectly overlaps the GT box is a true positive."""
        scores = torch.tensor([0.9])
        box = self._box(0, 0, 10, 10)
        _, matches, ignore, total_gt = self._run(scores, box, box)
        assert matches[0] == 1
        assert not ignore[0]
        assert total_gt == 1

    def test_disjoint_box_is_fp(self) -> None:
        """A prediction with no overlap with the GT box is a false positive."""
        scores = torch.tensor([0.9])
        pred = self._box(0, 0, 10, 10)
        gt = self._box(50, 50, 60, 60)
        _, matches, ignore, total_gt = self._run(scores, pred, gt)
        assert matches[0] == 0
        assert not ignore[0]
        assert total_gt == 1

    def test_iou_below_threshold_is_fp(self) -> None:
        """A detection with IoU < threshold must be marked as FP."""
        scores = torch.tensor([0.9])
        pred = self._box(0, 0, 5, 10)  # area = 50
        gt = self._box(6, 0, 10, 10)  # area = 40 — no overlap
        _, matches, _, _ = self._run(scores, pred, gt, iou_threshold=0.5)
        assert matches[0] == 0

    def test_greedy_matching_higher_score_wins(self) -> None:
        """When two predictions compete for one GT, the higher-score pred wins."""
        # Sorted descending: [0.9, 0.5] -> first gets TP, second gets FP.
        scores = torch.tensor([0.5, 0.9])
        preds = self._boxes([0, 0, 10, 10], [0, 0, 10, 10])
        gt = self._box(0, 0, 10, 10)
        scores_out, matches, _, _ = self._run(scores, preds, gt)
        assert list(scores_out) == pytest.approx([0.9, 0.5])
        assert matches[0] == 1  # highest score -> TP
        assert matches[1] == 0  # lower score -> FP

    def test_crowd_gt_match_is_ignored_not_fp(self) -> None:
        """A detection matched to a crowd GT is ignored, not a false positive."""
        scores = torch.tensor([0.9])
        box = self._box(0, 0, 10, 10)
        gt_crowd = torch.tensor([True])
        _, matches, ignore, total_gt = self._run(scores, box, box, gt_crowd=gt_crowd)
        assert matches[0] == 0  # not TP
        assert ignore[0]  # ignored -> not counted as FP
        assert total_gt == 0  # crowd GT excluded from denominator

    def test_non_crowd_gt_counts_in_total_gt(self) -> None:
        """Non-crowd GTs are counted in total_gt."""
        scores = torch.tensor([0.9])
        box = self._box(0, 0, 10, 10)
        gt_crowd = torch.tensor([False])
        _, _, _, total_gt = self._run(scores, box, box, gt_crowd=gt_crowd)
        assert total_gt == 1

    def test_mixed_crowd_only_non_crowd_in_total_gt(self) -> None:
        """Only non-crowd instances contribute to total_gt."""
        scores = torch.tensor([0.9])
        pred = self._box(0, 0, 5, 5)  # overlaps neither GT significantly
        gt_boxes = self._boxes([0, 0, 10, 10], [20, 20, 30, 30])
        gt_crowd = torch.tensor([False, True])  # second GT is crowd
        _, _, _, total_gt = self._run(scores, pred, gt_boxes, gt_crowd=gt_crowd)
        assert total_gt == 1

    def test_scores_returned_in_descending_order(self) -> None:
        """Output scores must be sorted in descending order."""
        scores = torch.tensor([0.3, 0.9, 0.6])
        preds = self._boxes([0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50])
        gt = self._box(20, 20, 30, 30)
        scores_out, _, _, _ = self._run(scores, preds, gt)
        assert list(scores_out) == pytest.approx([0.9, 0.6, 0.3])

    def test_segm_iou_type_identical_masks_is_tp(self) -> None:
        """Identical masks with iou_type='segm' should yield a TP."""
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        scores = torch.tensor([0.9])
        gt_crowd = torch.tensor([False])
        _, matches, _, total_gt = _match_single_class(scores, mask, mask, gt_crowd, 0.5, "segm")
        assert matches[0] == 1
        assert total_gt == 1


# ---------------------------------------------------------------------------
# build_matching_data
# ---------------------------------------------------------------------------


class TestBuildMatchingData:
    """Unit tests for build_matching_data()."""

    @staticmethod
    def _make_pred(
        boxes: list,
        scores: list,
        labels: list,
        masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
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
    """Return a compact matching dict as produced by ``build_matching_data()``."""
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
        with patch("rfdetr.evaluation.matching.utils.all_gather", return_value=[rank0, rank1]):
            result = distributed_merge_matching_data(rank0)
        assert set(result.keys()) == {0, 1}
        assert result[0]["total_gt"] == 1
        assert result[1]["total_gt"] == 2

    def test_two_ranks_overlapping_class_concatenates(self) -> None:
        """Two ranks sharing class 0 -> arrays concatenated, total_gt summed."""
        rank0 = {0: _make_matching_entry([0.9], [1], [False], 2)}
        rank1 = {0: _make_matching_entry([0.7, 0.5], [0, 1], [False, False], 3)}
        with patch("rfdetr.evaluation.matching.utils.all_gather", return_value=[rank0, rank1]):
            result = distributed_merge_matching_data(rank0)
        np.testing.assert_allclose(result[0]["scores"], [0.9, 0.7, 0.5], rtol=1e-6)
        assert result[0]["total_gt"] == 5

    def test_returns_new_dict_not_input(self) -> None:
        """Result is a new dict, not a reference to the local input."""
        local_data = {0: _make_matching_entry([0.9], [1], [False], 1)}
        result = distributed_merge_matching_data(local_data)
        assert result is not local_data
