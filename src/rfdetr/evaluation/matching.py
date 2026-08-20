# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Greedy matching and accumulation functions for evaluation metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.ops import box_iou

from rfdetr.utilities import all_gather

#: One class's contribution to the accumulator: its detection scores in descending order, their
#: true-positive flags, their crowd-ignore flags, and the class's non-crowd GT count.
_ClassContribution: TypeAlias = tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]

#: Positions of one class inside an image's prediction and GT arrays: the rows and the columns it
#: owns in the image-wide ``box_iou`` matrix. ``None`` instead means the image holds a single class,
#: which owns the whole matrix and needs no positions at all.
_ClassSlice: TypeAlias = tuple[
    np.ndarray[Any, np.dtype[np.intp]],
    np.ndarray[Any, np.dtype[np.intp]],
]

#: Column positions of a class that has predictions but no GT of its own. Shared, so the per-class
#: lookup that misses can hand back an empty selection without allocating one every time.
_NO_INDICES: np.ndarray[Any, np.dtype[np.intp]] = np.zeros(0, dtype=np.intp)


def _compute_mask_iou(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    """Compute pairwise boolean-mask IoU between N predictions and M ground truths.

    Args:
        pred_masks: Boolean mask tensor of shape [N, H, W].
        gt_masks: Boolean mask tensor of shape [M, H, W].

    Returns:
        IoU tensor of shape [N, M].
    """
    n = pred_masks.shape[0]
    m = gt_masks.shape[0]
    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        h, w = pred_masks.shape[-2:]
        gt_masks = F.interpolate(gt_masks.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
    pred_flat = pred_masks.bool().view(n, -1).float()  # [N, HW]
    gt_flat = gt_masks.bool().view(m, -1).float()  # [M, HW]
    inter = torch.mm(pred_flat, gt_flat.t())  # [N, M]
    pred_area = pred_flat.sum(dim=1, keepdim=True)  # [N, 1]
    gt_area = gt_flat.sum(dim=1, keepdim=True)  # [M, 1]
    union = pred_area + gt_area.t() - inter  # [N, M]
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _match_sorted_iou_matrix(
    iou_matrix_sorted: np.ndarray[Any, np.dtype[np.float32]],
    gt_crowd_np: np.ndarray[Any, np.dtype[np.bool_]],
    iou_threshold: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Apply COCO greedy matching to score-ordered NumPy IoUs for one class.

    Dispatches on whether the class has any crowd GT. Both loops below are per-detection Python
    loops whose cost is dominated by the number of NumPy calls each iteration makes, so the common
    crowd-free case gets its own loop rather than paying for masks that can only be no-ops.

    Args:
        iou_matrix_sorted: Pairwise IoUs whose rows are in descending detection-score order. Must not
            be modified here: the bbox caller derives it from the one ``box_iou`` matrix shared by
            every class of an image, so masking rows in place instead of copying them would leak one
            class's matching state into the classes matched after it.
        gt_crowd_np: Boolean crowd mask aligned to IoU columns.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple of true-positive flags, crowd-ignore flags, and the number of non-crowd ground truths.
        The scores the rows were ordered by are not returned — every caller already holds them.
    """
    if not gt_crowd_np.any():
        return _match_rows_without_crowd(iou_matrix_sorted, iou_threshold)
    return _match_rows_with_crowd(iou_matrix_sorted, gt_crowd_np, iou_threshold)


def _match_rows_without_crowd(
    iou_matrix_sorted: np.ndarray[Any, np.dtype[np.float32]],
    iou_threshold: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Greedily match score-ordered IoUs for a class whose ground truths are all non-crowd.

    Behaviourally a special case of ``_match_rows_with_crowd()`` — every GT is claimable, none can
    be ignored, and ``total_gt`` is simply the column count. What makes it worth its own loop is
    that the per-detection cost here is NumPy call overhead, not array work: at a few dozen GTs a
    call costs more than the elements it touches, so the win comes from making calls disappear.

    Two whole-matrix reductions do that for the two cases that need no search at all: a detection
    whose best IoU is below the threshold cannot match any GT, and one whose best GT is still free
    matches exactly that GT (masking can only lower other columns, so it cannot promote a different
    one, and ``argmax`` already resolved ties to the lowest column index). Only a detection whose
    best GT was claimed by a higher-scoring detection falls back to a masked search.

    Args:
        iou_matrix_sorted: Pairwise IoUs whose rows are in descending detection-score order. Read
            only, for the reason given on ``_match_sorted_iou_matrix()``.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple of true-positive flags, all-False crowd-ignore flags, and the ground-truth count.
    """
    n, m = iou_matrix_sorted.shape
    gt_matched_np = np.zeros(m, dtype=np.bool_)
    pred_match_np = np.zeros(n, dtype=np.int64)
    # argmax before max, so that a matrix with no GT columns still fails with the same "argmax of
    # an empty sequence" ValueError the per-detection search raises.
    best_columns = cast(list[int], iou_matrix_sorted.argmax(axis=1).tolist())
    best_ious = cast(list[float], iou_matrix_sorted.max(axis=1).tolist())

    for index in range(n):
        if best_ious[index] < iou_threshold:
            continue
        best_column = best_columns[index]
        if not gt_matched_np[best_column]:
            pred_match_np[index] = 1
            gt_matched_np[best_column] = True
            continue

        # Copy-then-mask rather than np.where(): measured cheaper, and it leaves the source matrix
        # — shared across the classes of an image on the bbox path — untouched either way.
        free_ious = cast(np.ndarray[Any, np.dtype[np.float32]], iou_matrix_sorted[index].copy())
        free_ious[gt_matched_np] = -1.0
        best_index = int(free_ious.argmax())
        if free_ious[best_index] >= iou_threshold:
            pred_match_np[index] = 1
            gt_matched_np[best_index] = True

    return pred_match_np, np.zeros(n, dtype=np.bool_), m


def _match_rows_with_crowd(
    iou_matrix_sorted: np.ndarray[Any, np.dtype[np.float32]],
    gt_crowd_np: np.ndarray[Any, np.dtype[np.bool_]],
    iou_threshold: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Greedily match score-ordered IoUs for a class that may have crowd ground truths.

    Handles crowd GTs in full generality, so it stays correct for an all-non-crowd input too — that
    is what makes it the reference ``_match_rows_without_crowd()`` is differentially tested against.

    Args:
        iou_matrix_sorted: Pairwise IoUs whose rows are in descending detection-score order. Read
            only, for the reason given on ``_match_sorted_iou_matrix()``.
        gt_crowd_np: Boolean crowd mask aligned to IoU columns.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple of true-positive flags, crowd-ignore flags, and the number of non-crowd ground truths.
    """
    n, m = iou_matrix_sorted.shape
    gt_matched_np = np.zeros(m, dtype=np.bool_)
    pred_match_np = np.zeros(n, dtype=np.int64)
    pred_ignore_np = np.zeros(n, dtype=np.bool_)
    any_crowd = bool(gt_crowd_np.any())
    not_crowd_np = ~gt_crowd_np

    # Each detection can claim at most one non-crowd target, so score order
    # remains sequential even though the IoU matrix is already vectorized.
    for index in range(len(iou_matrix_sorted)):
        ious = cast(np.ndarray[Any, np.dtype[np.float32]], iou_matrix_sorted[index])
        noncrowd_ious = ious.copy()
        noncrowd_ious[gt_crowd_np] = -1.0
        noncrowd_ious[gt_matched_np & not_crowd_np] = -1.0

        # The bound method, not np.argmax(): the module-level function re-dispatches on its
        # argument, which costs more per call than the search over a few dozen GTs.
        best_noncrowd_index = int(noncrowd_ious.argmax())
        if noncrowd_ious[best_noncrowd_index] >= iou_threshold:
            pred_match_np[index] = 1
            gt_matched_np[best_noncrowd_index] = True
        elif any_crowd:
            crowd_ious = ious.copy()
            crowd_ious[not_crowd_np] = -1.0
            if crowd_ious.max() >= iou_threshold:
                pred_ignore_np[index] = True

    return pred_match_np, pred_ignore_np, int(not_crowd_np.sum())


def _group_indices_by_label(
    label_ids_np: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[int, np.ndarray[Any, np.dtype[np.intp]]]:
    """Group array positions by label in one pass over an image's labels.

    Replaces one ``label_ids_np == class_id`` scan per class — O(N) per class, O(N*C) over an
    image — with a single stable sort. Within each group the positions stay ascending, which is the
    order a per-class scan produces and the order the greedy matcher's stable score sort then
    breaks ties on, so the grouping is a pure speedup and not a reordering.

    Args:
        label_ids_np: Label of every detection (or of every GT) of one image.

    Returns:
        Dict mapping each label present to its ascending positions in *label_ids_np*; empty for an
        empty input.
    """
    if label_ids_np.size == 0:
        return {}
    order = np.argsort(label_ids_np, kind="stable")
    sorted_labels = label_ids_np[order]
    boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    return {int(label_ids_np[group[0]]): group for group in np.split(order, boundaries)}


def _unmatched_contribution(
    scores_np: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> _ClassContribution:
    """Build the contribution of a class that has detections but no ground truth.

    Both ``iou_type`` paths reach this case and must report it identically: every detection is a
    false positive, none is crowd-ignored, and the class adds nothing to the GT denominator.

    Args:
        scores_np: Detection scores of one class, in the caller's own float dtype so that near-tied
            scores are ordered at full precision before the float32 cast.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` with the scores in descending order,
        all-zero matches, all-False ignore flags, and a ``total_gt`` of 0.
    """
    n = len(scores_np)
    order = np.argsort(-scores_np)
    return (
        scores_np[order].astype(np.float32, copy=False),
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=np.bool_),
        0,
    )


def _match_single_class_segm(
    pred_scores: Tensor,
    pred_masks: Tensor,
    gt_masks: Tensor,
    gt_crowd: Tensor,
    iou_threshold: float,
) -> _ClassContribution:
    """Greedy highest-score-first mask matching for one class in one image.

    Implements the COCO matching algorithm over boolean-mask IoU: each GT is matched at most once; detections are
    processed in descending score order; detections matched to crowd GTs are marked as ignored rather than false
    positives. This is the segmentation path only — the box path never reaches here, because it shares one image-wide
    ``box_iou`` matrix and calls ``_match_sorted_iou_matrix`` directly.

    Args:
        pred_scores: Float tensor of shape [N] with detection confidences.
        pred_masks: Boolean prediction masks of shape [N, H, W].
        gt_masks: Boolean ground-truth masks of shape [M, H, W].
        gt_crowd: Bool tensor of shape [M], True for crowd instances.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` where:
            - scores_np: float32 array [N] ordered by descending score.
            - matches_np: int array [N], 1 = TP, 0 = FP.
            - ignore_np: bool array [N], True if matched to a crowd GT.
            - total_gt: number of non-crowd GT instances.
    """
    # stable=True keeps tied scores in input order, which is the tie rule the bbox path gets from
    # np.argsort(kind="stable") and PostProcess._select_topk gets from the same flag. torch's
    # default sort is unstable past its ~32-element cutoff, so without it the greedy winner of a
    # tie -- and with it the TP/FP split -- would depend on the sort backend rather than the input.
    sort_idx = torch.argsort(pred_scores, descending=True, stable=True)
    pred_scores_sorted = pred_scores[sort_idx]
    pred_sorted = pred_masks[sort_idx]

    iou_matrix = _compute_mask_iou(pred_sorted, gt_masks)  # [N, M]

    # The greedy matching below is inherently sequential (each GT can only be claimed
    # once, so iteration i depends on the outcome of i-1) — it cannot be vectorized away.
    # But on CUDA, comparing a 0-dim tensor against a Python float inside the loop
    # (`if best_nc_iou >= iou_threshold`) forces a device-to-host sync every iteration,
    # turning what should be O(1) host-side work into O(N) GPU pipeline stalls. Move the
    # IoU matrix and crowd mask to host memory once, then run the loop on plain
    # numpy/Python values so no per-iteration tensor sync occurs (regression: #416).
    iou_matrix_np = iou_matrix.detach().float().cpu().numpy()  # [N, M] -- float() guards bf16/fp16 (no numpy dtype)
    gt_crowd_np = gt_crowd.detach().cpu().numpy()  # [M]

    matches_np, ignore_np, total_gt = _match_sorted_iou_matrix(iou_matrix_np, gt_crowd_np, iou_threshold)
    return (
        np.asarray(pred_scores_sorted.detach().float().cpu().numpy(), dtype=np.float32),
        matches_np,
        ignore_np,
        total_gt,
    )


def _accumulate_bbox_class(
    pred_scores_np: np.ndarray[Any, np.dtype[np.floating[Any]]],
    class_slice: _ClassSlice | None,
    gt_crowd_np: np.ndarray[Any, np.dtype[np.bool_]],
    bbox_iou_matrix_np: np.ndarray[Any, np.dtype[np.float32]] | None,
    iou_threshold: float,
) -> _ClassContribution:
    """Accumulate one class of one image against the image's shared box-IoU matrix.

    Everything needed here is already on the host, so the class is selected by slicing the per-image
    NumPy arrays instead of by indexing device tensors. A *class_slice* of ``None`` says the image
    holds one class only: it then owns every row and column of *bbox_iou_matrix_np*, and the
    two-axis fancy-index copy that isolates a class becomes a copy of the whole matrix onto itself,
    so only the rows are reordered and the crowd mask is taken as it stands.

    Args:
        pred_scores_np: Every detection score of the image, in the caller's own float dtype.
        class_slice: The class's prediction rows and GT columns, or ``None`` when the image holds a
            single class. A class with predictions but no GT gets an empty column selection, never
            ``None``.
        gt_crowd_np: Crowd flag per GT of the image.
        bbox_iou_matrix_np: The image's ``box_iou`` matrix, or ``None`` when the image has no
            detections or no GT at all. A class without GT returns before it is read.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` contributed by the class.
    """
    p_scores_np = pred_scores_np if class_slice is None else pred_scores_np[class_slice[0]]
    n_gt = gt_crowd_np.size if class_slice is None else class_slice[1].size

    if n_gt == 0:
        return _unmatched_contribution(p_scores_np)

    assert bbox_iou_matrix_np is not None, (
        "the image-wide box_iou matrix is built whenever the image has both detections and GTs, "
        "and this class has at least one of each"
    )
    order = np.argsort(-p_scores_np, kind="stable")
    if class_slice is None:
        iou_matrix_sorted, class_crowd_np = bbox_iou_matrix_np[order], gt_crowd_np
    else:
        pred_indices, gt_indices = class_slice
        iou_matrix_sorted = bbox_iou_matrix_np[np.ix_(pred_indices[order], gt_indices)]
        class_crowd_np = gt_crowd_np[gt_indices]

    matches_np, ignore_np, total_gt = _match_sorted_iou_matrix(iou_matrix_sorted, class_crowd_np, iou_threshold)
    return p_scores_np[order].astype(np.float32, copy=False), matches_np, ignore_np, total_gt


def _accumulate_segm_class(
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
    gt_crowd: Tensor,
    class_id: int,
    n_gt: int,
    iou_threshold: float,
) -> _ClassContribution:
    """Accumulate one class of one image on boolean-mask IoU.

    Mask IoU stays class-local rather than being shared image-wide like the box path: an all-pairs
    mask matrix is much larger, and most of it would cover pairs that cannot match.

    Args:
        preds: One image's predictions, keyed as documented on ``build_matching_data()``.
        targets: One image's ground truths, keyed as documented on ``build_matching_data()``.
        gt_crowd: Bool tensor [M] aligned to ``targets["labels"]``, already validated by the caller.
        class_id: Class to accumulate.
        n_gt: Number of GTs of *class_id* in this image. Taken from the caller's host-side label
            counts rather than read back out of ``targets["labels"]``, which would cost one
            device-to-host sync per class.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` contributed by *class_id*.

    Raises:
        ValueError: If ``masks`` is missing from *preds* or *targets*. A class with no GT of its own
            returns before that lookup, so a one-sided class is never reported.
    """
    pred_mask_c = preds["labels"] == class_id
    p_scores = preds["scores"][pred_mask_c]

    if n_gt == 0:
        # TODO: support bfloat16 natively once numpy adds bf16 dtype
        return _unmatched_contribution(np.asarray(p_scores.detach().float().cpu().numpy(), dtype=np.float32))

    pred_masks = preds.get("masks")
    gt_masks = targets.get("masks")
    if pred_masks is None or gt_masks is None:
        raise ValueError("iou_type='segm' requires 'masks' in both preds and targets")

    gt_mask_c = targets["labels"] == class_id
    return _match_single_class_segm(
        p_scores, pred_masks[pred_mask_c], gt_masks[gt_mask_c], gt_crowd[gt_mask_c], iou_threshold
    )


def build_matching_data(
    preds_list: list[dict[str, Tensor]],
    targets_list: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
    iou_type: str = "bbox",
) -> dict[int, dict[str, Any]]:
    """Build compact per-class matching data from a batch of predictions and targets.

    Implements greedy highest-score-first matching compatible with the COCO algorithm. The returned dict can be passed
    directly to ``merge_matching_data()`` and ultimately consumed by ``sweep_confidence_thresholds()`` after conversion
    to list form.

    Detections are ranked in the dtype of ``preds["scores"]``, so a ``float64`` input keeps the full precision that
    separates near-tied scores, and detections that really are tied keep their input order. Both ``iou_type`` paths
    share that rule, which makes the TP/FP split reproducible across devices and dtypes. The returned scores are
    float32 either way.

    Args:
        preds_list: Per-image predictions. Each dict must contain:

            - ``boxes``: float Tensor [N, 4] in absolute xyxy coordinates.
            - ``scores``: float Tensor [N].
            - ``labels``: int64 Tensor [N].
            - ``masks`` *(optional)*: bool Tensor [N, H, W] for segmentation.

        targets_list: Per-image ground truths. Each dict must contain:

            - ``boxes``: float Tensor [M, 4] in absolute xyxy coordinates.
            - ``labels``: int64 Tensor [M].
            - ``masks`` *(optional)*: bool Tensor [M, H, W] for segmentation.
            - ``iscrowd`` *(optional)*: int64 Tensor [M], 1 for crowd instances.

        iou_threshold: IoU threshold for positive matching. Defaults to 0.5.
        iou_type: ``"bbox"`` for bounding-box IoU; ``"segm"`` for boolean-mask
            IoU. Defaults to ``"bbox"``.

    Returns:
        Dict mapping ``class_id`` (int) to a compact matching dict with keys:

            - ``"scores"``: float32 ndarray of detection scores.
            - ``"matches"``: int ndarray (1 = TP, 0 = FP).
            - ``"ignore"``: bool ndarray (True if matched to a crowd GT).
            - ``"total_gt"``: int, count of non-crowd GT instances.

    Raises:
        ValueError: If a target's ``iscrowd`` is not a 1-D tensor with one entry per GT label, or if
            ``iou_type="segm"`` and ``masks`` is missing on either side for a class that has both
            predictions and ground truth. Classes present on only one side skip the mask lookup, so
            a missing ``masks`` key is not reported for an image whose classes are all one-sided.
    """
    acc: dict[int, dict[str, list[Any] | int]] = {}

    for preds, targets in zip(preds_list, targets_list):
        pred_boxes = preds["boxes"]  # [N, 4]
        pred_scores = preds["scores"]  # [N]
        pred_labels = preds["labels"]  # [N]

        gt_boxes = targets["boxes"]  # [M, 4]
        gt_labels = targets["labels"]  # [M]
        raw_crowd = targets.get(
            "iscrowd",
            torch.zeros(len(gt_labels), dtype=torch.long, device=gt_labels.device),
        )
        gt_crowd = raw_crowd.bool()
        # Checked on the shape, not on len(): a [M, 1] iscrowd has the right len() but its
        # tolist() rows are all truthy, which would silently drop every GT from total_gt.
        if gt_crowd.ndim != 1 or gt_crowd.shape[0] != gt_labels.numel():
            raise ValueError(
                f"'iscrowd' must be a 1-D tensor with one entry per GT label, got shape "
                f"{tuple(gt_crowd.shape)} for {gt_labels.numel()} labels"
            )

        gt_label_ids = cast(list[int], gt_labels.tolist())
        pred_label_ids = cast(list[int], pred_labels.tolist())
        gt_crowd_ids = cast(list[bool], gt_crowd.tolist())

        # Host-side counts, built once per image from the tolist()'d labels above, replace what
        # used to be a per-class `.sum().item()` device-to-host sync (two or three per class).
        # The crowd count feeds the `n_pred == 0` branch, which must skip crowd GTs.
        pred_count = Counter(pred_label_ids)
        gt_count = Counter(gt_label_ids)
        gt_noncrowd_count = Counter(label for label, crowd in zip(gt_label_ids, gt_crowd_ids) if not crowd)
        all_class_ids: set[int] = set(gt_count) | set(pred_count)

        # One image whose detections and GTs are all of the same class needs no per-class positions
        # at all: that class owns every row and column of the IoU matrix below.
        single_class = len(all_class_ids) == 1
        pred_groups: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
        gt_groups: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
        if iou_type == "bbox" and pred_count and not single_class:
            pred_groups = _group_indices_by_label(np.asarray(pred_label_ids, dtype=np.int64))
            gt_groups = _group_indices_by_label(np.asarray(gt_label_ids, dtype=np.int64))

        gt_crowd_np = np.asarray(gt_crowd_ids, dtype=np.bool_)
        bbox_iou_matrix_np: np.ndarray[Any, np.dtype[np.float32]] | None = None
        # Left in the caller's own dtype: this array orders the greedy loop, and a float32 cast
        # taken here would collapse near-tied float64 scores onto one value and hand the GT to
        # whichever detection happened to come first instead of to the higher-scoring one. Only
        # bfloat16 is cast, because numpy has no bfloat16 dtype. The scores that are handed back
        # are cast to float32 after the ordering is fixed, so the returned dtype is unchanged.
        pred_scores_np: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
        if iou_type == "bbox" and pred_count:
            pred_scores_cpu = pred_scores.detach().cpu()
            if pred_scores_cpu.dtype == torch.bfloat16:
                pred_scores_cpu = pred_scores_cpu.float()
            pred_scores_np = pred_scores_cpu.numpy()
        if iou_type == "bbox" and pred_count and gt_count:
            # One image-wide GPU IoU operation and one matrix transfer replace the
            # class-local launches/transfers. Host slicing below retains class isolation.
            bbox_iou_matrix_np = np.asarray(
                box_iou(pred_boxes, gt_boxes).detach().float().cpu().numpy(), dtype=np.float32
            )

        for class_id in all_class_ids:
            entry = acc.setdefault(
                class_id,
                {"scores": [], "matches": [], "ignore": [], "total_gt": 0},
            )

            if pred_count.get(class_id, 0) == 0:
                entry["total_gt"] = cast(int, entry["total_gt"]) + gt_noncrowd_count.get(class_id, 0)
                continue

            # The one place ``iou_type`` is branched on per class: each helper owns its own scores
            # variable and its own empty-GT case, so the two paths cannot drift out of agreement.
            if iou_type == "bbox":
                assert pred_scores_np is not None, (
                    "the image's scores are moved to host whenever it has detections, and this class has one"
                )
                class_slice = None if single_class else (pred_groups[class_id], gt_groups.get(class_id, _NO_INDICES))
                scores_np, matches_np, ignore_np, total_gt = _accumulate_bbox_class(
                    pred_scores_np,
                    class_slice,
                    gt_crowd_np,
                    bbox_iou_matrix_np,
                    iou_threshold,
                )
            else:
                scores_np, matches_np, ignore_np, total_gt = _accumulate_segm_class(
                    preds, targets, gt_crowd, class_id, gt_count.get(class_id, 0), iou_threshold
                )

            # ndarray.tolist() converts in C and yields the same Python scalars a per-element
            # float()/int()/bool() generator would, one interpreter round-trip instead of N.
            cast(list[float], entry["scores"]).extend(cast(list[float], scores_np.tolist()))
            cast(list[int], entry["matches"]).extend(cast(list[int], matches_np.tolist()))
            cast(list[bool], entry["ignore"]).extend(cast(list[bool], ignore_np.tolist()))
            entry["total_gt"] = cast(int, entry["total_gt"]) + total_gt

    return {
        class_id: {
            "scores": np.array(data["scores"], dtype=np.float32),
            "matches": np.array(data["matches"], dtype=np.int64),
            "ignore": np.array(data["ignore"], dtype=np.bool_),
            "total_gt": data["total_gt"],
        }
        for class_id, data in acc.items()
    }


def init_matching_accumulator() -> dict[int, dict[str, Any]]:
    """Return an empty matching accumulator compatible with ``merge_matching_data()``.

    Returns:
        Empty dict to be passed as the first argument to ``merge_matching_data()``.
    """
    return {}


def merge_matching_data(
    accumulator: dict[int, dict[str, Any]],
    new_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Merge *new_data* into *accumulator* in place.

    Both arguments share the dict schema produced by ``build_matching_data()``: each class-keyed sub-dict contains
    ``"scores"`` (float32 ndarray), ``"matches"`` (int64 ndarray), ``"ignore"`` (bool ndarray), and ``"total_gt"``
    (int).

    Args:
        accumulator: Running accumulator, modified in place.
        new_data: Batch-level matching data to merge in.

    Returns:
        The modified *accumulator* (same object, for method chaining).
    """
    for class_id, data in new_data.items():
        if class_id not in accumulator:
            accumulator[class_id] = {
                "scores": data["scores"].copy(),
                "matches": data["matches"].copy(),
                "ignore": data["ignore"].copy(),
                "total_gt": data["total_gt"],
            }
        else:
            entry = accumulator[class_id]
            entry["scores"] = np.concatenate([entry["scores"], data["scores"]])
            entry["matches"] = np.concatenate([entry["matches"], data["matches"]])
            entry["ignore"] = np.concatenate([entry["ignore"], data["ignore"]])
            entry["total_gt"] += data["total_gt"]
    return accumulator


def distributed_merge_matching_data(
    local_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Gather per-rank matching data from all DDP ranks and merge into one dict.

    Uses ``rfdetr.utilities.all_gather`` (pickle-based) so the data need not be a tensor. In single-process
    (non-distributed) mode, returns a merged copy of *local_data* unchanged.

    Args:
        local_data: Per-rank accumulator produced by ``merge_matching_data()``.

    Returns:
        Merged accumulator containing contributions from all ranks.
    """
    gathered: list[dict[int, dict[str, Any]]] = all_gather(local_data)
    merged: dict[int, dict[str, Any]] = {}
    for rank_data in gathered:
        merge_matching_data(merged, rank_data)
    return merged
