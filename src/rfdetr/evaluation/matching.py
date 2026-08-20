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
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.ops import box_iou

from rfdetr.utilities import all_gather


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
    pred_scores_sorted: np.ndarray[Any, np.dtype[np.float32]],
    iou_matrix_sorted: np.ndarray[Any, np.dtype[np.float32]],
    gt_crowd_np: np.ndarray[Any, np.dtype[np.bool_]],
    iou_threshold: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Apply COCO greedy matching to score-ordered NumPy IoUs for one class.

    Args:
        pred_scores_sorted: Detection scores in descending order.
        iou_matrix_sorted: Pairwise IoUs aligned to ``pred_scores_sorted``.
        gt_crowd_np: Boolean crowd mask aligned to IoU columns.
        iou_threshold: Minimum IoU to count as a positive match.

    Returns:
        Tuple of sorted scores, true-positive flags, crowd-ignore flags, and
        the number of non-crowd ground truths.
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

        best_noncrowd_index = int(np.argmax(noncrowd_ious))
        if noncrowd_ious[best_noncrowd_index] >= iou_threshold:
            pred_match_np[index] = 1
            gt_matched_np[best_noncrowd_index] = True
        elif any_crowd:
            crowd_ious = ious.copy()
            crowd_ious[not_crowd_np] = -1.0
            if crowd_ious.max() >= iou_threshold:
                pred_ignore_np[index] = True

    return pred_scores_sorted, pred_match_np, pred_ignore_np, int(not_crowd_np.sum())


def _match_single_class(
    pred_scores: Tensor,
    pred_items: Tensor,
    gt_items: Tensor,
    gt_crowd: Tensor,
    iou_threshold: float,
    iou_type: str,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Greedy highest-score-first matching for one class in one image.

    Implements the COCO matching algorithm: each GT is matched at most once; detections are processed in descending
    score order; detections matched to crowd GTs are marked as ignored rather than false positives.

    Args:
        pred_scores: Float tensor of shape [N] with detection confidences.
        pred_items: Predictions — boxes [N, 4] in xyxy coords or masks [N, H, W].
        gt_items: Ground truths — boxes [M, 4] in xyxy coords or masks [M, H, W].
        gt_crowd: Bool tensor of shape [M], True for crowd instances.
        iou_threshold: Minimum IoU to count as a positive match.
        iou_type: ``"bbox"`` for box IoU or ``"segm"`` for mask IoU.

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
    pred_sorted = pred_items[sort_idx]

    if iou_type == "bbox":
        iou_matrix = box_iou(pred_sorted, gt_items)  # [N, M]
    else:
        iou_matrix = _compute_mask_iou(pred_sorted, gt_items)  # [N, M]

    # The greedy matching below is inherently sequential (each GT can only be claimed
    # once, so iteration i depends on the outcome of i-1) — it cannot be vectorized away.
    # But on CUDA, comparing a 0-dim tensor against a Python float inside the loop
    # (`if best_nc_iou >= iou_threshold`) forces a device-to-host sync every iteration,
    # turning what should be O(1) host-side work into O(N) GPU pipeline stalls. Move the
    # IoU matrix and crowd mask to host memory once, then run the loop on plain
    # numpy/Python values so no per-iteration tensor sync occurs (regression: #416).
    iou_matrix_np = iou_matrix.detach().float().cpu().numpy()  # [N, M] -- float() guards bf16/fp16 (no numpy dtype)
    gt_crowd_np = gt_crowd.detach().cpu().numpy()  # [M]

    return _match_sorted_iou_matrix(
        np.asarray(pred_scores_sorted.float().cpu().numpy(), dtype=np.float32),
        iou_matrix_np,
        gt_crowd_np,
        iou_threshold,
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
        pred_masks = preds.get("masks")  # [N, H, W] | None

        gt_boxes = targets["boxes"]  # [M, 4]
        gt_labels = targets["labels"]  # [M]
        gt_masks = targets.get("masks")  # [M, H, W] | None
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

        pred_label_ids_np = np.asarray(pred_label_ids, dtype=np.int64)
        gt_label_ids_np = np.asarray(gt_label_ids, dtype=np.int64)
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
            n_pred = pred_count.get(class_id, 0)
            n_gt = gt_count.get(class_id, 0)

            entry = acc.setdefault(
                class_id,
                {"scores": [], "matches": [], "ignore": [], "total_gt": 0},
            )

            if n_pred == 0:
                entry["total_gt"] = cast(int, entry["total_gt"]) + gt_noncrowd_count.get(class_id, 0)
                continue

            if iou_type == "bbox":
                pred_indices = np.flatnonzero(pred_label_ids_np == class_id)
                assert pred_scores_np is not None
                p_scores_np = pred_scores_np[pred_indices]
            else:
                # Segmentation IoU remains class-local to avoid materializing a potentially
                # much larger all-pairs mask matrix for classes that cannot match.
                pred_mask_c = pred_labels == class_id
                p_scores = pred_scores[pred_mask_c]

            if n_gt == 0:
                # TODO: support bfloat16 natively once numpy adds bf16 dtype
                sc = p_scores_np if iou_type == "bbox" else np.asarray(p_scores.float().cpu().numpy(), dtype=np.float32)
                order = np.argsort(-sc)
                cast(list[float], entry["scores"]).extend(sc[order].tolist())
                cast(list[int], entry["matches"]).extend([0] * n_pred)
                cast(list[bool], entry["ignore"]).extend([False] * n_pred)
                continue

            if iou_type == "bbox":
                gt_indices = np.flatnonzero(gt_label_ids_np == class_id)
                assert bbox_iou_matrix_np is not None
                order = np.argsort(-p_scores_np, kind="stable")
                scores_np, matches_np, ignore_np, total_gt = _match_sorted_iou_matrix(
                    p_scores_np[order].astype(np.float32, copy=False),
                    bbox_iou_matrix_np[np.ix_(pred_indices[order], gt_indices)],
                    gt_crowd_np[gt_indices],
                    iou_threshold,
                )
            else:
                gt_mask_c = gt_labels == class_id
                gt_crowd_c = gt_crowd[gt_mask_c]
                if pred_masks is None or gt_masks is None:
                    raise ValueError("iou_type='segm' requires 'masks' in both preds and targets")
                scores_np, matches_np, ignore_np, total_gt = _match_single_class(
                    p_scores, pred_masks[pred_mask_c], gt_masks[gt_mask_c], gt_crowd_c, iou_threshold, iou_type
                )

            cast(list[float], entry["scores"]).extend(float(score) for score in scores_np)
            cast(list[int], entry["matches"]).extend(int(match) for match in matches_np)
            cast(list[bool], entry["ignore"]).extend(bool(ignore) for ignore in ignore_np)
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
