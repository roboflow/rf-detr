# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidence-threshold sweep for precision/recall/F1 computation."""

from __future__ import annotations

from typing import Any

import numpy as np


def sweep_confidence_thresholds(
    per_class_data: list[dict[str, Any]],
    conf_thresholds: Any,
    classes_with_gt: list[int],
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds and compute precision/recall/F1 at each.

    Args:
        per_class_data: Per-class matching data list indexed by class id.
            Each entry is a dict with keys ``"scores"``, ``"matches"``, ``"ignore"``, and ``"total_gt"``.
        conf_thresholds: Iterable of float confidence thresholds to evaluate.
        classes_with_gt: List of class indices that have at least one GT instance — used for macro-averaging.

    Returns:
        List of result dicts, one per threshold, each containing:
            - ``"confidence_threshold"``: float
            - ``"macro_f1"``: float
            - ``"macro_precision"``: float
            - ``"macro_recall"``: float
            - ``"per_class_prec"``: float ndarray
            - ``"per_class_rec"``: float ndarray
            - ``"per_class_f1"``: float ndarray
    """
    # Materialized exactly once: `conf_thresholds` is documented as any iterable, which a generator
    # would satisfy, and every use below (the length, the per-class searchsorted, and the per-threshold
    # results loop) needs its own full pass -- consuming a generator more than once would silently
    # return wrong-length or empty results from the second pass onward.
    conf_thresholds_arr = np.asarray(list(conf_thresholds), dtype=np.float64)
    num_classes = len(per_class_data)
    num_thresholds = len(conf_thresholds_arr)

    # Per-class TP/FP counts at every threshold, computed once per class instead of rescanning every
    # detection at every threshold (the loop below used to do exactly that: for each of T thresholds,
    # `scores >= conf_thresh` scans all of that class's N detections -- O(T*N) total). Sorting each
    # class's detections by score once (O(N log N)) and taking prefix sums over the sorted order lets
    # every threshold's TP/FP be a single `np.searchsorted` binary search plus an array lookup instead
    # of a full rescan, so the whole function becomes O(N log N + T log N) per class.
    per_class_tp = np.empty((num_classes, num_thresholds), dtype=np.int64)
    per_class_fp = np.empty((num_classes, num_thresholds), dtype=np.int64)
    total_gt_per_class = np.empty(num_classes, dtype=np.int64)

    for k in range(num_classes):
        data = per_class_data[k]
        scores = data["scores"]
        matches = data["matches"]
        ignore = data["ignore"]
        total_gt_per_class[k] = data["total_gt"]

        # Ascending sort: `np.searchsorted(..., side="left")` then gives, for a threshold, the index
        # of the first detection with score >= threshold -- everything from that index to the end is
        # the "above_thresh" set the original per-threshold boolean mask picked out. NumPy sorts NaN
        # scores to the end of an ascending sort, which would put them in the "above every threshold"
        # suffix -- but `scores >= conf_thresh` is False for a NaN score against any real threshold,
        # so the original loop always excluded them. Mask them out here the same way `ignore` is, or
        # a NaN score would flip from silently ignored to silently counted as TP/FP at every threshold.
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        valid = ~ignore[order] & ~np.isnan(sorted_scores)
        is_tp = valid & (matches[order] != 0)
        is_fp = valid & (matches[order] == 0)

        # Suffix sums: `suffix_tp[i]` = count of TPs among detections with score >= sorted_scores[i].
        # `np.cumsum(...)` is a prefix sum; reversing the input and output turns it into a suffix sum
        # without a second full pass.
        suffix_tp = np.concatenate((np.cumsum(is_tp[::-1])[::-1], [0]))
        suffix_fp = np.concatenate((np.cumsum(is_fp[::-1])[::-1], [0]))

        insertion_idx = np.searchsorted(sorted_scores, conf_thresholds_arr, side="left")
        per_class_tp[k] = suffix_tp[insertion_idx]
        per_class_fp[k] = suffix_fp[insertion_idx]

    results: list[dict[str, Any]] = []

    for t, conf_thresh in enumerate(conf_thresholds_arr):
        per_class_precisions: list[float] = []
        per_class_recalls: list[float] = []
        per_class_f1s: list[float] = []

        for k in range(num_classes):
            tp = per_class_tp[k, t]
            fp = per_class_fp[k, t]
            total_gt = total_gt_per_class[k]
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = float(np.mean([per_class_precisions[k] for k in classes_with_gt]))
            macro_recall = float(np.mean([per_class_recalls[k] for k in classes_with_gt]))
            macro_f1 = float(np.mean([per_class_f1s[k] for k in classes_with_gt]))
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results
