# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Torch-free multi-class top-k selection shared by the export inference helpers.

RF-DETR uses independent per-class sigmoids, not a mutually exclusive softmax, so a single query can legitimately score
above threshold on more than one class at once (e.g. "car" and "truck"). ``PostProcess._select_topk``
(``rfdetr/models/postprocess.py``) accounts for this: it flattens the ``(Q, C)`` score grid to ``Q * C`` query/class
pairs and takes the top ``num_select`` scoring pairs *before* any threshold is applied, so a query can contribute more
than one detection. Selecting the single highest-scoring class per query (``argmax`` over the class axis) silently drops
the rest.

The export inference helpers must reproduce that exact selection so exported ONNX/TFLite models match
``RFDETR.predict()`` detection-for-detection, not just box-for-box.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Matches ``PostProcess.__init__``'s default for callers that use the helper directly. Inference
# decoders pass their exported model's query count (or an explicit configured cap) so segmentation
# variants with smaller ``num_select`` values preserve the same selection contract.
DEFAULT_NUM_SELECT = 300


def _select_topk_multiclass(
    scores_all: NDArray[np.floating], threshold: float, num_select: int = DEFAULT_NUM_SELECT
) -> tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]]:
    """Select the top ``num_select`` query/class pairs, then threshold.

    Mirrors ``PostProcess._select_topk`` (flatten ``(Q, C)`` to ``Q * C``, take the top
    ``num_select`` scoring pairs in deterministic descending-score order) followed by the
    caller's own ``scores > threshold`` filter — ``PostProcess`` never bakes thresholding into
    ``_select_topk`` itself, it is always applied by the caller afterwards.

    Uses a deterministic lexicographic order: descending score, then ascending flattened
    query/class index. ``PostProcess._select_topk`` uses the same stable tie rule so exported
    inference remains reproducible when scores are equal.

    Args:
        scores_all: Per-query, per-class sigmoid probabilities, shape ``(Q, C)``.
        threshold: Confidence threshold; pairs at or below this score are dropped.
        num_select: Maximum number of query/class pairs to consider before thresholding.

    Returns:
        A ``(scores, labels, query_indices)`` tuple, each 1-D and sorted by descending score,
        containing only pairs that cleared ``threshold``. ``query_indices`` selects rows from
        box/mask outputs and, unlike a per-query argmax, can repeat when a query has more than one
        detection.
    """
    if scores_all.ndim != 2:
        raise ValueError(f"scores_all must have shape (Q, C); got {scores_all.shape}")
    if num_select < 0:
        raise ValueError(f"num_select must be non-negative; got {num_select}")

    num_queries, num_classes = scores_all.shape
    flat_scores = scores_all.reshape(-1)
    if num_select == 0 or flat_scores.size == 0:
        return (
            flat_scores[:0],
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    num_to_select = min(num_select, flat_scores.shape[0])
    flat_idx = np.arange(flat_scores.shape[0], dtype=np.int64)
    # PyTorch ranks NaNs ahead of finite values for descending argsort; preserve that ordering
    # so the subsequent ``> threshold`` filter drops the same malformed scores rather than
    # allowing a lower finite score to occupy the cap.
    sort_scores = np.where(np.isnan(flat_scores), np.inf, flat_scores)
    top_idx = np.lexsort((flat_idx, -sort_scores))[:num_to_select]

    topk_scores = flat_scores[top_idx]
    topk_query = top_idx // num_classes
    topk_labels = top_idx % num_classes

    keep = topk_scores > threshold
    return topk_scores[keep], topk_labels[keep], topk_query[keep]
