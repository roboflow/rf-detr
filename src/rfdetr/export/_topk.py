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
    # Composite-key construction only pays off for sparse selections; dense custom-class
    # grids retain the direct lexsort path.
    if (
        flat_scores.dtype == np.float32
        and num_to_select < flat_scores.shape[0]
        and num_to_select * 4 <= flat_scores.shape[0]
        and flat_scores.shape[0] <= np.iinfo(np.uint32).max
    ):
        # Map native IEEE float32 bits to ascending unsigned integers, then invert them for
        # descending score order. Canonicalising signed zero preserves lexsort's numeric tie,
        # while replacing NaNs above makes them tie with +inf exactly as the established path.
        score_bits = sort_scores.view(np.uint32)
        sign_bit = np.uint32(0x80000000)
        ordered_scores = score_bits ^ np.where(
            (score_bits & sign_bit) != 0,
            np.uint32(0xFFFFFFFF),
            sign_bit,
        )
        ordered_scores[(score_bits & np.uint32(0x7FFFFFFF)) == 0] = sign_bit

        # The low 32 bits make every key unique and encode the ascending flattened-index
        # tiebreak. Partitioning these composite keys therefore selects the exact same cutoff
        # set as lexsort, after which only the small selected set needs ordering.
        selection_keys = ((~ordered_scores).astype(np.uint64) << np.uint64(32)) | flat_idx.astype(np.uint64)
        partition_idx = np.argpartition(selection_keys, num_to_select - 1)[:num_to_select]
        top_idx = partition_idx[np.argsort(selection_keys[partition_idx], kind="stable")]
    else:
        top_idx = np.lexsort((flat_idx, -sort_scores))[:num_to_select]

    topk_scores = flat_scores[top_idx]
    topk_query = top_idx // num_classes
    topk_labels = top_idx % num_classes

    keep = topk_scores > threshold
    return topk_scores[keep], topk_labels[keep], topk_query[keep]
