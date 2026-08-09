# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Torch-free multi-class top-k selection shared by the export inference helpers.

RF-DETR uses independent per-class sigmoids, not a mutually exclusive softmax, so a single query
can legitimately score above threshold on more than one class at once (e.g. "car" and "truck").
``PostProcess._select_topk`` (``rfdetr/models/postprocess.py``) accounts for this: it flattens the
``(Q, C)`` score grid to ``Q * C`` query/class pairs and takes the top ``num_select`` scoring pairs
*before* any threshold is applied, so a query can contribute more than one detection. Selecting the
single highest-scoring class per query (``argmax`` over the class axis) silently drops the rest.

The export inference helpers must reproduce that exact selection so exported ONNX/TFLite models
match ``RFDETR.predict()`` detection-for-detection, not just box-for-box.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Matches PostProcess.__init__'s own ``num_select`` default (postprocess.py) and is the largest
# value used by any shipped RF-DETR variant — the detection sizes (Nano/Small/Medium/Large) all
# inherit 300 from RFDETRBaseConfig unchanged, but the segmentation configs override it lower
# (RFDETRSegNano/SegSmall use 100, RFDETRSegMedium/SegLarge use 200; see config.py).
# Exported ONNX/TFLite artifacts carry no training-time config, so there is no way for this decode
# to recover the exact value a given checkpoint was trained/exported with; 300 is used as a
# permissive ceiling. In the extreme case where a checkpoint configured with a smaller num_select
# produces more than that many query/class pairs above `threshold` in a single image, this can
# return a few more detections than `predict()` would for that same checkpoint — same order of
# rarity as the multi-label case this module exists to fix correctly, and never fewer detections.
DEFAULT_NUM_SELECT = 300


def _select_topk_multiclass(
    scores_all: NDArray[np.floating], threshold: float, num_select: int = DEFAULT_NUM_SELECT
) -> tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]]:
    """Select the top ``num_select`` query/class pairs, then threshold.

    Mirrors ``PostProcess._select_topk`` (flatten ``(Q, C)`` to ``Q * C``, take the top
    ``num_select`` scoring pairs by ``torch.topk``, in descending score order) followed by the
    caller's own ``scores > threshold`` filter — ``PostProcess`` never bakes thresholding into
    ``_select_topk`` itself, it is always applied by the caller afterwards.

    Uses ``np.argpartition`` + a stable ``np.argsort`` rather than a full sort, for the same
    result with lower complexity on a large flattened array. This has not been verified to break
    ties in the exact same order as ``torch.topk`` when two query/class pairs share the identical
    floating-point score — real network logits essentially never tie exactly, so this is not
    expected to matter in practice, but it is untested at that edge.

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
    num_queries, num_classes = scores_all.shape
    flat_scores = scores_all.reshape(-1)
    num_to_select = min(num_select, flat_scores.shape[0])
    if num_to_select < flat_scores.shape[0]:
        top_idx = np.argpartition(flat_scores, -num_to_select)[-num_to_select:]
    else:
        top_idx = np.arange(flat_scores.shape[0])
    # argpartition doesn't sort its output; PostProcess._select_topk relies on torch.topk's
    # descending order (consumers like early-exit display code assume the highest score is first).
    top_idx = top_idx[np.argsort(-flat_scores[top_idx], kind="stable")]

    topk_scores = flat_scores[top_idx]
    topk_query = top_idx // num_classes
    topk_labels = top_idx % num_classes

    keep = topk_scores > threshold
    return topk_scores[keep], topk_labels[keep], topk_query[keep]
