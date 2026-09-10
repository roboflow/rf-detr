# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Detection decoding shared by the export inference helpers.

Turning an exported model's raw ``dets``/``labels`` tensors into pixel-space detections is identical for every format:
per-class sigmoid, drop the background slot, rank query/class pairs globally, threshold, then convert normalised
``cxcywh`` to pixel ``xyxy``. Matching the raw tensors to their roles is *not* shared -- ONNX keeps its output names
while ``onnx2tf`` usually strips them -- so each format module does its own matching and calls :func:`decode_detections`
with the result.

This decode mirrors ``PostProcess.forward`` in ``rfdetr/models/postprocess.py``; the parity suite depends on it staying
that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._class_layout import _exclude_background_class
from rfdetr.export._topk import _select_topk_multiclass
from rfdetr.utilities.logger import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class DecodedDetections:
    """One image's decoded detections, ordered by descending confidence.

    Attributes are separate arrays rather than a :class:`supervision.Detections` so that a caller with an extra per-
    detection output -- segmentation masks, keypoints -- can gather it with *query_index* before assembling the final
    object.
    """

    #: Pixel-space boxes, shape ``(N, 4)`` in ``xyxy`` order.
    xyxy: NDArray[np.float32]
    #: Sigmoid confidence per detection, shape ``(N,)``.
    confidence: NDArray[np.floating[Any]]
    #: Exported class slot per detection, shape ``(N,)``.
    class_id: NDArray[np.int64]
    #: Source query row per detection, shape ``(N,)``. Repeats when one query clears the threshold on more than one
    #: class, so per-query outputs must be gathered with it rather than boolean-masked.
    query_index: NDArray[np.int64]


def decode_detections(
    boxes_cwh: NDArray[np.floating[Any]],
    logits: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    threshold: float = 0.3,
    num_select: int | None = None,
    background_class_id: int | None = -1,
) -> DecodedDetections:
    """Decode one image's raw model outputs into pixel-space detections.

    RF-DETR uses independent per-class sigmoids, not a mutually exclusive softmax, so the ``(Q, C)`` score grid is
    flattened to ``Q * C`` query/class pairs and the top-scoring pairs are taken *before* thresholding -- mirroring
    ``PostProcess._select_topk``. A per-query ``argmax`` would silently drop every extra class a query scores on.

    Args:
        boxes_cwh: Normalised ``cxcywh`` boxes for one image, shape ``(Q, 4)``.
        logits: Raw class logits for the same image, shape ``(Q, C)``.
        image_size: ``(width, height)`` of the source image at its original resolution, used to scale the normalised
            boxes back to pixels.
        threshold: Confidence threshold; detections at or below this score are dropped.
        num_select: Maximum query/class pairs selected before thresholding. ``None`` uses the exported model's query
            count, matching shipped RF-DETR configurations; pass an explicit value for custom exports.
        background_class_id: Exported class slot to exclude before selection. The default ``-1`` preserves the common
            final-slot background convention. Pass ``None`` for sparse COCO checkpoints, whose final slot is class 90,
            or ``0`` for legacy background-first keypoint checkpoints.

    Returns:
        The decoded detections, ordered by descending confidence.

    Examples:
        >>> import numpy as np
        >>> boxes = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
        >>> logits = np.array([[9.0, -9.0]], dtype=np.float32)
        >>> decoded = decode_detections(boxes, logits, (100, 50), background_class_id=None)
        >>> decoded.xyxy
        array([[  0.,   0., 100.,  50.]], dtype=float32)
        >>> decoded.class_id
        array([0])
    """
    # RF-DETR uses per-class sigmoid (not softmax) — mirrors PostProcess.forward in postprocess.py.
    if logits.size:
        logger.debug(
            "Logits stats: shape=%s min=%.3f max=%.3f mean=%.3f",
            logits.shape,
            float(logits.min()),
            float(logits.max()),
            float(logits.mean()),
        )
    else:
        logger.debug("Logits stats: empty shape=%s", logits.shape)
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    scores_all, class_ids = _exclude_background_class(scores_all, background_class_id)
    # The query count is read off the logits rather than the boxes; a valid RF-DETR export gives both the same Q.
    selection_cap = logits.shape[0] if num_select is None else num_select
    scores, cls, query_idx = _select_topk_multiclass(scores_all, threshold, num_select=selection_cap)
    cls = class_ids[cls]
    if scores_all.size:
        logger.debug(
            "Scores stats: min=%.3f max=%.3f — detections above threshold %.2f: %d",
            float(scores_all.min()),
            float(scores_all.max()),
            threshold,
            int(scores.shape[0]),
        )
    else:
        logger.debug("Scores stats: empty — detections above threshold %.2f: %d", threshold, int(scores.shape[0]))

    cx, cy, bw, bh = boxes_cwh[query_idx].T
    width, height = image_size
    scale = np.array([width, height, width, height], dtype=np.float32)
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy = np.clip(xyxy * scale, 0.0, scale)
    return DecodedDetections(xyxy=xyxy, confidence=scores, class_id=cls, query_index=query_idx)
