# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Class-layout handling shared by the export inference helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _exclude_background_class(
    scores_all: NDArray[np.floating],
    background_class_id: int | None,
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """Exclude one background slot while retaining the original exported class IDs.

    Args:
        scores_all: Per-query, per-class scores with shape ``(Q, C)``.
        background_class_id: Exported class slot to exclude. Negative indices follow NumPy conventions; ``None`` keeps
            every slot.

    Returns:
        The score grid to rank and its corresponding original exported class IDs.

    Raises:
        ValueError: If *scores_all* is not rank-2 or *background_class_id* is out of range.
        TypeError: If *background_class_id* is neither ``None`` nor an ``int`` (``bool`` is rejected too, since it is
            an ``int`` subclass and would otherwise silently act as index 0/1).
    """
    if scores_all.ndim != 2:
        raise ValueError(f"scores_all must have shape (Q, C); got {scores_all.shape}")

    num_classes = scores_all.shape[1]
    class_ids = np.arange(num_classes, dtype=np.int64)
    if background_class_id is None:
        return scores_all, class_ids
    if isinstance(background_class_id, bool) or not isinstance(background_class_id, (int, np.integer)):
        raise TypeError(
            f"background_class_id must be an int or None; got {background_class_id!r} "
            f"({type(background_class_id).__name__})"
        )
    if not -num_classes <= background_class_id < num_classes:
        raise ValueError(
            f"background_class_id must index one of {num_classes} exported class slots; got {background_class_id}"
        )

    foreground_mask = class_ids != background_class_id % num_classes
    return scores_all[:, foreground_mask], class_ids[foreground_mask]
