# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bilinear resize as a matrix product, for the Core ML pass that replaces ``upsample_bilinear``.

This module needs no coremltools, so the CPU test suite checks the matrices against ``F.interpolate``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bilinear_interpolation_matrix(size_in: int, size_out: int, align_corners: bool) -> NDArray[np.float32]:
    """The matrix that resizes one axis the way Core ML's ``upsample_bilinear`` does.

    Row ``i`` holds the weights of the two source cells around output cell ``i``, sampled and clamped as the
    ``upsample_bilinear`` documentation specifies. With one matrix per axis, ``R_h @ x @ R_w.T`` equals
    ``F.interpolate(x, size=..., mode="bilinear", align_corners=align_corners)``.

    Args:
        size_in: Length of the axis before the resize.
        size_out: Length of the axis after the resize.
        align_corners: Whether the centers of the first and last cells line up, as in ``F.interpolate``.

    Returns:
        A float32 ``[size_out, size_in]`` matrix whose rows each sum to 1.

    Examples:
        >>> bilinear_interpolation_matrix(2, 4, align_corners=False)
        array([[1.  , 0.  ],
               [0.75, 0.25],
               [0.25, 0.75],
               [0.  , 1.  ]], dtype=float32)
    """
    out = np.arange(size_out, dtype=np.float64)
    if align_corners:
        source = out * ((size_in - 1) / (size_out - 1) if size_out > 1 else 0.0)
    else:
        source = (out + 0.5) * size_in / size_out - 0.5
    source = np.clip(source, 0, size_in - 1)
    low = np.floor(source).astype(np.int64)
    high = np.minimum(low + 1, size_in - 1)
    fraction = source - low
    matrix = np.zeros((size_out, size_in), dtype=np.float32)
    rows = out.astype(np.int64)
    np.add.at(matrix, (rows, low), 1 - fraction)
    np.add.at(matrix, (rows, high), fraction)
    return matrix
