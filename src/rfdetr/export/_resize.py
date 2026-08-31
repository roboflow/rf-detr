# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Torch-free bilinear resize shared by the export inference helpers.

``RFDETR.predict()`` resizes with ``torchvision`` ``antialias=False`` bilinear (half-pixel centers, no low-pass filter).
The export inference paths must reproduce that exact convention so exported models score images the same way the PyTorch
model does. When ``torch`` is importable they call torchvision directly; this module provides the NumPy equivalent for
torch-free deployments.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _bilinear_resize_half_pixel(src: NDArray[np.float32], out_h: int, out_w: int) -> NDArray[np.float32]:
    """Numpy bilinear resize matching ``F.interpolate(mode="bilinear", align_corners=False)``.

    Half-pixel center convention with no antialias filter — the same convention
    ``torchvision.transforms.functional.resize(..., antialias=False)`` and
    ``RFDETR.predict()`` use. Serves as the torch-free fallback for both image
    preprocessing and mask decoding.

    Args:
        src: Source array of shape ``(K, src_h, src_w)``.
        out_h: Target height in pixels.
        out_w: Target width in pixels.

    Returns:
        Float32 array of shape ``(K, out_h, out_w)``.

    Note:
        Replaces ``PIL.Image.resize(BILINEAR)``, which applies an adaptive
        antialias filter when downscaling and a corner-aligned half-pixel
        convention, both of which diverge from ``F.interpolate``.
    """
    src_h, src_w = src.shape[-2], src.shape[-1]
    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    src_y = np.clip(src_y, 0.0, src_h - 1)
    src_x = np.clip(src_x, 0.0, src_w - 1)
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    dy = (src_y - y0)[:, None]
    dx = (src_x - x0)[None, :]
    if 3 * src_h < 4 * out_h:
        # Interpolate each source row horizontally once, then gather the two rows needed for
        # each output pixel. This preserves the existing arithmetic order while avoiding four
        # full output-grid gathers. Keep the three source-height work grids below the size of
        # those four output-height gathers; otherwise retain the bounded formulation below.
        left = np.take(src, x0, axis=-1)
        right = np.take(src, x1, axis=-1)
        horizontal = (1 - dx) * left + dx * right
        out = (1 - dy) * horizontal[..., y0, :] + dy * horizontal[..., y1, :]
    else:
        a = src[..., y0[:, None], x0[None, :]]
        b = src[..., y0[:, None], x1[None, :]]
        c = src[..., y1[:, None], x0[None, :]]
        d = src[..., y1[:, None], x1[None, :]]
        out = (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)
    return np.asarray(out, dtype=np.float32)
