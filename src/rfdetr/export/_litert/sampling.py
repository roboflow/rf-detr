# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bilinear sampling for the deformable attention, in a form that litert-torch lowers to a fast ``.tflite`` graph.

litert-torch lowers ``F.grid_sample`` to coordinate and index tensors that are expanded over every channel, and to
``GATHER_ND`` with bounds checks. These run on LiteRT's default single-threaded kernels, outside the XNNPACK delegate,
and they dominate the decoder time on CPU. :func:`pixel_row_grid_sample` computes the same values from whole pixel rows
instead: one index per sample point and corner, computed in float32 so that XNNPACK runs the arithmetic, and one
``EMBEDDING_LOOKUP`` (from ``F.embedding``) per corner, which has no bounds checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

import rfdetr.models.ops.functions.ms_deform_attn_func as deform_attention
from rfdetr.utilities.tensors import _bilinear_grid_sample

# float32 represents every integer below 2**24 exactly, so a pixel-row index below it survives the float arithmetic.
_EXACT_FLOAT32_INTEGERS = 2**24
# The module-level name that ms_deform_attn_core_pytorch looks up at call time.
_DEFORMABLE_SAMPLER = "_bilinear_grid_sample"


def pixel_row_grid_sample(
    input: Tensor,
    grid: Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    """Bilinear grid sampling from padded pixel rows, equal to ``F.grid_sample`` up to float32 rounding.

    The value map gets a one-pixel ring of zeros, and each pixel becomes one row of ``channels`` values. A corner
    outside the image is clamped onto the zero ring, so it contributes zero, as the ``"zeros"`` padding requires,
    and every index is in range.

    Args:
        input: Value map of shape ``(N, C, H, W)``.
        grid: Sampling grid of shape ``(N, Hg, Wg, 2)`` with ``(x, y)`` in ``[-1, 1]`` for points inside the image.
        padding_mode: Only ``"zeros"`` takes the pixel-row path. Other modes use the default sampler.
        align_corners: Only ``False`` takes the pixel-row path. ``True`` uses the default sampler.

    Returns:
        Sampled tensor of shape ``(N, C, Hg, Wg)``.

    Examples:
        >>> value = torch.arange(4.0).view(1, 1, 2, 2)
        >>> grid = torch.tensor([[[[0.0, 0.0], [-1.0, -1.0]]]])
        >>> pixel_row_grid_sample(value, grid).flatten().tolist()
        [1.5, 0.0]
    """
    batch, channels, height, width = input.shape
    padded_height, padded_width = height + 2, width + 2
    if padding_mode != "zeros" or align_corners or batch * padded_height * padded_width >= _EXACT_FLOAT32_INTEGERS:
        return _bilinear_grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)

    grid_height, grid_width = grid.shape[1], grid.shape[2]
    x = (grid[..., 0] + 1) * (width / 2) - 0.5
    y = (grid[..., 1] + 1) * (height / 2) - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    weight_right = (x - x0).unsqueeze(-1)
    weight_bottom = (y - y0).unsqueeze(-1)
    weight_left, weight_top = 1 - weight_right, 1 - weight_bottom

    # Padded coordinates: column 0 and column width + 1 are the zero ring.
    left, right = (x0 + 1).clamp(0, width + 1), (x0 + 2).clamp(0, width + 1)
    top, bottom = (y0 + 1).clamp(0, height + 1), (y0 + 2).clamp(0, height + 1)
    pixel_rows = F.pad(input, (1, 1, 1, 1)).permute(0, 2, 3, 1).reshape(-1, channels)
    image_offsets = (
        torch.arange(batch, dtype=input.dtype, device=input.device) * (padded_height * padded_width)
    ).reshape(batch, 1, 1)

    def corner(row: Tensor, column: Tensor) -> Tensor:
        index = (image_offsets + row * padded_width + column).reshape(-1).to(torch.int32)
        return F.embedding(index, pixel_rows).reshape(batch, grid_height, grid_width, channels)

    output = (
        weight_left * weight_top * corner(top, left)
        + weight_right * weight_top * corner(top, right)
        + weight_left * weight_bottom * corner(bottom, left)
        + weight_right * weight_bottom * corner(bottom, right)
    )
    return output.permute(0, 3, 1, 2)


@contextmanager
def pixel_row_sampling() -> Iterator[None]:
    """Make the deformable attention sample with :func:`pixel_row_grid_sample` inside the ``with`` block.

    The swap is process-wide for the duration of the block, so the LiteRT exporter holds it only around the capture.
    The segmentation head imports its own reference to the default sampler and is not affected.

    Examples:
        >>> original = deform_attention._bilinear_grid_sample
        >>> with pixel_row_sampling():
        ...     deform_attention._bilinear_grid_sample is pixel_row_grid_sample
        True
        >>> deform_attention._bilinear_grid_sample is original
        True
    """
    original = getattr(deform_attention, _DEFORMABLE_SAMPLER)
    setattr(deform_attention, _DEFORMABLE_SAMPLER, pixel_row_grid_sample)
    try:
        yield
    finally:
        setattr(deform_attention, _DEFORMABLE_SAMPLER, original)
