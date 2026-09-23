# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ATen decompositions RF-DETR needs on top of ``coreai_torch.get_decomp_table()``.

RF-DETR's deformable attention samples its value maps with :func:`torch.nn.functional.grid_sample`, which exports as
``aten.grid_sampler_2d``. coreai-torch has no lowering for that op, so :func:`grid_sampler_2d_gather` decomposes it
into floor, clamp, one flat ``gather`` per bilinear corner, and a weighted sum — ops every Core AI compute unit runs.

The in-bounds masks are computed with float arithmetic on integer-valued coordinates instead of the usual
``(x >= 0) & (x < W)`` comparison chain: the Core AI runtime can overwrite an unrelated, still-live tensor when such a
comparison -> bool -> float chain executes (apple/coreai-torch#11), and the float form is exact because the corner
coordinates are integers. The formulation follows the RF-DETR Core AI port in the community
`coreai-model-zoo <https://github.com/john-rocky/coreai-model-zoo>`_.

:func:`topk_in_float32` keeps a float16 export's two-stage query selection off the Neural Engine, whose float16
``topk`` returns corrupt indices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

#: ``interpolation_mode`` / ``padding_mode`` codes ``aten.grid_sampler_2d`` receives from ``F.grid_sample``.
_BILINEAR = 0
_ZEROS = 0
_BORDER = 1


def _inside_mask(coordinate: Tensor, clamped: Tensor) -> Tensor:
    """Return 1 where an integer-valued *coordinate* lies inside the image and 0 where clamping moved it.

    Args:
        coordinate: Integer-valued (floored) sample coordinate.
        clamped: *coordinate* clamped into the image.

    Returns:
        A float mask of the same shape and dtype as *coordinate*.

    Examples:
        >>> x = torch.tensor([-2.0, -1.0, 0.0, 3.0, 4.0])
        >>> _inside_mask(x, x.clamp(0, 3)).tolist()
        [0.0, 0.0, 1.0, 1.0, 0.0]
    """
    return 1.0 - (coordinate - clamped).abs().clamp(max=1.0)


def grid_sampler_2d_gather(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int = _BILINEAR,
    padding_mode: int = _ZEROS,
    align_corners: bool = False,
) -> Tensor:
    """Bilinear ``aten.grid_sampler_2d`` built from gathers, for converters that cannot lower the op itself.

    Matches ``F.grid_sample(input, grid, mode="bilinear", padding_mode=..., align_corners=...)`` for the
    ``"zeros"`` and ``"border"`` padding modes, the only ones RF-DETR uses.

    Args:
        input: Feature map of shape ``(N, C, H, W)``.
        grid: Sampling grid of shape ``(N, Hg, Wg, 2)`` with ``(x, y)`` in ``[-1, 1]``.
        interpolation_mode: ``aten.grid_sampler_2d`` interpolation code; only ``0`` (bilinear) is supported.
        padding_mode: ``aten.grid_sampler_2d`` padding code; ``0`` (zeros) or ``1`` (border).
        align_corners: Whether ``-1``/``1`` address the centres (``True``) or the outer edges of the corner pixels.

    Returns:
        Sampled tensor of shape ``(N, C, Hg, Wg)``.

    Raises:
        NotImplementedError: For nearest or bicubic interpolation, or reflection padding.

    Examples:
        >>> value = torch.arange(4.0).view(1, 1, 2, 2)
        >>> grid = torch.tensor([[[[0.0, 0.0], [-1.0, -1.0]]]])
        >>> grid_sampler_2d_gather(value, grid, 0, 0, True).flatten().tolist()
        [1.5, 0.0]
    """
    if interpolation_mode != _BILINEAR or padding_mode not in (_ZEROS, _BORDER):
        raise NotImplementedError(
            "Core AI export decomposes aten.grid_sampler_2d for bilinear interpolation with zeros or border padding"
            f" only, got interpolation_mode={interpolation_mode}, padding_mode={padding_mode}."
        )
    batch, channels, height, width = input.shape
    grid_height, grid_width = grid.shape[1], grid.shape[2]

    if align_corners:
        x = (grid[..., 0] + 1) * ((width - 1) / 2)
        y = (grid[..., 1] + 1) * ((height - 1) / 2)
    else:
        x = (grid[..., 0] + 1) * (width / 2) - 0.5
        y = (grid[..., 1] + 1) * (height / 2) - 0.5
    if padding_mode == _BORDER:
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    wx1 = (x - x0).unsqueeze(1)
    wy1 = (y - y0).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    flat = input.reshape(batch, channels, height * width)

    def corner(yi: Tensor, xi: Tensor) -> Tensor:
        xc = xi.clamp(0, width - 1)
        yc = yi.clamp(0, height - 1)
        # Flat indices are computed in float32 so a float16 graph stays exact beyond 2048 pixels.
        index = (yc.float() * width + xc.float()).to(torch.int64)
        index = index.reshape(batch, 1, grid_height * grid_width).expand(batch, channels, grid_height * grid_width)
        value = flat.gather(2, index).reshape(batch, channels, grid_height, grid_width)
        if padding_mode == _ZEROS:
            value = value * (_inside_mask(xi, xc) * _inside_mask(yi, yc)).unsqueeze(1)
        return value

    x1 = x0 + 1.0
    y1 = y0 + 1.0
    return (
        corner(y0, x0) * (wy0 * wx0)
        + corner(y0, x1) * (wy0 * wx1)
        + corner(y1, x0) * (wy1 * wx0)
        + corner(y1, x1) * (wy1 * wx1)
    )


def topk_in_float32(
    input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[Tensor, Tensor] | Any:
    """Run a float16 ``aten.topk`` in float32, and decline every other dtype.

    On the Neural Engine, which iOS and iPadOS 27 select for float16 graphs by default, a float16 ``topk`` returns its
    indices as 16-bit integers in an int32 buffer: every element packs two consecutive indices, so RF-DETR's
    two-stage query selection gathers garbage and the detector returns nothing. A float32 ``topk`` is not placed on
    the Neural Engine, and its values are cast back so the rest of the graph stays float16.

    Args:
        input: Scores to rank.
        k: Number of entries to keep.
        dim: Dimension to rank along.
        largest: Keep the largest (``True``) or smallest entries.
        sorted: Return the entries in sorted order.

    Returns:
        ``(values, indices)`` for a float16 *input*, else ``NotImplemented`` so the op is kept as it is.

    Examples:
        >>> values, indices = topk_in_float32(torch.tensor([[0.5, 2.0, 1.0]], dtype=torch.float16), 2, 1)
        >>> values.dtype, indices.tolist()
        (torch.float16, [[1, 2]])
    """
    if input.dtype != torch.float16:
        return NotImplemented
    values, indices = torch.ops.aten.topk.default(input.to(torch.float32), k, dim, largest, sorted)
    return values.to(input.dtype), indices


def coreai_decomposition_table(base: Mapping[Any, Callable[..., Any]]) -> dict[Any, Callable[..., Any]]:
    """Return *base* (normally ``coreai_torch.get_decomp_table()``) extended with RF-DETR's Core AI decompositions.

    Args:
        base: Decomposition table to extend. It is copied, not modified.

    Returns:
        A new table for ``ExportedProgram.run_decompositions()``.

    Examples:
        >>> table = coreai_decomposition_table({})
        >>> table[torch.ops.aten.grid_sampler_2d.default] is grid_sampler_2d_gather
        True
    """
    table = dict(base)
    table[torch.ops.aten.grid_sampler_2d.default] = grid_sampler_2d_gather
    table[torch.ops.aten.topk.default] = topk_in_float32
    return table
