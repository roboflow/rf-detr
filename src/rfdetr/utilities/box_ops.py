# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Utilities for bounding box manipulation and GIoU."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.ops.boxes import box_area

from rfdetr.utilities.compiler import is_compiling


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Compute pairwise IoU and union for two sets of boxes.

    Returns:
        iou: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        union: the NxM matrix containing the pairwise
            union values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    eps = 1e-7
    # Clamp only the degenerate (union==0) case so identical non-degenerate boxes
    # yield IoU==1.0 exactly; adding eps unconditionally would break that identity.
    iou = inter / union.clamp(min=eps)
    return iou, union


def _assert_equal_length(boxes1: Tensor, boxes2: Tensor) -> None:
    """Reject unequal-length operands so a length-1 side cannot silently broadcast."""
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            "elementwise box ops expect boxes1 and boxes2 to have the same length, "
            f"got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )


def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Compute IoU and union for pre-matched box pairs.

    Unlike ``box_iou``, this avoids materializing the full NxN pairwise matrix
    (of which only the diagonal is used for matched pairs), so peak memory is
    O(N) instead of O(N^2). ``boxes1`` and ``boxes2`` must have the same length
    and be in [x0, y0, x1, y1] format; a length mismatch raises ``ValueError``
    rather than silently broadcasting a length-1 operand.

    The numerics (op order and the ``eps=1e-7`` union clamp) are identical to the
    diagonal of ``box_iou``, so results carry no dtype dependence beyond that of
    the pairwise version under FP16/FP32.

    Returns:
        iou: the [N] tensor of IoU values for each matched pair.
        union: the [N] tensor of union areas for each matched pair.
    """
    _assert_equal_length(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    eps = 1e-7
    # Clamp only the degenerate (union==0) case so identical non-degenerate boxes
    # yield IoU==1.0 exactly; adding eps unconditionally would break that identity.
    iou = inter / union.clamp(min=eps)
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU from https://giou.stanford.edu/ for pre-matched box pairs.

    Equivalent to the diagonal of ``generalized_box_iou`` but without building the
    NxN matrix, giving O(N) instead of O(N^2) peak memory. The boxes should be in
    [x0, y0, x1, y1] format, and ``boxes1``/``boxes2`` must have the same length; a
    length mismatch raises ``ValueError`` rather than silently broadcasting.

    Returns a [N] tensor, one GIoU value per matched pair.
    """
    _assert_equal_length(boxes1, boxes2)
    # Degenerate (zero-area) boxes would divide 0/0; eps in the denominators keeps results finite.
    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    eps = 1e-7
    # Clamp only when enclosing area is zero (degenerate enclosing box) so normal boxes remain exact.
    return iou - (area - union) / area.clamp(min=eps)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format.

    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2).
    """
    # Degenerate (zero-area) boxes would divide 0/0; eps in the denominators keeps results finite.
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    eps = 1e-7
    # Clamp only when enclosing area is zero (degenerate enclosing box) so normal boxes remain exact.
    return iou - (area - union) / area.clamp(min=eps)


#: Element budget for the broadcast intermediate in :func:`pairwise_box_l1_cost`.
#: The intermediate holds ``rows * chunk * features`` values, so capping it keeps
#: peak memory flat across batch, query and target counts instead of growing with
#: their product. 32M elements is 128 MiB in float32, 256 MiB in float64; a narrower float
#: never costs less, since :func:`pairwise_box_l1_cost` reduces it in float32.
#: The figure is a chosen memory ceiling, not a measured hardware crossover -- unlike
#: ``_STACKED_COST_ELEMENT_LIMIT`` in :mod:`rfdetr.models.matcher`, which is placed between
#: two benchmarked points. No shape measured for this change was sensitive to its exact
#: value, so it bounds a working set rather than tuning one.
_L1_COST_ELEMENT_BUDGET = 32 * 1024 * 1024


def pairwise_box_l1_cost(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise L1 distance between two sets of boxes.

    Equivalent to ``torch.cdist(boxes1, boxes2, p=1)`` for the box-shaped inputs the
    matcher builds. In eager mode it is bit-identical to it on CPU and between this
    function's own chunked and single-shot branches on every device, for every shape
    and dtype the tests exercise. CUDA-vs-``cdist`` parity is exact in practice but is
    only tested to a numerical tolerance, since kernel reduction order is not guaranteed
    identical across CUDA runner/torch versions. That guarantee is scoped to an equal-dtype
    operand pair -- a mismatched pair delegates to ``torch.cdist`` itself -- with
    narrower-than-float32 operands reduced in float32, the form ``torch.cdist`` is
    handed under autocast and refuses outside one.
    ``torch.cdist`` is a general Minkowski-distance routine: for ``p=1`` on CUDA it
    dispatches to a generic kernel that cannot exploit how small the feature
    dimension is, which makes it the most expensive single operator in the matcher
    despite doing only four subtractions per pair. Broadcasting and reducing over
    the four box coordinates is memory-bound instead, and measured 15-29x faster on
    the shapes RF-DETR's matcher produces. That figure is a CUDA measurement, and the
    trade is device-specific: on CPU the same broadcast form measured 2.7-4.9x *slower*
    than ``torch.cdist``. Training builds this cost matrix on CUDA, so the CPU regression
    is confined to tests and small-scale debugging.

    The broadcast form would materialise ``[*leading, queries, targets, features]``,
    so the target dimension is processed in chunks sized from
    :data:`_L1_COST_ELEMENT_BUDGET`. Chunking cannot change the result because the
    reduction runs over the feature axis only, never across chunks.

    Under ``torch.compile``, this takes one broadcast reduction rather than the eager chunking
    loop. Inductor can fuse that expression without unrolling a target-dependent Python loop;
    eager calls retain the bounded-memory chunked implementation below. The compiled reduction
    adds the coordinates in a fixed left-to-right order, so on CPU it is bit-identical to the
    eager branches and to ``torch.cdist``; on CUDA it carries the same tolerance-only guarantee
    against ``torch.cdist`` as the eager path.

    Args:
        boxes1: Boxes of shape ``[*leading, queries, features]``.
        boxes2: Boxes of shape ``[*leading, targets, features]``, sharing the dtype and
            device of *boxes1*. Its leading dimensions broadcast against *boxes1*'s, as
            they do in ``torch.cdist``.

    Returns:
        Pairwise L1 cost of shape ``[*leading, queries, targets]``, in the operands' dtype
        except for floating dtypes narrower than float32, which are reduced in float32.

    Examples:
        >>> boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        >>> boxes2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        >>> pairwise_box_l1_cost(boxes1, boxes2).tolist()
        [[0.0], [2.0]]
    """
    if boxes1.dtype is not boxes2.dtype or boxes1.shape[-1] != boxes2.shape[-1]:
        # Broadcasting would silently promote mixed dtypes or expand a singleton
        # feature dimension, while ``torch.cdist`` rejects both outside autocast. Keep
        # those validation failures on the original op; neither is a hot path.
        # Under autocast the mismatch is not a failure at all: ``torch.cdist`` accepts a
        # narrow-float/float32 pair there because autocast promotes both operands to
        # float32 before the call. That is the live bfloat16-prediction/float32-target
        # training configuration, so this branch is a normal path for it rather than an
        # error path. A float32/float64 mix still raises under autocast, which is the
        # rejection the matcher's dtype gate relies on.
        return torch.cdist(boxes1, boxes2, p=1)

    if boxes1.dtype.is_floating_point and boxes1.dtype not in (torch.float32, torch.float64):
        # A narrower-than-float32 pair never reached a low-precision reduction before:
        # ``torch.cdist`` refuses bfloat16/float16 outright, and under autocast -- which the
        # advertised BF16 training configuration runs in -- it is handed both operands already
        # promoted to float32. Broadcasting carries no such promotion, so reducing in the input
        # dtype would quietly drop mantissa bits the replaced call always kept. One ``.float()``
        # per 4-wide operand restores it, and the matcher casts the assembled cost matrix to
        # float32 regardless, so nothing downstream needs the narrow dtype back.
        boxes1 = boxes1.float()
        boxes2 = boxes2.float()

    if is_compiling():
        # A compiled graph can fuse this reduction without materializing the eager chunks. Keep it
        # ahead of Python shape arithmetic so dynamic target sizes never specialize the graph by chunk count.
        # The coordinates are added one at a time, left to right, instead of through ``sum(-1)``: Inductor
        # is free to reassociate a reduction, and a differently rounded ULP is enough to flip a Hungarian
        # choice on a near-tie. Explicit binary adds fix the order, which keeps the compiled cost
        # bit-identical to the eager branches below (whose ``sum`` over four elements runs in that same
        # order). The feature axis is the box width, so specializing the graph on it costs nothing.
        difference = (boxes1.unsqueeze(-2) - boxes2.unsqueeze(-3)).abs()
        if difference.shape[-1] == 0:
            return difference.new_zeros(difference.shape[:-1])
        cost = difference[..., 0]
        for feature in range(1, difference.shape[-1]):
            cost = cost + difference[..., feature]
        return cost

    # ``torch.cdist`` broadcasts the leading dimensions, so an operand can carry fewer rows than
    # the result has. Sizing anything from ``boxes1`` alone therefore disagrees with it -- silently
    # for the single-shot branch, which broadcasts its way to the right answer regardless, and as a
    # ``RuntimeError`` for the chunked one, whose preallocated buffer is then too small to assign
    # into. ``rows`` counts the result's rows, which is what the broadcast intermediate materialises.
    leading = torch.broadcast_shapes(boxes1.shape[:-2], boxes2.shape[:-2])  # type: ignore[no-untyped-call]
    queries = boxes1.shape[-2]
    targets = boxes2.shape[-2]
    features = boxes1.shape[-1]
    cost_shape = (*leading, queries, targets)
    rows = leading.numel() * queries
    if features == 0 or rows == 0 or targets == 0:
        # A zero-width feature axis still has pairs to report, and ``torch.cdist`` reports every one
        # of them as zero, so the buffer has to be zeroed rather than merely allocated. The other two
        # cases hold no elements at all, which makes the fill value unobservable there.
        return boxes1.new_zeros(cost_shape)

    # Taking the absolute value in place keeps one intermediate alive instead of two, so the
    # budget above really does bound peak memory. That saving exists only where no graph is being
    # recorded: under autograd, ``abs_`` makes PyTorch save a separate copy of the pre-``abs``
    # values for backward, which reinstates the second buffer and leaves the in-place form buying
    # nothing. Both matcher call sites run under ``torch.no_grad()`` and take the saving; this is a
    # public function, so a grad-enabled caller keeps the plain out-of-place form instead of
    # depending on that implicit copy.
    reduce_in_place = not torch.is_grad_enabled()

    chunk = max(1, min(targets, _L1_COST_ELEMENT_BUDGET // max(1, rows * features)))
    if chunk >= targets:
        difference = boxes1.unsqueeze(-2) - boxes2.unsqueeze(-3)
        return (difference.abs_() if reduce_in_place else difference.abs()).sum(-1)

    cost = boxes1.new_empty(cost_shape)
    for start in range(0, targets, chunk):
        stop = min(start + chunk, targets)
        difference = boxes1.unsqueeze(-2) - boxes2[..., start:stop, :].unsqueeze(-3)
        cost[..., start:stop] = (difference.abs_() if reduce_in_place else difference.abs()).sum(-1)
    return cost


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Compute the bounding boxes around the provided masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensor, with the boxes in xyxy format.
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    boxes = torch.stack([x_min, y_min, x_max, y_max], 1)

    keep = masks.flatten(1).any(-1)
    boxes[~keep] = boxes.new_zeros(4)
    return boxes


def batch_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss: Tensor = 1 - (numerator + 1) / (denominator + 1)
    return loss


#: Preserve the historical scripted alias until Python 3.14 makes TorchScript
#: unsupported during import, where the eager function is the safe fallback.
batch_dice_loss_jit = batch_dice_loss if sys.version_info >= (3, 14) else torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute sigmoid cross-entropy loss for mask predictions.

    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor.
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


#: Backward-compatible alias; see ``batch_dice_loss_jit``.
batch_sigmoid_ce_loss_jit = (
    batch_sigmoid_ce_loss if sys.version_info >= (3, 14) else torch.jit.script(batch_sigmoid_ce_loss)
)
