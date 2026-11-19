# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sub-patch shift test-time augmentation for ``RFDETR.predict(..., shift_tta=N)``.

The backbone cuts its input into a grid of ``patch_size`` squares, and an object's class score depends a little on where
the object sits inside its square. Moving an image by a fraction of a patch can therefore swing a score across the
threshold while the box stays put (
https://github.com/roboflow/rf-detr/issues/1599).
Shift TTA runs the batch
``N`` times, each time with the image placed at a different fraction of a patch inside a slightly larger canvas,
maps each pass back to the image, and averages the matching detections of all passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from supervision import Detections

# Two detections of different passes are the same object when they share a class and overlap by more than this.
_SAME_OBJECT_IOU = 0.6


def validate_shift_tta(shift_tta: object, patch_size: int) -> int:
    """Check the ``shift_tta`` argument of ``predict``.

    Args:
        shift_tta: Number of passes; 1 turns shift TTA off.
        patch_size: Backbone patch size in model input pixels. Each pass needs its own sub-patch offset.

    Returns:
        The number of passes.

    Raises:
        ValueError: If ``shift_tta`` is not an integer from 1 to ``patch_size``.

    Examples:
        >>> validate_shift_tta(4, patch_size=16)
        4
    """
    if isinstance(shift_tta, bool) or not isinstance(shift_tta, int) or not 1 <= shift_tta <= patch_size:
        raise ValueError(f"shift_tta must be an integer from 1 to the patch size ({patch_size}), got {shift_tta!r}.")
    return shift_tta


def shift_offsets(patch_size: int, passes: int) -> list[int]:
    """Return evenly spaced sub-patch offsets, in model input pixels, one per pass.

    The image moves along the diagonal, by the same offset in x and y.

    Args:
        patch_size: Backbone patch size in model input pixels.
        passes: Number of passes.

    Returns:
        ``passes`` distinct offsets in ``[0, patch_size)``, starting at 0.

    Examples:
        >>> shift_offsets(16, 4)
        [0, 4, 8, 12]
        >>> shift_offsets(12, 2)
        [0, 6]
    """
    return [round(i * patch_size / passes) for i in range(passes)]


def place_on_canvas(batch: torch.Tensor, offset: int, margin: int) -> torch.Tensor:
    """Put a normalized ``(B, C, H, W)`` batch at ``(offset, offset)`` on a canvas ``margin`` pixels larger.

    The canvas is zero, the normalization mean colour, so the backbone sees the whole image and a plain border,
    never cropped content. ``margin`` should keep the canvas a valid model shape (a multiple of
    ``patch_size * num_windows``) and be larger than every offset.

    Args:
        batch: Normalized model input.
        offset: Shift down and right, in pixels.
        margin: How much larger the canvas is in each dimension.

    Returns:
        A ``(B, C, H + margin, W + margin)`` tensor.

    Examples:
        >>> place_on_canvas(torch.ones(1, 1, 2, 2), offset=1, margin=2)[0, 0].int().tolist()
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    """
    b, c, h, w = batch.shape
    canvas = batch.new_zeros((b, c, h + margin, w + margin))
    canvas[:, :, offset : offset + h, offset : offset + w] = batch
    return canvas


def map_to_image(
    outputs: dict[str, torch.Tensor], offset: int, size: tuple[int, int], margin: int
) -> dict[str, torch.Tensor]:
    """Map raw model outputs for a canvas from :func:`place_on_canvas` back to the image it holds.

    Boxes are normalized ``cxcywh`` and become relative to the image instead of the canvas. Mask logits are resampled
    from the image's region of the canvas, keeping their resolution relative to the image. The result reads as if
    the model had seen the image directly, so the usual post-processing applies unchanged.

    Args:
        outputs: Model outputs with ``pred_logits``, ``pred_boxes`` and optionally ``pred_masks``.
        offset: The offset the image was placed at.
        size: The image's ``(H, W)`` in model input pixels.
        margin: How much larger the canvas was.

    Returns:
        A copy of ``outputs`` with ``pred_boxes`` and ``pred_masks`` in the image's frame.

    Examples:
        >>> out = {"pred_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])}
        >>> map_to_image(out, offset=2, size=(6, 6), margin=2)["pred_boxes"].tolist()
        [[[0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408]]]
    """
    h, w = size
    canvas_h, canvas_w = h + margin, w + margin
    mapped = dict(outputs)
    boxes = outputs["pred_boxes"]
    scale = boxes.new_tensor([canvas_w / w, canvas_h / h, canvas_w / w, canvas_h / h])
    shift = boxes.new_tensor([offset / w, offset / h, 0.0, 0.0])
    mapped["pred_boxes"] = boxes * scale - shift

    masks = outputs.get("pred_masks")
    if masks is not None:
        mask_h, mask_w = masks.shape[-2:]
        out_h, out_w = max(1, round(mask_h * h / canvas_h)), max(1, round(mask_w * w / canvas_w))
        # Centres of the output mask pixels, in the canvas's [-1, 1] grid_sample coordinates (align_corners=False).
        ys = (offset + (torch.arange(out_h, device=masks.device) + 0.5) * h / out_h) * 2 / canvas_h - 1
        xs = (offset + (torch.arange(out_w, device=masks.device) + 0.5) * w / out_w) * 2 / canvas_w - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).to(masks.dtype).expand(masks.shape[0], -1, -1, -1)
        mapped["pred_masks"] = F.grid_sample(masks, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return mapped


def fuse_passes(passes: list[Detections]) -> Detections:
    """Average the detections of several shifted passes over the same image.

    Detections are grouped greedily, highest score first: each group takes at most one detection from every other
    pass, its highest-scoring one of the same class with box IoU above 0.6. A group's score is the sum of its scores
    divided by the number of passes, so a pass that missed the object counts as a score of 0. Its box is the mean box,
    its mask (if any) keeps the pixels that at least half of its masks cover, and its other fields come from its
    highest-scoring detection.

    Args:
        passes: One ``Detections`` per pass, with the same ``data`` keys.

    Returns:
        One detection per group, sorted by score, highest first.

    Examples:
        >>> import supervision as sv
        >>> a = sv.Detections(xyxy=np.array([[0, 0, 10, 10]], np.float32), confidence=np.array([0.9], np.float32),
        ...                   class_id=np.array([0]))
        >>> b = sv.Detections(xyxy=np.array([[1, 1, 11, 11]], np.float32), confidence=np.array([0.5], np.float32),
        ...                   class_id=np.array([0]))
        >>> fused = fuse_passes([a, b])
        >>> fused.xyxy.tolist(), [round(float(c), 2) for c in fused.confidence]
        ([[0.5, 0.5, 10.5, 10.5]], [0.7])
    """
    from supervision import Detections
    from supervision.detection.utils.iou_and_nms import box_iou_batch

    merged = Detections.merge(passes)
    if len(merged) == 0:
        return merged
    source = np.repeat(np.arange(len(passes)), [len(p) for p in passes])
    confidence = np.asarray(merged.confidence, dtype=np.float64)
    class_id = np.asarray(merged.class_id)
    mask = None if merged.mask is None else np.asarray(merged.mask, dtype=bool)
    iou = box_iou_batch(merged.xyxy, merged.xyxy)
    used = np.zeros(len(merged), dtype=bool)
    seeds: list[int] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    for i in np.argsort(-confidence, kind="stable"):
        if used[i]:
            continue
        used[i] = True
        members = [int(i)]
        for k in range(len(passes)):
            if k == source[i]:
                continue
            candidates = np.flatnonzero(~used & (source == k) & (class_id == class_id[i]) & (iou[i] > _SAME_OBJECT_IOU))
            if candidates.size:
                # The highest score, not the highest IoU: DETR often puts a near-copy of a box on a low-score query.
                j = int(candidates[np.argmax(confidence[candidates])])
                used[j] = True
                members.append(j)
        seeds.append(int(i))
        boxes.append(merged.xyxy[members].mean(axis=0))
        scores.append(float(confidence[members].sum() / len(passes)))
        if mask is not None:
            masks.append(mask[members].mean(axis=0) >= 0.5)

    rank = np.argsort(-np.array(scores), kind="stable")
    order = np.array(seeds)[rank]
    return Detections(
        xyxy=np.stack(boxes)[rank].astype(merged.xyxy.dtype),
        confidence=np.array(scores, dtype=np.asarray(merged.confidence).dtype)[rank],
        class_id=class_id[order],
        mask=np.stack(masks)[rank] if masks else None,
        data={
            key: value[order] if isinstance(value, np.ndarray) else [value[i] for i in order]
            for key, value in merged.data.items()
        },
        metadata=dict(merged.metadata),
    )
