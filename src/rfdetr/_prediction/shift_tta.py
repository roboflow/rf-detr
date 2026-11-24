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

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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


def _group_score(confidence: np.ndarray[Any, Any], members: list[int], num_passes: int) -> float:
    """Return a group's fused score: the sum of its members' scores over the number of passes.

    A pass with no member in the group adds 0, so the score is the mean over all passes.

    Args:
        confidence: Float64 scores of every detection, in merged order.
        members: The group's rows.
        num_passes: Number of passes.

    Returns:
        The fused score.

    Examples:
        >>> _group_score(np.array([0.9, 0.2, 0.7]), [0, 1], num_passes=2)
        0.55
    """
    return float(confidence[members].sum() / num_passes)


def group_passes(passes: list[Detections], can_seed: np.ndarray[Any, Any] | None = None) -> list[list[int]]:
    """Group the detections of several shifted passes over the same image that are the same object.

    Rows are numbered as in ``Detections.merge(passes)``: the rows of every pass, one pass after another. Detections are
    visited highest score first, ties in row order. Each unused detection that may seed starts a group, which then takes
    at most one unused detection from every other pass: its highest-scoring one of the same class with box IoU above
    0.6, the lower row on a tie.

    A group never scores more than its seed, because every member scored at most the seed's score when the seed was
    visited. A caller that keeps only groups scoring above a per-class threshold can therefore let only the rows above
    their class's threshold seed: each such row of a class is visited before every row of that class it leaves out, and
    a group takes only rows of its seed's class, so every group that can be kept comes out the same.

    Args:
        passes: One ``Detections`` per pass, each with ``confidence`` and ``class_id``.
        can_seed: Per merged row, whether it may start a group. ``None`` lets every row start one.

    Returns:
        Each group's rows, its seed first and then its other members in pass order. Groups are sorted by fused score
        (the sum of the members' scores divided by the number of passes), highest first, ties in seed order.

    Examples:
        >>> import supervision as sv
        >>> a = sv.Detections(xyxy=np.array([[0, 0, 10, 10]], np.float32), confidence=np.array([0.9], np.float32),
        ...                   class_id=np.array([0]))
        >>> b = sv.Detections(xyxy=np.array([[1, 1, 11, 11], [50, 50, 60, 60]], np.float32),
        ...                   confidence=np.array([0.2, 0.6], np.float32), class_id=np.array([0, 0]))
        >>> group_passes([a, b])
        [[0, 1], [2]]
        >>> group_passes([a, b], can_seed=np.array([True, False, False]))
        [[0, 1]]
    """
    from supervision.detection.utils.iou_and_nms import box_iou_batch

    source = np.repeat(np.arange(len(passes)), [len(p) for p in passes])
    xyxy = np.concatenate([p.xyxy for p in passes])
    confidence = np.concatenate([np.asarray(p.confidence, dtype=np.float64) for p in passes])
    class_id = np.concatenate([np.asarray(p.class_id) for p in passes])
    seeds = np.ones(len(source), dtype=bool) if can_seed is None else np.asarray(can_seed, dtype=bool)
    used = np.zeros(len(source), dtype=bool)
    groups: list[list[int]] = []
    scores: list[float] = []
    for i in np.argsort(-confidence, kind="stable"):
        if used[i] or not seeds[i]:
            continue
        used[i] = True
        members = [int(i)]
        # One seed's IoU row at a time: every pass keeps hundreds of low-score rows, too many for a full IoU matrix.
        candidates = np.flatnonzero(~used & (class_id == class_id[i]) & (source != source[i]))
        candidates = candidates[box_iou_batch(xyxy[i : i + 1], xyxy[candidates])[0] > _SAME_OBJECT_IOU]
        if candidates.size:
            # The highest score, not the highest IoU: DETR often puts a near-copy of a box on a low-score query. Sorted
            # by pass, then score, highest first, then row, the first candidate of each pass is the one it contributes.
            candidates = candidates[np.lexsort((candidates, -confidence[candidates], source[candidates]))]
            _, first = np.unique(source[candidates], return_index=True)
            picked = candidates[first]
            used[picked] = True
            members.extend(int(j) for j in picked)
        groups.append(members)
        scores.append(_group_score(confidence, members, len(passes)))
    return [groups[r] for r in np.argsort(-np.array(scores), kind="stable")]


def fuse_passes(passes: list[Detections], groups: list[list[int]] | None = None) -> Detections:
    """Average the detections of several shifted passes over the same image.

    Each group of matching detections from :func:`group_passes` becomes one detection. Its score is the sum of its
    members' scores divided by the number of passes, so a pass that missed the object counts as a score of 0 and a pass
    that scored it low still counts at its own score. Its box is the mean box, and its class and other ``data`` fields
    come from its seed, its highest-scoring detection. Masks are not fused here: :func:`vote_masks` fuses them from
    the mask logits, for the groups the caller keeps.

    Args:
        passes: One ``Detections`` per pass, with the same ``data`` keys.
        groups: Groups of merged rows from :func:`group_passes`, seed first. ``None`` groups every detection.

    Returns:
        One detection per group, in the order of ``groups``: by default sorted by score, highest first. When every pass
        is empty, a copy of the first pass, which keeps the passes' ``data`` keys and dtypes.

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

    if passes and all(len(detections) == 0 for detections in passes):
        # Detections.merge of empty inputs returns Detections.empty(), which drops the passes' data keys and dtypes.
        return copy.deepcopy(passes[0])
    groups = group_passes(passes) if groups is None else groups
    merged = Detections.merge(passes)
    confidence = np.asarray(merged.confidence, dtype=np.float64)
    seeds = [members[0] for members in groups]
    boxes = [merged.xyxy[members].mean(axis=0) for members in groups]
    return Detections(
        xyxy=np.array(boxes, dtype=merged.xyxy.dtype).reshape(-1, 4),
        confidence=np.array(
            [_group_score(confidence, members, len(passes)) for members in groups],
            dtype=np.asarray(merged.confidence).dtype,
        ),
        class_id=np.asarray(merged.class_id)[seeds],
        data={
            key: value[seeds] if isinstance(value, np.ndarray) else [value[i] for i in seeds]
            for key, value in merged.data.items()
        },
        metadata=dict(merged.metadata),
    )


def vote_masks(
    mask_logits: Sequence[torch.Tensor], members: Sequence[Sequence[tuple[int, int]]], size: tuple[int, int]
) -> torch.Tensor:
    """Fuse each group's masks, keeping the pixels that at least half of its members cover.

    A member's mask is its pass's mask logits for its query, resized to ``size`` and thresholded at 0, as
    :class:`~rfdetr.models.postprocess.PostProcess` builds masks. Only the members passed in are resized, one group at
    a time, so at most one group's planes are held at full size.

    Args:
        mask_logits: Per pass, one image's ``(Q, Hm, Wm)`` mask logits in the image's frame (see :func:`map_to_image`).
        members: Per group, the ``(pass, query)`` of each member.
        size: The ``(H, W)`` to resize the masks to.

    Returns:
        A ``(G, H, W)`` bool tensor on the device of ``mask_logits``.

    Examples:
        >>> logits = [torch.tensor([[[1.0, -1.0], [1.0, -1.0]]]), torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])]
        >>> vote_masks(logits, [[(0, 0), (1, 0)], [(1, 0)]], size=(2, 2)).int().tolist()
        [[[1, 1], [1, 0]], [[1, 1], [0, 0]]]
    """
    height, width = size
    fused = torch.empty((len(members), height, width), dtype=torch.bool, device=mask_logits[0].device)
    for g, group in enumerate(members):
        planes = torch.stack([mask_logits[k][q] for k, q in group]).unsqueeze(1)
        covered = F.interpolate(planes, size=(height, width), mode="bilinear", align_corners=False) > 0.0
        fused[g] = covered.sum(dim=0)[0] * 2 >= len(group)
    return fused
