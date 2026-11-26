# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Test-time augmentation views: sub-patch shifts and zoom, fused into one prediction.

The backbone cuts its input into a grid of ``patch_size`` squares, and an object's class score depends a little on where
the object sits inside its square. Moving an image by a fraction of a patch can therefore swing a score across the
threshold while the box stays put (
https://github.com/roboflow/rf-detr/issues/1599).
TTA runs the batch once per
*view*: the image is scaled and placed at a sub-patch offset on a canvas that is at least as large as the
image, so no view cuts content off. Each view is mapped back to the image, and the matching detections of all views are
averaged. ``RFDETRInference.predict(..., tta=TTAConfig(...))`` takes the views: ``shift`` sub-patch shifted
views and ``zoom`` scaled views.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import resize

if TYPE_CHECKING:
    from supervision import Detections

# Two detections of different passes are the same object when they share a class and overlap by more than this.
_SAME_OBJECT_IOU = 0.6

# Range of ``TTAConfig.zoom``. The cookbook measured scales from 0.85 to 1.5; the lower bound leaves some room below.
ZOOM_MIN = 0.75
ZOOM_MAX = 1.5

# The zoom views of ``TTAConfig()``: the plain image plus these three scales, four views. On COCO val2017 this recipe
# added 2.0 to 4.0 box AP on Nano, Small and Large, at 3 to 4 times the latency of one plain call.
DEFAULT_ZOOM = (0.85, 1.25, 1.5)


def validate_shift(shift: object, patch_size: int) -> int:
    """Check the ``shift`` field of ``TTAConfig`` against the backbone patch size.

    Args:
        shift: Number of shifted views; 1 adds no shift.
        patch_size: Backbone patch size in model input pixels. Each view needs its own sub-patch offset.

    Returns:
        The number of shifted views.

    Raises:
        ValueError: If ``shift`` is not an integer from 1 to ``patch_size``.

    Examples:
        >>> validate_shift(4, patch_size=16)
        4
    """
    if isinstance(shift, bool) or not isinstance(shift, int) or not 1 <= shift <= patch_size:
        raise ValueError(f"TTAConfig.shift must be an integer from 1 to the patch size ({patch_size}), got {shift!r}.")
    return shift


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


@dataclass(frozen=True)
class TTAConfig:
    """The test-time augmentation views of one ``RFDETRInference.predict`` call.

    Passing ``tta`` at all is the opt-in; ``tta=None`` is the plain prediction. ``TTAConfig()`` is the recommended
    recipe, the native-scale view plus three zoomed copies (``DEFAULT_ZOOM``), four views. With no explicit zoom
    scales, ``shift > 1`` suppresses the default zooms, so ``TTAConfig(shift=4)`` is four views.
    ``TTAConfig(zoom=())`` with the default shift runs one plain prediction. Every view
    costs one forward pass over a canvas at least as large as the image, so the cost grows with the number of views
    (see :func:`view_count`) and, for zoom, with the canvas area.

    Attributes:
        shift: Number of sub-patch shifted views, from 1 to the backbone patch size; 1 adds no shift.
        zoom: Extra views at these scales, each a resized copy of the image on a larger canvas. Every value is a finite
            number inside ``[ZOOM_MIN, ZOOM_MAX]`` and not 1. ``None``, the default, means ``DEFAULT_ZOOM`` when
            ``shift`` is 1, and no zoom otherwise; it is stored as the tuple it resolves to.

    Raises:
        TypeError: If a field has the wrong type, or ``zoom`` is a string, a mapping or a set.
        ValueError: If ``shift`` is below 1, or a ``zoom`` is not finite, equals 1, is out of range or repeats.

    Examples:
        >>> TTAConfig()
        TTAConfig(shift=1, zoom=(0.85, 1.25, 1.5))
        >>> TTAConfig(shift=4)
        TTAConfig(shift=4, zoom=())
        >>> TTAConfig(shift=2, zoom=[1.25])
        TTAConfig(shift=2, zoom=(1.25,))
        >>> view_count(TTAConfig(shift=2, zoom=(1.25,)))
        3
    """

    shift: int = 1
    zoom: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Check every field and store ``zoom`` as a tuple of floats."""
        if isinstance(self.shift, bool) or not isinstance(self.shift, int):
            raise TypeError(f"TTAConfig.shift must be an int, got {type(self.shift).__name__}.")
        if self.shift < 1:
            raise ValueError(f"TTAConfig.shift must be at least 1, got {self.shift!r}.")
        requested = self.zoom
        if requested is None:
            requested = DEFAULT_ZOOM if self.shift == 1 else ()
        if not isinstance(requested, (list, tuple)):
            raise TypeError(f"TTAConfig.zoom must be a list or tuple of numbers, got {type(requested).__name__}.")
        zoom: list[float] = []
        for value in requested:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"TTAConfig.zoom holds numbers only, got {type(value).__name__}.")
            if not ZOOM_MIN <= value <= ZOOM_MAX or value == 1.0:
                raise ValueError(
                    f"TTAConfig.zoom values must be finite, not 1 and in [{ZOOM_MIN}, {ZOOM_MAX}], got {value!r}."
                )
            zoom.append(float(value))
        if len(set(zoom)) != len(zoom):
            raise ValueError(f"TTAConfig.zoom must not repeat a value, got {requested!r}.")
        object.__setattr__(self, "zoom", tuple(zoom))


def view_count(config: TTAConfig) -> int:
    """Return how many views ``config`` runs: shifts plus zooms.

    Args:
        config: The requested views.

    Returns:
        The number of views; 1 is the plain path.

    Examples:
        >>> view_count(TTAConfig())
        4
        >>> view_count(TTAConfig(zoom=()))
        1
        >>> view_count(TTAConfig(shift=4))
        4
    """
    return config.shift + len(config.zoom or ())


@dataclass(frozen=True)
class _View:
    """One pass: the image is scaled by ``scale`` and placed at ``(offset, offset)`` on a canvas.

    Attributes:
        scale: 1 is native size, above 1 zooms in, below 1 zooms out. A view never crops.
        offset: Placement down and right, in model input pixels. Shift views use ``0`` to ``patch_size - 1``.
    """

    scale: float = 1.0
    offset: int = 0


def placed_size(view: _View, size: tuple[int, int]) -> tuple[int, int]:
    """Return the ``(height, width)`` the image takes on the canvas, rounded as the resize rounds each axis.

    Args:
        view: The view.
        size: The image's ``(H, W)`` in model input pixels.

    Returns:
        The placed content size, never below 1.

    Examples:
        >>> placed_size(_View(scale=1.25), (384, 384))
        (480, 480)
    """
    return max(1, round(view.scale * size[0])), max(1, round(view.scale * size[1]))


def build_views(config: TTAConfig, patch_size: int, size: tuple[int, int]) -> list[_View]:
    """Return the views of ``config`` for one model input size.

    Shift views come first (scale 1, the offsets of :func:`shift_offsets`), then one view per zoom (offset 0). The
    views are additive, not a product, so 4 shifts and 3 zooms are 7 views, not 12.

    Args:
        config: The requested views.
        patch_size: Backbone patch size in model input pixels.
        size: The model input ``(H, W)``; every image is resized to it before the views are placed.

    Returns:
        The views, in the order the fused result counts them.

    Raises:
        ValueError: If ``config.shift`` is above ``patch_size``, or a zoom places the image at the size of another view
            (it would repeat that view and double its weight).

    Examples:
        >>> [(v.scale, v.offset) for v in build_views(TTAConfig(shift=2, zoom=(1.25,)), 16, (64, 64))]
        [(1.0, 0), (1.0, 8), (1.25, 0)]
    """
    validate_shift(config.shift, patch_size)
    views = [_View(offset=offset) for offset in shift_offsets(patch_size, config.shift)]
    sizes = {placed_size(views[0], size)}
    for zoom in config.zoom or ():
        view = _View(scale=zoom)
        placed = placed_size(view, size)
        if placed in sizes:
            raise ValueError(f"TTAConfig.zoom {zoom!r} places the image at {placed}, the size of another view.")
        sizes.add(placed)
        views.append(view)
    return views


def canvas_shape(view: _View, size: tuple[int, int], block: int) -> tuple[int, int]:
    """Return the canvas ``(height, width)`` of a view.

    The canvas is one ``block`` larger than the image while the placed content and its offset fit, which is the room a
    shift needs and the canvas every shift view shares. A larger zoom gets the block multiple its content needs.

    Args:
        view: The view.
        size: The image's ``(H, W)`` in model input pixels, multiples of ``block``.
        block: ``patch_size * num_windows``, the unit every model input side is a multiple of.

    Returns:
        The canvas size, a multiple of ``block`` on each axis.

    Examples:
        >>> canvas_shape(_View(), (64, 64), 32)
        (96, 96)
        >>> canvas_shape(_View(scale=1.5), (64, 64), 32)
        (96, 96)
        >>> canvas_shape(_View(scale=1.75), (64, 64), 32)
        (128, 128)
    """
    sides = []
    for side, content in zip(size, placed_size(view, size)):
        end = content + view.offset
        sides.append(side + block if end <= side + block else math.ceil(end / block) * block)
    return sides[0], sides[1]


def place_view(
    batch: torch.Tensor,
    view: _View,
    canvas: tuple[int, int],
    *,
    antialias: bool = False,
    resize_on_cpu: bool = False,
) -> torch.Tensor:
    """Scale and place a normalized ``(B, C, H, W)`` batch on a zero canvas.

    The canvas is zero, the normalization mean colour, so the backbone sees the whole image and a plain border, never
    cropped content. A scaled view is resampled with the same call preprocessing uses.

    Args:
        batch: Normalized model input.
        view: The view to place.
        canvas: The canvas ``(height, width)`` from :func:`canvas_shape`.
        antialias: Antialias the resize of a scaled view; pass the flag the batch itself was resized with.
        resize_on_cpu: Resize on the CPU and move the result back, for a device with no antialiased resize kernel.

    Returns:
        A ``(B, C, Hc, Wc)`` tensor.

    Examples:
        >>> place_view(torch.ones(1, 1, 2, 2), _View(offset=1), (4, 4))[0, 0].int().tolist()
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
        >>> place_view(torch.tensor([[[[1.0, 2.0]]]]), _View(), (1, 2))[0, 0].tolist()
        [[1.0, 2.0]]
    """
    b, c, h, w = batch.shape
    content = batch
    ch, cw = placed_size(view, (h, w))
    if (ch, cw) != (h, w):
        resized = resize(content.cpu() if resize_on_cpu else content, [ch, cw], antialias=antialias)
        content = resized.to(batch.device)
    placed = batch.new_zeros((b, c, *canvas))
    placed[:, :, view.offset : view.offset + ch, view.offset : view.offset + cw] = content
    return placed


def map_to_image(
    outputs: dict[str, torch.Tensor],
    view: _View,
    size: tuple[int, int],
    canvas: tuple[int, int],
    mask_size: tuple[int, int] | None,
) -> dict[str, torch.Tensor]:
    """Map raw model outputs for a canvas from :func:`place_view` back to the image it holds.

    Boxes are normalized ``cxcywh`` and become relative to the image instead of the canvas. Boxes outside the image
    content have zero area. Mask logits are resampled from the image's region of the canvas to ``mask_size``. The
    result reads as if the model had seen the image directly, so the usual post-processing applies unchanged.

    Args:
        outputs: Model outputs with ``pred_logits``, ``pred_boxes`` and optionally ``pred_masks``.
        view: The view the batch was placed with.
        size: The image's ``(H, W)`` in model input pixels.
        canvas: The canvas ``(height, width)`` the batch was placed on.
        mask_size: The ``(height, width)`` of the returned mask logits, shared by every view of a request so the masks
            of different views stack; ignored without ``pred_masks``.

    Returns:
        A copy of ``outputs`` with ``pred_boxes`` and ``pred_masks`` in the image's frame.

    Examples:
        >>> out = {"pred_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])}
        >>> mapped = map_to_image(out, _View(offset=2), (6, 6), (8, 8), None)
        >>> [round(v, 4) for v in mapped["pred_boxes"][0, 0].tolist()]
        [0.3333, 0.3333, 0.3333, 0.3333]
    """
    canvas_h, canvas_w = canvas
    ch, cw = placed_size(view, size)
    mapped = dict(outputs)
    cx, cy, w, h = outputs["pred_boxes"].unbind(-1)
    # Test overlap in canvas coordinates before rescaling the center and size, which can round a touching edge inward.
    outside_x = (cx + w / 2 <= view.offset / canvas_w) | (cx - w / 2 >= (view.offset + cw) / canvas_w)
    outside_y = (cy + h / 2 <= view.offset / canvas_h) | (cy - h / 2 >= (view.offset + ch) / canvas_h)
    cx = (cx * canvas_w - view.offset) / cw
    cy = (cy * canvas_h - view.offset) / ch
    w = (w * canvas_w / cw).masked_fill(outside_x, 0)
    h = (h * canvas_h / ch).masked_fill(outside_y, 0)
    mapped["pred_boxes"] = torch.stack((cx, cy, w, h), dim=-1)

    masks = outputs.get("pred_masks")
    if masks is not None and mask_size is not None:
        out_h, out_w = mask_size
        # Centres of the output mask pixels as fractions of the image.
        frac_y = (torch.arange(out_h, device=masks.device) + 0.5) / out_h
        frac_x = (torch.arange(out_w, device=masks.device) + 0.5) / out_w
        # The same centres in the canvas's [-1, 1] grid_sample coordinates (align_corners=False).
        ys = (view.offset + frac_y * ch) * 2 / canvas_h - 1
        xs = (view.offset + frac_x * cw) * 2 / canvas_w - 1
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
