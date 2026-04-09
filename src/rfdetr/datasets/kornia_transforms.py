# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Kornia-based GPU augmentation pipeline for RF-DETR detection training.

This module provides GPU-side augmentation as an alternative to the CPU-based
Albumentations pipeline.  All transforms run on the device where the batch
already resides (typically CUDA), avoiding a CPU-GPU round-trip per sample.

Phase 1 supports detection bounding boxes only; segmentation masks are
deferred to phase 2.

Usage::

    from rfdetr.datasets.kornia_transforms import (
        build_kornia_pipeline,
        build_normalize,
        collate_boxes,
        unpack_boxes,
    )

    pipeline = build_kornia_pipeline(aug_config, resolution=560)
    normalize = build_normalize()

    # In on_after_batch_transfer:
    boxes_padded, valid = collate_boxes(targets, device)
    img_aug, boxes_aug = pipeline(img, boxes_padded)
    img_aug = normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, H, W)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: ImageNet channel-wise mean (RGB order).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
#: ImageNet channel-wise standard deviation (RGB order).
IMAGENET_STD = (0.229, 0.224, 0.225)


def _has_cuda_device() -> bool:
    """Return ``True`` when the runtime has a CUDA accelerator available.

    Uses the fork-safe global ``DEVICE`` constant from ``rfdetr.config`` so that
    the CUDA driver context is not created in the main process before forking
    (fork-based DDP and some notebook environments).

    Returns:
        ``True`` if at least one CUDA device is reachable; ``False`` otherwise.

    Examples:
        >>> _has_cuda_device()  # doctest: +SKIP
        False
    """
    from rfdetr.config import DEVICE

    return str(DEVICE).startswith("cuda")


def resolve_augmentation_backend(backend: str) -> str:
    """Resolve an ``augmentation_backend`` value to a concrete ``"cpu"`` or ``"gpu"``.

    ``"auto"`` resolves to ``"gpu"`` only when both CUDA and Kornia are available;
    otherwise it falls back to ``"cpu"``.  Explicit ``"cpu"`` and ``"gpu"`` values
    pass through unchanged; ``"gpu"`` is validated (CUDA + kornia presence).

    Args:
        backend: One of ``"cpu"``, ``"auto"``, or ``"gpu"``.

    Returns:
        ``"cpu"`` or ``"gpu"``.

    Raises:
        RuntimeError: When *backend* is ``"gpu"`` and no CUDA device is found.
        ImportError: When *backend* is ``"gpu"`` and kornia is not installed.
        ValueError: When *backend* is not one of ``"cpu"``, ``"auto"``, or ``"gpu"``.

    Examples:
        >>> resolve_augmentation_backend("cpu")
        'cpu'
    """
    if backend == "cpu":
        return "cpu"
    if backend == "auto":
        if not _has_cuda_device():
            return "cpu"
        try:
            import kornia.augmentation  # noqa: F401 # type: ignore[import-not-found]
        except ImportError:
            return "cpu"
        return "gpu"
    if backend == "gpu":
        if not _has_cuda_device():
            raise RuntimeError("augmentation_backend='gpu' requires a CUDA device")
        _require_kornia()
        return "gpu"
    raise ValueError(f"Unknown augmentation_backend {backend!r}; expected 'cpu', 'auto', or 'gpu'.")


def _require_kornia() -> None:
    """Verify that Kornia is importable, raising a clear error if not.

    Raises:
        ImportError: When ``kornia`` is not installed, with an install hint.
    """
    try:
        import kornia.augmentation  # noqa: F401
    except ImportError as e:
        raise ImportError("GPU augmentation requires kornia. Install with: pip install 'rfdetr[kornia]'") from e


# ---------------------------------------------------------------------------
# Registry: Albumentations key -> Kornia factory
# ---------------------------------------------------------------------------


def _make_horizontal_flip(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomHorizontalFlip`` from aug_config params."""
    from kornia.augmentation import RandomHorizontalFlip

    return RandomHorizontalFlip(p=params.get("p", 0.5))


def _make_vertical_flip(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomVerticalFlip`` from aug_config params."""
    from kornia.augmentation import RandomVerticalFlip

    return RandomVerticalFlip(p=params.get("p", 0.5))


def _make_rotate(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomRotation`` from aug_config params.

    The ``limit`` parameter may be a scalar (symmetric range) or a tuple.
    """
    from kornia.augmentation import RandomRotation

    limit = params.get("limit", 15)
    degrees = tuple(limit) if isinstance(limit, (list, tuple)) else (-limit, limit)
    return RandomRotation(degrees=degrees, p=params.get("p", 0.5))


def _make_affine(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomAffine`` from aug_config params.

    Albumentations ``translate_percent`` is a ``(min, max)`` signed range
    (e.g. ``(-0.1, 0.1)``).  Kornia ``translate`` is a non-negative
    per-axis max fraction ``(tx, ty)`` where translation is sampled from
    ``[-tx, tx]``.  The conversion takes ``max(|min|, |max|)`` for each
    axis, producing a symmetric range that matches the intent.
    """
    from kornia.augmentation import RandomAffine

    translate_percent = params.get("translate_percent")
    if translate_percent is not None:
        if isinstance(translate_percent, (list, tuple)) and len(translate_percent) == 2:
            t = max(abs(translate_percent[0]), abs(translate_percent[1]))
            translate: float | tuple[float, float] | None = (t, t)
        else:
            translate = translate_percent
    else:
        translate = None

    return RandomAffine(
        degrees=params.get("rotate", (-15, 15)),
        translate=translate,
        scale=params.get("scale"),
        shear=params.get("shear"),
        p=params.get("p", 0.5),
    )


def _make_color_jitter(params: dict[str, Any]) -> Any:
    """Build a ``K.ColorJiggle`` from aug_config ``ColorJitter`` params.

    Note: Kornia >=0.7 uses ``ColorJiggle``; the ``ColorJitter`` alias was
    added in later versions.  We use ``ColorJiggle`` for broad compatibility.
    """
    from kornia.augmentation import ColorJiggle

    return ColorJiggle(
        brightness=params.get("brightness", 0.0),
        contrast=params.get("contrast", 0.0),
        saturation=params.get("saturation", 0.0),
        hue=params.get("hue", 0.0),
        p=params.get("p", 0.5),
    )


def _make_random_brightness_contrast(params: dict[str, Any]) -> Any:
    """Build a ``K.ColorJiggle`` from ``RandomBrightnessContrast`` params."""
    from kornia.augmentation import ColorJiggle

    return ColorJiggle(
        brightness=params.get("brightness_limit", 0.2),
        contrast=params.get("contrast_limit", 0.2),
        p=params.get("p", 0.5),
    )


def _make_gaussian_blur(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomGaussianBlur`` from aug_config params.

    ``blur_limit`` is rounded up to an odd number for the kernel size.
    """
    from kornia.augmentation import RandomGaussianBlur

    blur_limit = params.get("blur_limit", 3)
    # Ensure blur_limit is odd and at least 3 (Kornia requires kernel_size >= 3)
    if blur_limit % 2 == 0:
        blur_limit = blur_limit + 1
    blur_limit = max(3, blur_limit)
    return RandomGaussianBlur(
        kernel_size=(blur_limit, blur_limit),
        sigma=(0.1, 2.0),
        p=params.get("p", 0.5),
    )


def _make_gauss_noise(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomGaussianNoise`` from aug_config params.

    Kornia takes a single ``std`` value; we use the upper bound of
    ``std_range`` as an acceptable approximation.
    """
    from kornia.augmentation import RandomGaussianNoise

    std_range = params.get("std_range", (0.01, 0.05))
    return RandomGaussianNoise(
        std=std_range[1],
        p=params.get("p", 0.5),
    )


_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "HorizontalFlip": _make_horizontal_flip,
    "VerticalFlip": _make_vertical_flip,
    "Rotate": _make_rotate,
    "Affine": _make_affine,
    "ColorJitter": _make_color_jitter,
    "RandomBrightnessContrast": _make_random_brightness_contrast,
    "GaussianBlur": _make_gaussian_blur,
    "GaussNoise": _make_gauss_noise,
}


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_kornia_pipeline(
    aug_config: dict[str, dict[str, Any]],
    resolution: int,
) -> Any:
    """Build a Kornia ``AugmentationSequential`` from an aug_config dict.

    Each key in *aug_config* is looked up in ``_REGISTRY`` and instantiated
    with the corresponding parameter dict.  Unknown keys raise ``ValueError``.

    Args:
        aug_config: Mapping of augmentation names to parameter dicts, identical
            to the format accepted by the Albumentations path (e.g.
            ``{"HorizontalFlip": {"p": 0.5}}``).
        resolution: Target image resolution in pixels (currently reserved for
            future resolution-aware augmentations).

    Returns:
        A ``kornia.augmentation.AugmentationSequential`` instance configured
        with ``data_keys=["input", "bbox_xyxy"]``.

    Raises:
        ValueError: If *aug_config* contains an unsupported augmentation key.
    """
    _require_kornia()
    from kornia.augmentation import AugmentationSequential

    transforms: list[Any] = []
    for name, params in aug_config.items():
        factory = _REGISTRY.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown augmentation key {name!r} for Kornia GPU backend. Supported keys: {sorted(_REGISTRY)}."
            )
        transforms.append(factory(params))

    return AugmentationSequential(
        *transforms,
        data_keys=["input", "bbox_xyxy"],
    )


def build_normalize(
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Any:
    """Build a Kornia ``Normalize`` transform for GPU-side normalization.

    Args:
        mean: Per-channel mean values.  Defaults to ImageNet statistics.
        std: Per-channel standard deviation values.  Defaults to ImageNet
            statistics.

    Returns:
        A ``kornia.augmentation.Normalize`` instance.
    """
    _require_kornia()
    from kornia.augmentation import Normalize

    return Normalize(
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def collate_boxes(
    targets: list[dict[str, Any]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Pack variable-length xyxy boxes into a padded tensor and valid mask.

    Kornia ``AugmentationSequential`` expects boxes as ``[B, N_max, 4]``.
    This function zero-pads each image's boxes to the maximum count in the
    batch and returns a boolean mask indicating which entries are real.

    Args:
        targets: List of target dicts (one per image), each containing a
            ``"boxes"`` key with an ``[N_i, 4]`` tensor in xyxy format.
        device: Device on which to allocate the output tensors.

    Returns:
        Tuple of:
            - ``boxes_padded`` — ``[B, N_max, 4]`` float tensor (zero-padded).
            - ``valid_mask``   — ``[B, N_max]`` bool tensor (``True`` = real box).

        When ``B == 0`` or all images have zero boxes, both tensors have
        ``N_max == 0``.
    """
    if len(targets) == 0:
        return (
            torch.zeros(0, 0, 4, device=device),
            torch.zeros(0, 0, dtype=torch.bool, device=device),
        )

    box_counts = [t["boxes"].shape[0] for t in targets]
    n_max = max(box_counts) if box_counts else 0
    batch_size = len(targets)

    if n_max == 0:
        return (
            torch.zeros(batch_size, 0, 4, device=device),
            torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )

    boxes_padded = torch.zeros(batch_size, n_max, 4, device=device)
    valid_mask = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)

    for i, t in enumerate(targets):
        n = t["boxes"].shape[0]
        if n > 0:
            boxes_padded[i, :n] = t["boxes"]
            valid_mask[i, :n] = True

    return boxes_padded, valid_mask


def unpack_boxes(
    boxes_aug: Tensor,
    valid: Tensor,
    targets: list[dict[str, Any]],
    image_height: int,
    image_width: int,
) -> list[dict[str, Any]]:
    """Unpack augmented boxes, clamp to image bounds, and remove zero-area boxes.

    After Kornia augmentation the padded ``[B, N_max, 4]`` tensor is unpacked
    back into per-image target dicts.  Boxes are clamped to ``[0, W] x [0, H]``
    and any that collapse to zero area are removed along with their
    corresponding ``labels``, ``area``, and ``iscrowd`` entries.

    Args:
        boxes_aug: Augmented boxes tensor ``[B, N_max, 4]`` in xyxy format.
        valid: Boolean mask ``[B, N_max]`` from :func:`collate_boxes`.
        targets: Original target dicts; each dict is shallow-copied before
            modification — the input list itself is not mutated.
        image_height: Image height in pixels (for clamping).
        image_width: Image width in pixels (for clamping).

    Returns:
        A new list of target dicts with updated ``boxes``, ``labels``,
        ``area``, and ``iscrowd`` entries.
    """
    new_targets: list[dict[str, Any]] = []
    for i, t in enumerate(targets):
        t = t.copy()
        n_orig = t["boxes"].shape[0]

        if n_orig == 0 or valid.shape[1] == 0:
            new_targets.append(t)
            continue

        # Extract valid boxes for this image
        v = valid[i, :n_orig]
        boxes_i = boxes_aug[i, :n_orig]

        # Clamp to image boundaries
        boxes_i = boxes_i.clone()
        boxes_i[:, 0].clamp_(min=0, max=image_width)
        boxes_i[:, 1].clamp_(min=0, max=image_height)
        boxes_i[:, 2].clamp_(min=0, max=image_width)
        boxes_i[:, 3].clamp_(min=0, max=image_height)

        # Remove zero-area boxes (after clamping)
        widths = boxes_i[:, 2] - boxes_i[:, 0]
        heights = boxes_i[:, 3] - boxes_i[:, 1]
        keep = v & (widths > 0) & (heights > 0)

        t["boxes"] = boxes_i[keep]
        if "labels" in t:
            t["labels"] = t["labels"][keep]
        if "area" in t:
            # Recompute area from clamped boxes
            kept_boxes = t["boxes"]
            t["area"] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
        if "iscrowd" in t:
            t["iscrowd"] = t["iscrowd"][keep]

        new_targets.append(t)

    return new_targets
