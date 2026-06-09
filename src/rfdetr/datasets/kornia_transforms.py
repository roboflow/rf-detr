# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Kornia-based augmentation pipeline for RF-DETR training.

This module provides Kornia transforms for both RF-DETR dataset-time CPU preprocessing and optional GPU-side batch
augmentation. Dataset-time transforms preserve the existing ``(PIL.Image, target)`` contract used by COCO and YOLO
datasets while replacing the previous Albumentations wrapper.

Supports detection (boxes only) and segmentation (boxes + instance masks).

Usage::

    from rfdetr.datasets.kornia_transforms import (
        build_kornia_pipeline,
        build_normalize,
        collate_boxes,
        collate_masks,
        unpack_boxes,
    )

    # Detection:
    pipeline = build_kornia_pipeline(aug_config, resolution=560)
    normalize = build_normalize()
    boxes_padded, valid = collate_boxes(targets, device)
    img_aug, boxes_aug = pipeline(img, boxes_padded)
    img_aug = normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, H, W)

    # Segmentation (Phase 2):
    pipeline = build_kornia_pipeline(aug_config, resolution=560, with_masks=True)
    normalize = build_normalize()
    boxes_padded, valid = collate_boxes(targets, device)
    masks_padded = collate_masks(targets, device, n_max=valid.shape[1], image_height=H, image_width=W)
    img_aug, boxes_aug, masks_aug = pipeline(img, boxes_padded, masks_padded)
    img_aug = normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, H, W, masks_aug=masks_aug)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from rfdetr.utilities.logger import get_logger

logger = get_logger()

__doctest_requires__ = {"build_kornia_pipeline": ["kornia"]}

#: ImageNet channel-wise mean (RGB order).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
#: ImageNet channel-wise standard deviation (RGB order).
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Threshold applied to float32 mask values produced by Kornia augmentation.
#: Kornia forces nearest-neighbour resampling for the ``"mask"`` data key, so
#: output values are already in {0.0, 1.0}; the threshold is a defensive cast.
#: Must be updated if the pipeline is ever switched to bilinear interpolation.
_MASK_BINARIZE_THRESHOLD: float = 0.5


def _has_cuda_device() -> bool:
    """Return ``True`` when the runtime has a CUDA accelerator available.

    Uses the fork-safe global ``DEVICE`` constant from ``rfdetr.config`` so that the CUDA driver context is not created
    in the main process before forking (fork-based DDP and some notebook environments).

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

    ``"auto"`` resolves to ``"gpu"`` only when both CUDA and Kornia are available; otherwise it falls back to ``"cpu"``.
    Explicit ``"cpu"`` and ``"gpu"`` values pass through unchanged; ``"gpu"`` is validated (CUDA + kornia presence).

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
        raise ImportError("Training augmentation requires kornia. Install with: pip install 'rfdetr[train]'") from e


# ---------------------------------------------------------------------------
# Registry: RF-DETR augmentation key -> Kornia factory
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
    rotation = RandomRotation(degrees=degrees, p=params.get("p", 0.5))

    # Kornia has changed the public parameter key for rotation ranges across releases.
    # Keep the legacy ``degrees`` entry available because our tests and downstream
    # callers inspect it directly.
    flags = getattr(rotation, "flags", None)
    if isinstance(flags, dict) and "degrees" not in flags:
        flags["degrees"] = degrees

    return rotation


def _make_affine(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomAffine`` from aug_config params.

    RF-DETR ``translate_percent`` is a ``(min, max)`` signed range (e.g. ``(-0.1, 0.1)``).  Kornia ``translate`` is a
    non-negative per-axis max fraction ``(tx, ty)`` where translation is sampled from ``[-tx, tx]``.  The
    conversion takes ``max(|min|, |max|)`` for each axis, producing a symmetric range that matches the intent.
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

    Kornia takes a single ``std`` value; we use the upper bound of ``std_range`` as an acceptable approximation.
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

_CONTAINER_KEYS = frozenset({"OneOf", "Sequential"})
_RESIZE_KEYS = frozenset({"Resize", "SmallestMaxSize", "LongestMaxSize", "RandomSizedCrop"})
SUPPORTED_KORNIA_TRANSFORMS = frozenset(_REGISTRY) | _CONTAINER_KEYS | _RESIZE_KEYS


# ---------------------------------------------------------------------------
# Dataset-time CPU transforms
# ---------------------------------------------------------------------------


def _as_config_entries(config_dict: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize an augmentation config into ordered single-key entries.

    Args:
        config_dict: Mapping or list of single-key mappings.

    Returns:
        Ordered list of single-key transform entries.

    Raises:
        TypeError: If *config_dict* has an unsupported type.
        ValueError: If any entry is not a single-key dictionary.
    """
    if isinstance(config_dict, list):
        entries = config_dict
    elif isinstance(config_dict, dict):
        entries = [{k: v} for k, v in config_dict.items()]
    else:
        raise TypeError(f"config_dict must be a dictionary or list, got {type(config_dict).__name__}")

    for entry in entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Each transform config entry must be a single-key dict, got {entry!r}")
    return entries


def _choose_size(size: int | Sequence[int]) -> int:
    """Choose an integer size from a scalar or sequence."""
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if not size:
            raise ValueError("size sequence must not be empty")
        index = int(torch.randint(0, len(size), ()).item())
        return int(size[index])
    return int(size)


def _pil_to_float_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image to a batched RGB float tensor in ``[0, 1]``."""
    image_np = np.asarray(image.convert("RGB")).copy()
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0)


def _float_tensor_to_pil(image: Tensor) -> Image.Image:
    """Convert a batched float tensor in ``[0, 1]`` to a PIL RGB image."""
    image = image.detach().squeeze(0).clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _resize_tensor(input_tensor: Tensor, size: tuple[int, int], interpolation: str) -> Tensor:
    """Resize a tensor using Kornia, with compatibility across Kornia minor versions."""
    from kornia.geometry.transform import resize

    if interpolation == "nearest":
        return resize(input_tensor, size, interpolation=interpolation)
    try:
        return resize(input_tensor, size, interpolation=interpolation, align_corners=False, antialias=True)
    except TypeError:
        return resize(input_tensor, size, interpolation=interpolation, align_corners=False)


def _resize_masks(masks: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize instance masks to ``size`` with nearest-neighbour interpolation."""
    if masks.numel() == 0:
        return torch.zeros((masks.shape[0], size[0], size[1]), dtype=torch.float32, device=masks.device)
    masks_4d = masks.to(dtype=torch.float32).unsqueeze(1)
    return _resize_tensor(masks_4d, size, interpolation="nearest").squeeze(1)


def _filter_instance_field(value: Any, keep: Tensor, n_orig: int) -> Any:
    """Filter a per-instance target field if its leading dimension matches boxes."""
    if torch.is_tensor(value):
        if value.ndim >= 1 and value.shape[0] == n_orig:
            return value[keep.to(device=value.device)]
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == n_orig:
        keep_list = keep.cpu().tolist()
        return [item for item, should_keep in zip(value, keep_list) if should_keep]
    return value


def _sanitize_target(
    target: dict[str, Any] | None,
    boxes: Tensor | None,
    image_height: int,
    image_width: int,
    masks: Tensor | None = None,
) -> dict[str, Any] | None:
    """Clamp boxes, remove invalid instances, sync per-instance fields, and update target size."""
    if target is None:
        return None

    target_out = target.copy()
    target_out["size"] = torch.as_tensor([image_height, image_width], dtype=torch.int64)
    if boxes is None or "boxes" not in target:
        return target_out

    n_orig = target["boxes"].shape[0]
    boxes = boxes.to(dtype=torch.float32).clone()
    if n_orig == 0:
        target_out["boxes"] = boxes.reshape(0, 4)
        if "labels" in target:
            target_out["labels"] = target["labels"].new_empty((0,))
        if "area" in target:
            target_out["area"] = target["area"].new_empty((0,))
        if "iscrowd" in target:
            target_out["iscrowd"] = target["iscrowd"].new_empty((0,))
        if "masks" in target:
            target_out["masks"] = torch.zeros((0, image_height, image_width), dtype=torch.bool)
        return target_out

    boxes[:, 0].clamp_(min=0, max=image_width)
    boxes[:, 1].clamp_(min=0, max=image_height)
    boxes[:, 2].clamp_(min=0, max=image_width)
    boxes[:, 3].clamp_(min=0, max=image_height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    global_fields = {"boxes", "orig_size", "size", "image_id"}
    for key, value in target.items():
        if key in global_fields:
            continue
        if key == "masks" and masks is not None:
            continue
        target_out[key] = _filter_instance_field(value, keep, n_orig)

    kept_boxes = boxes[keep]
    target_out["boxes"] = kept_boxes
    if "area" in target_out:
        target_out["area"] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
    if masks is not None:
        target_out["masks"] = masks[keep] > _MASK_BINARIZE_THRESHOLD
    elif "masks" in target:
        target_out["masks"] = target["masks"][keep].to(dtype=torch.bool)
    return target_out


def _resize_sample(
    image: Tensor,
    target: dict[str, Any] | None,
    size: tuple[int, int],
) -> tuple[Tensor, dict[str, Any] | None]:
    """Resize image, boxes, and masks to ``size``."""
    old_height, old_width = image.shape[-2:]
    new_height, new_width = size
    image_out = _resize_tensor(image, size, interpolation="bilinear")

    if target is None or "boxes" not in target:
        return image_out, _sanitize_target(target, None, new_height, new_width)

    scale = target["boxes"].new_tensor(
        [new_width / old_width, new_height / old_height, new_width / old_width, new_height / old_height]
    )
    boxes = target["boxes"] * scale
    masks = _resize_masks(target["masks"], size) if "masks" in target else None
    return image_out, _sanitize_target(target, boxes, new_height, new_width, masks)


def _crop_sample(
    image: Tensor,
    target: dict[str, Any] | None,
    top: int,
    left: int,
    height: int,
    width: int,
) -> tuple[Tensor, dict[str, Any] | None]:
    """Crop image, boxes, and masks to the requested rectangle."""
    image_out = image[..., top : top + height, left : left + width]
    if target is None or "boxes" not in target:
        return image_out, _sanitize_target(target, None, height, width)

    boxes = target["boxes"].clone()
    offset = boxes.new_tensor([left, top, left, top])
    boxes = boxes - offset
    masks = (
        target["masks"][..., top : top + height, left : left + width].to(dtype=torch.float32)
        if "masks" in target
        else None
    )
    return image_out, _sanitize_target(target, boxes, height, width, masks)


class _DatasetTransform:
    """Base protocol for dataset-time Kornia transforms."""

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Apply the transform to a batched image tensor and optional target."""
        raise NotImplementedError


class _SequentialTransform(_DatasetTransform):
    """Apply child transforms in order."""

    def __init__(self, transforms: list[_DatasetTransform]) -> None:
        """Initialize with ordered child transforms."""
        self.transforms = transforms

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Apply all child transforms in order."""
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class _OneOfTransform(_DatasetTransform):
    """Apply one child transform sampled uniformly."""

    def __init__(self, transforms: list[_DatasetTransform]) -> None:
        """Initialize with candidate child transforms."""
        if not transforms:
            raise ValueError("'OneOf' requires at least one transform")
        self.transforms = transforms

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Sample and apply one child transform."""
        index = int(torch.randint(0, len(self.transforms), ()).item())
        return self.transforms[index](image, target)


class _ResizeTransform(_DatasetTransform):
    """Resize to a fixed ``(height, width)``."""

    def __init__(self, height: int, width: int) -> None:
        """Initialize the target size."""
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Resize image and target."""
        return _resize_sample(image, target, (self.height, self.width))


class _SmallestMaxSizeTransform(_DatasetTransform):
    """Resize so the shortest side equals the selected size."""

    def __init__(self, max_size: int | Sequence[int]) -> None:
        """Initialize with one or more candidate short-side sizes."""
        self.max_size = max_size

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Resize image and target by shortest side."""
        height, width = image.shape[-2:]
        target_short = _choose_size(self.max_size)
        scale = target_short / min(height, width)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        return _resize_sample(image, target, (new_height, new_width))


class _LongestMaxSizeTransform(_DatasetTransform):
    """Cap the longest side at the selected size."""

    def __init__(self, max_size: int | Sequence[int]) -> None:
        """Initialize with one or more candidate long-side caps."""
        self.max_size = max_size

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Downscale image and target only when the longest side exceeds the cap."""
        height, width = image.shape[-2:]
        target_long = _choose_size(self.max_size)
        current_long = max(height, width)
        if current_long <= target_long:
            return image, _sanitize_target(target, target.get("boxes") if target is not None else None, height, width)
        scale = target_long / current_long
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        return _resize_sample(image, target, (new_height, new_width))


class _RandomSizedCropTransform(_DatasetTransform):
    """Random square crop followed by resize to the configured output size."""

    def __init__(self, min_max_height: Sequence[int], height: int, width: int) -> None:
        """Initialize crop height range and output size."""
        if len(min_max_height) != 2:
            raise ValueError("RandomSizedCrop.min_max_height must contain exactly two values")
        self.min_height = int(min_max_height[0])
        self.max_height = int(min_max_height[1])
        self.output_height = int(height)
        self.output_width = int(width)

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Apply a random crop and resize the result."""
        image_height, image_width = image.shape[-2:]
        max_crop = max(1, min(self.max_height, image_height, image_width))
        min_crop = max(1, min(self.min_height, max_crop))
        crop_size = int(torch.randint(min_crop, max_crop + 1, ()).item())
        max_top = image_height - crop_size
        max_left = image_width - crop_size
        top = int(torch.randint(0, max_top + 1, ()).item()) if max_top > 0 else 0
        left = int(torch.randint(0, max_left + 1, ()).item()) if max_left > 0 else 0
        image, target = _crop_sample(image, target, top, left, crop_size, crop_size)
        return _resize_sample(image, target, (self.output_height, self.output_width))


class _HorizontalFlipTransform(_DatasetTransform):
    """Horizontally flip a sample with pixel-edge box semantics."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the flip probability."""
        self.p = float(p)

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Flip image, boxes, and masks horizontally."""
        if torch.rand(()).item() >= self.p:
            return image, target

        from kornia.geometry.transform import hflip

        image_out = hflip(image)
        height, width = image.shape[-2:]
        if target is None or "boxes" not in target:
            return image_out, _sanitize_target(target, None, height, width)

        boxes = target["boxes"].clone()
        x_min = boxes[:, 0].clone()
        x_max = boxes[:, 2].clone()
        boxes[:, 0] = width - x_max
        boxes[:, 2] = width - x_min
        masks = hflip(target["masks"].to(dtype=torch.float32).unsqueeze(1)).squeeze(1) if "masks" in target else None
        return image_out, _sanitize_target(target, boxes, height, width, masks)


class _VerticalFlipTransform(_DatasetTransform):
    """Vertically flip a sample with pixel-edge box semantics."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the flip probability."""
        self.p = float(p)

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Flip image, boxes, and masks vertically."""
        if torch.rand(()).item() >= self.p:
            return image, target

        from kornia.geometry.transform import vflip

        image_out = vflip(image)
        height, width = image.shape[-2:]
        if target is None or "boxes" not in target:
            return image_out, _sanitize_target(target, None, height, width)

        boxes = target["boxes"].clone()
        y_min = boxes[:, 1].clone()
        y_max = boxes[:, 3].clone()
        boxes[:, 1] = height - y_max
        boxes[:, 3] = height - y_min
        masks = vflip(target["masks"].to(dtype=torch.float32).unsqueeze(1)).squeeze(1) if "masks" in target else None
        return image_out, _sanitize_target(target, boxes, height, width, masks)


class _KorniaAugmentationTransform(_DatasetTransform):
    """Apply a Kornia augmentation module to image, boxes, and optional masks."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """Initialize the augmentation by RF-DETR transform key."""
        factory = _REGISTRY.get(name)
        if factory is None:
            raise ValueError(
                f"Unsupported Kornia transform {name!r}. Supported keys: {sorted(SUPPORTED_KORNIA_TRANSFORMS)}"
            )
        self.name = name
        self.params = params

    def _build_pipeline(self, with_target: bool, with_masks: bool) -> Any:
        """Build a Kornia ``AugmentationSequential`` for the current input type."""
        from kornia.augmentation import AugmentationSequential

        module = _REGISTRY[self.name](self.params)
        if not with_target:
            data_keys = ["input"]
        else:
            data_keys = ["input", "bbox_xyxy", "mask"] if with_masks else ["input", "bbox_xyxy"]
        return AugmentationSequential(module, data_keys=data_keys)

    def __call__(self, image: Tensor, target: dict[str, Any] | None) -> tuple[Tensor, dict[str, Any] | None]:
        """Apply the Kornia augmentation and sanitize target fields."""
        if target is None or "boxes" not in target:
            pipeline = self._build_pipeline(with_target=False, with_masks=False)
            image_height, image_width = image.shape[-2:]
            return pipeline(image), _sanitize_target(target, None, image_height, image_width)

        if target["boxes"].shape[0] == 0:
            pipeline = self._build_pipeline(with_target=False, with_masks=False)
            image_out = pipeline(image)
            image_height, image_width = image_out.shape[-2:]
            return image_out, _sanitize_target(target, target["boxes"], image_height, image_width)

        boxes = target["boxes"].unsqueeze(0)
        if "masks" in target:
            masks = target["masks"].unsqueeze(0).to(dtype=torch.float32)
            pipeline = self._build_pipeline(with_target=True, with_masks=True)
            image_out, boxes_out, masks_out = pipeline(image, boxes, masks)
            image_height, image_width = image_out.shape[-2:]
            return image_out, _sanitize_target(
                target, boxes_out.squeeze(0), image_height, image_width, masks_out.squeeze(0)
            )

        pipeline = self._build_pipeline(with_target=True, with_masks=False)
        image_out, boxes_out = pipeline(image, boxes)
        image_height, image_width = image_out.shape[-2:]
        return image_out, _sanitize_target(target, boxes_out.squeeze(0), image_height, image_width)


def _build_dataset_transform(name: str, params: Any) -> _DatasetTransform:
    """Build one dataset-time Kornia transform from an RF-DETR config entry."""
    if isinstance(params, list) and name in _CONTAINER_KEYS:
        params = {"transforms": params}
    if not isinstance(params, dict):
        raise ValueError(f"Parameters for transform {name!r} must be a dict, got {type(params).__name__}")

    if name == "OneOf":
        raw_nested = params.get("transforms", [])
        if not isinstance(raw_nested, list):
            raise ValueError("'OneOf.transforms' must be a list")
        return _OneOfTransform([_build_dataset_transform(*next(iter(entry.items()))) for entry in raw_nested])
    if name == "Sequential":
        raw_nested = params.get("transforms", [])
        if not isinstance(raw_nested, list):
            raise ValueError("'Sequential.transforms' must be a list")
        return _SequentialTransform([_build_dataset_transform(*next(iter(entry.items()))) for entry in raw_nested])
    if name == "Resize":
        return _ResizeTransform(height=params["height"], width=params["width"])
    if name == "SmallestMaxSize":
        return _SmallestMaxSizeTransform(max_size=params["max_size"])
    if name == "LongestMaxSize":
        return _LongestMaxSizeTransform(max_size=params["max_size"])
    if name == "RandomSizedCrop":
        return _RandomSizedCropTransform(
            min_max_height=params["min_max_height"],
            height=params["height"],
            width=params["width"],
        )
    if name == "HorizontalFlip":
        return _HorizontalFlipTransform(p=params.get("p", 0.5))
    if name == "VerticalFlip":
        return _VerticalFlipTransform(p=params.get("p", 0.5))
    if name in _REGISTRY:
        return _KorniaAugmentationTransform(name, params)
    raise ValueError(f"Unsupported Kornia transform {name!r}. Supported keys: {sorted(SUPPORTED_KORNIA_TRANSFORMS)}")


class KorniaWrapper:
    """Apply Kornia dataset-time transforms to ``(PIL.Image, target)`` tuples."""

    def __init__(self, transform: _DatasetTransform) -> None:
        """Initialize the wrapper with a dataset-time transform."""
        self.transform = transform

    def __call__(self, image: Image.Image, target: dict[str, Any] | None) -> tuple[Image.Image, dict[str, Any] | None]:
        """Apply the wrapped transform and convert the image back to PIL."""
        _require_kornia()
        image_tensor = _pil_to_float_tensor(image)
        image_tensor, target = self.transform(image_tensor, target)
        return _float_tensor_to_pil(image_tensor), target

    @staticmethod
    def from_config(config_dict: dict[str, Any] | list[dict[str, Any]]) -> list["KorniaWrapper"]:
        """Build Kornia wrappers from an RF-DETR augmentation config.

        Args:
            config_dict: Either a transform-name mapping or an ordered list of single-key transform dictionaries.

        Returns:
            A list containing one sequential Kornia wrapper, or an empty list for an empty config.
        """
        entries = _as_config_entries(config_dict)
        if not entries:
            logger.warning("Empty augmentation config provided, no transforms will be applied")
            return []
        transforms = [_build_dataset_transform(*next(iter(entry.items()))) for entry in entries]
        logger.info("Built %d Kornia dataset transforms from config", len(transforms))
        return [KorniaWrapper(_SequentialTransform(transforms))]


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_kornia_pipeline(
    aug_config: dict[str, dict[str, Any]],
    resolution: int,
    with_masks: bool = False,
) -> Any:
    """Build a Kornia ``AugmentationSequential`` from an aug_config dict.

    Each key in *aug_config* is looked up in ``_REGISTRY`` and instantiated with the corresponding parameter dict.
    Unknown keys raise ``ValueError``.

    Args:
        aug_config: Mapping of RF-DETR Kornia augmentation names to parameter dicts
            (e.g. ``{"HorizontalFlip": {"p": 0.5}}``).
        resolution: Target image resolution in pixels (currently reserved for
            future resolution-aware augmentations).
        with_masks: When ``True``, include ``"mask"`` in ``data_keys`` so
            instance segmentation masks are augmented in sync with images and boxes.  The pipeline then expects three
            inputs ``(img, boxes, masks)`` and returns three outputs.  Defaults to ``False`` (detection-only, two
            inputs/outputs).

    Returns:
        A ``kornia.augmentation.AugmentationSequential`` instance.

    Raises:
        ValueError: If *aug_config* contains an unsupported augmentation key.

    Examples:
        >>> from rfdetr.datasets.aug_config import AUG_CONSERVATIVE
        >>> pipeline = build_kornia_pipeline(AUG_CONSERVATIVE, resolution=560)
        >>> pipeline_seg = build_kornia_pipeline(AUG_CONSERVATIVE, resolution=560, with_masks=True)
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

    data_keys = ["input", "bbox_xyxy", "mask"] if with_masks else ["input", "bbox_xyxy"]
    return AugmentationSequential(
        *transforms,
        data_keys=data_keys,
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

    Kornia ``AugmentationSequential`` expects boxes as ``[B, N_max, 4]``. This function zero-pads each image's boxes to
    the maximum count in the batch and returns a boolean mask indicating which entries are real.

    Args:
        targets: List of target dicts (one per image), each containing a
            ``"boxes"`` key with an ``[N_i, 4]`` tensor in xyxy format.
        device: Device on which to allocate the output tensors.

    Returns:
        Tuple of:
            - ``boxes_padded`` — ``[B, N_max, 4]`` float tensor (zero-padded).
            - ``valid_mask``   — ``[B, N_max]`` bool tensor (``True`` = real box).

        When ``B == 0`` or all images have zero boxes, both tensors have ``N_max == 0``.
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


def collate_masks(
    targets: list[dict[str, Any]],
    device: torch.device,
    n_max: int,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Pack variable-length instance masks into a zero-padded ``[B, N_max, H, W]`` tensor.

    Kornia ``AugmentationSequential`` expects masks as ``[B, N_max, H, W]`` when ``data_keys`` includes ``"mask"``.
    This function zero-pads each image's masks to *n_max* channels (matching the padding used by :func:`collate_boxes`)
    and converts boolean masks to ``float32`` for Kornia compatibility.

    Args:
        targets: List of target dicts (one per image).  Each dict may optionally
            contain a ``"masks"`` key with an ``[N_i, H, W]`` boolean tensor. Dicts without the key are treated as
            having zero instances.
        device: Device on which to allocate the output tensor.
        n_max: Maximum instance count across the batch — must equal
            ``collate_boxes(targets, device)[1].shape[1]`` to keep box/mask indices in sync.
        image_height: Spatial height ``H`` of each mask (pixels).
        image_width: Spatial width ``W`` of each mask (pixels).

    Returns:
        Float32 tensor of shape ``[B, N_max, H, W]``, zero-padded where ``N_i < N_max``.  Boolean input masks are cast
        to ``float32`` (``True → 1.0``, ``False → 0.0``).

    Examples:
        >>> import torch
        >>> targets = [{"masks": torch.ones(2, 8, 8, dtype=torch.bool)}]
        >>> out = collate_masks(targets, torch.device("cpu"), n_max=2, image_height=8, image_width=8)
        >>> out.shape
        torch.Size([1, 2, 8, 8])
        >>> out.dtype
        torch.float32
    """
    batch_size = len(targets)
    masks_padded = torch.zeros(batch_size, n_max, image_height, image_width, dtype=torch.float32, device=device)
    for i, t in enumerate(targets):
        if "masks" not in t or n_max == 0:
            continue
        masks_i = t["masks"].to(dtype=torch.float32, device=device)  # [N_i, H, W]
        n = min(masks_i.shape[0], n_max)
        if n > 0:
            masks_padded[i, :n] = masks_i[:n]
    return masks_padded


def unpack_boxes(
    boxes_aug: Tensor,
    valid: Tensor,
    targets: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    masks_aug: Tensor | None = None,
) -> list[dict[str, Any]]:
    """Unpack augmented boxes (and optionally masks), clamp to image bounds, remove zero-area boxes.

    After Kornia augmentation the padded ``[B, N_max, 4]`` tensor is unpacked back into per-image target dicts.  Boxes
    are clamped to ``[0, W] x [0, H]`` and any that collapse to zero area are removed along with their corresponding
    ``labels``, ``area``, ``iscrowd``, and (if provided) ``masks`` entries.

    Args:
        boxes_aug: Augmented boxes tensor ``[B, N_max, 4]`` in xyxy format.
        valid: Boolean mask ``[B, N_max]`` from :func:`collate_boxes`.
        targets: Original target dicts; each dict is shallow-copied before
            modification — the input list itself is not mutated.
        image_height: Image height in pixels (for clamping).
        image_width: Image width in pixels (for clamping).
        masks_aug: Optional augmented masks tensor ``[B, N_max, H, W]``
            (float32) from Kornia.  When provided, masks are filtered by the same ``keep`` mask as boxes, thresholded at
            ``> 0.5`` to bool, and stored under ``"masks"`` in each output target dict.  When ``None``, any existing
            ``"masks"`` entry in the target dict is preserved unchanged.

    Returns:
        A new list of target dicts with updated ``boxes``, ``labels``, ``area``, ``iscrowd``, and (when *masks_aug* is
        given) ``masks`` entries.
    """
    if masks_aug is not None:
        assert masks_aug.shape[:2] == valid.shape, (
            f"masks_aug batch/n_max dims {tuple(masks_aug.shape[:2])} must match "
            f"valid shape {tuple(valid.shape)}; ensure collate_masks is called with "
            "n_max=valid.shape[1] from collate_boxes"
        )
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
        if masks_aug is not None:
            masks_i = masks_aug[i, :n_orig]  # [N_orig, H, W]
            t["masks"] = masks_i[keep] > _MASK_BINARIZE_THRESHOLD

        new_targets.append(t)

    return new_targets
