# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Torchvision-native transforms for RF-DETR image/target pairs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple, TypeAlias, cast

import torch
import torch.nn.functional as torch_f
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional

_GLOBAL_TARGET_FIELDS = frozenset({"boxes", "labels", "orig_size", "size", "image_id"})
_ImageInput: TypeAlias = Image.Image | torch.Tensor
_Target: TypeAlias = Optional[Dict[str, Any]]
_TransformResult: TypeAlias = Tuple[_ImageInput, _Target]


def _image_size(image: Image.Image | torch.Tensor) -> tuple[int, int]:
    """Return image size as ``(height, width)``.

    Args:
        image: PIL image or tensor image.

    Returns:
        Image height and width.
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    return int(image.shape[-2]), int(image.shape[-1])


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize per-instance masks with nearest-neighbor interpolation.

    Args:
        masks: Boolean or numeric mask tensor of shape ``(N, H, W)``.
        size: Output ``(height, width)``.

    Returns:
        Boolean mask tensor of shape ``(N, size[0], size[1])``.
    """
    if masks.numel() == 0:
        return masks.new_zeros((masks.shape[0], size[0], size[1]), dtype=torch.bool)
    resized = torch_f.interpolate(masks[:, None].float(), size=size, mode="nearest")[:, 0]
    return resized.to(dtype=torch.bool)


def _filter_per_instance_fields(target: Dict[str, Any], keep: torch.Tensor, boxes: torch.Tensor) -> Dict[str, Any]:
    """Filter target fields whose first dimension matches the number of boxes.

    Args:
        target: Target dictionary before filtering.
        keep: Boolean tensor indicating surviving box rows.
        boxes: Already-filtered boxes.

    Returns:
        Target dictionary with per-instance fields filtered.
    """
    target_out: Dict[str, Any] = target.copy()
    num_boxes = int(target["boxes"].shape[0])
    keep_idx = keep.nonzero(as_tuple=False).flatten()
    target_out["boxes"] = boxes
    for key, value in target.items():
        if key in _GLOBAL_TARGET_FIELDS:
            continue
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == num_boxes:
            target_out[key] = value[keep_idx]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == num_boxes:
            target_out[key] = [value[int(i)] for i in keep_idx]

    if "labels" in target:
        target_out["labels"] = target["labels"][keep_idx]
    if "area" in target_out:
        target_out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return target_out


def _mark_invisible_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Clear keypoints that are invisible or outside the image.

    Args:
        keypoints: Keypoint tensor of shape ``(N, K, 3)``.
        height: Image height.
        width: Image width.

    Returns:
        Updated keypoint tensor.
    """
    if keypoints.numel() == 0:
        return keypoints
    visible = keypoints[..., 2] > 0
    inside = (
        (keypoints[..., 0] >= 0) & (keypoints[..., 0] < width) & (keypoints[..., 1] >= 0) & (keypoints[..., 1] < height)
    )
    invalid = ~(visible & inside)
    keypoints = keypoints.clone()
    keypoints[invalid] = 0.0
    return keypoints


def _update_size(target: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
    """Update the current image size in a target dictionary.

    Args:
        target: Target dictionary.
        height: Current image height.
        width: Current image width.

    Returns:
        Target dictionary with updated ``size``.
    """
    target_out = target.copy()
    target_out["size"] = torch.as_tensor([int(height), int(width)])
    return target_out


class RandomSelect:
    """Select one of two image/target transforms with probability ``p``.

    Args:
        transform1: Transform used when the sampled value is below ``p``.
        transform2: Transform used otherwise.
        p: Probability of selecting ``transform1``.
    """

    def __init__(self, transform1: Any, transform2: Any, p: float = 0.5) -> None:
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply one of the configured transforms.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Transformed image and target.
        """
        if torch.rand(()) < self.p:
            return cast(_TransformResult, self.transform1(image, target))
        return cast(_TransformResult, self.transform2(image, target))


class RandomChoice:
    """Select one transform uniformly from a sequence.

    Args:
        transforms: Candidate transforms.
    """

    def __init__(self, transforms: Sequence[Any]) -> None:
        if not transforms:
            raise ValueError("transforms must contain at least one transform")
        self.transforms = list(transforms)

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply one randomly selected transform.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Transformed image and target.
        """
        index = int(torch.randint(len(self.transforms), ()).item()) if len(self.transforms) > 1 else 0
        return cast(_TransformResult, self.transforms[index](image, target))


class Compose:
    """Compose RF-DETR image/target transforms.

    Args:
        transforms: Sequence of transforms accepting ``(image, target)``.
    """

    def __init__(self, transforms: Sequence[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply transforms in order.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Transformed image and target.
        """
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomResize:
    """Resize so the shortest side matches a sampled size, with an optional long-side cap.

    Args:
        sizes: Candidate shortest-side sizes.
        max_size: Optional maximum longest side after resizing.
    """

    def __init__(self, sizes: Sequence[int], max_size: int | None = None) -> None:
        if not sizes:
            raise ValueError("sizes must contain at least one value")
        self.sizes = [int(size) for size in sizes]
        self.max_size = max_size

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Resize the image and spatial target fields.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Resized image and target.
        """
        index = int(torch.randint(len(self.sizes), ()).item()) if len(self.sizes) > 1 else 0
        size = self.sizes[index]
        height, width = _image_size(image)
        new_height, new_width = self._get_size(height, width, size, self.max_size)
        return Resize((new_height, new_width))(image, target)

    @staticmethod
    def _get_size(height: int, width: int, size: int, max_size: int | None) -> tuple[int, int]:
        """Compute output size preserving aspect ratio.

        Args:
            height: Input height.
            width: Input width.
            size: Requested shortest side.
            max_size: Optional maximum longest side.

        Returns:
            Output ``(height, width)``.
        """
        min_original = float(min(height, width))
        max_original = float(max(height, width))
        if max_size is not None and max_original / min_original * size > max_size:
            size = int(round(max_size * min_original / max_original))

        if (width <= height and width == size) or (height <= width and height == size):
            return height, width

        if width < height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        return new_height, new_width


class Resize:
    """Resize to a fixed ``(height, width)``.

    Args:
        size: Output ``(height, width)``.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = (int(size[0]), int(size[1]))

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Resize the image and spatial target fields.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Resized image and target.
        """
        old_height, old_width = _image_size(image)
        new_height, new_width = self.size
        image = functional.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if target is None:
            return image, None

        target_out = target.copy()
        ratio_width = new_width / old_width
        ratio_height = new_height / old_height
        if "boxes" in target_out:
            scale = target_out["boxes"].new_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target_out["boxes"] = target_out["boxes"] * scale
        if "masks" in target_out:
            target_out["masks"] = _resize_masks(target_out["masks"], (new_height, new_width))
        if "keypoints" in target_out:
            keypoints = target_out["keypoints"].clone()
            keypoints[..., 0] = keypoints[..., 0] * ratio_width
            keypoints[..., 1] = keypoints[..., 1] * ratio_height
            keypoints = _mark_invisible_keypoints(keypoints, new_height, new_width)
            target_out["keypoints"] = keypoints
        if "area" in target_out:
            target_out["area"] = target_out["area"] * (ratio_width * ratio_height)
        return image, _update_size(target_out, new_height, new_width)


class RandomSizedCrop:
    """Crop a random square-ish region and resize it to a fixed output size.

    Args:
        min_max_height: Inclusive candidate range for the crop height.
        size: Output ``(height, width)`` after crop resizing.
    """

    def __init__(self, min_max_height: tuple[int, int], size: tuple[int, int]) -> None:
        self.min_max_height = (int(min_max_height[0]), int(min_max_height[1]))
        self.size = (int(size[0]), int(size[1]))

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Crop and resize the image and spatial target fields.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Cropped and resized image and target.
        """
        height, width = _image_size(image)
        min_crop, max_crop = self.min_max_height
        max_crop = min(max_crop, height, width)
        min_crop = min(min_crop, max_crop)
        crop_size = int(torch.randint(min_crop, max_crop + 1, ()).item()) if max_crop > min_crop else max_crop
        crop_height = min(crop_size, height)
        crop_width = min(crop_size, width)
        top = int(torch.randint(0, height - crop_height + 1, ()).item()) if height > crop_height else 0
        left = int(torch.randint(0, width - crop_width + 1, ()).item()) if width > crop_width else 0
        image, target = crop(image, target, top, left, crop_height, crop_width)
        return Resize(self.size)(image, target)


class RandomHorizontalFlip:
    """Horizontally flip an image/target pair with probability ``p``.

    Args:
        p: Flip probability.
        keypoint_flip_pairs: Flat list of keypoint indices to swap after flipping.
    """

    def __init__(self, p: float = 0.5, keypoint_flip_pairs: Sequence[int] | None = None) -> None:
        self.p = p
        self.keypoint_flip_pairs = list(keypoint_flip_pairs or [])

    def __call__(
        self, image: Image.Image | torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
        """Randomly flip the image and target.

        Args:
            image: Input image.
            target: Optional target dictionary.

        Returns:
            Possibly flipped image and target.
        """
        if torch.rand(()) >= self.p:
            return image, target

        height, width = _image_size(image)
        image = functional.horizontal_flip(image)
        if target is None:
            return image, None

        target_out = target.copy()
        if "boxes" in target_out:
            boxes = target_out["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target_out["boxes"] = boxes
        if "masks" in target_out:
            target_out["masks"] = functional.horizontal_flip(target_out["masks"])
        if "keypoints" in target_out:
            keypoints = target_out["keypoints"].clone()
            visible = keypoints[..., 2] > 0
            keypoints[..., 0] = (width - 1) - keypoints[..., 0]
            invisible = (~visible).unsqueeze(-1)  # (N, K, 1) for masked_fill on (N, K, 3)
            keypoints[..., :2] = keypoints[..., :2].masked_fill(invisible, 0.0)
            for i in range(0, len(self.keypoint_flip_pairs) - 1, 2):
                ai, bi = self.keypoint_flip_pairs[i], self.keypoint_flip_pairs[i + 1]
                if ai < keypoints.shape[1] and bi < keypoints.shape[1]:
                    tmp = keypoints[:, ai, :].clone()
                    keypoints[:, ai, :] = keypoints[:, bi, :]
                    keypoints[:, bi, :] = tmp
            target_out["keypoints"] = _mark_invisible_keypoints(keypoints, height, width)
        return image, target_out


def crop(
    image: Image.Image | torch.Tensor,
    target: Optional[Dict[str, Any]],
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[Image.Image | torch.Tensor, Optional[Dict[str, Any]]]:
    """Crop an image and associated spatial target fields.

    Args:
        image: Input image.
        target: Optional target dictionary.
        top: Crop top coordinate.
        left: Crop left coordinate.
        height: Crop height.
        width: Crop width.

    Returns:
        Cropped image and target.
    """
    image = functional.crop(image, top=top, left=left, height=height, width=width)
    if target is None:
        return image, None

    target_out = target.copy()
    if "boxes" in target_out:
        boxes = target_out["boxes"].clone()
        boxes[:, 0::2] -= left
        boxes[:, 1::2] -= top
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        target_out = _filter_per_instance_fields(target_out, keep, boxes)
    if "masks" in target_out:
        target_out["masks"] = functional.crop(target_out["masks"], top=top, left=left, height=height, width=width)
    if "keypoints" in target_out:
        keypoints = target_out["keypoints"].clone()
        keypoints[..., 0] -= left
        keypoints[..., 1] -= top
        target_out["keypoints"] = _mark_invisible_keypoints(keypoints, height, width)
    return image, _update_size(target_out, height, width)
