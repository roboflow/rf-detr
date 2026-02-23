# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from numbers import Number

import albumentations as A
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from rfdetr.util.box_ops import box_xyxy_to_cxcywh
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import interpolate

logger = get_logger()


def crop(
    image: PIL.Image.Image, target: Dict[str, Any], region: Tuple[int, int, int, int]
) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(
    image: PIL.Image.Image,
    target: Optional[Dict[str, Any]],
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(
        image_size: Tuple[int, int], size: int, max_size: Optional[int] = None
    ) -> Tuple[int, int]:
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(
        image_size: Tuple[int, int], size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = None
    ) -> Tuple[int, int]:
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(
    image: PIL.Image.Image, target: Optional[Dict[str, Any]], padding: Tuple[int, int]
) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        self.size = size

    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes: List[int], max_size: Optional[int] = None) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(
        self, img: PIL.Image.Image, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class SquareResize(object):
    def __init__(self, sizes: List[int]) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(
        self, img: PIL.Image.Image, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        rescaled_img = F.resize(img, (size, size))
        w, h = rescaled_img.size
        if target is None:
            return rescaled_img, None
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target["masks"] = interpolate(target["masks"][:, None].float(), (h, w), mode="nearest")[:, 0] > 0.5

        return rescaled_img, target


class RandomPad(object):
    def __init__(self, max_pad: int) -> None:
        self.max_pad = max_pad

    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class PILtoNdArray(object):
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return np.asarray(img), target


class NdArraytoPIL(object):
    def __call__(self, img: np.ndarray, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        return F.to_pil_image(img.astype("uint8")), target


class Pad(object):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        size_divisor: int = 32,
        pad_mode: int = 0,
        offsets: Optional[List[int]] = None,
        fill_value: Tuple[float, float, float] = (127.5, 127.5, 127.5),
    ) -> None:
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size: image target size, if None, pad to multiple of size_divisor, default None
            size_divisor: size divisor, default 32
            pad_mode: pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets: [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value: rgb value of pad area, default (127.5, 127.5, 127.5)
        """

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size))
            )

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [-1, 0, 1, 2], "currently only supports four modes [-1, 0, 1, 2]"
        if pad_mode == -1:
            assert offsets, "if pad_mode is -1, offsets should not be None"

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_bbox(self, bbox: np.ndarray, offsets: List[int]) -> np.ndarray:
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_image(self, image: np.ndarray, offsets: List[int], im_size: List[int], size: List[int]) -> np.ndarray:
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y : y + im_h, x : x + im_w, :] = image.astype(np.float32)
        return canvas

    def __call__(self, im: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert im_h <= h and im_w <= w, "(h, w) of target size should be greater than (im_h, im_w)"
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            return im.astype(np.float32), target

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        im = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            target["size"] = torch.tensor([h, w])
            return im, target
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = np.asarray(target["boxes"])
            target["boxes"] = torch.from_numpy(self.apply_bbox(boxes, offsets))
            target["size"] = torch.tensor([h, w])

        return im, target


class RandomExpand(object):
    """Random expand the canvas.
    Args:
        ratio: maximum expansion ratio.
        prob: probability to expand.
        fill_value: color value used to fill the canvas. in RGB order.
    """

    def __init__(
        self,
        ratio: float = 4.0,
        prob: float = 0.5,
        fill_value: Union[float, List[float], Tuple[float, float, float]] = (127.5, 127.5, 127.5),
    ) -> None:
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if np.random.uniform(0.0, 1.0) < self.prob:
            return img, target

        height, width = img.shape[:2]
        ratio = np.random.uniform(1.0, self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return img, target
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad_op = Pad(size, pad_mode=-1, offsets=offsets, fill_value=self.fill_value)

        return pad_op(img, target)


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1: Any, transforms2: Any, p: float = 0.5) -> None:
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img: Any, target: Any) -> Tuple[Any, Any]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(
        self, img: Union[PIL.Image.Image, np.ndarray], target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return F.to_tensor(img), target


class RandomErasing(object):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# Albumentations wrapper for RF-DETR

# Geometric transforms that affect bounding boxes
# These transforms modify spatial coordinates, so bounding boxes must be transformed accordingly.
# For custom geometric transforms, add the class name to this set.
GEOMETRIC_TRANSFORMS = {
    # Flips and transpositions
    "HorizontalFlip",
    "VerticalFlip",
    "Flip",
    "Transpose",
    "D4",
    # Rotations and affine transforms
    "Rotate",
    "RandomRotate90",
    "Affine",
    "ShiftScaleRotate",
    "SafeRotate",
    # Crops
    "RandomCrop",
    "RandomSizedCrop",
    "CenterCrop",
    "Crop",
    "CropNonEmptyMaskIfExists",
    "RandomCropNearBBox",
    "RandomCropFromBorders",
    "RandomSizedBBoxSafeCrop",
    "BBoxSafeRandomCrop",
    "AtLeastOneBBoxRandomCrop",
    "RandomResizedCrop",
    "CropAndPad",
    # Perspective and distortions
    "Perspective",
    "ElasticTransform",
    "GridDistortion",
    "GridElasticDeform",
    "OpticalDistortion",
    "PiecewiseAffine",
    "ThinPlateSpline",
    "RandomGridShuffle",
    # Resize operations
    "Resize",
    "SmallestMaxSize",
    "LongestMaxSize",
    "RandomScale",
    "Downscale",
    # Padding and symmetry
    "PadIfNeeded",
    "Pad",
    "SquareSymmetry",
}


class AlbumentationsWrapper:
    """Wrapper to apply Albumentations transforms to (image, target) tuples.

    This wrapper integrates Albumentations transforms with RF-DETR's data pipeline,
    automatically handling bounding box and segmentation mask transformations for
    geometric augmentations while preserving the (image, target) tuple format.

    The wrapper automatically detects transform types:
    - **Geometric transforms** (flips, rotations, crops): Bounding boxes and instance
      masks are transformed along with the image to maintain correct object localization.
    - **Pixel-level transforms** (blur, color adjustments, noise): Bounding boxes and
      masks remain unchanged as only pixel values are modified.

    Detection is based on the transform's class name matching the GEOMETRIC_TRANSFORMS set.
    For geometric transforms, bbox_params are automatically configured to handle coordinate
    transformations, clip boxes to image boundaries, and remove invalid boxes.

    Args:
        transform: Albumentations transform to apply (e.g., A.HorizontalFlip, A.GaussianBlur).

    Examples:
        >>> import albumentations as A
        >>> # Geometric transform - automatically transforms boxes
        >>> wrapper = AlbumentationsWrapper(A.HorizontalFlip(p=1.0))
        >>> image = Image.open("image.jpg")
        >>> target = {"boxes": torch.tensor([[10, 20, 100, 200]]), "labels": torch.tensor([1])}
        >>> aug_image, aug_target = wrapper(image, target)

        >>> # Pixel-level transform - automatically preserves boxes
        >>> wrapper = AlbumentationsWrapper(A.GaussianBlur(p=1.0))
        >>> aug_image, aug_target = wrapper(image, target)

    Note:
        For custom geometric transforms, add the transform class name to the
        GEOMETRIC_TRANSFORMS set at module level.
    """

    def __init__(self, transform: A.BasicTransform) -> None:
        # Auto-detect if transform is geometric based on its class name
        transform_name = transform.__class__.__name__
        self._is_geometric = transform_name in GEOMETRIC_TRANSFORMS

        if self._is_geometric:
            # Wrap geometric transform with bbox handling capabilities
            # bbox_params configure how Albumentations should transform bounding boxes:
            self.transform = A.Compose(
                [transform],
                bbox_params=A.BboxParams(
                    format="pascal_voc",  # Boxes are in (x1, y1, x2, y2) format
                    label_fields=["category_ids", "idxs"],  # Track labels and indices for per-instance field sync
                    min_visibility=0.0,  # Remove boxes with zero visibility/area after transformation
                    clip=True,  # Clip box coordinates to image boundaries after transformation
                ),
            )
        else:
            # Wrap non-geometric transform without bbox handling
            # Simpler composition since boxes don't need transformation
            self.transform = A.Compose([transform])

    def __repr__(self) -> str:
        """Return a readable string representation of the wrapper.

        Returns:
            Representation including the wrapped transform and type.
        """
        transform = None
        if isinstance(self.transform, A.Compose):
            for candidate in self.transform.transforms:
                if isinstance(candidate, A.BasicTransform):
                    transform = candidate
                    break
        elif isinstance(self.transform, A.BasicTransform):
            transform = self.transform

        if transform is None:
            return object.__repr__(self)

        transform_type = "geometric" if self._is_geometric else "pixel-level"
        return f"{self.__class__.__name__}(transform={transform}, type={transform_type})"

    @staticmethod
    def _boxes_to_numpy(boxes: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert boxes to numpy array and validate shape.

        >>> import torch
        >>> boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        >>> AlbumentationsWrapper._boxes_to_numpy(boxes).shape
        (1, 4)
        """
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.array(boxes)
        if len(boxes_np.shape) != 2 or boxes_np.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {boxes_np.shape}")
        return boxes_np

    @staticmethod
    def _clear_per_instance_fields(target: Dict[str, Any], num_boxes: int) -> Dict[str, Any]:
        """Clear all per-instance fields when no boxes remain.

        >>> import torch
        >>> target = {"area": torch.tensor([100, 200]), "iscrowd": torch.tensor([0, 1])}
        >>> cleared = AlbumentationsWrapper._clear_per_instance_fields(target, 2)
        >>> cleared["area"].shape
        torch.Size([0])
        """
        # Fields that are global properties, not per-instance
        global_fields = {"boxes", "labels", "orig_size", "size", "image_id"}

        result = {}
        for key, value in target.items():
            if key in global_fields:
                continue
            if torch.is_tensor(value):
                if value.ndim >= 1 and value.shape[0] == num_boxes:
                    result[key] = value.new_empty((0, *value.shape[1:]))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == num_boxes:
                    result[key] = []
        return result

    @staticmethod
    def _filter_per_instance_fields(target: Dict[str, Any], num_boxes: int, kept_idxs: List[int]) -> Dict[str, Any]:
        """Filter per-instance fields to match kept box indices.

        >>> import torch
        >>> target = {"area": torch.tensor([100, 200, 300]), "iscrowd": torch.tensor([0, 0, 1])}
        >>> filtered = AlbumentationsWrapper._filter_per_instance_fields(target, 3, [0, 2])
        >>> filtered["area"].tolist()
        [100, 300]
        """
        # Fields that are global properties, not per-instance
        global_fields = {"boxes", "labels", "orig_size", "size", "image_id"}

        result = {}
        kept_idxs_tensor = torch.as_tensor(kept_idxs, dtype=torch.long)
        for key, value in target.items():
            if key in global_fields:
                continue
            if torch.is_tensor(value):
                if value.ndim >= 1 and value.shape[0] == num_boxes:
                    result[key] = value[kept_idxs_tensor]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == num_boxes:
                    result[key] = [value[i] for i in kept_idxs]
        return result

    def _apply_geometric_transform(
        self, image_np: np.ndarray, target: Dict[str, Any], labels: List[int]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Apply geometric transform to image with boxes and optionally masks.

        Converts data to Albumentations format, applies the transform, and converts
        back to RF-DETR format. Handles box removal and per-instance field filtering.

        Args:
            image_np: Numpy array of image in HWC format.
            target: Target dictionary with 'boxes' and optionally 'masks'.
            labels: List of category labels.

        Returns:
            Tuple of (transformed PIL Image, transformed target dict).

        >>> import albumentations as A
        >>> import torch
        >>> wrapper = AlbumentationsWrapper(A.HorizontalFlip(p=1.0))
        >>> img = np.ones((100, 100, 3), dtype=np.uint8)
        >>> tgt = {"boxes": torch.tensor([[10, 20, 30, 40]]), "labels": torch.tensor([1])}
        >>> img_out, tgt_out = wrapper._apply_geometric_transform(img, tgt, [1])
        >>> tgt_out["boxes"].shape
        torch.Size([1, 4])
        """
        boxes_np = self._boxes_to_numpy(target["boxes"])
        num_boxes = boxes_np.shape[0]
        # Track indices to keep per-instance fields synchronized
        idxs = list(range(num_boxes))
        masks_list = None
        if "masks" in target:
            masks = target["masks"]
            masks_np = masks.cpu().numpy() if torch.is_tensor(masks) else np.array(masks)
            if masks_np.ndim != 3:
                raise ValueError(f"masks must have shape (N, H, W), got {masks_np.shape}")
            masks_np = masks_np.astype(np.uint8, copy=False)
            masks_list = [mask for mask in masks_np]
        # Apply transform
        transform_kwargs = {"image": image_np, "bboxes": boxes_np, "category_ids": labels, "idxs": idxs}
        if masks_list is not None and len(masks_list) > 0:
            transform_kwargs["masks"] = masks_list
        augmented = self.transform(**transform_kwargs)
        target_out: Dict[str, Any] = target.copy()
        bboxes_aug = augmented["bboxes"]
        kept_idxs = augmented.get("idxs", idxs)
        # Update target with transformed boxes and labels
        if len(bboxes_aug) == 0:
            target_out["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target_out["labels"] = torch.zeros((0,), dtype=torch.long)
            target_out.update(self._clear_per_instance_fields(target, num_boxes))
            # Override masks after _clear_per_instance_fields to ensure bool dtype.
            if "masks" in target:
                img_height, img_width = image_np.shape[:2]
                target_out["masks"] = torch.zeros((0, img_height, img_width), dtype=torch.bool)
        else:
            target_out["boxes"] = torch.as_tensor(bboxes_aug, dtype=torch.float32).reshape(-1, 4)
            target_out["labels"] = torch.tensor(augmented["category_ids"], dtype=torch.long)
            target_out.update(self._filter_per_instance_fields(target, num_boxes, kept_idxs))
        image_out = Image.fromarray(augmented["image"])
        if masks_list is not None and "masks" in augmented:
            height, width = augmented["image"].shape[:2]
            masks_aug = augmented["masks"]
            masks_aug = [masks_aug[int(i)] for i in kept_idxs]
            if len(masks_aug) == 0:
                target_out["masks"] = torch.zeros((0, height, width), dtype=torch.bool)
            else:
                target_out["masks"] = torch.as_tensor(np.stack(masks_aug), dtype=torch.bool)
        return image_out, target_out

    def __call__(self, image: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        """Apply the Albumentations transform to image and target.

        This method handles the data format conversion between RF-DETR and Albumentations:
        1. Converts PIL Image to numpy array (required by Albumentations)
        2. Converts PyTorch tensors to numpy/lists (required by Albumentations)
        3. Applies the transform
        4. Converts results back to PIL Image and PyTorch tensors

        For geometric transforms with bounding boxes, this method also:
        - Validates box shapes and coordinates
        - Handles boxes that may be removed by the transform (e.g., cropped out)
        - Ensures labels stay synchronized with their corresponding boxes
        - Transforms masks when present to stay aligned with the image

        Args:
            image: Input PIL Image in RGB format.
            target: Target dictionary containing:
                - 'labels': PyTorch tensor of shape (N,) with class labels
                - 'boxes' (optional): PyTorch tensor of shape (N, 4) in (x1, y1, x2, y2) format
                - 'masks' (optional): PyTorch tensor of shape (N, H, W) with instance segmentation masks.
                  For geometric transforms, masks are transformed alongside boxes to maintain alignment.
                  Requires 'boxes' to be present; a warning is logged if masks exist without boxes.

        Returns:
            Tuple of (transformed_image, transformed_target):
                - transformed_image: PIL Image after augmentation
                - transformed_target: Dictionary with augmented boxes and labels

        Raises:
            TypeError: If target is not a dictionary.
            KeyError: If target doesn't contain 'labels' key.
            ValueError: If boxes don't have shape (N, 4).

        Examples:
            >>> wrapper = AlbumentationsWrapper(A.HorizontalFlip(p=1.0))
            >>> image = Image.new('RGB', (100, 100))
            >>> target = {"boxes": torch.tensor([[10, 20, 90, 80]]), "labels": torch.tensor([1])}
            >>> aug_image, aug_target = wrapper(image, target)
        """
        # === Input Validation ===
        if not isinstance(target, dict):
            raise TypeError(f"target must be a dictionary, got {type(target)}")
        if "labels" not in target:
            raise KeyError("target must contain 'labels' key")

        # === Format Conversion: PyTorch/PIL → Albumentations ===
        # Convert PIL Image to numpy array (HWC format expected by Albumentations)
        image_np = np.array(image)

        # Convert labels tensor to Python list (required by Albumentations category_ids)
        labels = target["labels"].cpu().tolist() if torch.is_tensor(target["labels"]) else list(target["labels"])

        # === Apply Transform ===
        if self._is_geometric and "masks" in target and "boxes" not in target:
            logger.warning(
                "AlbumentationsWrapper: geometric transform requested with 'masks' but without 'boxes'. "
                "Masks will not be geometrically transformed because bounding boxes are missing."
            )
        if self._is_geometric and "boxes" in target:
            # Geometric path: transform image and boxes together
            image_out, target_out = self._apply_geometric_transform(image_np, target, labels)
        else:
            # Non-geometric path: transform image only
            augmented = self.transform(image=image_np)
            image_out = Image.fromarray(augmented["image"])
            target_out = target.copy()

        # Ensure 'size' (if present) matches the transformed image size (h, w)
        if "size" in target_out:
            # PIL.Image.size is (width, height); many detectors expect (height, width)
            width, height = image_out.size
            target_out["size"] = torch.as_tensor([height, width], dtype=torch.int64)
        return image_out, target_out

    @staticmethod
    def from_config(config_dict: Dict[str, Dict[str, Any]]) -> List["AlbumentationsWrapper"]:
        """Build list of AlbumentationsWrapper instances from configuration dictionary.

        Convenient way to create multiple augmentation wrappers from a config dictionary.
        Each transform is automatically wrapped with appropriate bbox handling based on
        whether it's geometric or pixel-level.

        Args:
            config_dict: Dictionary mapping transform names to parameters.
                Keys: Albumentations transform class names (e.g., "HorizontalFlip").
                Values: Parameter dictionaries to pass to the transform.

        Returns:
            List of AlbumentationsWrapper instances in the same order as config dict.

        Raises:
            TypeError: If config_dict is not a dictionary.

        Examples:
            >>> config = {
            ...     "HorizontalFlip": {"p": 0.5},
            ...     "Rotate": {"limit": 45, "p": 0.3},
            ...     "GaussianBlur": {"p": 0.2}
            ... }
            >>> transforms = AlbumentationsWrapper.from_config(config)
            >>> [t.transform.transforms[0].__class__.__name__ for t in transforms]
            ['HorizontalFlip', 'Rotate', 'GaussianBlur']

        Note:
            Invalid transforms or invalid parameters are logged and skipped gracefully.
        """
        if not isinstance(config_dict, dict):
            raise TypeError(f"config_dict must be a dictionary, got {type(config_dict)}")

        if not config_dict:
            logger.warning("Empty augmentation config provided, no transforms will be applied")
            return []

        transforms = []
        for aug_name, params in config_dict.items():
            if not isinstance(params, dict):
                logger.warning(f"Skipping {aug_name}: parameters must be a dictionary, got {type(params)}")
                continue

            base_aug = getattr(A, aug_name, None)
            if base_aug is None:
                logger.warning(f"Unknown Albumentations transform: {aug_name}. Skipping.")
                continue

            try:
                # AlbumentationsWrapper will auto-detect if transform is geometric
                # based on the transform class name matching GEOMETRIC_TRANSFORMS
                transforms.append(AlbumentationsWrapper(base_aug(**params)))
            except Exception as e:
                logger.warning(f"Failed to initialize {aug_name} with params {params}: {e}. Skipping.")
                continue

        logger.info(f"Built {len(transforms)} Albumentations transforms from config")
        return transforms


class ComposeAugmentations:
    """Compose multiple augmentation transforms into a single callable.

    This class sequentially applies a list of transforms to an (image, target) pair,
    following the same interface as torchvision.transforms.Compose but supporting
    the RF-DETR target dictionary format.

    Args:
        transforms: List of transforms to apply sequentially.

    Examples:
        >>> from rfdetr.augmentation_config import AUG_CONFIG
        >>> aug_transforms = AlbumentationsWrapper.from_config(AUG_CONFIG)
        >>> composed = ComposeAugmentations(aug_transforms)
        >>> image, target = composed(image, target)
    """

    def __init__(self, transforms: List[Any]) -> None:
        if not isinstance(transforms, list):
            raise TypeError(f"transforms must be a list, got {type(transforms)}")
        self.transforms = transforms

    def __call__(self, image: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        """Apply all transforms sequentially.

        Args:
            image: Input PIL Image.
            target: Target dictionary with labels and optionally boxes.

        Returns:
            Tuple of transformed image and target.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        """Return a readable representation of the composed augmentations."""
        format_string = f"{self.__class__.__name__}(\n"
        for t in self.transforms:
            format_string += f"\t{t!r}\n"
        format_string += ")"
        return format_string
