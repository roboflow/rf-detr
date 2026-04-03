# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""DOTA v1.0 dataset loader for oriented object detection."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.rotated_box_ops import corners_to_cxcywha

logger = get_logger()

DOTA_V1_CLASSES = (
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "ground-track-field",
    "harbor",
    "helicopter",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
)


def parse_dota_annotation(ann_path: Path) -> list[dict[str, Any]]:
    """Parse a DOTA annotation text file.

    Each line after the optional header has the format:
    ``x1 y1 x2 y2 x3 y3 x4 y4 category difficulty``

    Args:
        ann_path: Path to the annotation ``.txt`` file.

    Returns:
        List of annotation dicts with keys ``corners`` (8 floats)
        ``category`` (str), and ``difficulty`` (int).
    """
    annotations: list[dict[str, Any]] = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(parts[i]) for i in range(8)]
            except ValueError:
                continue
            category = parts[8]
            try:
                difficulty = int(parts[9]) if len(parts) > 9 else 0
            except ValueError:
                difficulty = 0
            annotations.append(
                {
                    "corners": coords,
                    "category": category,
                    "difficulty": difficulty,
                }
            )
    return annotations


def corners_list_to_tensor(corners: list[float]) -> torch.Tensor:
    """Convert flat 8-element corner list to ``(4, 2)`` tensor.

    Args:
        corners: Flat list ``[x1, y1, x2, y2, x3, y3, x4, y4]``.

    Returns:
        Tensor of shape ``(4, 2)``.
    """
    return torch.tensor(corners, dtype=torch.float32).reshape(4, 2)


class DotaDetection(Dataset):
    """DOTA v1.0 dataset for oriented object detection.

    Expects the standard DOTA directory layout::

        root/
          images/
            P0001.png
            P0002.png
            ...
          labelTxt/
            P0001.txt
            P0002.txt
            ...

    Each annotation file contains one object per line with 4-corner polygon
    coordinates, a category name, and a difficulty flag.

    Args:
        root: Path to the split directory (e.g. ``dota/train``).
        transforms: Transform pipeline applied to ``(image, target)`` pairs.
        class_names: Ordered tuple of class names. Defaults to DOTA v1.0 classes.
        include_difficult: If ``True``, include objects marked as difficult.
    """

    def __init__(
        self,
        root: str | Path,
        transforms: Compose | None = None,
        class_names: tuple[str, ...] = DOTA_V1_CLASSES,
        include_difficult: bool = False,
    ) -> None:
        self.root = Path(root)
        self._transforms = transforms
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.include_difficult = include_difficult

        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labelTxt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_files = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif")
        )
        logger.info(f"DOTA dataset loaded: {len(self.image_files)} images, {len(class_names)} classes")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        img_path = self.image_files[idx]
        ann_path = self.labels_dir / f"{img_path.stem}.txt"

        image = Image.open(img_path).convert("RGB")

        annotations = parse_dota_annotation(ann_path) if ann_path.exists() else []

        corners_list = []
        labels = []
        for ann in annotations:
            if not self.include_difficult and ann["difficulty"] == 1:
                continue
            cat = ann["category"]
            if cat not in self.class_to_idx:
                continue
            corners_list.append(corners_list_to_tensor(ann["corners"]))
            labels.append(self.class_to_idx[cat])

        if corners_list:
            all_corners = torch.stack(corners_list)
            boxes_obb = corners_to_cxcywha(all_corners)
        else:
            all_corners = torch.zeros((0, 4, 2), dtype=torch.float32)
            boxes_obb = torch.zeros((0, 5), dtype=torch.float32)

        target: dict[str, Any] = {
            "boxes_obb": boxes_obb,
            "corners": all_corners,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


def make_dota_transforms(
    image_set: str,
    resolution: int,
) -> Compose:
    """Build transform pipeline for DOTA dataset.

    Args:
        image_set: Split identifier — ``"train"`` or ``"val"``.
        resolution: Target square resolution in pixels.

    Returns:
        Composed transform pipeline.
    """
    to_image = ToImage()
    to_float = ToDtype(torch.float32, scale=True)
    normalize = DotaNormalize()

    if image_set == "train":
        resize_wrappers = AlbumentationsWrapper.from_config(
            [
                {"Resize": {"height": resolution, "width": resolution}},
            ]
        )
        aug_wrappers = AlbumentationsWrapper.from_config(
            [
                {"HorizontalFlip": {"p": 0.5}},
                {"VerticalFlip": {"p": 0.5}},
                {"RandomRotate90": {"p": 0.5}},
            ]
        )
        return Compose([*resize_wrappers, *aug_wrappers, to_image, to_float, normalize])

    resize_wrappers = AlbumentationsWrapper.from_config(
        [
            {"Resize": {"height": resolution, "width": resolution}},
        ]
    )
    return Compose([*resize_wrappers, to_image, to_float, normalize])


class DotaNormalize:
    """Normalize images and convert OBB corners to normalized cxcywha format.

    After geometric augmentations, recomputes ``boxes_obb`` from the
    (potentially transformed) ``corners`` keypoints, then normalizes
    spatial coordinates by image size.
    """

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        from torchvision.transforms import Normalize as _TVNormalize

        self._normalize = _TVNormalize(mean, std)

    def __call__(
        self, image: torch.Tensor, target: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        image = self._normalize(image)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]

        if "corners" in target and len(target["corners"]) > 0:
            corners = target["corners"]
            boxes_obb = corners_to_cxcywha(corners)
            scale = torch.tensor([w, h, w, h, 1.0], dtype=boxes_obb.dtype)
            boxes_obb = boxes_obb / scale
            target["boxes_obb"] = boxes_obb

        return image, target


def build_dota(image_set: str, args: Any, resolution: int) -> DotaDetection:
    """Build a DOTA dataset for the given split.

    Args:
        image_set: Split identifier — ``"train"`` or ``"val"``.
        args: Namespace with ``dataset_dir`` attribute.
        resolution: Target resolution in pixels.

    Returns:
        Configured DotaDetection dataset.
    """
    root = Path(args.dataset_dir) / image_set
    transforms = make_dota_transforms(image_set, resolution)
    return DotaDetection(root=root, transforms=transforms)
