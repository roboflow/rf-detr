# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""DOTAv1 format dataset loader for oriented bounding box detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class DotaDetection(VisionDataset):
    """Dataset for DOTAv1 format with oriented bounding boxes.

    DOTAv1 label format (one .txt file per image):
        x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

    Directory structure:
        root/{split}/images/{image_name}.png
        root/{split}/labels/{image_name}.txt

    Args:
        root: Root directory of the dataset.
        split: Dataset split (e.g., "train", "val", "test").
        class_names: List of class names for label-to-index mapping.
        transforms: Transform pipeline applied to (image, target) pairs.
        oriented: If True, include angle in the box representation (5-dim).
            If False, use axis-aligned bounding boxes (4-dim).
    """

    def __init__(
        self,
        root: str,
        split: str,
        class_names: List[str],
        transforms: Optional[Any] = None,
        oriented: bool = True,
    ) -> None:
        super().__init__(root)
        self._transforms = transforms
        self.oriented = oriented
        self.class_name_to_idx = {name: i for i, name in enumerate(class_names)}

        split_dir = Path(root) / split
        images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Collect all image paths with supported extensions
        self.image_paths: List[Path] = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        logger.info(
            "DotaDetection: %d images in %s split, %d classes, oriented=%s",
            len(self.image_paths),
            split,
            len(class_names),
            oriented,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_label_file(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Parse a DOTAv1 label file.

        Args:
            label_path: Path to the label .txt file.

        Returns:
            Tuple of (corners_list, class_indices) where corners_list contains
            [x1,y1,x2,y2,x3,y3,x4,y4] per object.
        """
        corners_list: List[List[float]] = []
        class_indices: List[int] = []

        if not label_path.exists():
            return corners_list, class_indices

        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                coords = [float(x) for x in parts[:8]]
                category = parts[8]
                if category not in self.class_name_to_idx:
                    logger.debug("Skipping unknown class '%s' in %s", category, label_path)
                    continue
                corners_list.append(coords)
                class_indices.append(self.class_name_to_idx[category])

        return corners_list, class_indices

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Find corresponding label file
        label_path = self.labels_dir / (img_path.stem + ".txt")
        corners_list, class_indices = self._parse_label_file(label_path)

        if len(corners_list) > 0:
            corners_tensor = torch.as_tensor(corners_list, dtype=torch.float32)
            labels_tensor = torch.as_tensor(class_indices, dtype=torch.int64)

            if self.oriented:
                # Store as xyxy for augmentation pipeline, then convert to cxcywh+angle in Normalize
                xs = corners_tensor[:, 0::2]
                ys = corners_tensor[:, 1::2]
                x_min = xs.min(dim=1).values
                y_min = ys.min(dim=1).values
                x_max = xs.max(dim=1).values
                y_max = ys.max(dim=1).values
                boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=1)

                # Store the original corners for keypoint-based augmentation
                target = {
                    "boxes": boxes_xyxy,
                    "labels": labels_tensor,
                    "obb_corners": corners_tensor,
                    "image_id": torch.tensor([idx]),
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
                    "orig_size": torch.as_tensor([h, w]),
                    "size": torch.as_tensor([h, w]),
                }
            else:
                # Use axis-aligned enclosing boxes
                xs = corners_tensor[:, 0::2]
                ys = corners_tensor[:, 1::2]
                x_min = xs.min(dim=1).values
                y_min = ys.min(dim=1).values
                x_max = xs.max(dim=1).values
                y_max = ys.max(dim=1).values
                boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
                target = {
                    "boxes": boxes,
                    "labels": labels_tensor,
                    "image_id": torch.tensor([idx]),
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
                    "orig_size": torch.as_tensor([h, w]),
                    "size": torch.as_tensor([h, w]),
                }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.as_tensor([h, w]),
                "size": torch.as_tensor([h, w]),
            }
            if self.oriented:
                target["obb_corners"] = torch.zeros((0, 8), dtype=torch.float32)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_dota(image_set: str, args: Any, resolution: int) -> DotaDetection:
    """Build a DOTAv1 format dataset.

    Args:
        image_set: Split identifier ("train", "val", "test").
        args: Namespace with dataset_dir, class_names, and other config.
        resolution: Target resolution for transforms.

    Returns:
        DotaDetection dataset instance.
    """
    root = Path(args.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"DOTA dataset path {root} does not exist")

    class_names = getattr(args, "class_names", None)
    if class_names is None:
        raise ValueError("class_names must be provided for DOTA datasets")

    oriented = getattr(args, "oriented", True)
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)

    if square_resize_div_64:
        transforms = make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=not do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
            aug_config=aug_config,
        )
    else:
        transforms = make_coco_transforms(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=not do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
            aug_config=aug_config,
        )

    # Map split names
    split = image_set
    if split == "val":
        # Try "val" first, fall back to "valid"
        if not (root / "val").exists() and (root / "valid").exists():
            split = "valid"

    logger.info("Building DOTA %s dataset at resolution %d", image_set, resolution)
    return DotaDetection(
        root=str(root),
        split=split,
        class_names=class_names,
        transforms=transforms,
        oriented=oriented,
    )
