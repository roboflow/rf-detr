# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""DOTA dataset loader for oriented object detection."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

try:
    import albumentations as A  # noqa: N812
except ImportError:
    A = None

from rfdetr.datasets.yolo import YOLO_IMAGE_EXTENSIONS
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
        List of annotation dicts with keys ``corners``, ``category`` and ``difficulty``.
    """
    annotations: list[dict[str, Any]] = []
    # DOTA writes these two metadata lines at the top of every file.
    header_prefixes = ("imagesource:", "gapsize:")
    with ann_path.open(encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith(header_prefixes):
                continue
            parts = line.split()
            if len(parts) < 9:
                logger.warning(
                    f"Skipping malformed line {line_num} in {ann_path}: expected >= 9 fields, got {len(parts)}."
                )
                continue
            try:
                coords = [float(parts[i]) for i in range(8)]
            except ValueError:
                logger.warning(f"Skipping line {line_num} in {ann_path}: coordinates are not numeric.")
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


class DotaDetection(Dataset[Any]):
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
        class_names: Ordered tuple of class names. Defaults to the 15 DOTA v1.0 classes.
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
        # Categories outside class_names are skipped; track them so the warning fires
        # once per dataset rather than once per annotation.
        self._unknown_categories: set[str] = set()

        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labelTxt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_files = sorted(
            p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTENSIONS
        )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        logger.info(f"DOTA dataset loaded: {len(self.image_files)} images, {len(class_names)} classes")

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Load one image and its oriented-box target.

        Args:
            idx: Index into the sorted image list.

        Returns:
            Tuple of the (optionally transformed) image and its target dict.
        """
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
                if cat not in self._unknown_categories:
                    self._unknown_categories.add(cat)
                    logger.warning(
                        f"Ignoring unknown DOTA category {cat!r}: it is not among the "
                        f"{len(self.class_names)} configured classes."
                    )
                continue
            corners_list.append(corners_list_to_tensor(ann["corners"]))
            labels.append(self.class_to_idx[cat])

        if corners_list:
            all_corners = torch.stack(corners_list)
            boxes_obb = corners_to_cxcywha(all_corners)
        else:
            all_corners = torch.zeros((0, 4, 2), dtype=torch.float32)
            boxes_obb = torch.zeros((0, 5), dtype=torch.float32)

        w, h = image.size
        target: dict[str, Any] = {
            "boxes_obb": boxes_obb,
            "boxes": boxes_obb[..., :4],  # [cx, cy, w, h] alias for COCO eval callback
            "corners": all_corners,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


class OBBGeometricTransform:
    """Apply an Albumentations geometric transform to an image and its oriented-box corners.

    Treats the 4 corners of each oriented box as individual keypoints so that
    geometric augmentations (flip, rotate, crop) correctly update the box
    geometry.  After the transform the ``corners`` tensor in the target is
    updated in place; ``DotaNormalize`` then recomputes ``boxes_obb`` from
    the updated corners.

    Args:
        transform: An Albumentations ``BasicTransform``, or a list of them to apply
            as a single pass. Passing the whole geometric chain at once avoids a
            PIL/numpy round-trip and a keypoint-processing pass per operation.
    """

    def __init__(self, transform: "A.BasicTransform | list[A.BasicTransform]") -> None:
        if A is None:
            raise ImportError("albumentations is required for OBBGeometricTransform")
        self._pipeline = A.Compose(
            list(transform) if isinstance(transform, list) else [transform],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["kp_instance_ids", "kp_point_ids"],
                remove_invisible=False,
            ),
        )

    def __call__(self, image: Image.Image, target: dict[str, Any] | None) -> tuple[Image.Image, dict[str, Any] | None]:
        """Apply the geometric transform to image and OBB corners.

        Args:
            image: Input PIL image.
            target: Target dict with ``corners`` tensor of shape ``(N, 4, 2)``
                in pixel coordinates, plus ``labels``.

        Returns:
            Augmented ``(image, target)`` pair with ``corners`` updated.
        """
        image_np = np.array(image)

        if target is None or "corners" not in target or len(target["corners"]) == 0:
            augmented = self._pipeline(image=image_np, keypoints=[], kp_instance_ids=[], kp_point_ids=[])
            return Image.fromarray(augmented["image"]), target

        corners: torch.Tensor = target["corners"]
        n_boxes = corners.shape[0]

        # Corners are passed through unclamped.  The four corners of a rotated box are
        # not independent, so clipping them individually turns the box into a different
        # quadrilateral: a box crossing an image edge loses ~30% of its width and gains
        # a double-digit angle error.  The pipeline is built with remove_invisible=False,
        # so albumentations preserves out-of-bounds keypoints without help.
        kp_xy = []
        inst_ids = []
        point_ids = []
        corners_np = corners.cpu().numpy()
        for i in range(n_boxes):
            for j in range(4):
                kp_xy.append((float(corners_np[i, j, 0]), float(corners_np[i, j, 1])))
                inst_ids.append(i)
                point_ids.append(j)

        augmented = self._pipeline(
            image=image_np,
            keypoints=kp_xy,
            kp_instance_ids=inst_ids,
            kp_point_ids=point_ids,
        )

        new_image_np = augmented["image"]

        if len(augmented["keypoints"]) != len(kp_xy):
            # out_corners is seeded with pre-transform coordinates, so a dropped keypoint
            # would leave one corner in the original frame while its siblings move to the
            # augmented one — a single box spanning two coordinate spaces, silently.
            raise ValueError(
                f"Albumentations returned {len(augmented['keypoints'])} keypoints for {len(kp_xy)} inputs. "
                "OBB corners require a transform that preserves every keypoint."
            )

        out_corners = corners_np.copy()
        for kp, inst, pt in zip(augmented["keypoints"], augmented["kp_instance_ids"], augmented["kp_point_ids"]):
            out_corners[int(inst), int(pt), 0] = float(kp[0])
            out_corners[int(inst), int(pt), 1] = float(kp[1])

        target = target.copy()
        target["corners"] = torch.from_numpy(out_corners).to(corners.dtype)
        return Image.fromarray(new_image_np), target


def make_dota_transforms(
    image_set: str,
    resolution: int,
) -> Compose:
    """Build transform pipeline for DOTA dataset.

    Args:
        image_set: Split identifier — ``"train"``, ``"val"`` or ``"test"``.
        resolution: Target square resolution in pixels.

    Returns:
        Composed transform pipeline.

    Raises:
        ValueError: If ``image_set`` is not a recognised split.
    """
    if A is None:
        raise ImportError("albumentations is required for DOTA transforms. Install with: pip install albumentations")
    if image_set not in ("train", "val", "test"):
        raise ValueError(f"unknown image_set {image_set!r}; expected 'train', 'val' or 'test'")

    # One Compose for the whole geometric chain: each OBBGeometricTransform costs a
    # PIL->numpy->PIL round-trip and a full keypoint pass, so chaining four of them
    # would convert every sample eight times.
    geometric: list[Any] = [A.Resize(height=resolution, width=resolution)]
    if image_set == "train":
        geometric += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]

    return Compose(
        [
            OBBGeometricTransform(geometric),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            DotaNormalize(),
        ]
    )


class DotaNormalize:
    """Normalize images and convert OBB corners to normalized cxcywha format.

    After geometric augmentations, recomputes ``boxes_obb`` from the (potentially transformed) ``corners`` keypoints,
    then normalizes spatial coordinates by image size.
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
        # ``corners`` stay in post-transform pixel space, so consumers need the
        # post-transform size to map them back to original-image coordinates.
        # Nothing else in the DOTA pipeline updates this (unlike the torchvision
        # transforms used by COCO), so it must be refreshed here.
        target["size"] = torch.as_tensor([int(h), int(w)])

        if "corners" in target and len(target["corners"]) > 0:
            boxes_obb = corners_to_cxcywha(target["corners"])
            # Drop zero-area boxes.  An augmentation can move a box entirely out of
            # frame, and a w=0 or h=0 target is not a crash downstream — _obb_to_gaussian
            # clamps it to a minimum size, so the model would silently train against a
            # meaningless target instead.  ConvertYolo applies the same guard.
            keep = (boxes_obb[:, 2] > 0) & (boxes_obb[:, 3] > 0)
            if not bool(keep.all()):
                boxes_obb = boxes_obb[keep]
                target["corners"] = target["corners"][keep]
                target["labels"] = target["labels"][keep]

            scale = boxes_obb.new_tensor([w, h, w, h, 1.0])
            target["boxes_obb"] = boxes_obb / scale
            target["boxes"] = target["boxes_obb"][..., :4]

        return image, target


def build_dota(image_set: str, args: Any, resolution: int) -> DotaDetection:
    """Build a DOTA dataset for the given split.

    Args:
        image_set: Split identifier — ``"train"`` or ``"val"``.
        args: Namespace with a ``dataset_dir`` attribute and an optional
            ``dota_include_difficult`` flag.
        resolution: Target resolution in pixels.

    Returns:
        Configured DotaDetection dataset.

    Raises:
        FileNotFoundError: If no directory exists for the requested split.
    """
    # multi_scale/expanded_scales default to True in TrainConfig but the OBB pipeline
    # resizes to a fixed square, so they would otherwise be accepted and ignored.
    unsupported = [name for name in ("multi_scale", "expanded_scales") if getattr(args, name, False)]
    if unsupported and image_set == "train":
        logger.warning(
            "DOTA training ignores %s: the OBB pipeline resizes to a fixed %dx%d square. "
            "Set them to False to silence this, or vary `resolution` between runs instead.",
            " and ".join(unsupported),
            resolution,
            resolution,
        )

    dataset_dir = Path(args.dataset_dir)
    # Roboflow-style exports name the validation split "valid"; DOTA uses "val".
    candidates = [dataset_dir / "valid"] if image_set == "val" else []
    root = dataset_dir / image_set
    if not root.exists():
        root = next((c for c in candidates if c.exists()), root)
    if not root.exists():
        raise FileNotFoundError(f"No directory found for split {image_set!r} under {dataset_dir}")

    return DotaDetection(
        root=root,
        transforms=make_dota_transforms(image_set, resolution),
        include_difficult=getattr(args, "dota_include_difficult", False),
    )
