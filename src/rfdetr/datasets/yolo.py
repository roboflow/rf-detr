# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import supervision as sv
from PIL import Image, ImageDraw
from torchvision.datasets import VisionDataset

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)

REQUIRED_YOLO_YAML_FILES = ["data.yaml", "data.yml"]
REQUIRED_SPLIT_DIRS = ["train", "valid"]
REQUIRED_DATA_SUBDIRS = ["images", "labels"]
YOLO_IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _parse_yolo_box(values: list[str]) -> np.ndarray:
    """Parse a YOLO center-width-height box into relative XYXY coordinates."""
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    """Convert a relative XYXY box into a 4-corner polygon."""
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
        dtype=np.float32,
    )


def _parse_yolo_polygon(values: list[str]) -> np.ndarray:
    """Parse a flattened YOLO polygon into relative XY points."""
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygon_to_mask(polygon: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon into a dense boolean mask.

    TODO: remove once supervision ships a direct CompactMask.from_polygon factory;
    at that point the dense intermediate array is no longer needed.
    """
    width, height = resolution_wh
    mask = Image.new("L", (width, height), 0)
    if polygon.size > 0:
        ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon.tolist()], fill=1)
    return np.array(mask, dtype=bool)


def _polygons_to_masks(polygons: tuple[np.ndarray, ...], resolution_wh: tuple[int, int]) -> np.ndarray:
    """Rasterize per-instance polygons into an ``(N, H, W)`` boolean array.

    TODO: remove once supervision ships a direct CompactMask.from_polygon factory;
    at that point the dense intermediate array is no longer needed.
    """
    if len(polygons) == 0:
        width, height = resolution_wh
        return np.zeros((0, height, width), dtype=bool)
    return np.stack([_polygon_to_mask(polygon, resolution_wh) for polygon in polygons])


def _list_yolo_image_paths(images_directory_path: str) -> list[str]:
    """List YOLO image files in a stable order."""
    return sorted(
        str(path)
        for path in Path(images_directory_path).iterdir()
        if path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
    )


def _extract_yolo_class_names(data_file: str) -> list[str]:
    """Read class names from a YOLO ``data.yaml`` file."""
    import yaml

    with Path(data_file).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in data file {data_file!r}, got {type(data).__name__}.")
    names = data.get("names")
    if isinstance(names, dict):
        # YOLO label files use integer class IDs. When ``names`` is a mapping, we
        # only support the standard numeric-keyed form where keys are a contiguous
        # 0-based range: {0: "cat", 1: "dog", ...}. This keeps class IDs consistent
        # with range checks downstream that assume valid IDs are 0..N-1.
        numeric_keys: list[int] = []
        non_numeric_keys: list[Any] = []
        for key in names.keys():
            key_str = str(key)
            if key_str.isdigit():
                numeric_keys.append(int(key_str))
            else:
                non_numeric_keys.append(key)

        if not numeric_keys:
            raise ValueError(
                "Unsupported 'names' mapping in data file "
                f"{data_file!r}: expected integer keys 0..N-1 when 'names' is a dict, "
                f"got only non-numeric keys {list(names.keys())!r}. "
                "Please provide 'names' as a list or as a dict with 0-based contiguous "
                "integer keys."
            )

        unique_sorted_keys = sorted(set(numeric_keys))
        expected_keys = list(range(len(unique_sorted_keys)))
        if unique_sorted_keys != expected_keys or non_numeric_keys:
            raise ValueError(
                "Unsupported 'names' mapping in data file "
                f"{data_file!r}: expected integer keys 0..N-1 with no gaps, "
                f"got numeric keys {unique_sorted_keys!r} and "
                f"non-numeric keys {non_numeric_keys!r}. "
                "This loader assumes class IDs are contiguous 0..N-1; please remap "
                "the 'names' keys or use the list form."
            )

        # At this point, keys are exactly 0..N-1; order them by numeric ID.
        return [str(names[idx]) for idx in unique_sorted_keys]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Expected 'names' to be a list or dict in {data_file!r}, got {type(names).__name__}.")


@dataclass(frozen=True)
class _LazyYoloSample:
    """Lightweight per-image YOLO metadata with polygons kept lazy until fetch time.

    Note: ``frozen=True`` prevents field *reassignment* but does NOT prevent
    in-place mutation of ``np.ndarray`` fields (e.g. ``sample.xyxy[0] = 999.0``
    would silently succeed).  This is safe across DataLoader workers because
    each worker receives a pickled copy of the dataset.
    """

    image_path: str
    width: int
    height: int
    xyxy: np.ndarray
    class_id: np.ndarray
    polygons: tuple[np.ndarray, ...]

    def to_detections(self) -> "sv.Detections":
        """Materialize the current sample as a supervision ``Detections`` object."""
        import supervision as sv

        if len(self.class_id) == 0:
            return sv.Detections.empty()
        if len(self.polygons) == 0:
            # Detection-only path: no masks were computed, return bare boxes.
            return sv.Detections(class_id=self.class_id, xyxy=self.xyxy)
        # TODO: once supervision v0.28 ships CompactMask, wrap the dense result:
        #   compact = sv.CompactMask.from_dense(mask, self.xyxy, (self.height, self.width))
        #   return sv.Detections(..., mask=compact)
        # CompactMask stores crop-RLE instead of a full H×W bool array, reducing memory
        # at the detections level for large images with sparse objects.
        # Note: _polygon_to_mask / _polygons_to_masks remain required as the intermediate
        # rasterization step until supervision provides a direct from_polygon factory.
        mask = _polygons_to_masks(self.polygons, (self.width, self.height))
        return sv.Detections(class_id=self.class_id, xyxy=self.xyxy, mask=mask)


class _LazyYoloDetectionDataset:
    """Lazy YOLO dataset that defers dense mask rasterization until ``__getitem__``."""

    def __init__(self, classes: list[str], samples: list[_LazyYoloSample]) -> None:
        self.classes = classes
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, "sv.Detections"]:
        import cv2

        sample = self._samples[idx]
        image = cv2.imread(sample.image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {sample.image_path}")
        return sample.image_path, image, sample.to_detections()

    def get_image_info(self, idx: int) -> _LazyYoloSample:
        """Return lightweight metadata without loading pixels or dense masks."""
        return self._samples[idx]


def _parse_yolo_label_line(
    values: list[str],
    line_num: int,
    label_path: Path,
    num_classes: int,
    width: int,
    height: int,
    *,
    parse_polygons: bool = True,
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """Parse one YOLO label line and return ``(class_id, xyxy_px, polygon_px)``.

    Args:
        values: Whitespace-split fields from the label line.
        line_num: 1-based line number (for error messages).
        label_path: Path to the label file (for error messages).
        num_classes: Total number of classes in the dataset (used for range check).
        width: Image width in pixels.
        height: Image height in pixels.
        parse_polygons: When ``False`` the pixel-space polygon array is not
            computed or returned (``polygon_px`` will be ``None``).  Set to
            ``False`` on the detection-only path to avoid allocating polygon
            arrays that would immediately be discarded.

    Returns:
        Tuple of ``(class_id, xyxy_px, polygon_px)`` where coordinates are in
        pixel space.  ``polygon_px`` is ``None`` when ``parse_polygons=False``.

    Raises:
        ValueError: If the line is malformed or the class ID is out of range.
    """
    if len(values) < 5:
        raise ValueError(
            f"Malformed label in {str(label_path)!r} at line {line_num}: "
            f"expected 5 (bbox) fields or ≥ 7 fields for polygons "
            f"(class_id + at least 3 (x, y) points), got {len(values)}."
        )
    if len(values) > 5 and len(values[1:]) % 2 != 0:
        raise ValueError(
            f"Malformed polygon in {str(label_path)!r} at line {line_num}: "
            f"polygon coordinates must be paired (x, y) values, "
            f"but got {len(values[1:])} coordinate values (odd count)."
        )
    try:
        cid = int(values[0])
    except ValueError as exc:
        raise ValueError(
            f"Label {str(label_path)!r} line {line_num}: invalid class ID {values[0]!r} (must be an integer)."
        ) from exc
    # num_classes equals len(class_names) which _extract_yolo_class_names guarantees
    # is a contiguous 0..N-1 range.  This assumption must remain consistent with the
    # class-name parser: accepting sparse keys there (e.g. {0: "cat", 2: "dog"} → 2
    # classes) would cause valid label files using the original IDs to be rejected here.
    if cid < 0 or cid >= num_classes:
        raise ValueError(
            f"Label {str(label_path)!r} line {line_num}: "
            f"class ID {cid} is out of range for dataset with {num_classes} classes "
            f"(valid range 0\u2013{num_classes - 1})."
        )
    if len(values) == 5:
        box = _parse_yolo_box(values[1:])
        # Skip polygon creation on the detection path — only the bbox is needed.
        polygon: np.ndarray | None = _box_to_polygon(box) if parse_polygons else None
    else:
        try:
            _raw_polygon = _parse_yolo_polygon(values[1:])
        except ValueError as exc:
            raise ValueError(
                f"Malformed polygon in {str(label_path)!r} at line {line_num}: "
                f"could not parse coordinate values as floats."
            ) from exc
        box = np.array(
            [
                np.min(_raw_polygon[:, 0]),
                np.min(_raw_polygon[:, 1]),
                np.max(_raw_polygon[:, 0]),
                np.max(_raw_polygon[:, 1]),
            ],
            dtype=np.float32,
        )
        # On the detection path, _raw_polygon was only needed for bbox extraction;
        # skip the pixel-space conversion to avoid a redundant allocation.
        polygon = _raw_polygon if parse_polygons else None
    xyxy_px = box * np.array([width, height, width, height], dtype=np.float32)
    if polygon is None:
        return cid, xyxy_px, None
    polygon_px = polygon * np.array([width, height], dtype=np.float32)
    polygon_px[:, 0] = np.clip(polygon_px[:, 0], 0.0, float(width - 1))
    polygon_px[:, 1] = np.clip(polygon_px[:, 1], 0.0, float(height - 1))
    return cid, xyxy_px, polygon_px.astype(np.float32)


def _build_yolo_samples(
    img_folder: str, lb_folder: str, data_file: str, *, include_polygons: bool
) -> tuple[list[str], list[_LazyYoloSample]]:
    """Build the class list and sample list shared by both YOLO builder functions.

    Iterates over every image in ``img_folder``, reads image dimensions via PIL
    (header-only, no full decode), and parses the matching ``.txt`` label file
    when present.  Images without a label file are included as *background*
    samples with empty detections.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.
        include_polygons: When ``True`` polygon coordinates are stored in each
            :class:`_LazyYoloSample` (segmentation path).  When ``False``
            polygon coordinates returned by :func:`_parse_yolo_label_line` are
            discarded and ``polygons=()`` is stored instead (detection-only path).

    Returns:
        A ``(classes, samples)`` tuple where ``classes`` is the ordered list of
        class names and ``samples`` is a list of :class:`_LazyYoloSample` objects.

    Examples:
        >>> # Used internally by _build_lazy_yolo_detection_dataset and
        >>> # _build_lazy_yolo_segmentation_dataset — not part of the public API.
        >>> pass
    """
    classes = _extract_yolo_class_names(data_file)
    samples: list[_LazyYoloSample] = []

    for image_path in _list_yolo_image_paths(img_folder):
        label_path = Path(lb_folder) / f"{Path(image_path).stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size

        xyxy: list[np.ndarray] = []
        class_id: list[int] = []
        polygons: list[np.ndarray] = []
        if label_path.exists():
            with label_path.open(encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            for i, line in enumerate(lines):
                cid, xyxy_px, polygon_px = _parse_yolo_label_line(
                    line.split(),
                    i + 1,
                    label_path,
                    len(classes),
                    width,
                    height,
                    parse_polygons=include_polygons,
                )
                class_id.append(cid)
                xyxy.append(xyxy_px)
                if include_polygons and polygon_px is not None:
                    polygons.append(polygon_px)

        samples.append(
            _LazyYoloSample(
                image_path=image_path,
                width=width,
                height=height,
                xyxy=np.array(xyxy, dtype=np.float32).reshape(-1, 4),
                class_id=np.array(class_id, dtype=np.int64),
                polygons=tuple(polygons),
            )
        )

    return classes, samples


def _build_lazy_yolo_detection_dataset(img_folder: str, lb_folder: str, data_file: str) -> _LazyYoloDetectionDataset:
    """Build a YOLO detection dataset that stores bounding boxes lazily.

    Unlike :func:`_build_lazy_yolo_segmentation_dataset`, this function does
    not store polygon coordinates or dense masks — only ``xyxy`` boxes are
    retained, keeping peak memory proportional to the number of annotations.

    Images without a matching ``.txt`` label file are included as
    *background* samples with empty detections, so datasets that mix labelled
    and unlabelled images are handled correctly.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.

    Returns:
        A :class:`_LazyYoloDetectionDataset` whose ``__getitem__`` loads pixel
        data on demand and returns ``sv.Detections`` without mask information.
    """
    classes, samples = _build_yolo_samples(img_folder, lb_folder, data_file, include_polygons=False)
    return _LazyYoloDetectionDataset(classes=classes, samples=samples)


def _build_lazy_yolo_segmentation_dataset(img_folder: str, lb_folder: str, data_file: str) -> _LazyYoloDetectionDataset:
    """Build a YOLO dataset that stores polygons and rasterizes masks on demand.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.

    Returns:
        A :class:`_LazyYoloDetectionDataset` whose ``__getitem__`` loads pixel
        data on demand and rasterizes polygon masks into dense boolean tensors.
    """
    classes, samples = _build_yolo_samples(img_folder, lb_folder, data_file, include_polygons=True)
    return _LazyYoloDetectionDataset(classes=classes, samples=samples)


def is_valid_yolo_dataset(dataset_dir: str) -> bool:
    """
    Checks if the specified dataset directory is in yolo format.

    We accept a dataset to be in yolo format if the following conditions are met:
    - The dataset_dir contains a data.yaml or data.yml file
    - The dataset_dir contains "train" and "valid" subdirectories, each containing "images" and "labels" subdirectories
    - The "test" subdirectory is optional

    Returns a boolean indicating whether the dataset is in correct yolo format.
    """
    contains_required_yolo_yaml = any(
        os.path.exists(os.path.join(dataset_dir, yaml_file)) for yaml_file in REQUIRED_YOLO_YAML_FILES
    )
    contains_required_split_dirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir)) for split_dir in REQUIRED_SPLIT_DIRS
    )
    contains_required_data_subdirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir, data_subdir))
        for split_dir in REQUIRED_SPLIT_DIRS
        for data_subdir in REQUIRED_DATA_SUBDIRS
    )
    return contains_required_yolo_yaml and contains_required_split_dirs and contains_required_data_subdirs


class ConvertYolo:
    """
    Converts supervision Detections to the target dict format expected by RF-DETR.

    Args:
        include_masks: whether to include segmentation masks

    Examples:
        >>> import numpy as np
        >>> import supervision as sv
        >>> from PIL import Image
        >>> # Create a sample image and target
        >>> image = Image.new("RGB", (100, 100))
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[10, 20, 30, 40]]),
        ...     class_id=np.array([0])
        ... )
        >>> target = {"image_id": 0, "detections": detections}
        >>> # Create converter
        >>> converter = ConvertYolo(include_masks=False)
        >>> # Call converter
        >>> img, result = converter(image, target)
        >>> sorted(result.keys())
        ['area', 'boxes', 'image_id', 'iscrowd', 'labels', 'orig_size', 'size']
        >>> result["boxes"].shape
        torch.Size([1, 4])
        >>> result["labels"].tolist()
        [0]
        >>> result["image_id"].tolist()
        [0]
    """

    def __init__(self, include_masks: bool = False):
        self.include_masks = include_masks

    def __call__(self, image: Image.Image, target: dict) -> tuple:
        """
        Convert image and YOLO detections to RF-DETR format.

        Args:
            image: PIL Image
            target: dict with 'image_id' and 'detections' (sv.Detections)

        Returns:
            tuple of (image, target_dict)
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        detections = target["detections"]

        if len(detections) > 0:
            boxes = torch.from_numpy(detections.xyxy).to(torch.float32)
            classes = torch.from_numpy(detections.class_id).to(torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # clamp and filter
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id

        # compute area after clamp
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_out["area"] = area

        iscrowd = torch.zeros((classes.shape[0],), dtype=torch.int64)
        target_out["iscrowd"] = iscrowd

        if self.include_masks:
            if detections.mask is not None and np.size(detections.mask) > 0:
                masks = torch.from_numpy(detections.mask[keep.cpu().numpy()]).to(torch.uint8)
                target_out["masks"] = masks
            else:
                target_out["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target_out["masks"] = target_out["masks"].bool()

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        return image, target_out


class _MockSvDataset:
    """Mock supervision dataset for testing CocoLikeAPI."""

    classes = ["cat", "dog"]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        import numpy as np
        import supervision as sv

        det = sv.Detections(xyxy=np.array([[10 * i, 20, 30, 40]]), class_id=np.array([i]))
        return f"img_{i}.jpg", np.zeros((100, 100, 3), dtype=np.uint8), det


class CocoLikeAPI:
    """
    A minimal COCO-compatible API wrapper for YOLO datasets.

    This provides the necessary interface for CocoEvaluator to work with
    YOLO format datasets.

    Examples:
        >>> mock = _MockSvDataset()
        >>> coco = CocoLikeAPI(mock.classes, mock)
        >>> # dataset structure
        >>> len(coco.dataset["images"]), len(coco.dataset["categories"]), len(coco.dataset["annotations"])
        (2, 2, 2)
        >>> # getAnnIds
        >>> coco.getAnnIds()
        [0, 1]
        >>> coco.getAnnIds(imgIds=[0])
        [0]
        >>> coco.getAnnIds(catIds=[1])
        [1]
        >>> # getCatIds
        >>> sorted(coco.getCatIds())
        [0, 1]
        >>> coco.getCatIds(catNms=["cat"])
        [0]
        >>> # getImgIds
        >>> sorted(coco.getImgIds())
        [0, 1]
        >>> coco.getImgIds(catIds=[0])
        [0]
        >>> # loadAnns
        >>> ann = coco.loadAnns([0])[0]
        >>> ann["category_id"], ann["image_id"]
        (0, 0)
        >>> # loadCats
        >>> coco.loadCats([0])[0]["name"]
        'cat'
        >>> len(coco.loadCats())
        2
        >>> # loadImgs
        >>> coco.loadImgs([1])[0]["file_name"]
        'img_1.jpg'
    """

    def __init__(self, classes: list[str], dataset: Any):
        self.classes = classes
        # ``dataset`` may be a supervision DetectionDataset or a lazy adapter exposing
        # ``get_image_info(idx)`` in addition to ``__len__``/``__getitem__``.
        self.sv_dataset = dataset

        # Build the dataset dict that COCO API expects
        self.dataset = self._build_coco_dataset()
        self.imgs = {img["id"]: img for img in self.dataset["images"]}
        self.anns = {ann["id"]: ann for ann in self.dataset["annotations"]}
        self.cats = {cat["id"]: cat for cat in self.dataset["categories"]}

        # Build imgToAnns index
        self.imgToAnns = {}
        for ann in self.dataset["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []
            self.imgToAnns[img_id].append(ann)

        # Ensure all images have an entry
        for img_id in self.imgs:
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []

        # Build catToImgs index
        self.catToImgs = {}
        for cat_id in self.cats:
            self.catToImgs[cat_id] = []
        for ann in self.dataset["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if img_id not in self.catToImgs[cat_id]:
                self.catToImgs[cat_id].append(img_id)

    def _build_coco_dataset(self) -> dict:
        """Build a COCO-format dataset dict from YOLO data.

        ``dataset`` is duck-typed: it must implement ``__len__`` and
        ``__getitem__`` (returning ``(image_path, cv2_image, sv.Detections)``).
        When it additionally exposes ``get_image_info(idx)`` returning a
        :class:`_LazyYoloSample`, that lighter-weight path is used instead of
        loading pixel data just to extract image dimensions.
        """
        images = []
        annotations = []
        categories = []

        # Build categories (0-indexed class IDs in YOLO)
        for idx, class_name in enumerate(self.classes):
            categories.append({"id": idx, "name": class_name, "supercategory": "none"})

        ann_id = 0
        use_lazy_path = hasattr(self.sv_dataset, "get_image_info")
        for img_id in range(len(self.sv_dataset)):
            if use_lazy_path:
                sample = self.sv_dataset.get_image_info(img_id)
                image_path = sample.image_path
                h, w = sample.height, sample.width
                xyxy = sample.xyxy
                class_id = sample.class_id
                has_masks = len(sample.polygons) > 0
            else:
                image_path, cv2_image, detections = self.sv_dataset[img_id]
                h, w = cv2_image.shape[:2]
                xyxy = detections.xyxy
                class_id = detections.class_id
                has_masks = detections.mask is not None

            images.append({"id": img_id, "file_name": str(image_path), "height": h, "width": w})

            if len(xyxy) == 0:
                continue
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                bbox_x, bbox_y, bbox_w, bbox_h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id[i]),
                    "bbox": [float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                }

                # Add segmentation if available
                if has_masks:
                    # For now, use empty polygon - evaluation will still work for bbox
                    ann["segmentation"] = []

                annotations.append(ann)
                ann_id += 1

        return {
            "info": {"description": "RF-DETR YOLO dataset"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter by image IDs (list or single ID)
            catIds: Filter by category IDs (list or single ID)
            areaRng: Filter by area range [min, max]
            iscrowd: Filter by iscrowd flag (0 or 1)

        Returns:
            List of annotation IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        areaRng = areaRng or []

        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        catIds = catIds if isinstance(catIds, list) else [catIds]

        if len(imgIds) == 0:
            anns = self.dataset["annotations"]
        else:
            anns = []
            for img_id in imgIds:
                anns.extend(self.imgToAnns.get(img_id, []))

        if len(catIds) > 0:
            anns = [ann for ann in anns if ann["category_id"] in catIds]

        if len(areaRng) == 2:
            anns = [ann for ann in anns if ann["area"] >= areaRng[0] and ann["area"] <= areaRng[1]]

        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

        return [ann["id"] for ann in anns]

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs that satisfy given filter conditions.

        Args:
            catNms: Filter by category names (list)
            supNms: Filter by supercategory names (list, not used)
            catIds: Filter by category IDs (list)

        Returns:
            List of category IDs matching the filter conditions
        """
        catNms = catNms or []
        # supNms = supNms or []
        catIds = catIds or []

        cats = self.dataset["categories"]

        if len(catNms) > 0:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if len(catIds) > 0:
            cats = [cat for cat in cats if cat["id"] in catIds]

        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=None, catIds=None):
        """Get image IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter to these image IDs (list)
            catIds: Filter by images containing these category IDs (list)

        Returns:
            List of image IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        imgIds = set(imgIds) if imgIds else set(self.imgs.keys())

        if len(catIds) > 0:
            # Find all images that contain at least one of the specified categories
            matching_img_ids = set()
            for cat_id in catIds:
                matching_img_ids.update(self.catToImgs.get(cat_id, []))

            # Intersect with existing imgIds filter
            imgIds &= matching_img_ids

        return list(imgIds)

    def loadAnns(self, ids=None):
        """Load annotations with the specified IDs.

        Args:
            ids: Annotation IDs to load (list or single ID)

        Returns:
            List of annotation dicts with keys: id, image_id, category_id, bbox, area, iscrowd
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.anns[ann_id] for ann_id in ids if ann_id in self.anns]

    def loadCats(self, ids=None):
        """Load categories with the specified IDs.

        Args:
            ids: Category IDs to load (list or single ID). If None, returns all categories.

        Returns:
            List of category dicts with keys: id, name, supercategory
        """
        if ids is None:
            return list(self.cats.values())
        ids = ids if isinstance(ids, list) else [ids]
        return [self.cats[cat_id] for cat_id in ids if cat_id in self.cats]

    def loadImgs(self, ids=None):
        """Load images with the specified IDs.

        Args:
            ids: Image IDs to load (list or single ID)

        Returns:
            List of image dicts with keys: id, file_name, height, width
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.imgs[img_id] for img_id in ids if img_id in self.imgs]


class YoloDetection(VisionDataset):
    """YOLO format dataset with lazy image loading and optional mask support.

    Both detection (``include_masks=False``) and segmentation
    (``include_masks=True``) paths use a lazy backend: image pixels are loaded
    on demand inside ``__getitem__`` rather than at construction time, which
    keeps peak RAM proportional to the number of annotations rather than to
    ``N × H × W``.

    Images without a matching ``.txt`` label file are treated as *background*
    images and produce empty detections.  This ensures that datasets containing
    a mix of annotated and unannotated images are handled correctly in both
    single-GPU and multi-GPU training.

    This class provides a VisionDataset interface compatible with RF-DETR training,
    matching the API of CocoDetection.

    Args:
        img_folder: Path to the directory containing images
        lb_folder: Path to the directory containing YOLO annotation .txt files
        data_file: Path to data.yaml file containing class names and dataset info
        transforms: Optional transforms to apply to images and targets
        include_masks: Whether to load segmentation masks (for YOLO segmentation format).
            When True polygons are parsed and rasterized on demand; when False only
            bounding-box coordinates are stored.
    """

    def __init__(
        self,
        img_folder: str,
        lb_folder: str,
        data_file: str,
        transforms=None,
        include_masks: bool = False,
    ):
        super(YoloDetection, self).__init__(img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertYolo(include_masks=include_masks)

        if include_masks:
            self.sv_dataset = _build_lazy_yolo_segmentation_dataset(img_folder, lb_folder, data_file)
        else:
            self.sv_dataset = _build_lazy_yolo_detection_dataset(img_folder, lb_folder, data_file)

        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))

        # Create COCO-compatible API for evaluation
        self.coco = CocoLikeAPI(self.classes, self.sv_dataset)

    def __len__(self) -> int:
        return len(self.sv_dataset)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path, cv2_image, detections = self.sv_dataset[idx]

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2_image[:, :, ::-1]
        img = Image.fromarray(rgb_image)

        target = {"image_id": image_id, "detections": detections}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_roboflow_from_yolo(image_set: str, args: Any, resolution: int) -> YoloDetection:
    """Build a Roboflow YOLO-format dataset.

    This uses Roboflow's standard YOLO directory structure
    (train/valid/test folders with images/ and labels/ subdirectories).

    Args:
        image_set: Dataset split to load. One of ``"train"``, ``"val"``, or
            ``"test"``.
        args: Argument namespace. The following attributes are consumed:
            ``dataset_dir``, ``square_resize_div_64``, ``aug_config``,
            ``segmentation_head``, ``multi_scale``, ``expanded_scales``,
            ``do_random_resize_via_padding``, ``patch_size``, ``num_windows``.
            ``aug_config`` is forwarded to the transform builder; when
            ``None`` the builder falls back to the default
            :data:`~rfdetr.datasets.aug_config.AUG_CONFIG`.
        resolution: Target square resolution in pixels.

    Returns:
        A :class:`YoloDetection` dataset instance ready for use with a
        DataLoader.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    # YOLO format uses images/ and labels/ subdirectories
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "valid" / "images", root / "valid" / "labels"),
        "test": (root / "test" / "images", root / "test" / "labels"),
    }

    # Prefer data.yaml; fall back to data.yml if present; default to data.yaml for error reporting
    data_file = next((root / f for f in REQUIRED_YOLO_YAML_FILES if (root / f).exists()), root / "data.yaml")
    img_folder, lb_folder = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", None)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", None)
    num_windows = getattr(args, "num_windows", None)
    aug_config = getattr(args, "aug_config", None)

    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
            ),
            include_masks=include_masks,
        )
    else:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
            ),
            include_masks=include_masks,
        )
    return dataset
