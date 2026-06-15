# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Parquet DatasetRecordWithBBox dataset support."""

from __future__ import annotations

import bisect
import io
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Sampler

from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.utilities.logger import get_logger

logger = get_logger()

IMAGE_COLUMN = "GroundTruthPageImages"
BBOX_COLUMN = "GroundTruthBboxOnPageImages"


def _load_parquet_module() -> Any:
    """Import pyarrow.parquet with a targeted optional-dependency error."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet DatasetRecordWithBBox training requires pyarrow. "
            "Install training dependencies with `pip install 'rfdetr[train]'`."
        ) from exc
    return pq


def _normalize_label_mapping(label_mapping: dict[int, str]) -> tuple[dict[int, int], list[str]]:
    """Return source-category to contiguous-label mapping and ordered class names."""
    if not label_mapping:
        raise ValueError("parquet_label_mapping must not be empty.")
    normalized = {int(category_id): name for category_id, name in label_mapping.items()}
    ordered_category_ids = sorted(normalized)
    class_names: list[str] = []
    category_to_label: dict[int, int] = {}
    for label_id, category_id in enumerate(ordered_category_ids):
        name = normalized[category_id]
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid label name for category_id={category_id!r}: {name!r}")
        category_to_label[category_id] = label_id
        class_names.append(name)
    return category_to_label, class_names


def _parse_bbox_mapping(value: Any) -> dict[Any, Any]:
    """Parse a GroundTruthBboxOnPageImages cell into a mapping."""
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{BBOX_COLUMN} must be a dict or JSON object string, got {type(value).__name__}.")
    return value


def _lookup_page_bboxes(bbox_mapping: dict[Any, Any], page_index: int) -> list[dict[str, Any]]:
    """Return bbox records for page_index, accepting int and string keys."""
    raw = bbox_mapping.get(page_index)
    if raw is None:
        raw = bbox_mapping.get(str(page_index), [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{BBOX_COLUMN}[{page_index!r}] must be a list, got {type(raw).__name__}.")
    return raw


def _to_pil_image(value: Any) -> Image.Image:
    """Convert a parquet image cell value to an RGB PIL image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    raise ValueError(f"Cannot convert image value of type {type(value).__name__} to PIL.Image.")


def _iter_page_samples_from_row(row: dict[str, Any]) -> Iterable[tuple[Image.Image, list[dict[str, Any]], int]]:
    """Yield page image, bbox list, and page index from a raw parquet row."""
    try:
        from docling_eval.datamodels.dataset_record import DatasetRecordWithBBox
    except ImportError:
        DatasetRecordWithBBox = None

    if DatasetRecordWithBBox is not None:
        record = DatasetRecordWithBBox.model_validate(row)
        get_samples = getattr(record, "get_page_images_with_bboxes", None)
        if get_samples is not None:
            for page_index, (image, bbox_list) in enumerate(get_samples()):
                yield _to_pil_image(image), list(bbox_list or []), page_index
            return

    if IMAGE_COLUMN not in row:
        raise ValueError(f"Missing required parquet column: {IMAGE_COLUMN}")
    if BBOX_COLUMN not in row:
        raise ValueError(f"Missing required parquet column: {BBOX_COLUMN}")

    page_images = row[IMAGE_COLUMN]
    if page_images is None:
        page_images = []
    if not isinstance(page_images, list):
        raise ValueError(f"{IMAGE_COLUMN} must be a list, got {type(page_images).__name__}.")

    bbox_mapping = _parse_bbox_mapping(row[BBOX_COLUMN])
    for page_index, image_value in enumerate(page_images):
        yield _to_pil_image(image_value), _lookup_page_bboxes(bbox_mapping, page_index), page_index


def _bbox_to_ltrb(bbox_record: dict[str, Any]) -> list[float] | None:
    """Convert a bbox record to [left, top, right, bottom], or None when absent."""
    if bbox_record.get("ltrb") is not None:
        values = bbox_record["ltrb"]
        if len(values) != 4:
            raise ValueError(f"ltrb must contain 4 values, got {values!r}")
        return [float(v) for v in values]
    if bbox_record.get("bbox") is not None:
        values = bbox_record["bbox"]
        if len(values) != 4:
            raise ValueError(f"bbox must contain 4 values, got {values!r}")
        x, y, width, height = (float(v) for v in values)
        return [x, y, x + width, y + height]
    return None


class ShardSequentialSampler(Sampler[int]):
    """Sampler that iterates parquet rows file-by-file for cache locality."""

    def __init__(self, file_offsets: list[int], file_num_rows: list[int], *, shuffle: bool = False) -> None:
        self._file_offsets = file_offsets
        self._file_num_rows = file_num_rows
        self._shuffle = shuffle
        self._epoch = 0
        self._total = sum(file_num_rows)

    def __len__(self) -> int:
        """Return the number of indexed rows."""
        return self._total

    def set_epoch(self, epoch: int) -> None:
        """Set epoch seed for deterministic per-epoch shuffle."""
        self._epoch = epoch

    def __iter__(self) -> Iterable[int]:
        """Yield global row indices."""
        generator = torch.Generator()
        generator.manual_seed(self._epoch)

        file_count = len(self._file_offsets)
        file_order = torch.randperm(file_count, generator=generator).tolist() if self._shuffle else range(file_count)
        for file_idx in file_order:
            offset = self._file_offsets[file_idx]
            row_count = self._file_num_rows[file_idx]
            row_order = torch.randperm(row_count, generator=generator).tolist() if self._shuffle else range(row_count)
            for local_idx in row_order:
                yield offset + int(local_idx)


class FlattenPageSamplesCollate:
    """Flatten row-level page sample lists before delegating to RF-DETR collate."""

    def __init__(self, base_collate: Callable[[list[tuple[Any, ...]]], tuple[Any, ...]]) -> None:
        self.base_collate = base_collate

    def __call__(self, items: list[list[tuple[Any, ...]]]) -> tuple[Any, ...]:
        """Flatten row samples and call the base collate function."""
        flat_items = [sample for row_samples in items for sample in row_samples]
        if not flat_items:
            raise RuntimeError("Collate received no page samples to batch.")
        return self.base_collate(flat_items)


class DatasetRecordWithBBoxParquetDataset(torch.utils.data.Dataset):
    """Lazy parquet dataset where each row contains DatasetRecordWithBBox-style data."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        split: str | None = None,
        transforms: Callable[[Image.Image, dict[str, Any]], tuple[Any, dict[str, Any]]] | None = None,
        label_mapping: dict[int, str],
        max_objects: int | None = 400,
        max_skip_retries: int = 10,
    ) -> None:
        self.transforms = transforms
        self.max_objects = max_objects
        self.max_skip_retries = max_skip_retries
        self.category_to_label, self.class_names = _normalize_label_mapping(label_mapping)
        self._skipped_rows = 0
        self._skipped_pages = 0

        path = Path(parquet_path)
        if path.is_dir():
            pattern = f"{split}-*.parquet" if split else "*.parquet"
            files = sorted(path.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No parquet files matching {pattern!r} in {path}")
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        pq = _load_parquet_module()
        self._files = files
        self.file_offsets: list[int] = []
        self.file_num_rows: list[int] = []
        total_rows = 0
        for file_path in files:
            parquet_file = pq.ParquetFile(str(file_path))
            row_count = int(parquet_file.metadata.num_rows)
            self.file_offsets.append(total_rows)
            self.file_num_rows.append(row_count)
            total_rows += row_count
        self._total_rows = total_rows
        self._cached_file_idx: int | None = None
        self._cached_table: Any | None = None

        logger.info(
            "Indexed %d parquet rows from %d file(s) for DatasetRecordWithBBox training.",
            self._total_rows,
            len(files),
        )

    def __len__(self) -> int:
        """Return total parquet row count."""
        return self._total_rows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        """Resolve a global row index into file index and local row index."""
        file_idx = bisect.bisect_right(self.file_offsets, idx) - 1
        if file_idx < 0:
            raise IndexError(idx)
        local_idx = idx - self.file_offsets[file_idx]
        return file_idx, local_idx

    def _get_row(self, idx: int) -> dict[str, Any]:
        """Read one parquet row, loading only the containing file into the cache."""
        file_idx, local_idx = self._resolve_index(idx)
        if self._cached_file_idx != file_idx:
            pq = _load_parquet_module()
            self._cached_table = pq.read_table(str(self._files[file_idx]))
            self._cached_file_idx = file_idx
        table = self._cached_table
        if table is None:
            raise RuntimeError("Parquet table cache was not initialized.")
        return {column: table.column(column)[local_idx].as_py() for column in table.column_names}

    def _retry_row(self, idx: int, retry_count: int, reason: str) -> list[tuple[Any, dict[str, Any]]]:
        """Retry another row from the same parquet file after an invalid row/page."""
        self._skipped_rows += 1
        if self._skipped_rows % 100 == 1:
            logger.warning("Skipped %d parquet rows; latest idx=%d: %s", self._skipped_rows, idx, reason)
        if retry_count >= self.max_skip_retries:
            raise RuntimeError(
                f"Could not find a usable parquet row after {self.max_skip_retries} retries; latest reason: {reason}"
            )
        file_idx, _ = self._resolve_index(idx)
        offset = self.file_offsets[file_idx]
        row_count = self.file_num_rows[file_idx]
        replacement_idx = offset + random.randint(0, row_count - 1)
        return self.load_item(replacement_idx, retry_count + 1)

    def _build_target(
        self,
        row_idx: int,
        page_index: int,
        image_size: tuple[int, int],
        bbox_list: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Build one RF-DETR target dict from page bboxes."""
        width, height = image_size
        boxes: list[list[float]] = []
        labels: list[int] = []

        for bbox_record in bbox_list:
            if not isinstance(bbox_record, dict):
                raise ValueError(f"bbox records must be dicts, got {type(bbox_record).__name__}.")
            if "category_id" not in bbox_record:
                raise ValueError(f"bbox record is missing category_id: {bbox_record!r}")
            category_id = int(bbox_record["category_id"])
            if category_id not in self.category_to_label:
                raise ValueError(f"category_id={category_id} is not present in parquet_label_mapping.")
            ltrb = _bbox_to_ltrb(bbox_record)
            if ltrb is None:
                continue
            left, top, right, bottom = ltrb
            left = min(max(left, 0.0), float(width))
            right = min(max(right, 0.0), float(width))
            top = min(max(top, 0.0), float(height))
            bottom = min(max(bottom, 0.0), float(height))
            if right <= left or bottom <= top:
                continue
            boxes.append([left, top, right, bottom])
            labels.append(self.category_to_label[category_id])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes_tensor) > 0
            else torch.zeros(0, dtype=torch.float32)
        )
        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([row_idx * 100000 + page_index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
            "orig_size": torch.as_tensor([height, width], dtype=torch.int64),
            "size": torch.as_tensor([height, width], dtype=torch.int64),
        }

    def load_item(self, idx: int, retry_count: int = 0) -> list[tuple[Any, dict[str, Any]]]:
        """Load and transform all usable page samples from one parquet row."""
        try:
            row = self._get_row(idx)
            raw_samples = list(_iter_page_samples_from_row(row))
        except Exception as exc:
            return self._retry_row(idx, retry_count, f"row parsing failed: {type(exc).__name__}: {exc}")

        samples: list[tuple[Any, dict[str, Any]]] = []
        for image, bbox_list, page_index in raw_samples:
            target = self._build_target(idx, page_index, image.size, bbox_list)
            if self.max_objects is not None and len(target["boxes"]) > self.max_objects:
                self._skipped_pages += 1
                continue
            if len(target["boxes"]) == 0:
                continue
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            samples.append((image, target))

        if samples:
            return samples
        return self._retry_row(idx, retry_count, "row produced no usable page samples")

    def __getitem__(self, idx: int) -> list[tuple[Any, dict[str, Any]]]:
        """Return transformed page-level samples for a parquet row."""
        return self.load_item(idx)


def resolve_parquet_train_source(dataset_dir: str | Path) -> tuple[Path, str | None]:
    """Resolve supported parquet layouts to a path and optional split prefix."""
    root = Path(dataset_dir)
    if root.is_file():
        return root, None
    data_dir = root / "data"
    if data_dir.is_dir() and any(data_dir.glob("train-*.parquet")):
        return data_dir, "train"
    train_dir = root / "train"
    if train_dir.is_dir() and any(train_dir.glob("*.parquet")):
        return train_dir, None
    raise FileNotFoundError(
        f"No train parquet files found under {root}. Expected {root / 'data'}/train-*.parquet "
        f"or {root / 'train'}/*.parquet."
    )


def build_parquet_bbox(image_set: str, args: Any, resolution: int) -> DatasetRecordWithBBoxParquetDataset:
    """Build a DatasetRecordWithBBox parquet dataset for RF-DETR training."""
    if image_set != "train":
        raise ValueError("dataset_file='parquet_bbox' only supports train datasets; dedicated evals are out of scope.")
    root = getattr(args, "dataset_dir", None)
    if not root:
        raise ValueError("dataset_dir is required for dataset_file='parquet_bbox'.")
    label_mapping = getattr(args, "parquet_label_mapping", None)
    if not label_mapping:
        raise ValueError("parquet_label_mapping is required for dataset_file='parquet_bbox'.")

    parquet_path, split = resolve_parquet_train_source(root)
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)
    keypoint_flip_pairs: list[int] = getattr(args, "keypoint_flip_pairs", []) or []

    transform_builder = make_coco_transforms_square_div_64 if square_resize_div_64 else make_coco_transforms
    transforms = transform_builder(
        "train",
        resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        skip_random_resize=not do_random_resize_via_padding,
        patch_size=patch_size,
        num_windows=num_windows,
        aug_config=aug_config,
        gpu_postprocess=False,
        keypoint_flip_pairs=keypoint_flip_pairs,
    )
    return DatasetRecordWithBBoxParquetDataset(
        parquet_path,
        split=split,
        transforms=transforms,
        label_mapping=label_mapping,
        max_objects=getattr(args, "parquet_max_objects", 400),
    )
