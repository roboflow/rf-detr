# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DatasetRecordWithBBox parquet datasets."""

import io
import json

import pytest
import torch
from PIL import Image

from rfdetr.datasets.parquet_bbox import (
    DatasetRecordWithBBoxParquetDataset,
    FlattenPageSamplesCollate,
    resolve_parquet_train_source,
)
from rfdetr.utilities.tensors import make_collate_fn

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _png_bytes(color: str = "white", size: tuple[int, int] = (32, 24)) -> bytes:
    """Return a PNG byte string for a tiny RGB image."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_parquet(path, rows):
    """Write rows to parquet."""
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _row(page_count=1, bboxes=None):
    """Build one DatasetRecordWithBBox-like parquet row."""
    return {
        "GroundTruthPageImages": [{"bytes": _png_bytes(size=(32, 24)), "path": None} for _ in range(page_count)],
        "GroundTruthBboxOnPageImages": json.dumps(bboxes or {}),
    }


def test_dataset_indexes_metadata_and_expands_page_samples(tmp_path):
    """Dataset should index rows and expand one row into page-level samples."""
    parquet_path = tmp_path / "train-00000.parquet"
    _write_parquet(
        parquet_path,
        [
            _row(
                page_count=2,
                bboxes={
                    "0": [{"category_id": 0, "bbox": [1, 2, 10, 12]}],
                    "1": [{"category_id": 1, "ltrb": [3, 4, 20, 22]}],
                },
            )
        ],
    )

    dataset = DatasetRecordWithBBoxParquetDataset(
        parquet_path,
        label_mapping={0: "text", 1: "table"},
        transforms=None,
    )

    assert len(dataset) == 1
    samples = dataset[0]
    assert len(samples) == 2
    assert torch.equal(samples[0][1]["labels"], torch.tensor([0]))
    assert torch.equal(samples[1][1]["labels"], torch.tensor([1]))
    assert torch.allclose(samples[0][1]["boxes"], torch.tensor([[1.0, 2.0, 11.0, 14.0]]))
    assert torch.allclose(samples[1][1]["boxes"], torch.tensor([[3.0, 4.0, 20.0, 22.0]]))


def test_dataset_drops_invalid_boxes_and_empty_pages(tmp_path):
    """Invalid boxes are skipped and pages without boxes are dropped."""
    parquet_path = tmp_path / "train-00000.parquet"
    _write_parquet(
        parquet_path,
        [
            _row(
                page_count=2,
                bboxes={
                    "0": [
                        {"category_id": 0, "bbox": [5, 5, -1, 10]},
                        {"category_id": 0, "bbox": [1, 1, 4, 4]},
                    ],
                    "1": [],
                },
            )
        ],
    )

    dataset = DatasetRecordWithBBoxParquetDataset(parquet_path, label_mapping={0: "text"}, transforms=None)

    samples = dataset[0]
    assert len(samples) == 1
    assert torch.allclose(samples[0][1]["boxes"], torch.tensor([[1.0, 1.0, 5.0, 5.0]]))


def test_dataset_rejects_category_not_in_label_mapping(tmp_path):
    """BBox category ids must be present in the explicit label mapping."""
    parquet_path = tmp_path / "train-00000.parquet"
    _write_parquet(
        parquet_path,
        [_row(page_count=1, bboxes={"0": [{"category_id": 3, "bbox": [1, 1, 4, 4]}]})],
    )

    dataset = DatasetRecordWithBBoxParquetDataset(
        parquet_path,
        label_mapping={0: "text"},
        transforms=None,
        max_skip_retries=0,
    )

    with pytest.raises(RuntimeError, match="category_id=3"):
        dataset[0]


def test_resolve_parquet_train_source_supports_data_layout(tmp_path):
    """HF-style data/train-*.parquet layout should resolve to split prefix train."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_parquet(data_dir / "train-00000-of-00001.parquet", [_row(bboxes={"0": []})])

    path, split = resolve_parquet_train_source(tmp_path)

    assert path == data_dir
    assert split == "train"


def test_flatten_page_samples_collate_delegates_to_base_collate():
    """Flatten collate should turn row samples into a normal RF-DETR batch."""
    collate = FlattenPageSamplesCollate(make_collate_fn(block_size=16))
    image = torch.zeros(3, 8, 8)
    target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "labels": torch.tensor([0]),
        "orig_size": torch.tensor([8, 8]),
        "size": torch.tensor([8, 8]),
    }

    samples, targets = collate([[(image, target)], [(image, target)]])

    assert samples.tensors.shape == (2, 3, 16, 16)
    assert len(targets) == 2
