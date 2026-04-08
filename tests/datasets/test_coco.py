# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for COCO dataset handling.

Tests cover:
- Sparse COCO category ID remapping in ``ConvertCoco``
- ``_load_classes`` hierarchy detection (GitHub #609)
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from rfdetr.datasets.coco import ConvertCoco
from rfdetr.detr import RFDETR

# Minimal image shared across all tests
_IMAGE = Image.new("RGB", (100, 100))

# Sparse COCO-style category IDs (as in the real COCO dataset: 1-90 with gaps)
# e.g. COCO skips IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91
_SPARSE_CAT_IDS = [1, 2, 3, 7, 8]  # sparse, non-zero-indexed

_ANNOTATIONS = [
    {"bbox": [10, 10, 30, 30], "category_id": 1, "area": 900, "iscrowd": 0},
    {"bbox": [50, 50, 20, 20], "category_id": 7, "area": 400, "iscrowd": 0},
]

_CAT2LABEL = {cat_id: i for i, cat_id in enumerate(sorted(_SPARSE_CAT_IDS))}
# {1: 0, 2: 1, 3: 2, 7: 3, 8: 4}


def _make_target(annotations=_ANNOTATIONS):
    return {"image_id": 1, "annotations": annotations}


class TestConvertCocoWithoutMapping:
    """Without cat2label, sparse IDs pass through unchanged — demonstrating the bug."""

    def test_labels_are_raw_category_ids(self):
        converter = ConvertCoco(cat2label=None)
        _, target = converter(_IMAGE, _make_target())
        # Raw COCO IDs — NOT safe to use as indices into an 80-class tensor
        assert target["labels"].tolist() == [1, 7]

    def test_raw_ids_would_exceed_num_classes(self):
        """Illustrates why raw IDs cause CUDA out-of-bounds with num_classes=80."""
        converter = ConvertCoco(cat2label=None)
        _, target = converter(_IMAGE, _make_target())
        num_classes = len(_SPARSE_CAT_IDS)  # 5 — same as model would see
        assert any(lbl >= num_classes for lbl in target["labels"].tolist()), (
            "At least one raw category_id should exceed num_classes, "
            "triggering an out-of-bounds index in the matcher/loss."
        )


class TestConvertCocoWithMapping:
    """With cat2label, sparse IDs are remapped to contiguous 0-indexed labels."""

    def test_labels_are_remapped_to_zero_indexed(self):
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        # category_id 1 → 0, category_id 7 → 3
        assert target["labels"].tolist() == [0, 3]

    def test_all_labels_within_num_classes(self):
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        num_classes = len(_SPARSE_CAT_IDS)
        assert all(lbl < num_classes for lbl in target["labels"].tolist())

    def test_roboflow_zero_indexed_is_identity(self):
        """Roboflow datasets already use 0-indexed IDs — mapping must be identity."""
        roboflow_cat2label = {0: 0, 1: 1, 2: 2}
        annotations = [
            {"bbox": [10, 10, 30, 30], "category_id": 0, "area": 900, "iscrowd": 0},
            {"bbox": [50, 50, 20, 20], "category_id": 2, "area": 400, "iscrowd": 0},
        ]
        converter = ConvertCoco(cat2label=roboflow_cat2label)
        _, target = converter(_IMAGE, _make_target(annotations))
        assert target["labels"].tolist() == [0, 2]

    def test_label_tensor_dtype(self):
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        assert target["labels"].dtype == torch.int64


def _write_coco_json(path: Path, categories: List[Dict]) -> None:
    """Write a minimal valid COCO annotation file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"images": [], "annotations": [], "categories": categories}
    path.write_text(json.dumps(data))


class TestLoadClassesHierarchy:
    """Regression tests for ``_load_classes`` supercategory filtering (#609).

    When all categories have ``supercategory: "none"`` (flat COCO datasets),
    ``_load_classes`` previously returned an empty list. It should only filter
    when a Roboflow hierarchical export is detected.
    """

    def test_roboflow_hierarchy_filters_parent(self, tmp_path: Path) -> None:
        """Roboflow exports include a parent node — only leaf categories kept."""
        categories = [
            {"id": 0, "name": "annotations", "supercategory": "none"},
            {"id": 1, "name": "dog", "supercategory": "annotations"},
            {"id": 2, "name": "cat", "supercategory": "annotations"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_flat_none_supercategory_keeps_all(self, tmp_path: Path) -> None:
        """Flat datasets where every category has supercategory 'none' (#609)."""
        categories = [
            {"id": 1, "name": "dog", "supercategory": "none"},
            {"id": 2, "name": "cat", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_mixed_supercategories_keeps_all(self, tmp_path: Path) -> None:
        """Mix of 'none' and non-'none' supercategories where no category is a parent of another.

        'animal' appears as a supercategory but is not itself a category name, so
        ``has_children`` is empty and all categories pass the ``name not in has_children``
        filter — both 'dog' and 'cat' are returned.
        """
        categories = [
            {"id": 1, "name": "dog", "supercategory": "none"},
            {"id": 2, "name": "cat", "supercategory": "animal"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_category_named_none_does_not_empty_list(self, tmp_path: Path) -> None:
        """If a category is literally named 'none' and all supercategories
        are placeholders, the loader must return all class names instead of [].
        """
        categories = [
            {"id": 1, "name": "none", "supercategory": "none"},
            {"id": 2, "name": "dog", "supercategory": "none"},
            {"id": 3, "name": "cat", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["none", "dog", "cat"]

    def test_mixed_hierarchy_leaf_and_standalone_forwarding(self, tmp_path: Path) -> None:
        """Mixed hierarchy: only leaf classes + standalone top-level categories
        should be forwarded. Parent/grouping nodes are dropped.
        """
        categories = [
            {"id": 1, "name": "animals", "supercategory": "none"},
            {"id": 2, "name": "mammal", "supercategory": "animals"},
            {"id": 3, "name": "dog", "supercategory": "mammal"},
            {"id": 4, "name": "cat", "supercategory": "mammal"},
            {"id": 5, "name": "bird", "supercategory": "animals"},
            {"id": 6, "name": "eagle", "supercategory": "bird"},
            {"id": 7, "name": "pigeon", "supercategory": "bird"},
            {"id": 8, "name": "objects", "supercategory": "none"},
            {"id": 9, "name": "vehicle", "supercategory": "objects"},
            {"id": 10, "name": "car", "supercategory": "vehicle"},
            {"id": 11, "name": "truck", "supercategory": "vehicle"},
            {"id": 12, "name": "appliance", "supercategory": "objects"},
            {"id": 13, "name": "toaster", "supercategory": "appliance"},
            {"id": 14, "name": "microwave", "supercategory": "appliance"},
            {"id": 15, "name": "person", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        expected = [
            "dog",
            "cat",
            "eagle",
            "pigeon",
            "car",
            "truck",
            "toaster",
            "microwave",
            "person",
        ]
        assert result == expected

    def test_placeholder_values_treated_as_no_parent(self, tmp_path: Path) -> None:
        """Placeholders like None, '', and 'null' should be treated the same
        as 'none'.
        """
        categories = [
            {"id": 1, "name": "dog", "supercategory": None},
            {"id": 2, "name": "cat", "supercategory": ""},
            {"id": 3, "name": "elephant", "supercategory": "null"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat", "elephant"]

    def test_unsorted_category_ids_return_id_sorted_class_order(self, tmp_path: Path) -> None:
        """Returned class names must follow category-ID order for stable index mapping."""
        categories = [
            {"id": 30, "name": "truck", "supercategory": "vehicle"},
            {"id": 10, "name": "vehicle", "supercategory": "none"},
            {"id": 20, "name": "car", "supercategory": "vehicle"},
            {"id": 40, "name": "person", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["car", "truck", "person"]
