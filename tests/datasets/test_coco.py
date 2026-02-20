# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for sparse COCO category ID remapping in ConvertCoco.

COCO category IDs are sparse (1–90 with gaps). Without remapping they are used
directly as tensor indices, causing out-of-bounds errors during training when
the model has only 80 classes.  ConvertCoco must remap them to contiguous
0-indexed labels via the ``cat2label`` mapping built from the annotation file.
"""

from typing import Dict, List

import pytest
import torch
from PIL import Image
from pycocotools.coco import COCO

from rfdetr.datasets.coco import ConvertCoco
from rfdetr.datasets.coco_eval import CocoEvaluator

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


@pytest.fixture
def coco_gt() -> COCO:
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "width": 10, "height": 10}],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cat_1"},
            {"id": 3, "name": "cat_3"},
        ],
    }
    coco.createIndex()
    setattr(coco, "label2cat", {0: 1, 1: 3})
    return coco


@pytest.fixture
def base_prediction() -> Dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=torch.float32),
        "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
    }


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


class TestCocoEvaluatorCategoryResolution:
    @pytest.mark.parametrize(
        ("labels", "expected_category_ids"),
        [
            pytest.param([0, 1], [1, 3], id="contiguous-labels"),
            pytest.param([1, 3], [1, 3], id="raw-coco-category-ids"),
        ],
    )
    def test_prepare_detection_resolves_category_ids(
        self,
        coco_gt: COCO,
        base_prediction: Dict[str, torch.Tensor],
        labels: List[int],
        expected_category_ids: List[int],
    ) -> None:
        evaluator = CocoEvaluator(coco_gt, ["bbox"])
        predictions = {
            1: {
                **base_prediction,
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        }
        results = evaluator.prepare_for_coco_detection(predictions)
        assert [result["category_id"] for result in results] == expected_category_ids
