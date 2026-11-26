# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the one category rule that keypoint schema inference and the COCO loader share.

Covers which categories own a keypoint slot (``keypoint_category_ids``).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema, keypoint_category_ids
from rfdetr.datasets.coco import _build_keypoint_cat2label, _category_ids_with_keypoints

_PERSON_KEYPOINT_NAMES = [f"joint_{index}" for index in range(17)]
_FOUR_KEYPOINTS = [1.0, 2.0, 2.0] * 4
_SEVENTEEN_KEYPOINTS = [1.0, 2.0, 2.0] * 17


def _annotation(annotation_id: int, category_id: int, keypoints: list[float]) -> dict[str, Any]:
    """Return a COCO keypoint annotation as a plain dictionary (a data builder, not a test).

    Examples:
        >>> _annotation(7, 2, [1.0, 2.0, 2.0])["num_keypoints"]
        1
    """
    return {
        "id": annotation_id,
        "image_id": 1,
        "category_id": category_id,
        "bbox": [0, 0, 8, 8],
        "area": 64,
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": len(keypoints) // 3,
    }


# Category 1 has annotation-only keypoints (4), category 2 has metadata keypoints (17): inference yields [4, 17].
_PARTIAL_METADATA_CATEGORIES = [
    {"id": 1, "name": "helmet"},
    {"id": 2, "name": "person", "keypoints": _PERSON_KEYPOINT_NAMES},
]
_PARTIAL_METADATA_ANNOTATIONS = [
    _annotation(1, 1, _FOUR_KEYPOINTS),
    _annotation(2, 2, _SEVENTEEN_KEYPOINTS),
]


def _write_split(path: Path, categories: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> Path:
    """Write a COCO annotation file holding no images and return its path.

    Examples:
        >>> import tempfile
        >>> target = _write_split(Path(tempfile.mkdtemp()) / "split" / "ann.json", [{"id": 1, "name": "a"}], [])
        >>> json.loads(target.read_text())["categories"]
        [{'id': 1, 'name': 'a'}]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": [], "categories": categories, "annotations": annotations}), encoding="utf-8")
    return path


def _as_coco(categories: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> types.SimpleNamespace:
    """Return the ``cats``/``anns`` view of a pycocotools object for the given raw lists.

    Examples:
        >>> _as_coco([{"id": 3, "name": "a"}], [{"id": 9, "category_id": 3}]).anns
        {9: {'id': 9, 'category_id': 3}}
    """
    return types.SimpleNamespace(
        cats={category["id"]: category for category in categories},
        anns={annotation["id"]: annotation for annotation in annotations},
    )


class TestKeypointCategoryIds:
    """Unit tests for the shared ``keypoint_category_ids`` rule."""

    @pytest.mark.parametrize(
        ("categories", "annotations", "expected"),
        [
            pytest.param([{"id": 1, "keypoints": ["a"]}], [], [1], id="metadata-only"),
            pytest.param([{"id": 1}], [{"category_id": 1, "keypoints": [1, 2, 2]}], [1], id="annotation-only"),
            pytest.param(
                [{"id": 1}, {"id": 2, "keypoints": ["a"]}],
                [{"category_id": 1, "keypoints": [1, 2, 2]}],
                [1, 2],
                id="mixed-metadata-and-annotation-keeps-both",
            ),
            pytest.param([{"id": 1, "keypoints": []}], [{"category_id": 1, "keypoints": []}], [], id="empty-vectors"),
            pytest.param([{"id": 1}], [{"category_id": 1}], [], id="no-keypoints-at-all"),
            pytest.param([{"id": 2, "keypoints": ["a"]}, {"id": 1, "keypoints": ["a"]}], [], [1, 2], id="sorted"),
        ],
    )
    def test_rule(self, categories: list[dict], annotations: list[dict], expected: list[int]) -> None:
        """A category owns a slot when it has keypoint metadata or a non-empty annotation vector."""
        assert keypoint_category_ids(categories, annotations) == expected


class TestInferenceAndLoaderShareOneRule:
    """Schema inference and the loader must assign slots to the same categories."""

    def test_partial_metadata_keeps_the_annotation_only_category(self, tmp_path: Path) -> None:
        """Metadata on one category does not hide the annotation-only keypoints of another from the loader."""
        ann_file = _write_split(tmp_path / "ann.json", _PARTIAL_METADATA_CATEGORIES, _PARTIAL_METADATA_ANNOTATIONS)

        schema = infer_coco_keypoint_schema(ann_file).num_keypoints_per_class
        loader_ids = _category_ids_with_keypoints(_as_coco(_PARTIAL_METADATA_CATEGORIES, _PARTIAL_METADATA_ANNOTATIONS))

        assert schema == [4, 17]
        assert loader_ids == [1, 2]

    def test_partial_metadata_slots_follow_the_inferred_schema(self) -> None:
        """With schema ``[4, 17]`` the helmet takes slot 0 and the person slot 1, not the other way round."""
        coco = _as_coco(_PARTIAL_METADATA_CATEGORIES, _PARTIAL_METADATA_ANNOTATIONS)

        assert _build_keypoint_cat2label(coco, [4, 17]) == {1: 0, 2: 1}

    def test_slot_counts_agree_for_every_keypoint_category(self, tmp_path: Path) -> None:
        """The categories that infer a non-zero count are exactly the ones the loader gives a slot."""
        categories = [*_PARTIAL_METADATA_CATEGORIES, {"id": 3, "name": "car"}]
        ann_file = _write_split(tmp_path / "ann.json", categories, _PARTIAL_METADATA_ANNOTATIONS)

        counts = infer_coco_keypoint_schema(ann_file).num_keypoints_per_class
        loader_ids = _category_ids_with_keypoints(_as_coco(categories, _PARTIAL_METADATA_ANNOTATIONS))

        assert [category_id for category_id, count in zip([1, 2, 3], counts) if count > 0] == loader_ids
