# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for private COCO keypoint schema inference helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfdetr.datasets._keypoint_schema import CocoKeypointSchema, infer_coco_keypoint_schema


def _write_coco_annotations(
    path: Path,
    *,
    categories: list[dict],
    annotations: list[dict] | None = None,
) -> None:
    """Write a minimal COCO annotation file.

    Args:
        path: Destination JSON path.
        categories: COCO category objects.
        annotations: Optional COCO annotation objects.

    Returns:
        ``None``.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> import tempfile
        >>> output = Path(tempfile.mkdtemp()) / "annotations.json"
        >>> _write_coco_annotations(output, categories=[{"id": 0, "name": "person"}])
        >>> output.exists()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"images": [], "annotations": annotations or [], "categories": categories}),
        encoding="utf-8",
    )


def test_infer_coco_keypoint_schema_uses_declared_category_keypoints(tmp_path: Path) -> None:
    """Declared category keypoints should produce a category-aligned keypoint schema."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[{"id": 0, "name": "person", "keypoints": ["nose", "left_eye"], "skeleton": []}],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema == CocoKeypointSchema(
        class_names=["person"],
        num_keypoints_per_class=[2],
        keypoint_oks_sigmas=[0.1, 0.1],
    )


def test_infer_coco_keypoint_schema_uses_annotation_keypoints_when_category_metadata_is_missing(
    tmp_path: Path,
) -> None:
    """Annotation keypoint arrays should define the count when category metadata is absent."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[{"id": 7, "name": "pose"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 7, "keypoints": [1, 2, 2, 3, 4, 1]}],
    )

    schema = infer_coco_keypoint_schema(annotation_path, keypoint_oks_sigma=0.1)

    assert schema.class_names == ["pose"]
    assert schema.num_keypoints_per_class == [2]
    assert schema.keypoint_oks_sigmas == [0.1, 0.1]


def test_infer_coco_keypoint_schema_places_detection_only_categories_in_free_slots(tmp_path: Path) -> None:
    """Detection-only categories should stay category-aligned with zero keypoint counts."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {"id": 0, "name": "person", "keypoints": ["nose", "left_eye"]},
            {"id": 1, "name": "helmet"},
            {"id": 2, "name": "vest"},
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.class_names == ["person", "helmet", "vest"]
    assert schema.num_keypoints_per_class == [2, 0, 0]
    assert schema.keypoint_oks_sigmas == [0.1, 0.1]


def test_infer_coco_keypoint_schema_supports_multiple_keypoint_categories_with_same_count(tmp_path: Path) -> None:
    """Multiple keypoint classes with the same keypoint count should stay category-aligned."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {"id": 3, "name": "adult", "keypoints": ["head", "foot"]},
            {"id": 9, "name": "child", "keypoints": ["head", "foot"]},
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.class_names == ["adult", "child"]
    assert schema.num_keypoints_per_class == [2, 2]
    assert schema.keypoint_oks_sigmas == [0.1, 0.1]


def test_infer_coco_keypoint_schema_rejects_missing_keypoints(tmp_path: Path) -> None:
    """Detection-only COCO files should fail fast instead of silently training without keypoints."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(annotation_path, categories=[{"id": 0, "name": "person"}])

    with pytest.raises(ValueError, match="does not contain keypoint annotations"):
        infer_coco_keypoint_schema(annotation_path)


def test_infer_coco_keypoint_schema_supports_mixed_keypoint_counts(tmp_path: Path) -> None:
    """Different keypoint counts are represented per class and padded later by the dataset."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {"id": 0, "name": "person", "keypoints": ["nose"]},
            {"id": 1, "name": "animal", "keypoints": ["head", "tail"]},
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.class_names == ["person", "animal"]
    assert schema.num_keypoints_per_class == [1, 2]
    assert schema.keypoint_oks_sigmas == [0.1, 0.1]


def test_infer_coco_keypoint_schema_rejects_malformed_annotation_keypoint_length(tmp_path: Path) -> None:
    """COCO keypoint arrays must be flattened ``x, y, visibility`` triples."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[{"id": 0, "name": "person"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 0, "keypoints": [1, 2]}],
    )

    with pytest.raises(ValueError, match="length divisible by 3"):
        infer_coco_keypoint_schema(annotation_path)
