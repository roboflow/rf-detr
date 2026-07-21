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

from rfdetr.datasets._keypoint_schema import (
    CocoKeypointSchema,
    YoloKeypointSchema,
    _infer_keypoint_flip_pairs_from_names,
    _merge_category_keypoint_flip_pairs,
    infer_coco_keypoint_schema,
    infer_yolo_keypoint_schema,
)


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


def test_infer_coco_keypoint_schema_infers_left_right_flip_pairs(tmp_path: Path) -> None:
    """COCO category keypoint names should infer horizontal flip pairs when unambiguous."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {
                "id": 0,
                "name": "person",
                "keypoints": ["nose", "left_eye", "right_eye", "left_wrist", "right_wrist"],
                "skeleton": [],
            }
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.keypoint_flip_pairs == [1, 2, 3, 4]


def test_infer_coco_keypoint_schema_does_not_invent_missing_mirror_pairs(tmp_path: Path) -> None:
    """A left/right token without its counterpart should keep the keypoint slots unswapped."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {
                "id": 0,
                "name": "person",
                "keypoints": ["nose", "left_eye", "left_wrist"],
                "skeleton": [],
            }
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.num_keypoints_per_class == [3]
    assert schema.keypoint_flip_pairs == []


def test_infer_coco_keypoint_schema_drops_pairs_when_keypoint_categories_disagree(tmp_path: Path) -> None:
    """A global flip-pair list is unsafe when keypoint classes use different slot layouts."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[
            {
                "id": 0,
                "name": "standing_person",
                "keypoints": ["left_eye", "right_eye", "nose"],
                "skeleton": [],
            },
            {
                "id": 1,
                "name": "seated_person",
                "keypoints": ["nose", "left_eye", "right_eye"],
                "skeleton": [],
            },
        ],
    )

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.num_keypoints_per_class == [3, 3]
    assert schema.keypoint_flip_pairs == []


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

    with pytest.raises(ValueError, match="has no keypoint metadata"):
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


def test_infer_coco_keypoint_schema_raises_file_not_found(tmp_path: Path) -> None:
    """Non-existent annotation file should raise FileNotFoundError before any parsing."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        infer_coco_keypoint_schema(missing)


def test_infer_coco_keypoint_schema_raises_key_error_for_missing_categories_key(tmp_path: Path) -> None:
    """JSON file missing the 'categories' key should raise KeyError."""
    annotation_path = tmp_path / "no_categories.json"
    annotation_path.write_text('{"images": [], "annotations": []}', encoding="utf-8")

    with pytest.raises(KeyError):
        infer_coco_keypoint_schema(annotation_path)


def test_infer_coco_keypoint_schema_raises_value_error_for_list_root(tmp_path: Path) -> None:
    """JSON file whose root is a list (not an object) should raise ValueError."""
    annotation_path = tmp_path / "list_root.json"
    annotation_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        infer_coco_keypoint_schema(annotation_path)


@pytest.mark.parametrize(
    "counts,expected",
    [
        pytest.param([0, 17, 25], [17, 25], id="leading-zero-filtered"),
        pytest.param([0, 0], [], id="all-zero-returns-empty"),
        pytest.param([5, 17], [5, 17], id="all-nonzero-returned-unchanged"),
        pytest.param([], [], id="empty-input-returns-empty"),
    ],
)
def test_active_keypoint_counts_filters_zeros(counts: list[int], expected: list[int]) -> None:
    """active_keypoint_counts should return only positive counts in schema order."""
    from rfdetr.datasets._keypoint_schema import active_keypoint_counts

    result = active_keypoint_counts(counts)

    assert result == expected, f"active_keypoint_counts({counts!r}) = {result!r}, expected {expected!r}"


def test_infer_yolo_keypoint_schema_reads_pose_yaml_metadata(tmp_path: Path) -> None:
    """YOLO pose YAML should define class names, keypoint count, names, and flip pairs."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text(
        "names:\n  0: person\nkpt_shape: [2, 3]\nflip_idx: [0, 1]\nkpt_names:\n  0:\n    - left_eye\n    - right_eye\n",
        encoding="utf-8",
    )

    schema = infer_yolo_keypoint_schema(data_file)

    assert schema == YoloKeypointSchema(
        class_names=["person"],
        num_keypoints_per_class=[2],
        keypoint_oks_sigmas=[0.1, 0.1],
        keypoint_names=["left_eye", "right_eye"],
        flip_idx=[0, 1],
        keypoint_dim=3,
    )


def test_infer_yolo_keypoint_schema_rejects_detection_yaml(tmp_path: Path) -> None:
    """Detection-only YOLO YAML should fail fast in keypoint schema inference."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names:\n  - person\n", encoding="utf-8")

    with pytest.raises(ValueError, match="kpt_shape"):
        infer_yolo_keypoint_schema(data_file)


@pytest.mark.parametrize("kpt_shape", ["[17, 1]", "[0, 3]", "[17]", "[17, 4]"])
def test_infer_yolo_keypoint_schema_rejects_invalid_kpt_shape(tmp_path: Path, kpt_shape: str) -> None:
    """YOLO pose kpt_shape must be [positive_count, 2_or_3]."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text(f"names:\n  - person\nkpt_shape: {kpt_shape}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="kpt_shape"):
        infer_yolo_keypoint_schema(data_file)


@pytest.mark.parametrize(
    "flip_idx_text, expected_match",
    [
        pytest.param("[0, 5]", "permutation", id="out_of_range"),
        pytest.param("[0, 0]", "permutation", id="duplicate"),
        pytest.param("[0]", "integer indexes", id="wrong_length"),
    ],
)
def test_infer_yolo_keypoint_schema_rejects_invalid_flip_idx(
    tmp_path: Path, flip_idx_text: str, expected_match: str
) -> None:
    """flip_idx must be a valid permutation of 0..N-1 matching kpt_shape count."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text(
        f"names:\n  0: person\nkpt_shape: [2, 3]\nflip_idx: {flip_idx_text}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=expected_match):
        infer_yolo_keypoint_schema(data_file)


# ---------------------------------------------------------------------------
# _infer_keypoint_flip_pairs_from_names edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names,expected",
    [
        pytest.param([], [], id="empty-input"),
        pytest.param(["left_eye"], [], id="single-directional-no-mirror"),
        pytest.param(["Left-Eye", "left_eye"], [], id="duplicate-normalized-names"),
        pytest.param(["left_right_wrist"], [], id="two-directional-tokens-ambiguous"),
        pytest.param(["nose", "left_eye", "right_eye"], [1, 2], id="standard-coco-pair"),
    ],
)
def test_infer_keypoint_flip_pairs_from_names_edge_cases(names: list[str], expected: list[int]) -> None:
    """Edge-case inputs should return the expected flat pair list without raising."""
    assert _infer_keypoint_flip_pairs_from_names(names) == expected


# ---------------------------------------------------------------------------
# _merge_category_keypoint_flip_pairs success path
# ---------------------------------------------------------------------------


def test_merge_category_keypoint_flip_pairs_returns_common_pairs_when_all_agree() -> None:
    """All-agreeing categories should return the shared pair list."""
    assert _merge_category_keypoint_flip_pairs([[1, 2], [1, 2], [1, 2]]) == [1, 2]


def test_merge_category_keypoint_flip_pairs_single_category() -> None:
    """Single-category input should return that category's pairs unchanged."""
    assert _merge_category_keypoint_flip_pairs([[3, 5, 1, 2]]) == [3, 5, 1, 2]


# ---------------------------------------------------------------------------
# YoloKeypointSchema.keypoint_flip_pairs
# ---------------------------------------------------------------------------


def test_infer_yolo_keypoint_schema_populates_keypoint_flip_pairs(tmp_path: Path) -> None:
    """YOLO flip_idx should produce an equivalent keypoint_flip_pairs list on the schema."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text(
        "names:\n  0: person\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n",
        encoding="utf-8",
    )

    schema = infer_yolo_keypoint_schema(data_file)

    assert schema.flip_idx == [0, 2, 1]
    assert schema.keypoint_flip_pairs == [1, 2]


def test_infer_yolo_keypoint_schema_empty_flip_idx_gives_empty_pairs(tmp_path: Path) -> None:
    """Missing flip_idx should result in an empty keypoint_flip_pairs list."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text(
        "names:\n  0: person\nkpt_shape: [2, 3]\n",
        encoding="utf-8",
    )

    schema = infer_yolo_keypoint_schema(data_file)

    assert schema.flip_idx == []
    assert schema.keypoint_flip_pairs == []


# ---------------------------------------------------------------------------
# Public re-export from rfdetr.datasets
# ---------------------------------------------------------------------------


def test_infer_coco_keypoint_schema_importable_from_datasets_package(tmp_path: Path) -> None:
    """infer_coco_keypoint_schema should be importable from the public rfdetr.datasets package."""
    from rfdetr.datasets import infer_coco_keypoint_schema as public_fn

    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(
        annotation_path,
        categories=[{"id": 0, "name": "person", "keypoints": ["nose", "left_eye", "right_eye"], "skeleton": []}],
    )
    schema = public_fn(annotation_path)
    assert schema.keypoint_flip_pairs == [1, 2]


def test_infer_yolo_keypoint_schema_importable_from_datasets_package(tmp_path: Path) -> None:
    """infer_yolo_keypoint_schema should be importable from the public rfdetr.datasets package."""
    from rfdetr.datasets import infer_yolo_keypoint_schema as public_fn

    data_file = tmp_path / "data.yaml"
    data_file.write_text("names:\n  0: person\nkpt_shape: [1, 3]\n", encoding="utf-8")
    schema = public_fn(data_file)
    assert schema.num_keypoints_per_class == [1]
