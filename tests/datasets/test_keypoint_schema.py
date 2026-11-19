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


@pytest.mark.parametrize(
    "categories,class_names,counts",
    [
        pytest.param(
            [
                {"id": 0, "name": "person", "keypoints": ["nose", "left_eye"]},
                {"id": 1, "name": "helmet"},
                {"id": 2, "name": "vest"},
            ],
            ["person", "helmet", "vest"],
            [2, 0, 0],
            id="detection-only-categories-in-free-slots",
        ),
        pytest.param(
            [
                {"id": 3, "name": "adult", "keypoints": ["head", "foot"]},
                {"id": 9, "name": "child", "keypoints": ["head", "foot"]},
            ],
            ["adult", "child"],
            [2, 2],
            id="same-count-keypoint-categories",
        ),
        pytest.param(
            [
                {"id": 0, "name": "person", "keypoints": ["nose"]},
                {"id": 1, "name": "animal", "keypoints": ["head", "tail"]},
            ],
            ["person", "animal"],
            [1, 2],
            id="mixed-keypoint-counts",
        ),
    ],
)
def test_infer_coco_keypoint_schema_stays_category_aligned(
    tmp_path: Path, categories: list[dict], class_names: list[str], counts: list[int]
) -> None:
    """Class names and per-class keypoint counts follow category order whatever the counts or ids (padded later)."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(annotation_path, categories=categories)

    schema = infer_coco_keypoint_schema(annotation_path)

    assert schema.class_names == class_names
    assert schema.num_keypoints_per_class == counts
    assert schema.keypoint_oks_sigmas == [0.1, 0.1]


def test_infer_coco_keypoint_schema_rejects_missing_keypoints(tmp_path: Path) -> None:
    """Detection-only COCO files should fail fast instead of silently training without keypoints."""
    annotation_path = tmp_path / "annotations.json"
    _write_coco_annotations(annotation_path, categories=[{"id": 0, "name": "person"}])

    with pytest.raises(ValueError, match="has no keypoint metadata"):
        infer_coco_keypoint_schema(annotation_path)


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


@pytest.mark.parametrize(
    "content,error",
    [
        pytest.param('{"images": [], "annotations": []}', KeyError, id="missing-categories-key"),
        pytest.param("[]", ValueError, id="list-root"),
    ],
)
def test_infer_coco_keypoint_schema_rejects_malformed_json_root(tmp_path: Path, content: str, error: type) -> None:
    """A JSON file without a 'categories' key raises KeyError and one with a list root raises ValueError."""
    annotation_path = tmp_path / "malformed.json"
    annotation_path.write_text(content, encoding="utf-8")

    with pytest.raises(error):
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


@pytest.mark.parametrize(
    "per_category,expected",
    [
        pytest.param([[1, 2], [1, 2], [1, 2]], [1, 2], id="all-agree"),
        pytest.param([[3, 5, 1, 2]], [3, 5, 1, 2], id="single-category"),
    ],
)
def test_merge_category_keypoint_flip_pairs_returns_shared_pairs(
    per_category: list[list[int]], expected: list[int]
) -> None:
    """Agreeing categories return the shared pair list, and a single category's pairs come back unchanged."""
    assert _merge_category_keypoint_flip_pairs(per_category) == expected


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
