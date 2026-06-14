# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private helpers for COCO keypoint schema extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CocoKeypointSchema", "KeypointSchema", "active_keypoint_counts", "infer_coco_keypoint_schema"]


@dataclass(frozen=True, slots=True)
class CocoKeypointSchema:
    """Keypoint schema inferred from COCO category metadata.

    Args:
        class_names: Category names sorted by category id.
        num_keypoints_per_class: Number of keypoints for each sorted category.
        keypoint_oks_sigmas: Default OKS sigmas matching the largest keypoint class.

    Returns:
        Immutable schema value used to configure keypoint training.

    Raises:
        This value object does not raise.

    Example:
        >>> CocoKeypointSchema(["person"], [17], [0.1] * 17).num_keypoints_per_class
        [17]
    """

    class_names: list[str]
    num_keypoints_per_class: list[int]
    keypoint_oks_sigmas: list[float]


# Forward-compat alias: KeypointSchema is kept as the public name so future schema
# variants (e.g. YOLOKeypointSchema) can be swapped in without touching call sites.
KeypointSchema = CocoKeypointSchema


def _load_coco_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load a COCO annotation JSON file.

    Args:
        annotation_path: Path to a COCO annotation JSON file.

    Returns:
        Parsed COCO annotation mapping.

    Raises:
        ValueError: If the JSON root is not an object.
        OSError: If the file cannot be read.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "annotations.json"
        >>> _ = path.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
        >>> sorted(_load_coco_annotation(path))
        ['annotations', 'categories', 'images']
    """
    with annotation_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected COCO annotation root to be an object, got {type(data).__name__}.")
    return data


def _validate_categories(categories: Any) -> list[dict[str, Any]]:
    """Validate and sort COCO categories by category id.

    Args:
        categories: Raw ``categories`` value from a COCO annotation file.

    Returns:
        Category dictionaries sorted by ``id``.

    Raises:
        ValueError: If categories are missing or malformed.

    Example:
        >>> _validate_categories([{"id": 2, "name": "b"}, {"id": 1, "name": "a"}])
        [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]
    """
    if not isinstance(categories, list) or not categories:
        raise ValueError("Expected COCO annotations to contain a non-empty 'categories' list.")

    validated: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"Expected each COCO category to be an object, got {type(category).__name__}.")
        if "id" not in category:
            raise ValueError("Expected each COCO category to contain an 'id' field.")
        if "name" not in category:
            raise ValueError(f"Expected COCO category_id {category['id']!r} to contain a 'name' field.")
        validated.append(category)
    return sorted(validated, key=lambda item: int(item["id"]))


def _validate_annotations(annotations: Any) -> list[dict[str, Any]]:
    """Validate COCO annotations container type.

    Args:
        annotations: Raw ``annotations`` value from a COCO annotation file.

    Returns:
        COCO annotation dictionaries.

    Raises:
        ValueError: If annotations are not a list of objects.

    Example:
        >>> _validate_annotations([{"category_id": 1}])
        [{'category_id': 1}]
    """
    if not isinstance(annotations, list):
        raise ValueError("Expected COCO annotations to contain an 'annotations' list.")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Expected each COCO annotation to be an object, got {type(annotation).__name__}.")
    return annotations


def _keypoint_count_from_annotations(annotations: list[dict[str, Any]], category_id: int) -> int:
    """Infer keypoint count for one category from annotation vectors.

    Args:
        annotations: COCO annotation dictionaries.
        category_id: Category id whose annotations should be inspected.

    Returns:
        Maximum keypoint vector length found for the category.

    Raises:
        ValueError: If keypoint annotation length is not divisible by three.

    Example:
        >>> anns = [{"category_id": 1, "keypoints": [1, 2, 2, 3, 4, 2]}]
        >>> _keypoint_count_from_annotations(anns, 1)
        2
    """
    keypoint_count = 0
    for annotation in annotations:
        if int(annotation.get("category_id", -1)) != category_id:
            continue
        raw_keypoints = annotation.get("keypoints")
        if raw_keypoints is None or raw_keypoints == []:
            continue
        if not isinstance(raw_keypoints, list):
            continue
        if len(raw_keypoints) % 3 != 0:
            raise ValueError(
                f"COCO annotation for category_id {category_id!r} has {len(raw_keypoints)} keypoint values; "
                "expected a flat [x, y, v] list with length divisible by 3."
            )
        keypoint_count = max(keypoint_count, len(raw_keypoints) // 3)
    return keypoint_count


def infer_coco_keypoint_schema(
    annotation_path: str | Path,
    *,
    keypoint_oks_sigma: float = 0.1,
) -> CocoKeypointSchema:
    """Infer a keypoint schema from a COCO annotation JSON file.

    Args:
        annotation_path: Path to a COCO annotation JSON file.
        keypoint_oks_sigma: Default OKS sigma to repeat for the largest keypoint class.

    Returns:
        Category-aligned class names, ``num_keypoints_per_class``, and OKS sigmas.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ValueError: If the annotation file has no categories, no keypoint metadata,
            malformed COCO fields, or malformed keypoint vectors.
        KeyError: If required COCO keys are missing.
    """
    path = Path(annotation_path)
    data = _load_coco_annotation(path)
    categories = _validate_categories(data["categories"])
    annotations = _validate_annotations(data.get("annotations", []))

    class_names: list[str] = []
    num_keypoints_per_class: list[int] = []
    for category in categories:
        category_id = int(category["id"])
        class_names.append(str(category["name"]))
        category_keypoints = category.get("keypoints")
        if isinstance(category_keypoints, list) and category_keypoints:
            num_keypoints_per_class.append(len(category_keypoints))
        else:
            num_keypoints_per_class.append(_keypoint_count_from_annotations(annotations, category_id))

    if not any(count > 0 for count in num_keypoints_per_class):
        raise ValueError(
            f"COCO annotation file '{path}' has no keypoint metadata. "
            "Expected category 'keypoints' entries or annotation keypoint vectors."
        )

    max_keypoints = max(num_keypoints_per_class, default=0)
    return CocoKeypointSchema(
        class_names=class_names,
        num_keypoints_per_class=num_keypoints_per_class,
        keypoint_oks_sigmas=[keypoint_oks_sigma] * max_keypoints,
    )


def active_keypoint_counts(num_keypoints_per_class: list[int]) -> list[int]:
    """Return non-zero keypoint counts from a model schema.

    Args:
        num_keypoints_per_class: Model keypoint schema.

    Returns:
        Positive keypoint counts in schema order.

    Raises:
        This helper does not raise.

    Example:
        >>> active_keypoint_counts([0, 17, 25])
        [17, 25]
    """
    return [count for count in num_keypoints_per_class if count > 0]
