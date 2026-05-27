# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private helpers for inferring RF-DETR keypoint schemas from COCO annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CocoKeypointSchema", "infer_coco_keypoint_schema"]


@dataclass(frozen=True, slots=True)
class CocoKeypointSchema:
    """Container for keypoint model schema inferred from a COCO annotation file.

    The schema mirrors RF-DETR keypoint dataset loading, where keypoint-bearing
    categories are assigned to non-zero model label slots and non-keypoint
    categories fill the remaining slots.

    Args:
        class_names: Class names in model label-slot order.
        num_keypoints_per_class: Number of keypoints per model label slot.
        keypoint_oks_sigmas: Per-keypoint OKS sigmas for COCO keypoint evaluation.

    Returns:
        Frozen keypoint schema metadata.

    Raises:
        TypeError: If constructed with incompatible field values.

    Example:
        >>> CocoKeypointSchema(
        ...     class_names=["", "person"],
        ...     num_keypoints_per_class=[0, 17],
        ...     keypoint_oks_sigmas=[0.05] * 17,
        ... ).num_keypoints_per_class
        [0, 17]
    """

    class_names: list[str]
    num_keypoints_per_class: list[int]
    keypoint_oks_sigmas: list[float]


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


def _category_keypoint_count(category: dict[str, Any], annotations: list[dict[str, Any]]) -> int:
    """Return the declared or observed keypoint count for one COCO category.

    Args:
        category: COCO category object.
        annotations: COCO annotation objects from the same file.

    Returns:
        Number of keypoints associated with the category, or ``0`` for a
        detection-only category.

    Raises:
        ValueError: If keypoint annotation length is not divisible by three.

    Example:
        >>> _category_keypoint_count({"id": 1, "keypoints": ["a", "b"]}, [])
        2
        >>> _category_keypoint_count({"id": 1}, [{"category_id": 1, "keypoints": [1, 2, 2]}])
        1
    """
    declared = category.get("keypoints") or []
    if declared:
        return len(declared)

    category_id = category["id"]
    for annotation in annotations:
        if annotation.get("category_id") != category_id or not annotation.get("keypoints"):
            continue
        keypoints = annotation["keypoints"]
        if len(keypoints) % 3 != 0:
            raise ValueError(
                f"COCO annotation for category_id {category_id!r} has {len(keypoints)} keypoint values; "
                "expected a flat [x, y, v] list with length divisible by 3."
            )
        return len(keypoints) // 3
    return 0


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


def _class_names_for_keypoint_schema(
    categories: list[dict[str, Any]],
    keypoint_counts: list[int],
    num_keypoints_per_class: list[int],
) -> list[str]:
    """Return class names in RF-DETR keypoint label-slot order.

    Args:
        categories: COCO categories sorted by category id.
        keypoint_counts: Keypoint count for each category.
        num_keypoints_per_class: Inferred RF-DETR keypoint schema.

    Returns:
        Class names ordered by model label slot.

    Raises:
        ValueError: If no label slot is available for a category.

    Example:
        >>> cats = [{"id": 0, "name": "person"}, {"id": 1, "name": "helmet"}]
        >>> _class_names_for_keypoint_schema(cats, [17, 0], [0, 17])
        ['helmet', 'person']
    """
    active_slots = [slot for slot, count in enumerate(num_keypoints_per_class) if count > 0]
    keypoint_categories = [category for category, count in zip(categories, keypoint_counts) if count > 0]
    required_slots = max(len(categories), max(active_slots) + 1)
    slot_names = [""] * required_slots
    assigned_slots: set[int] = set()
    assigned_category_ids: set[int] = set()

    for category, slot in zip(keypoint_categories, active_slots):
        slot_names[slot] = str(category["name"])
        assigned_slots.add(slot)
        assigned_category_ids.add(int(category["id"]))

    free_slots = [slot for slot in range(required_slots) if slot not in assigned_slots]
    for category in categories:
        if int(category["id"]) in assigned_category_ids:
            continue
        if not free_slots:
            raise ValueError(f"No free model label slot remains for category_id {category['id']!r}.")
        slot_names[free_slots.pop(0)] = str(category["name"])
    return slot_names


def infer_coco_keypoint_schema(
    annotation_path: str | Path,
    *,
    keypoint_oks_sigma: float = 0.05,
) -> CocoKeypointSchema:
    """Infer RF-DETR keypoint schema metadata from a COCO annotation file.

    Keypoint-bearing categories are assigned to non-zero label slots so the
    inferred schema matches the keypoint COCO loader and the preview model
    convention. Detection-only categories fill the remaining zero-keypoint slots.

    Args:
        annotation_path: Path to a COCO annotation JSON file.
        keypoint_oks_sigma: Default OKS sigma to repeat for each keypoint.

    Returns:
        Inferred class names, ``num_keypoints_per_class``, and OKS sigmas.

    Raises:
        ValueError: If the file has no keypoint category, malformed COCO fields,
            or multiple keypoint counts that cannot share one OKS sigma vector.
        OSError: If the annotation file cannot be read.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "annotations.json"
        >>> _ = path.write_text(
        ...     '{"images": [], "annotations": [], '
        ...     '"categories": [{"id": 0, "name": "person", "keypoints": ["nose"]}]}',
        ...     encoding="utf-8",
        ... )
        >>> infer_coco_keypoint_schema(path).num_keypoints_per_class
        [0, 1]
    """
    annotation_path = Path(annotation_path)
    data = _load_coco_annotation(annotation_path)
    categories = _validate_categories(data.get("categories"))
    annotations = _validate_annotations(data.get("annotations", []))

    keypoint_counts = [_category_keypoint_count(category, annotations) for category in categories]
    active_keypoint_counts = [count for count in keypoint_counts if count > 0]
    if not active_keypoint_counts:
        raise ValueError(f"COCO annotation file {annotation_path} does not contain keypoint annotations.")
    unique_keypoint_counts = sorted(set(active_keypoint_counts))
    if len(unique_keypoint_counts) != 1:
        raise ValueError(
            f"Expected one keypoint count across keypoint classes, got {unique_keypoint_counts} in {annotation_path}."
        )

    num_keypoints_per_class = [0, *active_keypoint_counts]
    class_names = _class_names_for_keypoint_schema(categories, keypoint_counts, num_keypoints_per_class)
    num_keypoints_per_class.extend([0] * (len(class_names) - len(num_keypoints_per_class)))
    return CocoKeypointSchema(
        class_names=class_names,
        num_keypoints_per_class=num_keypoints_per_class,
        keypoint_oks_sigmas=[keypoint_oks_sigma] * active_keypoint_counts[0],
    )
