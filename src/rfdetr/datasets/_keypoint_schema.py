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

__all__ = [
    "CocoKeypointSchema",
    "KeypointSchema",
    "YoloKeypointSchema",
    "active_keypoint_counts",
    "infer_coco_keypoint_schema",
    "infer_yolo_keypoint_schema",
]


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


@dataclass(frozen=True, slots=True)
class YoloKeypointSchema:
    """Keypoint schema inferred from an Ultralytics YOLO pose YAML file.

    Args:
        class_names: Class names ordered by YOLO class id.
        num_keypoints_per_class: Number of keypoints for each YOLO class slot.
        keypoint_oks_sigmas: Default OKS sigmas matching the global keypoint count.
        keypoint_names: Keypoint names ordered by keypoint index.
        flip_idx: Optional Ultralytics horizontal-flip index mapping.
        keypoint_dim: Number of keypoint dimensions in label files, either 2 or 3.

    Returns:
        Immutable schema value used to configure YOLO pose training.

    Raises:
        This value object does not raise.

    Example:
        >>> YoloKeypointSchema(["person"], [1], [0.1], ["nose"], [], 3).keypoint_dim
        3
    """

    class_names: list[str]
    num_keypoints_per_class: list[int]
    keypoint_oks_sigmas: list[float]
    keypoint_names: list[str]
    flip_idx: list[int]
    keypoint_dim: int


# Public union alias covering both concrete schema types.
KeypointSchema = CocoKeypointSchema | YoloKeypointSchema


def _load_yaml_mapping(yaml_path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping root.

    Args:
        yaml_path: Path to a YAML data file.

    Returns:
        Parsed YAML mapping.

    Raises:
        ValueError: If the YAML root is not a mapping.
        OSError: If the file cannot be read.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "data.yaml"
        >>> _ = path.write_text("names: [person]\\nkpt_shape: [1, 3]\\n", encoding="utf-8")
        >>> sorted(_load_yaml_mapping(path))
        ['kpt_shape', 'names']
    """
    import yaml

    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in data file {str(yaml_path)!r}, got {type(data).__name__}.")
    return data


def _extract_yolo_class_names_from_data(data: dict[str, Any], data_file: Path) -> list[str]:
    """Extract contiguous YOLO class names from parsed YAML data."""
    names = data.get("names")
    if isinstance(names, dict):
        numeric_keys: list[int] = []
        non_numeric_keys: list[Any] = []
        for key in names.keys():
            key_str = str(key)
            if key_str.isdigit():
                numeric_keys.append(int(key_str))
            else:
                non_numeric_keys.append(key)

        unique_sorted_keys = sorted(set(numeric_keys))
        if not unique_sorted_keys or unique_sorted_keys != list(range(len(unique_sorted_keys))) or non_numeric_keys:
            raise ValueError(
                "Unsupported 'names' mapping in data file "
                f"{str(data_file)!r}: expected integer keys 0..N-1 with no gaps."
            )
        return [str(names[idx]) for idx in unique_sorted_keys]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Expected 'names' to be a list or dict in {str(data_file)!r}, got {type(names).__name__}.")


def _validate_yolo_kpt_shape(raw_kpt_shape: Any, data_file: Path) -> tuple[int, int]:
    """Validate and normalize a YOLO pose ``kpt_shape`` entry."""
    if not isinstance(raw_kpt_shape, (list, tuple)) or len(raw_kpt_shape) != 2:
        raise ValueError(f"YOLO pose data file {str(data_file)!r} must define kpt_shape as [num_keypoints, 2_or_3].")
    try:
        num_keypoints = int(raw_kpt_shape[0])
        keypoint_dim = int(raw_kpt_shape[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YOLO pose data file {str(data_file)!r} has invalid kpt_shape={raw_kpt_shape!r}; expected integer values."
        ) from exc
    if num_keypoints <= 0 or keypoint_dim not in (2, 3):
        raise ValueError(
            f"YOLO pose data file {str(data_file)!r} has invalid kpt_shape={raw_kpt_shape!r}; "
            "expected [positive_num_keypoints, 2_or_3]."
        )
    return num_keypoints, keypoint_dim


def _extract_yolo_keypoint_names(data: dict[str, Any], num_keypoints: int) -> list[str]:
    """Extract YOLO keypoint names or synthesize stable placeholders."""
    raw_kpt_names = data.get("kpt_names")
    keypoint_names: Any = None
    if isinstance(raw_kpt_names, dict) and raw_kpt_names:
        keypoint_names = raw_kpt_names.get(0, raw_kpt_names.get("0"))
    elif isinstance(raw_kpt_names, list):
        keypoint_names = raw_kpt_names

    if keypoint_names is None:
        return [f"keypoint_{idx}" for idx in range(num_keypoints)]
    if not isinstance(keypoint_names, list) or len(keypoint_names) != num_keypoints:
        raise ValueError(
            f"YOLO pose kpt_names length must match kpt_shape keypoint count {num_keypoints}, got {keypoint_names!r}."
        )
    return [str(name) for name in keypoint_names]


def _extract_yolo_flip_idx(data: dict[str, Any], num_keypoints: int) -> list[int]:
    """Extract and validate optional YOLO ``flip_idx`` metadata."""
    raw_flip_idx = data.get("flip_idx")
    if raw_flip_idx is None:
        return []
    if not isinstance(raw_flip_idx, list) or len(raw_flip_idx) != num_keypoints:
        raise ValueError(f"YOLO pose flip_idx must contain {num_keypoints} integer indexes.")
    try:
        flip_idx = [int(idx) for idx in raw_flip_idx]
    except (TypeError, ValueError) as exc:
        raise ValueError("YOLO pose flip_idx must contain integer indexes.") from exc
    if sorted(flip_idx) != list(range(num_keypoints)):
        raise ValueError(f"YOLO pose flip_idx must be a permutation of 0..{num_keypoints - 1}.")
    return flip_idx


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


def infer_yolo_keypoint_schema(
    data_file: str | Path,
    *,
    keypoint_oks_sigma: float = 0.1,
) -> YoloKeypointSchema:
    """Infer a keypoint schema from an Ultralytics YOLO pose YAML file.

    Args:
        data_file: Path to a YOLO ``data.yaml`` or ``data.yml`` file.
        keypoint_oks_sigma: Default OKS sigma to repeat for the global keypoint count.

    Returns:
        Class-aligned keypoint counts plus optional YOLO keypoint names and flip indexes.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required YOLO pose fields are missing or malformed.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "data.yaml"
        >>> _ = path.write_text("names: [person]\\nkpt_shape: [1, 3]\\n", encoding="utf-8")
        >>> infer_yolo_keypoint_schema(path).num_keypoints_per_class
        [1]
    """
    path = Path(data_file)
    data = _load_yaml_mapping(path)
    class_names = _extract_yolo_class_names_from_data(data, path)
    if "kpt_shape" not in data:
        raise ValueError(f"YOLO pose data file {str(path)!r} is missing required kpt_shape metadata.")
    num_keypoints, keypoint_dim = _validate_yolo_kpt_shape(data["kpt_shape"], path)
    keypoint_names = _extract_yolo_keypoint_names(data, num_keypoints)
    flip_idx = _extract_yolo_flip_idx(data, num_keypoints)
    return YoloKeypointSchema(
        class_names=class_names,
        num_keypoints_per_class=[num_keypoints] * len(class_names),
        keypoint_oks_sigmas=[keypoint_oks_sigma] * num_keypoints,
        keypoint_names=keypoint_names,
        flip_idx=flip_idx,
        keypoint_dim=keypoint_dim,
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
