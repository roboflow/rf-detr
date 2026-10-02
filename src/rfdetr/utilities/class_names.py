# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Utilities for mapping prediction class IDs to names."""

from __future__ import annotations

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
from rfdetr.utilities.keypoints import _is_bg_first_schema


def is_coco_pretrained(class_names: list[str], num_classes: int) -> bool:
    """Identify the sparse COCO class layout used by released detection models.

    Args:
        class_names: Names in model order.
        num_classes: Number of logit slots.

    Returns:
        Whether class IDs follow the sparse COCO category map.

    Examples:
        >>> is_coco_pretrained(COCO_CLASS_NAMES, 90)
        True
        >>> is_coco_pretrained(["cat", "dog"], 2)
        False
    """
    return num_classes > len(class_names) and class_names == COCO_CLASS_NAMES


def class_id_to_name(
    class_names: list[str],
    num_classes: int,
    num_keypoints_per_class: list[int],
) -> dict[int, str]:
    """Map model class IDs to names for COCO, keypoint, and regular models.

    Args:
        class_names: Names for active classes in model order.
        num_classes: Number of logit slots in the model.
        num_keypoints_per_class: Keypoint counts for each model slot.

    Returns:
        A mapping from prediction class IDs to class names.

    Examples:
        >>> class_id_to_name(["cat", "dog"], 2, [])
        {0: 'cat', 1: 'dog'}
        >>> class_id_to_name(["person"], 2, [0, 17])
        {1: 'person'}
        >>> class_id_to_name(COCO_CLASS_NAMES, 90, [])[18]
        'dog'
    """
    if is_coco_pretrained(class_names, num_classes):
        return {
            category_id: class_names[index]
            for index, category_id in enumerate(COCO_CLASSES)
            if index < len(class_names)
        }
    if _is_bg_first_schema(num_keypoints_per_class):
        active_slots = [slot for slot, count in enumerate(num_keypoints_per_class) if count > 0]
        return {slot: class_names[index] for index, slot in enumerate(active_slots) if index < len(class_names)}
    return dict(enumerate(class_names))
