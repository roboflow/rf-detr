# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Utilities for mapping prediction class IDs to names."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
from rfdetr.utilities.keypoints import _is_bg_first_schema

if TYPE_CHECKING:
    from rfdetr.config import ModelConfig
    from rfdetr.detr import RFDETR


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


class PredictionLabels(NamedTuple):
    """Hold the class layout that a model's predictions use.

    Attributes:
        class_names: Names for active classes in model order.
        num_classes: Number of logit slots in the model.
        num_keypoints_per_class: Keypoint counts for each model slot; empty for non-keypoint models.
        class_id_to_name: Mapping from prediction class IDs to class names.
    """

    class_names: list[str]
    num_classes: int
    num_keypoints_per_class: list[int]
    class_id_to_name: dict[int, str]


def prediction_labels(model: RFDETR) -> PredictionLabels:
    """Derive the class layout of a live model's predictions from its names and training arguments.

    Native prediction and export metadata both read the layout here, so an exported artifact maps class IDs to the
    same names that ``predict()`` returns. A model context without ``args`` takes its logit-slot count and keypoint
    schema from ``model_config``, which describes the head that was built. Its name mapping still uses one slot per
    name to keep class IDs 0-indexed even for COCO names, independently of the head's class count.

    Args:
        model: Live RF-DETR wrapper whose ``class_names``, ``model.args`` and ``model_config`` define the layout.

    Returns:
        Class names, logit-slot count, keypoint schema, and the class-ID-to-name map built from them.

    Examples:
        >>> from types import SimpleNamespace
        >>> config = SimpleNamespace(num_keypoints_per_class=[])
        >>> args = SimpleNamespace(num_classes=3, num_keypoints_per_class=[0, 17, 4])
        >>> trained = SimpleNamespace(args=args)
        >>> names = ["person", "car"]
        >>> labels = prediction_labels(SimpleNamespace(class_names=names, model=trained, model_config=config))
        >>> labels.num_classes, labels.class_id_to_name
        (3, {1: 'person', 2: 'car'})
        >>> bare = SimpleNamespace()
        >>> labels = prediction_labels(SimpleNamespace(class_names=COCO_CLASS_NAMES, model=bare, model_config=config))
        >>> labels.num_classes, labels.num_keypoints_per_class, labels.class_id_to_name[1]
        (80, [], 'bicycle')
    """
    names = list(model.class_names)
    args = getattr(model.model, "args", None)
    config = getattr(model, "model_config", None)
    num_classes = int(getattr(args, "num_classes", getattr(config, "num_classes", len(names))))
    # Missing training args historically mean zero-indexed names, even when the head has extra COCO slots.
    mapping_num_classes = int(getattr(args, "num_classes", len(names)))
    schema = getattr(args, "num_keypoints_per_class", None)
    if schema is None:
        schema = getattr(config, "num_keypoints_per_class", None)
    keypoint_schema = [int(count) for count in (schema or [])]
    mapping = {
        int(class_id): name for class_id, name in class_id_to_name(names, mapping_num_classes, keypoint_schema).items()
    }
    return PredictionLabels(names, num_classes, keypoint_schema, mapping)


def _task_of(config: ModelConfig) -> Literal["detect", "segment", "keypoints"]:
    """Identify the prediction task from the configured model heads.

    Args:
        config: Model configuration defining the segmentation and keypoint heads.

    Returns:
        The prediction task, giving segmentation precedence over keypoints.

    Examples:
        >>> from types import SimpleNamespace
        >>> _task_of(SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False))
        'detect'
        >>> _task_of(SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=True))
        'keypoints'
        >>> _task_of(SimpleNamespace(segmentation_head=True, use_grouppose_keypoints=False))
        'segment'
        >>> _task_of(SimpleNamespace(segmentation_head=True, use_grouppose_keypoints=True))
        'segment'
    """
    if config.segmentation_head:
        return "segment"
    if config.use_grouppose_keypoints:
        return "keypoints"
    return "detect"
