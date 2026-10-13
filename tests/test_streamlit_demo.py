# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the optional Streamlit demo helpers."""

from unittest.mock import Mock

import numpy as np
import supervision as sv
from PIL import Image

from app import detect_image, filter_small_detections, get_class_names


def test_get_class_names_preserves_names_for_sparse_ids() -> None:
    """Use RF-DETR's prediction names rather than indexing by sparse COCO IDs."""
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20]]),
        class_id=np.array([17]),
        data={"class_name": np.array(["cat"])},
    )

    assert get_class_names(detections) == ["cat"]


def test_get_class_names_falls_back_to_class_ids() -> None:
    """Return class IDs as readable labels when prediction names are absent."""
    detections = sv.Detections(xyxy=np.array([[0, 0, 20, 20]]), class_id=np.array([17]))

    assert get_class_names(detections) == ["17"]


def test_filter_small_detections_keeps_only_boxes_meeting_both_dimensions() -> None:
    """Filter boxes independently by width and height."""
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20], [0, 0, 20, 8], [0, 0, 8, 20]]),
        class_id=np.array([1, 2, 3]),
    )

    filtered = filter_small_detections(detections, min_width=12, min_height=12)

    assert filtered.class_id.tolist() == [1]


def test_filter_small_detections_returns_empty_input_unchanged() -> None:
    """Avoid indexing when there are no detections to filter."""
    detections = sv.Detections.empty()

    assert filter_small_detections(detections) is detections


def test_detect_image_passes_pil_image_to_model() -> None:
    """Run inference on the uploaded image directly, without temporary files."""
    image = Image.new("RGB", (32, 32))
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20]]),
        confidence=np.array([0.9]),
        class_id=np.array([17]),
        data={"class_name": np.array(["cat"])},
    )
    model = Mock()
    model.predict.return_value = detections

    annotated_image, actual_detections = detect_image(image, 0.4, model)

    model.predict.assert_called_once_with(image, threshold=0.4)
    np.testing.assert_array_equal(actual_detections.class_id, detections.class_id)
    assert annotated_image.shape == (32, 32, 3)
