# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for explicit parquet label mapping."""

from rfdetr.datasets.parquet_bbox import _normalize_label_mapping


def test_label_mapping_is_ordered_by_category_id():
    """Category ids should map to contiguous labels in sorted id order."""
    category_to_label, class_names = _normalize_label_mapping({10: "table", 2: "text"})

    assert category_to_label == {2: 0, 10: 1}
    assert class_names == ["text", "table"]
