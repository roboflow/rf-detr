# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""No-network tests for private COCO developer helper URL selection."""

from rfdetr.datasets._develop import _COCO_URLS, get_coco_download_url


def test_coco_helper_train2017_url_selection() -> None:
    """``train2017`` should resolve to the official COCO train archive URL."""
    assert get_coco_download_url("train2017") == _COCO_URLS["train2017"]
    assert get_coco_download_url("train2017").endswith("/train2017.zip")


def test_coco_helper_val2017_url_selection() -> None:
    """``val2017`` should resolve to the official COCO val archive URL."""
    assert get_coco_download_url("val2017") == _COCO_URLS["val2017"]
    assert get_coco_download_url("val2017").endswith("/val2017.zip")


def test_coco_helper_annotations_url_selection() -> None:
    """``annotations`` should resolve to the COCO train/val annotations archive URL."""
    assert get_coco_download_url("annotations") == _COCO_URLS["annotations"]
    assert get_coco_download_url("annotations").endswith("/annotations_trainval2017.zip")
