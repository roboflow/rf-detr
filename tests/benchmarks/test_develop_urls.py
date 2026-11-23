# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""No-network tests for private COCO developer helper URL selection."""

import pytest

from rfdetr.datasets._develop import _COCO_URLS, get_coco_download_url


@pytest.mark.parametrize(
    ("split", "archive_name"),
    [("train2017", "train2017.zip"), ("val2017", "val2017.zip"), ("annotations", "annotations_trainval2017.zip")],
)
def test_coco_helper_url_selection(split: str, archive_name: str) -> None:
    """Each COCO split resolves to its official archive URL from ``_COCO_URLS``."""
    assert get_coco_download_url(split) == _COCO_URLS[split]
    assert get_coco_download_url(split).endswith(f"/{archive_name}")
