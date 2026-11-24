# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for how the WebDataset reader turns a shard member tagged with an EXIF orientation upright."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfdetr.datasets.webdataset.load import WebDatasetDetection
from rfdetr.datasets.webdataset.pack import pack_coco_to_shards
from rfdetr.utilities.imports import _IS_WEBDATASET_INSTALLED

_requires_webdataset = pytest.mark.skipif(
    not _IS_WEBDATASET_INSTALLED, reason="webdataset not installed; skip streaming tests (optional `data` extra)"
)


@_requires_webdataset
class TestWebDatasetExifOrientation:
    """A phone-tagged shard member streams upright, as the loose-file readers and the labels see the photo."""

    def test_tagged_member_is_upright_and_its_draft_scales_follow_the_upright_axes(self, tmp_path: Path) -> None:
        """``draft_size`` rescales the boxes of a quarter-turned member by the upright axes' decode scales.

        The stored 100 x 37 JPEG, tagged as a quarter turn, drafts by 2 to 50 x 19; the upright 37 x 100 photo comes
        back 19 x 50, so the scales reaching the annotation rescale are ``(19 / 37, 0.5)``, not the stored axes'.
        """
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        upright = np.zeros((100, 37, 3), dtype=np.uint8)
        upright[20:60, 8:28] = 255
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90).save(
            image_dir / "img_0000.jpg", quality=95, exif=exif.tobytes()
        )
        payload = {
            "images": [{"id": 1000, "file_name": "img_0000.jpg", "height": 100, "width": 37}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 1000,
                    "category_id": 3,
                    "bbox": [8.0, 20.0, 20.0, 40.0],
                    "area": 800.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 3, "name": "cat"}],
        }
        annotations = tmp_path / "annotations.json"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")

        ((image, target),) = list(
            WebDatasetDetection(shard_dir, "train", transforms=None, cat2label={3: 0}, draft_size=16)
        )

        assert image.size == (19, 50)
        np.testing.assert_allclose(
            target["boxes"][0].numpy(), [8 * 19 / 37, 20 * 0.5, 28 * 19 / 37, 60 * 0.5], atol=1e-3
        )
