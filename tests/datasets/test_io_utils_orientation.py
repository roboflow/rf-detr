# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""EXIF orientation checks of ``rfdetr.datasets.io_utils`` as the COCO-style readers use it.

Covers the warning for boxes drawn on the stored pixels of a quarter-turned photo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.io_utils import _boxes_fit_only_turned
from rfdetr.datasets.webdataset.load import WebDatasetDetection
from rfdetr.datasets.webdataset.pack import pack_coco_to_shards
from rfdetr.utilities.imports import _IS_WEBDATASET_INSTALLED
from tests._oriented_photo import write_oriented_photo

_requires_webdataset = pytest.mark.skipif(not _IS_WEBDATASET_INSTALLED, reason="webdataset is not installed")

#: Part of the warning for boxes that fit a quarter-turned photo only as its file stores the pixels.
_STORED_FRAME_WARNING = "boxes fit the photo only as the file stores the pixels"

# The photos below are stored 160 x 100. Orientation 6 shows them 100 x 160; orientations 1 and 3 keep 160 x 100.
#: Right edge at 150: past the upright width of 100, inside the stored width of 160.
_STORED_FRAME_BOX = [100.0, 10.0, 50.0, 30.0]
#: Bottom edge at 140: inside the upright height of 160.
_UPRIGHT_BOX = [10.0, 100.0, 50.0, 40.0]
#: Bottom edge at 140: past a 160 x 100 photo, inside it turned a quarter.
_TURNED_FRAME_BOX = [10.0, 110.0, 20.0, 30.0]


def _write_tagged_coco_split(root: Path, orientation: int, bbox: list[float], count: int = 2) -> tuple[Path, Path]:
    """Write ``count`` JPEG photos stored 160 x 100 and tagged ``orientation``, each with one box, as a COCO split.

    Args:
        root: Directory to write the ``images`` folder and ``annotations.json`` into.
        orientation: EXIF orientation of every photo.
        bbox: COCO ``[x, y, width, height]`` box of every photo.
        count: Number of photos.

    Returns:
        The image directory and the annotation file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     image_dir, ann_file = _write_tagged_coco_split(Path(tmp), 6, _UPRIGHT_BOX)
        ...     sorted(path.name for path in image_dir.iterdir()), ann_file.name
        (['photo_0.jpg', 'photo_1.jpg'], 'annotations.json')
    """
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for index in range(count):
        name = f"photo_{index}.jpg"
        write_oriented_photo(image_dir / name, orientation, width=160, height=100)
        images.append({"id": index + 1, "file_name": name, "width": 160, "height": 100})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )
    ann_file = root / "annotations.json"
    ann_file.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "carton"}]})
    )
    return image_dir, ann_file


class TestBoxesFitOnlyTurned:
    """Only a clear overrun of the upright photo that fits it turned counts as a sign of stored-frame labels."""

    @pytest.mark.parametrize(
        "annotations",
        [
            pytest.param([{"bbox": [50.0, 10.0, 50.5, 20.0]}], id="overrun-within-tolerance"),
            pytest.param([{"bbox": _STORED_FRAME_BOX}, {"bbox": [1.0, 2.0, 3.0]}], id="malformed-box"),
            pytest.param([], id="no-boxes"),
        ],
    )
    def test_no_sign_without_a_clear_overrun(self, annotations: list[dict[str, Any]]) -> None:
        """Rounding at the edge, a box that is not four numbers, and an empty sample give no sign."""
        assert not _boxes_fit_only_turned(annotations, (100, 160))


class TestStoredFrameLabelWarning:
    """The readers warn once when boxes fit a quarter-turned photo only as its file stores the pixels."""

    def test_coco_labels_on_stored_pixels_warn_once(self, tmp_path: Path) -> None:
        """Two such samples give one warning per dataset, naming the first photo."""
        image_dir, ann_file = _write_tagged_coco_split(tmp_path, 6, _STORED_FRAME_BOX)
        dataset = CocoDetection(image_dir, ann_file, transforms=None, remap_category_ids=True)

        with pytest.warns(UserWarning, match=_STORED_FRAME_WARNING) as record:
            [dataset[index] for index in range(len(dataset))]

        messages = [str(warning.message) for warning in record if _STORED_FRAME_WARNING in str(warning.message)]
        assert len(messages) == 1
        assert "photo_0.jpg" in messages[0]

    def test_coco_upright_labels_do_not_warn(self, tmp_path: Path, recwarn: pytest.WarningsRecorder) -> None:
        """Boxes drawn on the upright photo fit it, so the reader reads no header and says nothing."""
        image_dir, ann_file = _write_tagged_coco_split(tmp_path, 6, _UPRIGHT_BOX)
        dataset = CocoDetection(image_dir, ann_file, transforms=None, remap_category_ids=True)

        [dataset[index] for index in range(len(dataset))]

        assert not [warning for warning in recwarn if _STORED_FRAME_WARNING in str(warning.message)]

    @pytest.mark.parametrize("orientation", [1, 3])
    def test_coco_photo_without_a_quarter_turn_does_not_warn(
        self, tmp_path: Path, recwarn: pytest.WarningsRecorder, orientation: int
    ) -> None:
        """A box past the photo that would fit it turned is no sign when the photo is not turned a quarter."""
        image_dir, ann_file = _write_tagged_coco_split(tmp_path, orientation, _TURNED_FRAME_BOX)
        dataset = CocoDetection(image_dir, ann_file, transforms=None, remap_category_ids=True)

        [dataset[index] for index in range(len(dataset))]

        assert not [warning for warning in recwarn if _STORED_FRAME_WARNING in str(warning.message)]

    @_requires_webdataset
    def test_webdataset_labels_on_stored_pixels_warn_once(self, tmp_path: Path) -> None:
        """A packed split warns the same way when its shards are streamed, naming the sample."""
        image_dir, ann_file = _write_tagged_coco_split(tmp_path, 6, _STORED_FRAME_BOX)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, ann_file, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)

        with pytest.warns(UserWarning, match=_STORED_FRAME_WARNING) as record:
            list(dataset)

        messages = [str(warning.message) for warning in record if _STORED_FRAME_WARNING in str(warning.message)]
        assert len(messages) == 1
        assert messages[0].startswith("WebDataset sample ")
