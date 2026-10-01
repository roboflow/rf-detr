# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the COCO val2017 accuracy helpers the export cookbooks score each exported model with.

Everything runs on a tiny synthetic COCO file written to ``tmp_path``; no network access and no real model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from rfdetr.export._benchmark import (
    CocoValSubset,
    evaluate_coco_map,
    fetch_coco_val2017,
    select_coco_val_ids,
)

#: Ground truth per image: one box in pixel ``xywh`` and its sparse COCO category ID.
_GT_BOXES = {1: ([8.0, 8.0, 32.0, 24.0], 18), 2: ([20.0, 4.0, 16.0, 40.0], 90), 3: ([0.0, 0.0, 30.0, 30.0], 1)}
#: Each image gets its own ``(width, height)``, so a fake runtime can tell which image it was handed.
_IMAGE_SIZES = {1: (64, 64), 2: (64, 80), 3: (80, 64)}


def _write_coco(root: Path) -> CocoValSubset:
    """Write a three-image COCO ``instances_val2017.json`` plus blank JPEGs under *root*.

    Args:
        root: Directory that receives ``annotations/`` and ``val2017/``.

    Returns:
        A subset over all three images.

    Examples:
        >>> import tempfile
        >>> len(_write_coco(Path(tempfile.mkdtemp())).image_ids)
        3
    """
    images_dir, annotations_dir = root / "val2017", root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    images, annotations = [], []
    for image_id, (bbox, category_id) in _GT_BOXES.items():
        file_name = f"{image_id:012d}.jpg"
        width, height = _IMAGE_SIZES[image_id]
        Image.new("RGB", (width, height)).save(images_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "coco_url": f"http://images.cocodataset.org/val2017/{file_name}",
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "category_id": category_id,
                "iscrowd": 0,
            }
        )
    categories = [{"id": category_id, "name": str(category_id)} for category_id in (1, 18, 90)]
    annotations_path = annotations_dir / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return CocoValSubset(images_dir=images_dir, annotations_path=annotations_path, image_ids=(1, 2, 3))


def _image_id(image: Image.Image) -> int:
    """Recover the COCO image ID from the image's unique size.

    Examples:
        >>> _image_id(Image.new("RGB", (64, 80)))
        2
    """
    return next(image_id for image_id, size in _IMAGE_SIZES.items() if size == image.size)


def _gt_detections(image: Image.Image) -> sv.Detections:
    """Return the ground-truth box of *image* as already-decoded detections.

    Examples:
        >>> _gt_detections.__name__
        '_gt_detections'
    """
    (x, y, w, h), category_id = _GT_BOXES[_image_id(image)]
    return sv.Detections(
        xyxy=np.array([[x, y, x + w, y + h]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([category_id]),
    )


def _gt_raw_outputs(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``dets``/``labels`` arrays (batch of one) whose top query decodes to the ground-truth box.

    Boxes are normalized ``cxcywh`` and the 91-slot logits put the only confident score at the category's own slot,
    the way an official sparse-ID COCO checkpoint does.

    Examples:
        >>> _gt_raw_outputs.__name__
        '_gt_raw_outputs'
    """
    (x, y, w, h), category_id = _GT_BOXES[_image_id(image)]
    width, height = image.size
    boxes = np.full((1, 4, 4), 0.25, dtype=np.float32)
    boxes[0, 0] = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
    logits = np.full((1, 4, 91), -10.0, dtype=np.float32)
    logits[0, 0, category_id] = 5.0
    return boxes, logits


class TestSelectCocoValIds:
    """``select_coco_val_ids`` picks a seeded, reproducible subset of image IDs."""

    def test_same_seed_gives_same_ids(self, tmp_path: Path) -> None:
        """The same seed selects the same images, so every cookbook scores the identical subset.

        Cross-cookbook mAP comparisons only mean something when the evaluated images are the same everywhere.
        """
        subset = _write_coco(tmp_path)
        assert select_coco_val_ids(subset.annotations_path, 2, seed=3) == select_coco_val_ids(
            subset.annotations_path, 2, seed=3
        )

    def test_none_selects_every_image_sorted(self, tmp_path: Path) -> None:
        """``n_images=None`` is the full split, in image-ID order, for the ``FULL_VAL`` path."""
        subset = _write_coco(tmp_path)
        assert select_coco_val_ids(subset.annotations_path, None) == [1, 2, 3]

    def test_more_images_than_split_raises(self, tmp_path: Path) -> None:
        """Asking for more images than exist fails loudly instead of silently evaluating fewer."""
        subset = _write_coco(tmp_path)
        with pytest.raises(ValueError, match="only 3"):
            select_coco_val_ids(subset.annotations_path, 4)


class TestFetchCocoVal2017:
    """``fetch_coco_val2017`` downloads only the selected images that are not on disk yet."""

    def test_downloads_only_missing_images(self, tmp_path: Path) -> None:
        """An image already on disk is not fetched again; a missing one is fetched from its ``coco_url``.

        Cookbooks rerun this cell on every session, so a second run must not re-download the subset.
        """
        _write_coco(tmp_path)
        (tmp_path / "val2017" / "000000000002.jpg").unlink()
        with mock.patch("rfdetr.export._benchmark._download") as download:
            subset = fetch_coco_val2017(tmp_path, n_images=None)
        download.assert_called_once_with(
            "http://images.cocodataset.org/val2017/000000000002.jpg", tmp_path / "val2017" / "000000000002.jpg"
        )
        assert subset.image_ids == (1, 2, 3)


class TestEvaluateCocoMap:
    """``evaluate_coco_map`` scores one runtime's batch-1 outputs against COCO ground truth."""

    def test_perfect_decoded_detections_score_one(self, tmp_path: Path) -> None:
        """Already-decoded detections that equal the ground truth reach mAP 1.0 (the PyTorch ``predict()`` path)."""
        result = evaluate_coco_map(_gt_detections, _write_coco(tmp_path), progress=False)
        assert (result.map50_95, result.map50, result.n_images) == (1.0, 1.0, 3)

    def test_perfect_raw_outputs_score_one(self, tmp_path: Path) -> None:
        """Raw ``dets``/``labels`` decode with sparse COCO IDs, so category 90 is kept and the score is 1.0.

        The default ``background_class_id=None`` matters: the decoder's own default (``-1``) would drop slot 90 and the
        image whose only object is category 90 would score zero.
        """
        result = evaluate_coco_map(_gt_raw_outputs, _write_coco(tmp_path), progress=False)
        assert (result.map50_95, result.map50) == (1.0, 1.0)

    def test_no_detections_score_zero(self, tmp_path: Path) -> None:
        """A runtime that returns nothing scores 0.0 instead of crashing on an empty result set."""
        result = evaluate_coco_map(lambda image: sv.Detections.empty(), _write_coco(tmp_path), progress=False)
        assert (result.map50_95, result.map50) == (0.0, 0.0)
