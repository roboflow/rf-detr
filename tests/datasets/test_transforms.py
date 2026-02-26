# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Optional

import pytest
import torch
from PIL import Image

from rfdetr.datasets.transforms import RandomResize, RandomSizeCrop, SquareResize


def test_random_resize_max_size_matches_legacy_behavior_without_target() -> None:
    """RandomResize should keep short side at requested size when max_size is not exceeded."""
    transform = RandomResize([312], max_size=1333)
    image = Image.new("RGB", (640, 480))

    image_out, target_out = transform(image, None)

    assert image_out.size == (416, 312)
    assert target_out is None


def test_random_resize_max_size_updates_target_mask_shape() -> None:
    """RandomResize should keep image/target size in sync for segmentation targets."""
    transform = RandomResize([312], max_size=1333)
    image = Image.new("RGB", (640, 480))
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 110.0, 120.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.ones((1, 480, 640), dtype=torch.bool),
        "size": torch.tensor([480, 640], dtype=torch.int64),
        "orig_size": torch.tensor([480, 640], dtype=torch.int64),
    }

    image_out, target_out = transform(image, target)

    assert image_out.size == (416, 312)
    assert target_out["size"].tolist() == [312, 416]
    assert tuple(target_out["masks"].shape) == (1, 312, 416)


def test_square_resize_empty_masks_shape_updated() -> None:
    """Empty masks must adopt the new spatial dimensions after a resize transform.

    Regression test: when an image has zero annotations, the empty masks tensor
    previously kept its original dimensions. This caused torch.cat in the matcher
    to fail when batching images with different original sizes.
    """
    transform = SquareResize([480])
    image = Image.new("RGB", (640, 312))  # non-square original
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, 312, 640), dtype=torch.bool),
        "size": torch.tensor([312, 640], dtype=torch.int64),
        "orig_size": torch.tensor([312, 640], dtype=torch.int64),
    }

    image_out, target_out = transform(image, target)

    assert image_out.size == (480, 480)
    assert tuple(target_out["masks"].shape) == (0, 480, 480), (
        f"Empty masks should match resized image dims, got {tuple(target_out['masks'].shape)}"
    )


def test_square_resize_non_empty_masks_shape_updated() -> None:
    """Non-empty masks must adopt the new spatial dimensions after SquareResize."""
    transform = SquareResize([320])
    image = Image.new("RGB", (640, 480))
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 110.0, 120.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.ones((1, 480, 640), dtype=torch.bool),
        "size": torch.tensor([480, 640], dtype=torch.int64),
        "orig_size": torch.tensor([480, 640], dtype=torch.int64),
    }

    image_out, target_out = transform(image, target)

    assert image_out.size == (320, 320)
    assert tuple(target_out["masks"].shape) == (1, 320, 320)


def test_random_resize_empty_masks_shape_updated() -> None:
    """Empty masks must adopt the new dimensions after RandomResize.

    Parallel regression test to test_square_resize_empty_masks_shape_updated:
    RandomResize should also update empty mask shapes when the target has no annotations.
    """
    transform = RandomResize([312], max_size=1333)
    image = Image.new("RGB", (640, 480))
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, 480, 640), dtype=torch.bool),
        "size": torch.tensor([480, 640], dtype=torch.int64),
        "orig_size": torch.tensor([480, 640], dtype=torch.int64),
    }

    image_out, target_out = transform(image, target)

    assert image_out.size == (416, 312)
    assert tuple(target_out["masks"].shape) == (0, 312, 416), (
        f"Empty masks should match resized image dims, got {tuple(target_out['masks'].shape)}"
    )


def test_random_size_crop_updates_target_size() -> None:
    """RandomSizeCrop should update target size and mask dimensions after cropping."""
    transform = RandomSizeCrop(min_size=100, max_size=200)
    image = Image.new("RGB", (640, 480))
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 110.0, 120.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.ones((1, 480, 640), dtype=torch.bool),
        "size": torch.tensor([480, 640], dtype=torch.int64),
        "orig_size": torch.tensor([480, 640], dtype=torch.int64),
    }

    image_out, target_out = transform(image, target)

    out_w, out_h = image_out.size
    assert 100 <= out_w <= 200
    assert 100 <= out_h <= 200
    assert target_out["size"].tolist() == [out_h, out_w]
    assert tuple(target_out["masks"].shape)[1:] == (out_h, out_w)


@pytest.mark.parametrize(
    "image_size, short_side, max_size, expected",
    [
        pytest.param((640, 480), 312, None, 312, id="no_max_size_returns_unchanged"),
        pytest.param((640, 480), 800, 1333, 800, id="constraint_not_needed"),
        # Wide image (1280x480): scale=1280/480*800=2133>1333 → clamped to int(round(1333*480/1280))=500
        pytest.param((1280, 480), 800, 1333, 500, id="constraint_clamps_short_side"),
    ],
)
def test_get_constrained_short_side(
    image_size: tuple[int, int], short_side: int, max_size: Optional[int], expected: int
) -> None:
    """_get_constrained_short_side should clamp to max_size constraint when needed."""
    result = RandomResize._get_constrained_short_side(image_size, short_side, max_size)
    assert result == expected
