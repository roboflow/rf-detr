# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Albumentations augmentation wrappers."""

import json

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from rfdetr import RFDETRSegNano
from rfdetr.datasets._develop import _SimpleDataset
from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, _build_albu_transform
from rfdetr.util.misc import collate_fn


class TestAlbumentationsWrapper:
    """Tests for AlbumentationsWrapper class."""

    @pytest.mark.parametrize(
        "transform_class,params,box_in,box_out",
        [
            (A.HorizontalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [70.0, 20.0, 90.0, 40.0]),
            (A.VerticalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [10.0, 60.0, 30.0, 80.0]),
        ],
    )
    def test_flip_transforms_with_boxes(self, transform_class, params, box_in, box_out):
        """Test flip transforms correctly transform bounding boxes."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([box_in]), "labels": torch.tensor([1])}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert torch.allclose(aug_target["boxes"], torch.tensor([box_out]), atol=1.0)
        assert torch.equal(aug_target["labels"], target["labels"])

    def test_non_geometric_transform_preserves_boxes(self):
        """Test that non-geometric transforms preserve bounding boxes."""
        transform = A.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "labels": torch.tensor([1])}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Boxes should be unchanged
        assert torch.equal(aug_target["boxes"], target["boxes"])
        assert torch.equal(aug_target["labels"], target["labels"])

    def test_empty_boxes_handling(self):
        """Test wrapper handles empty boxes correctly."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)

    def test_multiple_boxes(self):
        """Test wrapper handles multiple bounding boxes."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                ]
            ),
            "labels": torch.tensor([1, 2]),
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (2, 4)
        assert aug_target["labels"].shape == (2,)
        assert torch.equal(aug_target["labels"], target["labels"])

    def test_none_target_inference_mode(self):
        """Test wrapper accepts None target for inference (no ground-truth annotations)."""
        transform = A.Resize(height=64, width=64)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        aug_image, aug_target = wrapper(image, None)

        assert isinstance(aug_image, Image.Image)
        assert aug_image.size == (64, 64)
        assert aug_target is None

    def test_invalid_target_type(self):
        """Test wrapper raises error for invalid target type."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))

        with pytest.raises(TypeError, match="target must be a dictionary"):
            wrapper(image, "invalid_target")

    def test_missing_labels_key(self):
        """Test wrapper raises error when labels key is missing."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]])}

        with pytest.raises(KeyError, match="target must contain 'labels' key"):
            wrapper(image, target)

    def test_invalid_boxes_shape(self):
        """Test wrapper raises error for invalid boxes shape."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([10.0, 20.0, 30.0]),  # Invalid shape
            "labels": torch.tensor([1]),
        }

        with pytest.raises(ValueError, match="boxes must have shape"):
            wrapper(image, target)

    def test_orig_size_preserved_with_two_boxes(self):
        """Test that orig_size is not treated as per-instance field when num_boxes=2.

        Regression test for bug where orig_size (shape [2] for [h, w]) was incorrectly
        treated as a per-instance field when there were exactly 2 boxes, causing
        orig_size to be filtered/indexed incorrectly and leading to inconsistent
        tensor shapes in batches.
        """
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (640, 480))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0], [300.0, 100.0, 500.0, 400.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2]),
            "orig_size": torch.tensor([480, 640]),  # shape [2], same as num_boxes!
            "size": torch.tensor([480, 640]),
            "image_id": torch.tensor([123]),
            "area": torch.tensor([100.0, 200.0]),
            "iscrowd": torch.tensor([0, 0]),
        }

        aug_image, aug_target = wrapper(image, target)

        # Verify orig_size is still [2] elements (h, w), not filtered as per-instance
        assert aug_target["orig_size"].shape == torch.Size([2]), (
            f"orig_size should have shape [2], got {aug_target['orig_size'].shape}"
        )
        assert torch.equal(aug_target["orig_size"], target["orig_size"]), "orig_size should be unchanged"

        # Verify other global fields are also preserved
        assert aug_target["size"].shape == torch.Size([2])
        assert aug_target["image_id"].shape == torch.Size([1])
        assert torch.equal(aug_target["image_id"], target["image_id"])

    def test_orig_size_preserved_with_two_boxes_and_masks(self):
        """Test that orig_size and masks are handled correctly when num_boxes=2.

        Critical regression test: With 2 boxes, both orig_size and masks have
        first dimension = 2, but they must be treated differently:
        - orig_size (shape [2]): global field, should NOT be filtered
        - masks (shape [2, H, W]): per-instance field, SHOULD be transformed
        """
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (640, 480))
        # Create masks for 2 boxes (use uint8 for Albumentations compatibility)
        masks = torch.zeros((2, 480, 640), dtype=torch.uint8)
        masks[0, 50:150, 50:150] = 1  # Mask for first box
        masks[1, 200:300, 300:500] = 1  # Mask for second box

        target = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0], [300.0, 100.0, 500.0, 400.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2]),
            "masks": masks,  # shape [2, 480, 640], same first dim as orig_size!
            "orig_size": torch.tensor([480, 640]),  # shape [2]
            "size": torch.tensor([480, 640]),
            "image_id": torch.tensor([123]),
            "area": torch.tensor([100.0, 200.0]),
            "iscrowd": torch.tensor([0, 0]),
        }

        aug_image, aug_target = wrapper(image, target)

        # Verify orig_size is preserved (global field)
        assert aug_target["orig_size"].shape == torch.Size([2]), (
            f"orig_size should have shape [2], got {aug_target['orig_size'].shape}"
        )
        assert torch.equal(aug_target["orig_size"], target["orig_size"]), "orig_size should be unchanged"

        # Verify masks are transformed (per-instance field)
        assert aug_target["masks"].shape == torch.Size([2, 480, 640]), (
            f"masks should have shape [2, 480, 640], got {aug_target['masks'].shape}"
        )
        assert aug_target["masks"].dtype == torch.bool, "masks should be converted to bool after transform"
        # Masks should be flipped - verify they're different
        assert not torch.equal(aug_target["masks"], target["masks"].bool()), (
            "masks should be transformed (flipped) for geometric transform"
        )

        # Verify we still have 2 boxes and 2 masks
        assert len(aug_target["boxes"]) == 2, "Should have 2 boxes after transform"
        assert len(aug_target["labels"]) == 2, "Should have 2 labels after transform"
        assert aug_target["masks"].shape[0] == 2, "Should have 2 masks after transform"

        # Verify other global fields are preserved
        assert aug_target["size"].shape == torch.Size([2])
        assert aug_target["image_id"].shape == torch.Size([1])
        assert torch.equal(aug_target["image_id"], target["image_id"])

    @pytest.mark.parametrize(
        "transform_class,params",
        [
            (A.HorizontalFlip, {"p": 1.0}),
            (A.VerticalFlip, {"p": 1.0}),
            (A.Rotate, {"limit": 45, "p": 1.0}),
        ],
    )
    def test_various_geometric_transforms(self, transform_class, params):
        """Test various geometric transforms work correctly."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "labels": torch.tensor([1])}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Albumentations can return multiple boxes for a single input box on some Python versions.
        assert aug_target["boxes"].shape[1] == 4
        assert aug_target["labels"].shape[0] == aug_target["boxes"].shape[0]
        assert aug_target["labels"].numel() >= 1

    def test_masks_transform_with_horizontal_flip(self):
        """Masks should be transformed consistently with boxes for geometric transforms."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        # Create test image (100x100)
        height, width = 100, 100
        image = Image.new("RGB", (width, height), color="red")

        # Single box and corresponding mask
        box = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # x1, y1, x2, y2
        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        # Fill the mask inside the box region
        x1, y1, x2, y2 = box[0].to(dtype=torch.long)
        masks[0, y1:y2, x1:x2] = 1

        target = {
            "boxes": box,
            "labels": torch.tensor([1]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "masks" in aug_target
        assert aug_target["masks"].shape[0] == aug_target["boxes"].shape[0]

        # Check that the transformed mask's bounding box matches the transformed box
        aug_mask = aug_target["masks"][0]
        ys, xs = torch.nonzero(aug_mask, as_tuple=True)
        assert ys.numel() > 0 and xs.numel() > 0
        mask_bbox = torch.tensor(
            [
                xs.min().item(),
                ys.min().item(),
                xs.max().item() + 1,
                ys.max().item() + 1,
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(mask_bbox, aug_target["boxes"][0].to(dtype=torch.float32), atol=1.0)

    @pytest.mark.parametrize(
        "transform_class,params",
        [
            (A.HorizontalFlip, {"p": 1.0}),
            (A.VerticalFlip, {"p": 1.0}),
            (A.Rotate, {"limit": 15, "p": 1.0}),  # Small angle to avoid boxes going out
        ],
    )
    def test_various_geometric_transforms_with_masks(self, transform_class, params):
        """Test various geometric transforms correctly transform masks."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # Create mask covering the box region (more centered to avoid edge issues with rotation)
        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 40:60, 40:60] = 1

        target = {
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([1]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "masks" in aug_target
        # Number of boxes may change with rotation (boxes can be removed if they go out of bounds)
        assert aug_target["masks"].shape[0] == aug_target["boxes"].shape[0]
        if aug_target["boxes"].shape[0] > 0:
            # Mask should still have content (not all zeros)
            assert aug_target["masks"].any()

    @pytest.mark.parametrize(
        "transform_class,params",
        [
            (A.GaussianBlur, {"blur_limit": 3, "p": 1.0}),
            (A.RandomBrightnessContrast, {"p": 1.0}),
            (A.GaussNoise, {"p": 1.0}),
        ],
    )
    def test_pixel_transforms_preserve_masks(self, transform_class, params):
        """Test pixel-level transforms preserve masks unchanged."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 20:40, 10:30] = 1

        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Pixel transforms should not modify masks
        assert torch.equal(aug_target["masks"], target["masks"])

    def test_multiple_masks_with_geometric_transform(self):
        """Test multiple masks are correctly transformed together."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # Two masks for two boxes
        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 10:30, 10:30] = 1  # First mask
        masks[1, 50:70, 50:70] = 1  # Second mask

        target = {
            "boxes": torch.tensor(
                [
                    [10.0, 10.0, 30.0, 30.0],
                    [50.0, 50.0, 70.0, 70.0],
                ]
            ),
            "labels": torch.tensor([1, 2]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_target["masks"].shape == (2, height, width)
        assert aug_target["boxes"].shape[0] == 2
        assert aug_target["labels"].shape[0] == 2

    def test_empty_masks_handling(self):
        """Test wrapper correctly handles empty masks (no 'masks' key when empty)."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # When boxes are empty, don't include masks field
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)

    def test_geometric_transform_with_empty_masks_tensor(self):
        """Test that a geometric transform does not crash when masks tensor is empty (0 instances).

        Regression test for: when a prior crop removes all annotations, target["masks"]
        has shape (0, H, W). Passing an empty list to albumentations raises
        ValueError: masks cannot be empty.
        """
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # Simulate what happens after RandomSizeCrop removes all annotations:
        # target["masks"] has shape (0, H, W)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "masks": torch.zeros((0, height, width), dtype=torch.uint8),
        }

        # Should not raise ValueError: masks cannot be empty
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)
        assert "masks" in aug_target
        assert aug_target["masks"].shape[0] == 0
        assert aug_target["masks"].dtype == torch.bool

    def test_pixel_transform_with_masks_no_boxes(self):
        """Test that pixel transforms work with masks but no boxes."""
        # Use a non-geometric transform which doesn't need boxes
        transform = A.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks_orig = torch.zeros((1, height, width), dtype=torch.uint8)
        masks_orig[0, 20:40, 10:30] = 1

        target = {
            "labels": torch.tensor([1]),
            "masks": masks_orig.clone(),  # No boxes!
        }

        aug_image, aug_target = wrapper(image, target)

        # Pixel transforms should preserve masks
        assert torch.equal(aug_target["masks"], masks_orig)

    def test_invalid_mask_shape_raises_error(self):
        """Test that invalid mask shape raises ValueError."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # Invalid mask shape (2D instead of 3D)
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros((height, width), dtype=torch.uint8),
        }

        with pytest.raises(ValueError, match="masks must have shape"):
            wrapper(image, target)

    @pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.float32])
    def test_mask_dtype_handling(self, mask_dtype):
        """Test wrapper handles different mask dtypes correctly (uint8, float32)."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((1, height, width), dtype=mask_dtype)
        masks[0, 20:40, 10:30] = 1

        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "masks" in aug_target
        # Output masks should be bool after Albumentations processing
        assert aug_target["masks"].dtype == torch.bool

    def test_masks_transform_with_dropped_boxes(self):
        """Test wrapper filters masks appropriately when boxes are dropped by transform."""
        # Use a crop transform to ensure a box is dropped
        # Original image 100x100
        # Box 1: [10, 10, 20, 20] (will be kept if we crop top-left)
        # Box 2: [80, 80, 90, 90] (will be dropped if we crop top-left to 50x50)
        transform = A.Crop(x_min=0, y_min=0, x_max=50, y_max=50, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 10:20, 10:20] = 1
        masks[1, 80:90, 80:90] = 1

        target = {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [80.0, 80.0, 90.0, 90.0]]),
            "labels": torch.tensor([1, 2]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert len(aug_target["boxes"]) == 1
        assert len(aug_target["labels"]) == 1
        assert "masks" in aug_target
        assert len(aug_target["masks"]) == 1
        assert aug_target["masks"].shape == (1, 50, 50)


class TestAlbumentationsWrapperFromConfig:
    """Tests for AlbumentationsWrapper.from_config() static method."""

    def test_build_from_valid_config(self):
        """Test building transforms from valid configuration."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.3},
        }

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 2
        assert all(isinstance(t, AlbumentationsWrapper) for t in transforms)
        # Validate transform names match config in correct order
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

    def test_build_from_empty_config(self):
        """Test building from empty config returns empty list."""
        config = {}

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 0

    def test_unknown_transform_skipped(self):
        """Test that unknown transforms are skipped with warning."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "NonExistentTransform": {"p": 0.5},
        }

        transforms = AlbumentationsWrapper.from_config(config)

        # Only valid transform should be included
        assert len(transforms) == 1
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == ["HorizontalFlip"]

    def test_invalid_params_skipped(self):
        """Test that transforms with invalid parameters are skipped."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "Rotate": {"invalid_param": "value"},  # Will fail initialization
        }

        transforms = AlbumentationsWrapper.from_config(config)

        # At least HorizontalFlip should succeed
        assert len(transforms) >= 1
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names[0] == "HorizontalFlip"

    def test_invalid_config_type(self):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config_dict must be a dictionary or list"):
            AlbumentationsWrapper.from_config("invalid")

    def test_mixed_geometric_and_pixel_transforms(self):
        """Test building mix of geometric and pixel-level transforms."""
        config = {
            "HorizontalFlip": {"p": 1.0},  # Geometric
            "GaussianBlur": {"p": 1.0},  # Pixel-level
        }

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 2
        # Validate transform names match config in correct order
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

    def test_config_with_complex_params(self):
        """Test building transforms with complex parameter structures."""
        config = {
            "Rotate": {"limit": (90, 90), "p": 0.5},
            "Affine": {"scale": (0.9, 1.1), "translate_percent": (0.1, 0.1), "p": 0.3},
        }

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 2
        # Validate transform names match config in correct order
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

    def test_non_dict_params_skipped(self):
        """Test that transforms with non-dict params are skipped."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "InvalidTransform": "not_a_dict",
        }

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == ["HorizontalFlip"]


class TestAlbumentationsWrapperNestedConfig:
    """Tests for nested container (OneOf, SomeOf, Sequential) support in from_config."""

    def test_one_of_geometric_detection(self):
        """OneOf containing a geometric transform is treated as geometric."""
        wrapper = AlbumentationsWrapper(A.OneOf([A.HorizontalFlip(p=1.0), A.GaussianBlur(p=1.0)]))
        assert wrapper._is_geometric is True

    def test_one_of_pixel_detection(self):
        """OneOf containing only pixel transforms is treated as pixel-level."""
        wrapper = AlbumentationsWrapper(A.OneOf([A.GaussianBlur(p=1.0), A.Blur(p=1.0)]))
        assert wrapper._is_geometric is False

    def test_sequential_geometric_detection(self):
        """Sequential containing a geometric transform is treated as geometric."""
        wrapper = AlbumentationsWrapper(A.Sequential([A.Rotate(limit=45, p=1.0), A.GaussianBlur(p=1.0)]))
        assert wrapper._is_geometric is True

    def test_from_config_nested_one_of(self):
        """from_config builds a OneOf wrapper from nested config; p is ignored."""
        config = {
            "OneOf": {
                "transforms": [
                    {"HorizontalFlip": {"p": 1.0}},
                    {"VerticalFlip": {"p": 1.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        wrapper = transforms[0]
        assert isinstance(wrapper, AlbumentationsWrapper)
        assert wrapper._is_geometric is True
        # The inner Albumentations transform should be OneOf
        inner = wrapper.transform.transforms[0]
        assert isinstance(inner, A.OneOf)
        assert len(inner.transforms) == 2

    def test_from_config_nested_one_of_pixel_only(self):
        """from_config OneOf with only pixel transforms is non-geometric."""
        config = {
            "OneOf": {
                "transforms": [
                    {"GaussianBlur": {"p": 1.0}},
                    {"Blur": {"p": 1.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        assert transforms[0]._is_geometric is False

    def test_from_config_deeply_nested(self):
        """from_config handles nested containers (OneOf inside Sequential)."""
        config = {
            "Sequential": {
                "transforms": [
                    {
                        "OneOf": {
                            "transforms": [
                                {"HorizontalFlip": {"p": 1.0}},
                                {"VerticalFlip": {"p": 1.0}},
                            ],
                        }
                    },
                    {"GaussianBlur": {"p": 1.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        assert transforms[0]._is_geometric is True
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, A.Sequential)
        assert isinstance(inner.transforms[0], A.OneOf)

    def test_from_config_shorthand_list(self):
        """from_config supports shorthand {OneOf: [...]} without explicit transforms key."""
        config = {
            "OneOf": [
                {"HorizontalFlip": {"p": 1.0}},
                {"VerticalFlip": {"p": 1.0}},
            ]
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, A.OneOf)
        assert len(inner.transforms) == 2

    def test_from_config_nested_sequential(self):
        """from_config builds a Sequential wrapper from nested config."""
        config = {
            "Sequential": {
                "transforms": [
                    {"Rotate": {"limit": 45, "p": 1.0}},
                    {"GaussianBlur": {"p": 1.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 1
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, A.Sequential)
        assert len(inner.transforms) == 2

    def test_from_config_list_format(self):
        """from_config accepts list-of-single-key-dicts format."""
        config = [
            {"HorizontalFlip": {"p": 0.5}},
            {
                "OneOf": {
                    "transforms": [
                        {"VerticalFlip": {"p": 1.0}},
                        {"Rotate": {"limit": 45, "p": 1.0}},
                    ],
                }
            },
        ]
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 2
        assert isinstance(transforms[0], AlbumentationsWrapper)
        assert isinstance(transforms[1].transform.transforms[0], A.OneOf)

    def test_from_config_mixed_flat_and_nested(self):
        """from_config handles mix of flat and nested transforms."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "OneOf": {
                "transforms": [
                    {"GaussianBlur": {"p": 1.0}},
                    {"Blur": {"p": 1.0}},
                ],
            },
            "Rotate": {"limit": 15, "p": 0.3},
        }
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 3

    def test_from_config_one_of_applies_correctly_geometric(self):
        """OneOf geometric wrapper correctly transforms boxes (always fires)."""
        config = {
            "OneOf": {
                "transforms": [
                    {"HorizontalFlip": {"p": 1.0}},
                    {"VerticalFlip": {"p": 0.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)
        wrapper = transforms[0]

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            "labels": torch.tensor([1]),
        }
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        expected_boxes = torch.tensor([[50.0, 20.0, 90.0, 60.0]])
        torch.testing.assert_close(aug_target["boxes"], expected_boxes)

    def test_from_config_one_of_applies_correctly_pixel(self):
        """OneOf pixel-level wrapper preserves boxes unchanged."""
        config = {
            "OneOf": {
                "transforms": [
                    {"GaussianBlur": {"blur_limit": 3, "p": 1.0}},
                    {"Blur": {"blur_limit": 3, "p": 1.0}},
                ],
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)
        wrapper = transforms[0]

        image = Image.new("RGB", (100, 80))
        original_boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        target = {
            "boxes": original_boxes.clone(),
            "labels": torch.tensor([1]),
        }
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        torch.testing.assert_close(aug_target["boxes"], original_boxes)

    def test_one_of_p_in_config_is_ignored(self):
        """Any p supplied for OneOf in config is ignored; container always fires."""
        config = {
            "OneOf": {
                "transforms": [{"HorizontalFlip": {"p": 1.0}}],
                "p": 0.0,  # would suppress the container if respected
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, A.OneOf)
        assert inner.p == pytest.approx(1.0)

    def test_one_of_empty_transforms_raises(self):
        """OneOf with no transforms raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            _build_albu_transform("OneOf", {"transforms": []})

    def test_sequential_p_in_config_is_ignored(self):
        """Any p supplied for Sequential in config is ignored; container always fires."""
        config = {
            "Sequential": {
                "transforms": [{"HorizontalFlip": {"p": 1.0}}],
                "p": 0.0,  # would suppress the container if respected
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, A.Sequential)
        assert inner.p == pytest.approx(1.0)

    def test_some_of_single_p_still_works(self):
        """SomeOf with a plain p (block probability) still works without probs."""
        config = {
            "SomeOf": {
                "transforms": [
                    {"HorizontalFlip": {}},
                    {"VerticalFlip": {}},
                ],
                "n": 1,
                "p": 0.5,
            }
        }
        transforms = AlbumentationsWrapper.from_config(config)
        inner = transforms[0].transform.transforms[0]

        assert isinstance(inner, A.SomeOf)
        assert inner.p == pytest.approx(0.5)


class TestIntegration:
    """Integration tests for full augmentation pipeline."""

    def test_full_pipeline_from_config(self):
        """Test complete pipeline from config to application."""
        config = {
            "HorizontalFlip": {"p": 1.0},
            "VerticalFlip": {"p": 0.0},  # Will not apply
        }

        # Build transforms from config
        transforms = AlbumentationsWrapper.from_config(config)

        # Validate transform names match config in correct order
        assert len(transforms) == 2
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

        # Compose them
        composed = Compose(transforms)

        # Apply to data
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (1, 4)
        assert aug_target["labels"].shape == (1,)

    def test_pipeline_with_no_boxes(self):
        """Test pipeline works when target has no boxes."""
        config = {
            "GaussianBlur": {"p": 1.0},
        }

        transforms = AlbumentationsWrapper.from_config(config)

        # Validate transform names match config
        assert len(transforms) == 1
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

        composed = Compose(transforms)

        image = Image.new("RGB", (100, 100))
        target = {"labels": torch.tensor([1])}

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "labels" in aug_target

    def test_realistic_augmentation_config(self):
        """Test with realistic augmentation configuration."""
        aug_config = {
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.5},
            "Rotate": {"limit": 15, "p": 0.5},  # Better keep small angles
        }
        transforms = AlbumentationsWrapper.from_config(aug_config)

        # Validate transform names match in correct order
        assert len(transforms) == 3
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(aug_config.keys())

        composed = Compose(transforms)

        image = Image.new("RGB", (640, 480))
        target = {
            "boxes": torch.tensor([[50.0, 60.0, 200.0, 300.0], [300.0, 100.0, 500.0, 400.0]]),
            "labels": torch.tensor([1, 2]),
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        # Boxes might be filtered out by augmentations, so check shape is valid
        assert aug_target["boxes"].shape[1] == 4
        assert aug_target["labels"].shape[0] == aug_target["boxes"].shape[0]

    def test_full_pipeline_with_masks(self):
        """Test complete pipeline with masks from config to application."""
        config = {
            "HorizontalFlip": {"p": 1.0},
            "VerticalFlip": {"p": 0.0},  # Don't apply to make test deterministic
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = Compose(transforms)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 10:30, 10:30] = 1
        masks[1, 50:70, 50:70] = 1

        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 70.0, 70.0]]),
            "labels": torch.tensor([1, 2]),
            "masks": masks,
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "masks" in aug_target
        assert aug_target["boxes"].shape[0] == aug_target["masks"].shape[0]
        assert aug_target["labels"].shape[0] == aug_target["masks"].shape[0]
        assert aug_target["masks"].any()  # Masks should have content

    def test_pipeline_mixed_geometric_pixel_with_masks(self):
        """Test pipeline with mix of geometric and pixel transforms on masks."""
        config = {
            "HorizontalFlip": {"p": 1.0},  # Geometric
            "GaussianBlur": {"p": 1.0},  # Pixel
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = Compose(transforms)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 20:40, 10:30] = 1

        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "masks": masks,
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert "masks" in aug_target
        assert aug_target["masks"].shape == (1, height, width)
        assert aug_target["masks"].any()

    @pytest.mark.parametrize("num_instances", [1, 2, 5])
    def test_pipeline_scales_with_instances(self, num_instances):
        """Test pipeline handles varying numbers of instances correctly."""
        config = {
            "HorizontalFlip": {"p": 1.0},
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = Compose(transforms)

        height, width = 100, 100
        image = Image.new("RGB", (width, height))

        # Create multiple boxes and masks
        boxes = []
        masks = torch.zeros((num_instances, height, width), dtype=torch.uint8)
        for i in range(num_instances):
            x = i * 15 + 10
            y = i * 15 + 10
            boxes.append([float(x), float(y), float(x + 15), float(y + 15)])
            x1, y1, x2, y2 = int(x), int(y), int(x + 15), int(y + 15)
            masks[i, y1:y2, x1:x2] = 1

        target = {
            "boxes": torch.tensor(boxes),
            "labels": torch.arange(1, num_instances + 1),
            "masks": masks,
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape[0] <= num_instances  # May be filtered
        assert aug_target["masks"].shape[0] == aug_target["boxes"].shape[0]
        assert aug_target["labels"].shape[0] == aug_target["boxes"].shape[0]


class TestTrainingLoop:
    """Test augmentations work correctly in training loop scenario."""

    def test_augmentation_in_dataloader(self):
        """Test that augmentations work correctly when used with DataLoader.

        This is a critical integration test that simulates actual training conditions
        where multiple samples with different numbers of boxes are batched together.
        It specifically tests that orig_size remains consistent across the batch.
        """
        # Create augmentations
        aug_transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=0.5)),
            AlbumentationsWrapper(A.Rotate(limit=10, p=0.5)),
        ]
        transforms = Compose(aug_transforms)

        # Create dataset and dataloader
        dataset = _SimpleDataset(num_samples=12, transforms=transforms)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

        # Run through batches
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Check orig_size consistency
            orig_sizes = [t["orig_size"] for t in targets]

            # Verify all orig_sizes have shape [2]
            for i, orig_size in enumerate(orig_sizes):
                assert orig_size.shape == torch.Size([2]), (
                    f"Batch {batch_idx}, target {i}: orig_size has shape {orig_size.shape}, expected [2]"
                )

            # Critical test: This is what fails in training if orig_size is corrupted
            orig_target_sizes = torch.stack(orig_sizes, dim=0)
            assert orig_target_sizes.shape == torch.Size([len(targets), 2]), (
                f"Batch {batch_idx}: stacked orig_sizes has shape {orig_target_sizes.shape}"
            )

            # Verify images and targets are valid
            assert images.tensors.shape[0] == len(targets)
            num_boxes = [len(t["boxes"]) for t in targets]
            assert all(n > 0 for n in num_boxes), "All targets should have at least one box"

            # Only test a few batches for speed
            if batch_idx >= 1:
                break

    def test_augmentation_with_varying_box_counts(self):
        """Test that samples with 1, 2, and 3 boxes all work correctly in same batch.

        This specifically tests the edge case where some samples have 2 boxes
        (which matches orig_size shape [2]), ensuring they don't get mixed up.
        """
        aug_transforms = [AlbumentationsWrapper(A.HorizontalFlip(p=0.5))]
        transforms = Compose(aug_transforms)

        # Create dataset with samples that have different numbers of boxes
        dataset = _SimpleDataset(num_samples=9, transforms=transforms)  # Will cycle through 1,2,3 boxes
        dataloader = DataLoader(
            dataset,
            batch_size=6,  # Batch will contain mix of 1,2,3 box samples
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Get one batch
        images, targets = next(iter(dataloader))

        # Verify we have samples with different numbers of boxes
        num_boxes_list = [len(t["boxes"]) for t in targets]
        assert 1 in num_boxes_list, "Should have samples with 1 box"
        assert 2 in num_boxes_list, "Should have samples with 2 boxes (critical edge case)"
        assert 3 in num_boxes_list, "Should have samples with 3 boxes"

        # Verify all orig_sizes are consistent
        for i, target in enumerate(targets):
            assert target["orig_size"].shape == torch.Size([2]), (
                f"Target {i} (with {num_boxes_list[i]} boxes): orig_size shape is {target['orig_size'].shape}"
            )

        # Verify they can be stacked
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        assert orig_sizes.shape == torch.Size([len(targets), 2])

    @pytest.mark.parametrize(
        "transform_class,transform_kwargs",
        [
            (A.HorizontalFlip, {"p": 1.0}),
            (A.VerticalFlip, {"p": 1.0}),
            (A.RandomRotate90, {"p": 1.0}),
        ],
        ids=["horizontal_flip", "vertical_flip", "random_rotate_90"],
    )
    @pytest.mark.parametrize("include_masks", [False, True], ids=["detection", "segmentation"])
    def test_geometric_dataloader_compatibility(self, include_masks, transform_class, transform_kwargs):
        """Test geometric Albumentations transforms work in DataLoader for detection and segmentation."""

        class _TinyTrainDataset:
            def __init__(self, transforms):
                self._transforms = transforms

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                height, width = 64, 64
                image = Image.new("RGB", (width, height))
                target = {
                    "boxes": torch.tensor([[8.0, 12.0, 24.0, 28.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "orig_size": torch.tensor([height, width]),
                    "size": torch.tensor([height, width]),
                    "image_id": torch.tensor([idx]),
                    "area": torch.tensor([256.0]),
                    "iscrowd": torch.tensor([0]),
                }
                if include_masks:
                    masks = torch.zeros((1, height, width), dtype=torch.bool)
                    masks[0, 12:28, 8:24] = True
                    target["masks"] = masks

                image, target = self._transforms(image, target)
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                return image, target

        transforms = Compose([AlbumentationsWrapper(transform_class(**transform_kwargs))])
        dataloader = DataLoader(_TinyTrainDataset(transforms), batch_size=2, collate_fn=collate_fn, num_workers=0)
        images, targets = next(iter(dataloader))

        assert images.tensors.shape[0] == 2
        for target in targets:
            assert target["boxes"].shape == (1, 4)
            assert target["labels"].shape == (1,)
            if include_masks:
                assert target["masks"].shape == (1, 64, 64)
                assert target["masks"].dtype == torch.bool

    def test_segmentation_nano_training_with_tiny_mock_coco_dataset(self, tmp_path, monkeypatch):
        """Integration test: run a minimal segmentation training step on a mocked COCO dataset."""

        def _write_split(split_name: str) -> None:
            split_dir = tmp_path / "tiny_seg_dataset" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            image_path = split_dir / "sample.jpg"
            Image.new("RGB", (64, 64), color="white").save(image_path)

            annotations = {
                "images": [{"id": 1, "width": 64, "height": 64, "file_name": "sample.jpg"}],
                "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [8.0, 8.0, 16.0, 16.0],
                        "area": 256.0,
                        "iscrowd": 0,
                        "segmentation": [[8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0]],
                    }
                ],
            }
            (split_dir / "_annotations.coco.json").write_text(json.dumps(annotations))

        for split in ("train", "valid"):
            _write_split(split)

        def _fake_evaluate(*args, **kwargs):
            return {
                "coco_eval_masks": [0.0, 0.0],
                "results_json": {"map": 0.0, "f1_score": 0.0, "class_map": {}},
            }, None

        monkeypatch.setattr("rfdetr.main.evaluate", _fake_evaluate)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        model = RFDETRSegNano(pretrain_weights=None, device="cpu")
        model.train(
            dataset_dir=str(tmp_path / "tiny_seg_dataset"),
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            device="cpu",
            num_workers=0,
            resolution=64,
            amp=False,
            use_ema=False,
            run_test=False,
            tensorboard=False,
            dont_save_weights=True,
            min_batches=1,
            output_dir=str(output_dir),
        )

    def test_segmentation_training_saves_results_mask_json_with_run_test(self, tmp_path, monkeypatch):
        """results_mask.json must mirror results.json structure: valid+test keys and top-level scalars."""

        def _write_split(split_name: str) -> None:
            split_dir = tmp_path / "tiny_seg_dataset" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color="white").save(split_dir / "sample.jpg")
            annotations = {
                "images": [{"id": 1, "width": 64, "height": 64, "file_name": "sample.jpg"}],
                "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [8.0, 8.0, 16.0, 16.0],
                        "area": 256.0,
                        "iscrowd": 0,
                        "segmentation": [[8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0]],
                    }
                ],
            }
            (split_dir / "_annotations.coco.json").write_text(json.dumps(annotations))

        for split in ("train", "valid", "test"):
            _write_split(split)

        def _fake_evaluate_with_masks(*args, **kwargs):
            return {
                "coco_eval_masks": [0.5, 0.5],
                "results_json": {
                    "map": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1_score": 0.5,
                    "class_map": [],
                },
                "results_json_masks": {
                    "map": 0.5,
                    "precision": 0.6,
                    "recall": 0.7,
                    "f1_score": 0.65,
                    "class_map": [
                        {
                            "class": "object",
                            "map@50:95": 0.5,
                            "map@50": 0.6,
                            "precision": 0.6,
                            "recall": 0.7,
                            "f1_score": 0.65,
                        }
                    ],
                },
            }, None

        monkeypatch.setattr("rfdetr.main.evaluate", _fake_evaluate_with_masks)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        model = RFDETRSegNano(pretrain_weights=None, device="cpu")
        model.train(
            dataset_dir=str(tmp_path / "tiny_seg_dataset"),
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            device="cpu",
            num_workers=0,
            resolution=64,
            amp=False,
            use_ema=False,
            run_test=True,
            tensorboard=False,
            dont_save_weights=False,
            min_batches=1,
            output_dir=str(output_dir),
        )

        results_mask_path = output_dir / "results_mask.json"
        assert results_mask_path.exists(), "results_mask.json should be written for segmentation models"
        with open(results_mask_path) as f:
            results_mask = json.load(f)

        assert "class_map" in results_mask
        assert "valid" in results_mask["class_map"], "results_mask.json must have 'valid' key in class_map"
        assert "test" in results_mask["class_map"], "results_mask.json must have 'test' key in class_map"
        assert "map" in results_mask, "results_mask.json must have top-level 'map' scalar"
        assert "precision" in results_mask, "results_mask.json must have top-level 'precision' scalar"
        assert "recall" in results_mask, "results_mask.json must have top-level 'recall' scalar"
        assert "f1_score" in results_mask, "results_mask.json must have top-level 'f1_score' scalar"


class TestMakeCocoTransformsAugConfig:
    """Tests for aug_config propagation in make_coco_transforms / make_coco_transforms_square_div_64."""

    @pytest.mark.parametrize(
        "make_transforms",
        [
            make_coco_transforms,
            make_coco_transforms_square_div_64,
        ],
    )
    def test_default_none_uses_aug_config(self, make_transforms):
        """Omitting aug_config uses the module-level AUG_CONFIG default (HorizontalFlip)."""
        pipeline = make_transforms("train", 640)
        # Train pipeline: [resize_wrapper, *aug_wrappers, normalize]
        # First AlbumentationsWrapper is the resize OneOf; remaining are from aug_config.
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        aug_wrappers = wrappers[1:]

        expected_names = list(AUG_CONFIG.keys())
        actual_names = [w.transform.transforms[0].__class__.__name__ for w in aug_wrappers]
        assert actual_names == expected_names

    @pytest.mark.parametrize(
        "make_transforms",
        [
            make_coco_transforms,
            make_coco_transforms_square_div_64,
        ],
    )
    def test_empty_dict_disables_augmentations(self, make_transforms):
        """aug_config={} means no aug wrappers beyond the resize wrapper."""
        pipeline = make_transforms("train", 640, aug_config={})
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        aug_wrappers = wrappers[1:]  # skip resize wrapper

        assert aug_wrappers == []

    @pytest.mark.parametrize(
        "make_transforms",
        [
            make_coco_transforms,
            make_coco_transforms_square_div_64,
        ],
    )
    def test_custom_dict_is_used(self, make_transforms):
        """aug_config with a custom dict wires up exactly those transforms."""
        custom = {"HorizontalFlip": {"p": 1.0}}
        pipeline = make_transforms("train", 640, aug_config=custom)
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        aug_wrappers = wrappers[1:]  # skip resize wrapper

        assert len(aug_wrappers) == 1
        assert aug_wrappers[0].transform.transforms[0].__class__.__name__ == "HorizontalFlip"

    @pytest.mark.parametrize(
        "make_transforms,expected_resize_wrappers",
        [
            # make_coco_transforms val: SmallestMaxSize + LongestMaxSize = 2 wrappers
            pytest.param(make_coco_transforms, 2, id="make_coco_transforms"),
            # make_coco_transforms_square_div_64 val: Resize = 1 wrapper
            pytest.param(make_coco_transforms_square_div_64, 1, id="make_coco_transforms_square_div_64"),
        ],
    )
    def test_aug_config_not_applied_on_val(self, make_transforms, expected_resize_wrappers):
        """aug_config is ignored for val splits — only resize wrappers are present."""
        pipeline = make_transforms("val", 640, aug_config={"HorizontalFlip": {"p": 1.0}})
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]

        assert len(wrappers) == expected_resize_wrappers

    @pytest.mark.parametrize(
        "make_transforms",
        [
            make_coco_transforms,
            make_coco_transforms_square_div_64,
        ],
    )
    def test_aug_config_not_applied_on_val_speed(self, make_transforms):
        """aug_config is ignored for val_speed splits — only the resize wrapper is present."""
        pipeline = make_transforms("val_speed", 640, aug_config={"HorizontalFlip": {"p": 1.0}})
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]

        assert len(wrappers) == 1

    def test_aug_config_not_applied_on_test(self):
        """aug_config is ignored for the test split in make_coco_transforms_square_div_64."""
        pipeline = make_coco_transforms_square_div_64("test", 640, aug_config={"HorizontalFlip": {"p": 1.0}})
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]

        assert len(wrappers) == 1
