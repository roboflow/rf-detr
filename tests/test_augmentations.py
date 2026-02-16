# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Albumentations augmentation wrappers."""

import albumentations as A
import pytest
import torch
from PIL import Image

from rfdetr.augmentation_config import AUG_CONFIG
from rfdetr.datasets.transforms import (
    AlbumentationsWrapper,
    ComposeAugmentations,
)


class TestAlbumentationsWrapper:
    """Tests for AlbumentationsWrapper class."""

    @pytest.mark.parametrize("transform_class,params,box_in,box_out", [
        (A.HorizontalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [70.0, 20.0, 90.0, 40.0]),
        (A.VerticalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [10.0, 60.0, 30.0, 80.0]),
    ])
    def test_flip_transforms_with_boxes(self, transform_class, params, box_in, box_out):
        """Test flip transforms correctly transform bounding boxes."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([box_in]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert torch.allclose(aug_target['boxes'], torch.tensor([box_out]), atol=1.0)
        assert torch.equal(aug_target['labels'], target['labels'])

    def test_non_geometric_transform_preserves_boxes(self):
        """Test that non-geometric transforms preserve bounding boxes."""
        transform = A.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Boxes should be unchanged
        assert torch.equal(aug_target['boxes'], target['boxes'])
        assert torch.equal(aug_target['labels'], target['labels'])

    def test_empty_boxes_handling(self):
        """Test wrapper handles empty boxes correctly."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.zeros((0, 4)),
            'labels': torch.zeros((0,), dtype=torch.long)
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (0, 4)
        assert aug_target['labels'].shape == (0,)

    def test_multiple_boxes(self):
        """Test wrapper handles multiple bounding boxes."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 80.0]
            ]),
            'labels': torch.tensor([1, 2])
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (2, 4)
        assert aug_target['labels'].shape == (2,)
        assert torch.equal(aug_target['labels'], target['labels'])

    def test_invalid_target_type(self):
        """Test wrapper raises error for invalid target type."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))

        with pytest.raises(TypeError, match="target must be a dictionary"):
            wrapper(image, "invalid_target")

    def test_missing_labels_key(self):
        """Test wrapper raises error when labels key is missing."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]])}

        with pytest.raises(KeyError, match="target must contain 'labels' key"):
            wrapper(image, target)

    def test_invalid_boxes_shape(self):
        """Test wrapper raises error for invalid boxes shape."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([10.0, 20.0, 30.0]),  # Invalid shape
            'labels': torch.tensor([1])
        }

        with pytest.raises(ValueError, match="boxes must have shape"):
            wrapper(image, target)

    @pytest.mark.parametrize("transform_class,params", [
        (A.HorizontalFlip, {"p": 1.0}),
        (A.VerticalFlip, {"p": 1.0}),
        (A.Rotate, {"limit": 45, "p": 1.0}),
    ])
    def test_various_geometric_transforms(self, transform_class, params):
        """Test various geometric transforms work correctly."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Albumentations can return multiple boxes for a single input box on some Python versions.
        assert aug_target['boxes'].shape[1] == 4
        assert aug_target['labels'].shape[0] == aug_target['boxes'].shape[0]
        assert aug_target['labels'].numel() >= 1

    def test_masks_transform_with_horizontal_flip(self):
        """Masks should be transformed consistently with boxes for geometric transforms."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        # Create test image (100x100)
        height, width = 100, 100
        image = Image.new('RGB', (width, height), color='red')

        # Single box and corresponding mask
        box = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # x1, y1, x2, y2
        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        # Fill the mask inside the box region
        x1, y1, x2, y2 = box[0].to(dtype=torch.long)
        masks[0, y1:y2, x1:x2] = 1

        target = {
            'boxes': box,
            'labels': torch.tensor([1]),
            'masks': masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'masks' in aug_target
        assert aug_target['masks'].shape[0] == aug_target['boxes'].shape[0]

        # Check that the transformed mask's bounding box matches the transformed box
        aug_mask = aug_target['masks'][0]
        ys, xs = torch.nonzero(aug_mask, as_tuple=True)
        assert ys.numel() > 0 and xs.numel() > 0
        mask_bbox = torch.tensor([
            xs.min().item(),
            ys.min().item(),
            xs.max().item() + 1,
            ys.max().item() + 1,
        ], dtype=torch.float32)
        assert torch.allclose(mask_bbox, aug_target['boxes'][0].to(dtype=torch.float32), atol=1.0)


    @pytest.mark.parametrize("transform_class,params", [
        (A.HorizontalFlip, {"p": 1.0}),
        (A.VerticalFlip, {"p": 1.0}),
        (A.Rotate, {"limit": 15, "p": 1.0}),  # Small angle to avoid boxes going out
    ])
    def test_various_geometric_transforms_with_masks(self, transform_class, params):
        """Test various geometric transforms correctly transform masks."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        # Create mask covering the box region (more centered to avoid edge issues with rotation)
        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 40:60, 40:60] = 1

        target = {
            'boxes': torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            'labels': torch.tensor([1]),
            'masks': masks
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'masks' in aug_target
        # Number of boxes may change with rotation (boxes can be removed if they go out of bounds)
        assert aug_target['masks'].shape[0] == aug_target['boxes'].shape[0]
        if aug_target['boxes'].shape[0] > 0:
            # Mask should still have content (not all zeros)
            assert aug_target['masks'].any()

    @pytest.mark.parametrize("transform_class,params", [
        (A.GaussianBlur, {"blur_limit": 3, "p": 1.0}),
        (A.RandomBrightnessContrast, {"p": 1.0}),
        (A.GaussNoise, {"p": 1.0}),
    ])
    def test_pixel_transforms_preserve_masks(self, transform_class, params):
        """Test pixel-level transforms preserve masks unchanged."""
        transform = transform_class(**params)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 20:40, 10:30] = 1

        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1]),
            'masks': masks
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Pixel transforms should not modify masks
        assert torch.equal(aug_target['masks'], target['masks'])

    def test_multiple_masks_with_geometric_transform(self):
        """Test multiple masks are correctly transformed together."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        # Two masks for two boxes
        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 10:30, 10:30] = 1  # First mask
        masks[1, 50:70, 50:70] = 1  # Second mask

        target = {
            'boxes': torch.tensor([
                [10.0, 10.0, 30.0, 30.0],
                [50.0, 50.0, 70.0, 70.0]
            ]),
            'labels': torch.tensor([1, 2]),
            'masks': masks
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_target['masks'].shape == (2, height, width)
        assert aug_target['boxes'].shape[0] == 2
        assert aug_target['labels'].shape[0] == 2

    def test_empty_masks_handling(self):
        """Test wrapper correctly handles empty masks (no 'masks' key when empty)."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        # When boxes are empty, don't include masks field
        target = {
            'boxes': torch.zeros((0, 4)),
            'labels': torch.zeros((0,), dtype=torch.long),
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (0, 4)
        assert aug_target['labels'].shape == (0,)

    def test_pixel_transform_with_masks_no_boxes(self):
        """Test that pixel transforms work with masks but no boxes."""
        # Use a non-geometric transform which doesn't need boxes
        transform = A.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        masks_orig = torch.zeros((1, height, width), dtype=torch.uint8)
        masks_orig[0, 20:40, 10:30] = 1

        target = {
            'labels': torch.tensor([1]),
            'masks': masks_orig.clone()  # No boxes!
        }

        aug_image, aug_target = wrapper(image, target)

        # Pixel transforms should preserve masks
        assert torch.equal(aug_target['masks'], masks_orig)

    def test_invalid_mask_shape_raises_error(self):
        """Test that invalid mask shape raises ValueError."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        # Invalid mask shape (2D instead of 3D)
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1]),
            'masks': torch.zeros((height, width), dtype=torch.uint8)
        }

        with pytest.raises(ValueError, match="masks must have shape"):
            wrapper(image, target)

    @pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.float32])
    def test_mask_dtype_handling(self, mask_dtype):
        """Test wrapper handles different mask dtypes correctly (uint8, float32)."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        masks = torch.zeros((1, height, width), dtype=mask_dtype)
        masks[0, 20:40, 10:30] = 1

        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1]),
            'masks': masks
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'masks' in aug_target
        # Output masks should be bool after Albumentations processing
        assert aug_target['masks'].dtype == torch.bool


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
        with pytest.raises(TypeError, match="config_dict must be a dictionary"):
            AlbumentationsWrapper.from_config("invalid")

    def test_mixed_geometric_and_pixel_transforms(self):
        """Test building mix of geometric and pixel-level transforms."""
        config = {
            "HorizontalFlip": {"p": 1.0},  # Geometric
            "GaussianBlur": {"p": 1.0},     # Pixel-level
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
            "Affine": {
                "scale": (0.9, 1.1),
                "translate_percent": (0.1, 0.1),
                "p": 0.3
            }
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


class TestComposeAugmentations:
    """Tests for ComposeAugmentations class."""

    def test_compose_initialization(self):
        """Test ComposeAugmentations initialization."""
        transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0)),
            AlbumentationsWrapper(A.VerticalFlip(p=1.0)),
        ]

        composed = ComposeAugmentations(transforms)

        assert composed.transforms == transforms
        assert len(composed.transforms) == 2
        # Validate transform names in correct order
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in composed.transforms]
        assert transform_names == ["HorizontalFlip", "VerticalFlip"]

    def test_compose_applies_all_transforms(self):
        """Test that all transforms are applied sequentially."""
        transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0)),
            AlbumentationsWrapper(A.VerticalFlip(p=1.0)),
        ]
        composed = ComposeAugmentations(transforms)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        # After both flips, both coordinates should be mirrored
        assert aug_target['boxes'].shape == (1, 4)

    def test_compose_empty_transforms(self):
        """Test composing with empty transforms list."""
        composed = ComposeAugmentations([])

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = composed(image, target)

        # Should return unchanged
        assert aug_image == image
        assert torch.equal(aug_target['boxes'], target['boxes'])

    def test_compose_invalid_transforms_type(self):
        """Test that invalid transforms type raises TypeError."""
        with pytest.raises(TypeError, match="transforms must be a list"):
            ComposeAugmentations("invalid")

    def test_compose_single_transform(self):
        """Test composing with single transform."""
        transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0))
        ]
        composed = ComposeAugmentations(transforms)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (1, 4)


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
        composed = ComposeAugmentations(transforms)

        # Apply to data
        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (1, 4)
        assert aug_target['labels'].shape == (1,)

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

        composed = ComposeAugmentations(transforms)

        image = Image.new('RGB', (100, 100))
        target = {'labels': torch.tensor([1])}

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'labels' in aug_target

    def test_realistic_augmentation_config(self):
        """Test with realistic augmentation configuration."""
        transforms = AlbumentationsWrapper.from_config(AUG_CONFIG)

        # Validate transform names match AUG_CONFIG in correct order
        assert len(transforms) == 3
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(AUG_CONFIG.keys())

        composed = ComposeAugmentations(transforms)

        image = Image.new('RGB', (640, 480))
        target = {
            'boxes': torch.tensor([
                [50.0, 60.0, 200.0, 300.0],
                [300.0, 100.0, 500.0, 400.0]
            ]),
            'labels': torch.tensor([1, 2])
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        # Boxes might be filtered out by augmentations, so check shape is valid
        assert aug_target['boxes'].shape[1] == 4
        assert aug_target['labels'].shape[0] == aug_target['boxes'].shape[0]

    def test_full_pipeline_with_masks(self):
        """Test complete pipeline with masks from config to application."""
        config = {
            "HorizontalFlip": {"p": 1.0},
            "VerticalFlip": {"p": 0.0},  # Don't apply to make test deterministic
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = ComposeAugmentations(transforms)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 10:30, 10:30] = 1
        masks[1, 50:70, 50:70] = 1

        target = {
            'boxes': torch.tensor([
                [10.0, 10.0, 30.0, 30.0],
                [50.0, 50.0, 70.0, 70.0]
            ]),
            'labels': torch.tensor([1, 2]),
            'masks': masks
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'masks' in aug_target
        assert aug_target['boxes'].shape[0] == aug_target['masks'].shape[0]
        assert aug_target['labels'].shape[0] == aug_target['masks'].shape[0]
        assert aug_target['masks'].any()  # Masks should have content

    def test_pipeline_mixed_geometric_pixel_with_masks(self):
        """Test pipeline with mix of geometric and pixel transforms on masks."""
        config = {
            "HorizontalFlip": {"p": 1.0},  # Geometric
            "GaussianBlur": {"p": 1.0},    # Pixel
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = ComposeAugmentations(transforms)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

        masks = torch.zeros((1, height, width), dtype=torch.uint8)
        masks[0, 20:40, 10:30] = 1

        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1]),
            'masks': masks
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'masks' in aug_target
        assert aug_target['masks'].shape == (1, height, width)
        assert aug_target['masks'].any()

    @pytest.mark.parametrize("num_instances", [1, 2, 5])
    def test_pipeline_scales_with_instances(self, num_instances):
        """Test pipeline handles varying numbers of instances correctly."""
        config = {
            "HorizontalFlip": {"p": 1.0},
        }

        transforms = AlbumentationsWrapper.from_config(config)
        composed = ComposeAugmentations(transforms)

        height, width = 100, 100
        image = Image.new('RGB', (width, height))

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
            'boxes': torch.tensor(boxes),
            'labels': torch.arange(1, num_instances + 1),
            'masks': masks
        }

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape[0] <= num_instances  # May be filtered
        assert aug_target['masks'].shape[0] == aug_target['boxes'].shape[0]
        assert aug_target['labels'].shape[0] == aug_target['boxes'].shape[0]
