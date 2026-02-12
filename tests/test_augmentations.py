# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Albumentations augmentation wrappers."""

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.transforms import (
    AlbumentationsWrapper,
    ComposeAugmentations,
    build_albumentations_from_config,
)


class TestAlbumentationsWrapper:
    """Tests for AlbumentationsWrapper class."""

    def test_wrapper_initialization_bbox_safe(self):
        """Test wrapper initialization with bbox_safe=True."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        assert wrapper.bbox_safe is True
        assert wrapper.transform is not None

    def test_wrapper_initialization_not_bbox_safe(self):
        """Test wrapper initialization with bbox_safe=False."""
        transform = A.GaussianBlur(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=False)

        assert wrapper.bbox_safe is False
        assert wrapper.transform is not None

    def test_horizontal_flip_with_boxes(self):
        """Test horizontal flip correctly transforms bounding boxes."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        # Create test image (100x100)
        image = Image.new('RGB', (100, 100), color='red')
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),  # x1, y1, x2, y2
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        # After horizontal flip, x-coordinates should be mirrored
        # Original: [10, 20, 30, 40] -> Flipped: [70, 20, 90, 40]
        expected_boxes = torch.tensor([[70.0, 20.0, 90.0, 40.0]])

        assert isinstance(aug_image, Image.Image)
        assert torch.allclose(aug_target['boxes'], expected_boxes, atol=1.0)
        assert torch.equal(aug_target['labels'], target['labels'])

    def test_vertical_flip_with_boxes(self):
        """Test vertical flip correctly transforms bounding boxes."""
        transform = A.VerticalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        # Create test image (100x100)
        image = Image.new('RGB', (100, 100), color='blue')
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        # After vertical flip, y-coordinates should be mirrored
        # Original: [10, 20, 30, 40] -> Flipped: [10, 60, 30, 80]
        expected_boxes = torch.tensor([[10.0, 60.0, 30.0, 80.0]])

        assert isinstance(aug_image, Image.Image)
        assert torch.allclose(aug_target['boxes'], expected_boxes, atol=1.0)

    def test_non_geometric_transform_preserves_boxes(self):
        """Test that non-geometric transforms preserve bounding boxes."""
        transform = A.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=False)

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
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

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
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

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
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        image = Image.new('RGB', (100, 100))

        with pytest.raises(TypeError, match="target must be a dictionary"):
            wrapper(image, "invalid_target")

    def test_missing_labels_key(self):
        """Test wrapper raises error when labels key is missing."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        image = Image.new('RGB', (100, 100))
        target = {'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]])}

        with pytest.raises(KeyError, match="target must contain 'labels' key"):
            wrapper(image, target)

    def test_invalid_boxes_shape(self):
        """Test wrapper raises error for invalid boxes shape."""
        transform = A.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

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
        wrapper = AlbumentationsWrapper(transform, bbox_safe=True)

        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'labels': torch.tensor([1])
        }

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target['boxes'].shape == (1, 4)
        assert aug_target['labels'].shape == (1,)


class TestBuildAlbumentationsFromConfig:
    """Tests for build_albumentations_from_config function."""

    def test_build_from_valid_config(self):
        """Test building transforms from valid configuration."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.3},
        }

        transforms = build_albumentations_from_config(config)

        assert len(transforms) == 2
        assert all(isinstance(t, AlbumentationsWrapper) for t in transforms)

    def test_build_from_empty_config(self):
        """Test building from empty config returns empty list."""
        config = {}

        transforms = build_albumentations_from_config(config)

        assert len(transforms) == 0

    def test_unknown_transform_skipped(self):
        """Test that unknown transforms are skipped with warning."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "NonExistentTransform": {"p": 0.5},
        }

        transforms = build_albumentations_from_config(config)

        # Only valid transform should be included
        assert len(transforms) == 1

    def test_invalid_params_skipped(self):
        """Test that transforms with invalid parameters are skipped."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "Rotate": {"invalid_param": "value"},  # Will fail initialization
        }

        transforms = build_albumentations_from_config(config)

        # At least HorizontalFlip should succeed
        assert len(transforms) >= 1

    def test_invalid_config_type(self):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config_dict must be a dictionary"):
            build_albumentations_from_config("invalid")

    def test_geometric_transform_detection(self):
        """Test that geometric transforms are correctly identified."""
        config = {
            "HorizontalFlip": {"p": 1.0},  # Geometric
            "GaussianBlur": {"p": 1.0},     # Non-geometric
        }

        transforms = build_albumentations_from_config(config)

        assert len(transforms) == 2
        assert transforms[0].bbox_safe is True   # HorizontalFlip
        assert transforms[1].bbox_safe is False  # GaussianBlur

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

        transforms = build_albumentations_from_config(config)

        assert len(transforms) == 2

    def test_non_dict_params_skipped(self):
        """Test that transforms with non-dict params are skipped."""
        config = {
            "HorizontalFlip": {"p": 0.5},
            "InvalidTransform": "not_a_dict",
        }

        transforms = build_albumentations_from_config(config)

        assert len(transforms) == 1


class TestComposeAugmentations:
    """Tests for ComposeAugmentations class."""

    def test_compose_initialization(self):
        """Test ComposeAugmentations initialization."""
        transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0), bbox_safe=True),
            AlbumentationsWrapper(A.VerticalFlip(p=1.0), bbox_safe=True),
        ]

        composed = ComposeAugmentations(transforms)

        assert composed.transforms == transforms
        assert len(composed.transforms) == 2

    def test_compose_applies_all_transforms(self):
        """Test that all transforms are applied sequentially."""
        transforms = [
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0), bbox_safe=True),
            AlbumentationsWrapper(A.VerticalFlip(p=1.0), bbox_safe=True),
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
            AlbumentationsWrapper(A.HorizontalFlip(p=1.0), bbox_safe=True)
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
        transforms = build_albumentations_from_config(config)

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

        transforms = build_albumentations_from_config(config)
        composed = ComposeAugmentations(transforms)

        image = Image.new('RGB', (100, 100))
        target = {'labels': torch.tensor([1])}

        aug_image, aug_target = composed(image, target)

        assert isinstance(aug_image, Image.Image)
        assert 'labels' in aug_target

    def test_realistic_augmentation_config(self):
        """Test with realistic augmentation configuration."""
        from rfdetr.augmentation_config import AUG_CONFIG

        transforms = build_albumentations_from_config(AUG_CONFIG)
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
