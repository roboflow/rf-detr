# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Albumentations augmentations inside the COCO transform pipelines and the training data flow.

Also covers keypoint handling: horizontal-flip filtering, flip parity and keypoint scaling across resolutions.
"""

import math
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from rfdetr.datasets._aug_utils import filter_keypoint_hflip_augmentations
from rfdetr.datasets._develop import _SimpleDataset
from rfdetr.datasets._torchvision import RandomHorizontalFlip
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.utilities import collate_fn
from tests._markers import _requires_albumentations, _requires_kornia


@_requires_albumentations
class TestKeypointHFlipFiltering:
    """Shared keypoint hflip filtering behavior for augmentation backends."""

    def test_drops_hflip_and_warns_for_keypoint_pipeline(self) -> None:
        """Keypoint augmentation config drops hflip transforms while keeping other transforms."""
        config = {"HorizontalFlip": {"p": 0.5}, "VerticalFlip": {"p": 0.5}}
        warning = mock.Mock()

        filtered = filter_keypoint_hflip_augmentations(config, warn=warning)

        assert filtered == {"VerticalFlip": {"p": 0.5}}
        warning.assert_called_once()
        assert "HorizontalFlip" in str(warning.call_args)

    def test_keeps_hflip_when_keypoints_are_not_active(self) -> None:
        """Detection augmentation config keeps hflip transforms unchanged."""
        config = {"HorizontalFlip": {"p": 0.5}}
        warning = mock.Mock()

        filtered = filter_keypoint_hflip_augmentations(config, warn=warning, include_keypoints=False)

        assert filtered == config
        warning.assert_not_called()

    def test_drops_nested_hflip_from_container(self) -> None:
        """Keypoint augmentation config drops hflip transforms inside containers."""
        config = {
            "OneOf": {
                "transforms": [
                    {"HorizontalFlip": {"p": 1.0}},
                    {"VerticalFlip": {"p": 1.0}},
                ],
                "p": 0.5,
            }
        }
        warning = mock.Mock()

        filtered = filter_keypoint_hflip_augmentations(config, warn=warning)

        assert filtered == {
            "OneOf": {
                "transforms": [
                    {"VerticalFlip": {"p": 1.0}},
                ],
                "p": 0.5,
            }
        }
        warning.assert_called_once()
        assert "HorizontalFlip" in str(warning.call_args)


@_requires_albumentations
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


@_requires_albumentations
class TestTrainingLoop:
    """Test augmentations work correctly in training loop scenario."""

    def test_augmentation_in_dataloader(self):
        """Test that augmentations work correctly when used with DataLoader.

        This is a critical integration test that simulates actual training conditions where multiple samples with
        different numbers of boxes are batched together. It specifically tests that orig_size remains consistent across
        the batch.
        """
        # Create augmentations
        import albumentations as alb

        aug_transforms = [
            AlbumentationsWrapper(alb.HorizontalFlip(p=0.5)),
            AlbumentationsWrapper(alb.Rotate(limit=10, p=0.5)),
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

        This specifically tests the edge case where some samples have 2 boxes (which matches orig_size shape [2]),
        ensuring they don't get mixed up.
        """
        import albumentations as alb

        aug_transforms = [AlbumentationsWrapper(alb.HorizontalFlip(p=0.5))]
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
        "transform_name,transform_kwargs",
        [
            pytest.param("HorizontalFlip", {"p": 1.0}, id="horizontal_flip"),
            pytest.param("VerticalFlip", {"p": 1.0}, id="vertical_flip"),
            pytest.param("RandomRotate90", {"p": 1.0}, id="random_rotate_90"),
        ],
    )
    @pytest.mark.parametrize(
        "include_masks", [pytest.param(False, id="detection"), pytest.param(True, id="segmentation")]
    )
    def test_geometric_dataloader_compatibility(self, include_masks, transform_name, transform_kwargs):
        """Test geometric Albumentations transforms work in DataLoader for detection and segmentation."""
        import albumentations as alb

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

        transforms = Compose([AlbumentationsWrapper(getattr(alb, transform_name)(**transform_kwargs))])
        dataloader = DataLoader(_TinyTrainDataset(transforms), batch_size=2, collate_fn=collate_fn, num_workers=0)
        images, targets = next(iter(dataloader))

        assert images.tensors.shape[0] == 2
        for target in targets:
            assert target["boxes"].shape == (1, 4)
            assert target["labels"].shape == (1,)
            if include_masks:
                assert target["masks"].shape == (1, 64, 64)
                assert target["masks"].dtype == torch.bool


@_requires_albumentations
class TestMakeCocoTransformsAugConfig:
    """Tests for aug_config propagation in make_coco_transforms / make_coco_transforms_square_div_64."""

    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_default_none_uses_aug_config(self, make_transforms):
        """Omitting aug_config uses the torchvision-native default HorizontalFlip."""
        pipeline = make_transforms("train", 640)

        assert any(isinstance(t, RandomHorizontalFlip) for t in pipeline.transforms)
        assert not any(isinstance(t, AlbumentationsWrapper) for t in pipeline.transforms)

    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_empty_dict_disables_augmentations(self, make_transforms):
        """aug_config={} disables the default torchvision HorizontalFlip."""
        pipeline = make_transforms("train", 640, aug_config={})

        assert not any(isinstance(t, RandomHorizontalFlip) for t in pipeline.transforms)
        assert not any(isinstance(t, AlbumentationsWrapper) for t in pipeline.transforms)

    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_custom_dict_is_used(self, make_transforms):
        """A custom non-empty aug_config uses the optional Albumentations path."""
        custom = {"HorizontalFlip": {"p": 1.0}}
        pipeline = make_transforms("train", 640, aug_config=custom)
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        aug_wrappers = [w for w in wrappers if "HorizontalFlip" in _transform_leaf_names(w.transform)]

        assert len(aug_wrappers) == 1
        assert aug_wrappers[0].transform.transforms[0].__class__.__name__ == "HorizontalFlip"

    @pytest.mark.parametrize("split", ["val", "val_speed", "test"])
    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_aug_config_ignored_outside_train(self, make_transforms, split):
        """aug_config is ignored for val, val_speed and test splits, which keep the torchvision resize."""
        pipeline = make_transforms(split, 640, aug_config={"HorizontalFlip": {"p": 1.0}})

        assert not any(isinstance(t, AlbumentationsWrapper) for t in pipeline.transforms)


def _transform_leaf_names(transform: Any) -> set[str]:
    """Collect the class names of every transform nested inside *transform* (itself included).

    Albumentations containers expose their children through ``.transforms``, so walking that attribute flattens
    ``OneOf``/``Sequential`` wrappers down to their leaves.

    Args:
        transform: An Albumentations transform, possibly a container.

    Returns:
        The class names of *transform* and every transform nested inside it.

    Examples:
        >>> from types import SimpleNamespace
        >>> outer = SimpleNamespace(transforms=[SimpleNamespace(transforms=[]), SimpleNamespace(transforms=[])])
        >>> sorted(_transform_leaf_names(outer))
        ['SimpleNamespace']
    """
    names = {type(transform).__name__}
    for child in getattr(transform, "transforms", []) or []:
        names |= _transform_leaf_names(child)
    return names


def _line_angle(mask: np.ndarray) -> float:
    """Return the angle (degrees from horizontal) of the principal axis of a boolean *mask*.

    Args:
        mask: Two-dimensional boolean array marking the line's pixels.

    Returns:
        The line's angle in degrees, in ``(-90, 90]``.

    Examples:
        >>> _line_angle(np.eye(5, dtype=bool))
        45.0
    """
    ys, xs = np.nonzero(mask)
    covariance = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    values, vectors = np.linalg.eigh(covariance)
    dx, dy = vectors[:, np.argmax(values)]
    return (math.degrees(math.atan2(dy, dx)) + 90) % 180 - 90


class TestAugmentationCompatibility:
    """Preserve existing augmentation configurations when geometry moves before resizing."""

    @staticmethod
    def _sample() -> tuple[Image.Image, dict[str, Any]]:
        """Return a small image with one detection."""
        image = Image.new("RGB", (80, 60), color=(100, 50, 25))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 40.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(1),
            "orig_size": torch.tensor([60, 80]),
            "size": torch.tensor([60, 80]),
        }
        return image, target

    @_requires_albumentations
    def test_fixed_crop_larger_than_source_is_padded(self):
        """A fixed crop accepts a smaller source image and retains the final training size."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"RandomCrop": {"height": 100, "width": 100, "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
        )
        image, target = pipeline(*self._sample())
        assert image.shape[-2:] == (128, 128)
        assert target["boxes"].shape == (1, 4)

    @_requires_albumentations
    def test_nested_crop_then_rotation_keeps_source_geometry(self):
        """Nested crops pad small sources and leave later rotation before the square resize."""
        config = {
            "OneOf": {"transforms": [{"RandomCrop": {"height": 100, "width": 100, "p": 1.0}}], "p": 1.0},
            "Rotate": {"limit": (10, 10), "p": 1.0},
        }
        pipeline = make_coco_transforms_square_div_64(
            "train", 128, aug_config=config, scale_jitter=False, skip_random_resize=True
        )

        image, result = pipeline(*self._sample())

        assert image.shape[-2:] == (128, 128)
        assert result["boxes"].shape == (1, 4)

    @_requires_albumentations
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"Crop": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100, "p": 1.0}}, id="crop"),
            pytest.param(
                {"RandomSizedCrop": {"min_max_height": (100, 100), "size": (100, 100), "p": 1.0}},
                id="random-sized-crop",
            ),
        ],
    )
    def test_other_fixed_crops_pad_small_images(self, config):
        """Fixed-coordinate and sized crops accept images smaller than their sampled crop."""
        pipeline = make_coco_transforms_square_div_64(
            "train", 128, aug_config=config, scale_jitter=False, skip_random_resize=True
        )

        image, _ = pipeline(*self._sample())

        assert image.shape[-2:] == (128, 128)

    @_requires_albumentations
    def test_skipped_crop_does_not_pad_image(self):
        """Padding follows the crop probability rather than running for a skipped crop."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"RandomCrop": {"height": 100, "width": 100, "p": 0.0}},
            scale_jitter=False,
            skip_random_resize=True,
        )
        crop = next(t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper))

        image, target = crop(*self._sample())

        assert image.size == (80, 60)
        torch.testing.assert_close(target["boxes"], torch.tensor([[10.0, 10.0, 50.0, 40.0]]))

    @_requires_albumentations
    def test_crop_then_rotation_runs_before_final_resize(self):
        """A crop followed by rotation retains the crop's non-square frame until resizing."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"CenterCrop": {"height": 100, "width": 120, "p": 1.0}, "Rotate": {"limit": (10, 10), "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
        )
        image, target = self._sample()
        crop, rotate = pipeline.transforms[:2]

        image, target = crop(image, target)
        image, _ = rotate(image, target)

        assert image.size == (120, 100)

    @_requires_kornia
    def test_gpu_prefix_preserves_kornia_grayscale_channels(self):
        """ToGray's ignored channel option stays ignored when rotation follows it."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"ToGray": {"num_output_channels": 1, "p": 1.0}, "Rotate": {"limit": 0, "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
        )
        image, target = pipeline(*self._sample())
        assert image.shape == (3, 128, 128)
        assert target["boxes"].shape == (1, 4)

    @_requires_kornia
    def test_gpu_prefix_keeps_masks_and_keypoints_aligned(self):
        """A pre-resize flip transforms boxes, masks, and relabeled keypoints together."""
        image, target = self._sample()
        mask = torch.zeros((1, 60, 80), dtype=torch.bool)
        mask[:, 10:40, 10:50] = True
        target["masks"] = mask
        target["keypoints"] = torch.tensor([[[20.0, 20.0, 2.0], [40.0, 30.0, 2.0]]])
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"HorizontalFlip": {"p": 1.0}, "Rotate": {"limit": (0, 0), "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
            keypoint_flip_pairs=[0, 1],
        )

        _, result = pipeline(image, target)

        torch.testing.assert_close(result["boxes"], torch.tensor([[48.0, 128 / 6, 112.0, 256 / 3]]), atol=2, rtol=0)
        torch.testing.assert_close(result["keypoints"][0, :, 0], torch.tensor([64.0, 96.0]), atol=1, rtol=0)
        assert result["masks"][0, 40, 64]
        assert not result["masks"][0, 40, 16]

    @_requires_kornia
    def test_gpu_prefix_preserves_values_outside_unit_range(self):
        """Kornia noise is not clipped or quantized before the batch pipeline."""
        torch.manual_seed(0)
        image, target = self._sample()
        image = Image.new("RGB", image.size, color="white")
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"GaussNoise": {"std_range": (0.5, 0.5), "p": 1.0}, "Rotate": {"limit": (0, 0), "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
        )

        result, _ = pipeline(image, target)

        assert result.max() > 1.0

    @_requires_kornia
    def test_gpu_prefix_filters_custom_fields_with_removed_boxes(self):
        """Instance metadata stays aligned when a pre-resize transform removes a box."""
        image, target = self._sample()
        target["boxes"] = torch.tensor([[10.0, 10.0, 20.0, 20.0], [70.0, 10.0, 79.0, 20.0]])
        target["labels"] = torch.tensor([1, 2])
        target["track_ids"] = torch.tensor([11, 22])
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"Affine": {"translate_px": {"x": 40, "y": 0}, "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
        )

        _, result = pipeline(image, target)

        assert result["labels"].tolist() == [1]
        assert result["track_ids"].tolist() == [11]

    @_requires_albumentations
    @_requires_kornia
    def test_gpu_prefix_accepts_pixel_translation_with_rotation(self):
        """A combined pixel shift and rotation remains accepted in the pre-resize prefix."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"Affine": {"translate_px": {"x": 4, "y": 0}, "rotate": (10, 10), "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
        )

        image, result = pipeline(*self._sample())

        assert image.shape == (3, 128, 128)
        assert result["boxes"].shape == (1, 4)

    @_requires_albumentations
    @_requires_kornia
    def test_mixed_pixel_affine_preserves_fit_output(self):
        """Mixed pixel affine keeps options supported by Albumentations before the resize."""
        config = {"Affine": {"translate_px": 4, "rotate": (10, 10), "fit_output": True, "p": 1.0}}
        pipeline = make_coco_transforms_square_div_64(
            "train", 128, aug_config=config, scale_jitter=False, skip_random_resize=True, gpu_postprocess=True
        )

        image, _ = pipeline(*self._sample())

        assert image.shape == (3, 128, 128)

    @_requires_albumentations
    @_requires_kornia
    def test_mixed_pixel_affine_fallback_preserves_grayscale_channels(self):
        """The Albumentations fallback retains Kornia's three-channel grayscale contract."""
        config = {
            "ToGray": {"num_output_channels": 1, "p": 1.0},
            "Affine": {"translate_px": 4, "rotate": (10, 10), "fit_output": True, "p": 1.0},
        }
        pipeline = make_coco_transforms_square_div_64(
            "train", 128, aug_config=config, scale_jitter=False, skip_random_resize=True, gpu_postprocess=True
        )

        image, _ = pipeline(*self._sample())

        assert image.shape == (3, 128, 128)

    @_requires_kornia
    def test_gpu_prefix_rejects_conflicting_translation_options(self):
        """Mixed affine geometry must not discard an additional percentage translation."""
        config = {"Affine": {"translate_px": 4, "translate_percent": 0.1, "rotate": (10, 10), "p": 1.0}}

        with pytest.raises(ValueError, match="either translate_px or translate_percent"):
            make_coco_transforms_square_div_64("train", 128, aug_config=config, gpu_postprocess=True)

    @_requires_kornia
    def test_gpu_prefix_combines_rotation_and_pixel_translation(self):
        """The configured pixel shift moves the rotated box in source-image coordinates."""
        config = {"Affine": {"translate_px": {"x": 0, "y": 0}, "rotate": (10, 10), "p": 1.0}}
        kwargs = {"scale_jitter": False, "skip_random_resize": True, "gpu_postprocess": True}
        unshifted = make_coco_transforms_square_div_64("train", 128, aug_config=config, **kwargs)
        config_shifted = {"Affine": {"translate_px": {"x": 4, "y": 0}, "rotate": (10, 10), "p": 1.0}}
        shifted = make_coco_transforms_square_div_64("train", 128, aug_config=config_shifted, **kwargs)

        _, target_before = unshifted(*self._sample())
        _, target_after = shifted(*self._sample())

        torch.testing.assert_close(
            target_after["boxes"] - target_before["boxes"], torch.tensor([[6.4, 0.0, 6.4, 0.0]]), atol=0.1, rtol=0
        )

    @_requires_kornia
    def test_gpu_rotation_does_not_require_albumentations(self, monkeypatch):
        """The Kornia path applies rotation before resize without Albumentations."""
        import rfdetr.datasets.transforms as transforms

        monkeypatch.setattr(transforms, "alb", None)
        pipeline = make_coco_transforms_square_div_64(
            "train",
            128,
            aug_config={"Rotate": {"limit": (10, 10), "p": 1.0}},
            scale_jitter=False,
            skip_random_resize=True,
            gpu_postprocess=True,
        )
        image, target = pipeline(*self._sample())
        assert image.shape == (3, 128, 128)
        assert target["boxes"].shape == (1, 4)


@_requires_albumentations
class TestAugmentBeforeResize:
    """Regression tests for #1769: geometric aug_config entries run before the square resize.

    A rotation applied after the square resize turns rigidly in the network input, a shape no camera produces. On a
    non-square image the augmentation must run on the image as loaded, so the resize then preserves the
    camera-realistic slant: the two axes of a tilted object turn by different angles.
    """

    _ROTATION = 7.5

    def _rotated_line_angles(self, pipeline: Compose) -> tuple[float, float]:
        """Run the reporter's 800x1500 line image through *pipeline* and measure the two line angles.

        Args:
            pipeline: Train transform pipeline built with a fixed 7.5 degree ``Affine`` rotation.

        Returns:
            ``(horizontal_angle, vertical_lean)`` in degrees: the horizontal line's turn from horizontal and the
            vertical line's lean from vertical, as the network input sees them.
        """
        width, height = 800, 1500
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        draw.line([(100, height // 2), (width - 100, height // 2)], fill=(255, 0, 0), width=9)
        draw.line([(width // 2, 150), (width // 2, height - 150)], fill=(0, 255, 0), width=9)
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor(1),
            "orig_size": torch.tensor([height, width]),
            "size": torch.tensor([height, width]),
        }
        output, _ = pipeline(image, target)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        if any(isinstance(t, Normalize) for t in pipeline.transforms):
            output = output * std + mean
        rgb = (output.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        horizontal = _line_angle((rgb[..., 0] > 128) & (rgb[..., 1] < 100))
        vertical = _line_angle((rgb[..., 1] > 128) & (rgb[..., 0] < 100))
        return horizontal, (vertical - 90 if vertical > 0 else vertical + 90)

    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_geometric_aug_wrapper_precedes_resize_wrapper(self, make_transforms):
        """The geometric aug wrapper sits before the resize wrapper in the train pipeline."""
        pipeline = make_transforms(
            "train",
            576,
            aug_config={"Affine": {"rotate": [self._ROTATION, self._ROTATION], "p": 1.0}},
            scale_jitter=False,
        )
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        aug_index = next(i for i, w in enumerate(wrappers) if "Affine" in _transform_leaf_names(w.transform))
        resize_index = next(
            i for i, w in enumerate(wrappers) if _transform_leaf_names(w.transform) & {"Resize", "SmallestMaxSize"}
        )

        assert aug_index < resize_index

    def test_rotation_on_non_square_image_keeps_camera_realistic_slant(self):
        """A 7.5 degree rotation of an 800x1500 image slants the two axes by different angles."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            576,
            aug_config={"Affine": {"rotate": [self._ROTATION, self._ROTATION], "p": 1.0}},
            scale_jitter=False,
        )

        horizontal, vertical_lean = self._rotated_line_angles(pipeline)

        # Rotate-then-resize reference values from the issue: a rigid -7.5 / -7.5 would mean the rotation ran
        # after the square resize.
        assert horizontal == pytest.approx(-4.0, abs=1.5)
        assert vertical_lean == pytest.approx(-13.9, abs=1.5)

    @_requires_kornia
    def test_gpu_postprocess_still_runs_geometry_before_resize(self):
        """With gpu_postprocess the geometric entry still runs on the CPU before the resize."""
        pipeline = make_coco_transforms_square_div_64(
            "train",
            576,
            aug_config={"Affine": {"rotate": [self._ROTATION, self._ROTATION], "p": 1.0}},
            scale_jitter=False,
            gpu_postprocess=True,
        )
        horizontal, vertical_lean = self._rotated_line_angles(pipeline)

        assert horizontal == pytest.approx(4.0, abs=1.5)
        assert vertical_lean == pytest.approx(13.9, abs=1.5)
        assert not any(isinstance(t, Normalize) for t in pipeline.transforms)

    _TRANSLATE_PX = 100

    def _translated_bar_centroid(self, pipeline: Compose) -> float:
        """Run a non-square image with one white vertical bar through *pipeline* and return the bar's x-centroid.

        Args:
            pipeline: Train transform pipeline built with a fixed 100-pixel ``Affine`` x-translation.

        Returns:
            The output x-centroid of the white bar, in output pixels.
        """
        width, height = 800, 600
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(200, 0), (209, height - 1)], fill=(255, 255, 255))
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor(1),
            "orig_size": torch.tensor([height, width]),
            "size": torch.tensor([height, width]),
        }
        output, _ = pipeline(image, target)
        if any(isinstance(t, Normalize) for t in pipeline.transforms):
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            output = output * std + mean
        gray = output.clamp(0, 1).mean(dim=0).numpy()
        xs = np.where(gray > 0.5)[1]
        return float(xs.mean())

    def test_translate_px_shifts_by_source_pixels_on_both_backends(self):
        """A fixed translate_px shift is in source-image pixels on the CPU and the GPU path.

        The 10-pixel bar starts centred at x=204.5 on an 800-pixel-wide image and shifts 100 source pixels before
        the square resize to 576, so its output centroid is (204.5 + 100) * 576 / 800 = 219.2. A shift applied after
        the resize would be 100 resized pixels (centroid 247.6), and the old split dropped it from the CPU pipeline
        entirely, leaving the centroid at 147.2.
        """
        aug_config = {"Affine": {"translate_px": {"x": self._TRANSLATE_PX}, "p": 1.0}}
        cpu_pipeline = make_coco_transforms_square_div_64(
            "train",
            576,
            aug_config=aug_config,
            scale_jitter=False,
        )
        gpu_pipeline = make_coco_transforms_square_div_64(
            "train",
            576,
            aug_config=aug_config,
            scale_jitter=False,
            gpu_postprocess=True,
        )

        assert self._translated_bar_centroid(cpu_pipeline) == pytest.approx(219.2, abs=6.0)
        assert self._translated_bar_centroid(gpu_pipeline) == pytest.approx(219.2, abs=6.0)

    @_requires_kornia
    def test_gpu_postprocess_keeps_flip_before_rotation(self):
        """With gpu_postprocess a flip configured before a rotation still runs before it.

        Flips do not commute with rotations, so the split must be a prefix cut in configured order: a partition by key
        would run the rotation on the CPU before the resize and the flip on the GPU after it, reversing the tilt
        relative to the CPU path.
        """
        pipeline = make_coco_transforms_square_div_64(
            "train",
            576,
            aug_config={"HorizontalFlip": {"p": 1.0}, "Rotate": {"limit": (30, 30), "p": 1.0}},
            scale_jitter=False,
            gpu_postprocess=True,
        )
        # A horizontal line is mirrored first; rotation then tilts it in the same direction as a plain rotation.
        horizontal, _ = self._rotated_line_angles(pipeline)
        assert horizontal == pytest.approx(-16.1, abs=2.0)


@_requires_albumentations
class TestMakeCocoTransformsOutputSize:
    """Regression tests for #979: transforms must resize high-resolution images to the target resolution.

    These tests verify that ``make_coco_transforms`` and ``make_coco_transforms_square_div_64`` actually produce output
    images at the requested ``resolution``, not at the original image size. Existing tests only check pipeline
    *structure*; these check actual output *dimensions*.
    """

    # 1920x1080 (landscape) — larger than any typical training resolution.
    # PIL size is (width, height), so Image.new("RGB", (1920, 1080)) gives a 1920-wide, 1080-tall image.
    _INPUT_W = 1920
    _INPUT_H = 1080
    _RESOLUTION = 640

    def _make_image(self) -> Image.Image:
        return Image.new("RGB", (self._INPUT_W, self._INPUT_H))

    def test_square_val_resizes_large_image(self) -> None:
        """Square val transform resizes 1920x1080 to exactly 640x640."""
        transform = make_coco_transforms_square_div_64("val", self._RESOLUTION)
        tensor, _ = transform(self._make_image(), None)
        assert tensor.shape[-2:] == (self._RESOLUTION, self._RESOLUTION)

    def test_square_train_resizes_large_image(self) -> None:
        """Square train transform resizes 1920x1080 to 640x640 regardless of OneOf branch."""
        transform = make_coco_transforms_square_div_64("train", self._RESOLUTION, aug_config={})
        tensor, _ = transform(self._make_image(), None)
        assert tensor.shape[-2:] == (self._RESOLUTION, self._RESOLUTION)

    def test_nonsquare_val_resizes_and_caps_longest_side(self) -> None:
        """Non-square val transform resizes the image and keeps the longest side within 1333 px.

        Avoid asserting an exact output dimension here because Albumentations resize behavior can vary across supported
        versions. The stable contract is that the image is resized and the longest side does not exceed the configured
        maximum.
        """
        transform = make_coco_transforms("val", self._RESOLUTION)
        tensor, _ = transform(self._make_image(), None)
        height, width = tensor.shape[-2], tensor.shape[-1]
        assert (height, width) != (self._INPUT_H, self._INPUT_W)
        assert max(height, width) <= 1333

    def test_nonsquare_val_longest_side_at_most_1333(self) -> None:
        """Non-square val transform caps the longest side at 1333 px.

        Use an input that still exceeds 1333 px on its longest side after SmallestMaxSize(640), so this assertion
        specifically validates that LongestMaxSize(1333) is applied.
        """
        transform = make_coco_transforms("val", self._RESOLUTION)
        image = Image.new("RGB", (4000, 1000))
        tensor, _ = transform(image, None)
        height, width = tensor.shape[-2], tensor.shape[-1]
        assert max(height, width) <= 1333

    def test_square_val_does_not_pass_original_dimensions(self) -> None:
        """Square val transform must not emit the original 1920x1080 dimensions — the core regression."""
        transform = make_coco_transforms_square_div_64("val", self._RESOLUTION)
        tensor, _ = transform(self._make_image(), None)
        height, width = tensor.shape[-2], tensor.shape[-1]
        assert (height, width) != (self._INPUT_H, self._INPUT_W), (
            f"Transform emitted original {self._INPUT_H}x{self._INPUT_W} — resize was not applied"
        )

    def test_output_is_float_tensor(self) -> None:
        """Transform pipeline produces a float32 tensor, not a PIL Image."""
        transform = make_coco_transforms_square_div_64("val", self._RESOLUTION)
        tensor, _ = transform(self._make_image(), None)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32


@_requires_albumentations
class TestKeypointScalingAcrossResolutions:
    """Keypoint coordinates are correctly normalised when training at non-default resolutions.

    The full transform pipeline — Resize → ToImage → ToDtype → Normalize — applies Normalize last. Normalize divides
    keypoint x by image width and y by image height, so the model always receives relative [0, 1] coordinates regardless
    of the configured training resolution.  These tests confirm that contract holds for all supported resolutions (576
    default, 640, 768, 960).
    """

    # Input image: 480 wide, 360 tall. Box occupies the centre quadrant.
    _INPUT_W = 480
    _INPUT_H = 360
    # Keypoint at exactly 60 % of each axis — well inside the box, visibility=2.
    _KP_FRAC = 0.6
    _KP_X = _KP_FRAC * _INPUT_W  # 288.0
    _KP_Y = _KP_FRAC * _INPUT_H  # 216.0

    def _make_target(self) -> dict:
        return {
            "boxes": torch.tensor([[120.0, 90.0, 360.0, 270.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[self._KP_X, self._KP_Y, 2.0], [0.0, 0.0, 0.0]]]),
        }

    @pytest.mark.parametrize("resolution", [576, 640, 768, 960])
    def test_val_pipeline_keypoints_normalised_to_unit_range(self, resolution: int) -> None:
        """After Normalize, visible keypoint coords are in [0, 1] for every resolution."""
        transform = make_coco_transforms_square_div_64("val", resolution)
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        _, transformed = transform(image, self._make_target())

        kp = transformed["keypoints"]
        assert kp[0, 0, 0].item() == pytest.approx(self._KP_FRAC, abs=1e-3), (
            f"normalised keypoint x must equal original fraction {self._KP_FRAC} at resolution={resolution}"
        )
        assert kp[0, 0, 1].item() == pytest.approx(self._KP_FRAC, abs=1e-3), (
            f"normalised keypoint y must equal original fraction {self._KP_FRAC} at resolution={resolution}"
        )
        assert kp[0, 0, 2].item() == pytest.approx(2.0), "visibility must be preserved after normalisation"

    @pytest.mark.parametrize("resolution", [576, 640, 768, 960])
    def test_val_pipeline_invisible_keypoints_stay_zero(self, resolution: int) -> None:
        """Zero-visibility keypoints must remain at (0, 0, 0) after the full pipeline."""
        transform = make_coco_transforms_square_div_64("val", resolution)
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        _, transformed = transform(image, self._make_target())

        kp = transformed["keypoints"]
        torch.testing.assert_close(
            kp[0, 1],
            torch.zeros(3),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize("resolution", [576, 640, 768, 960])
    def test_train_pipeline_visible_keypoints_normalised_in_unit_range(self, resolution: int) -> None:
        """Train transform (random resize/crop) must produce normalised keypoints in [0, 1]."""
        transform = make_coco_transforms_square_div_64("train", resolution, aug_config={})
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        _, transformed = transform(image, self._make_target())

        kp = transformed["keypoints"]  # shape (N, K, 3); x and y already normalised by Normalize
        visible_mask = kp[:, :, 2] > 0  # (N, K) bool; an empty selection passes vacuously
        visible_x = kp[:, :, 0][visible_mask]
        visible_y = kp[:, :, 1][visible_mask]
        assert (visible_x >= 0).all() and (visible_x <= 1.0).all(), (
            f"normalised keypoint x out of [0, 1] at resolution={resolution}: {visible_x}"
        )
        assert (visible_y >= 0).all() and (visible_y <= 1.0).all(), (
            f"normalised keypoint y out of [0, 1] at resolution={resolution}: {visible_y}"
        )

    @pytest.mark.parametrize("resolution", [576, 640, 768, 960])
    def test_output_tensor_shape_matches_resolution(self, resolution: int) -> None:
        """Transform output image tensor must be (C, resolution, resolution) for all resolutions."""
        transform = make_coco_transforms_square_div_64("val", resolution, aug_config={})
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        tensor, _ = transform(image, None)

        assert tensor.shape[-2:] == (resolution, resolution), (
            f"expected (C, {resolution}, {resolution}), got {tuple(tensor.shape)}"
        )


@_requires_albumentations
class TestNormalize:
    """Unit tests for Normalize.__call__."""

    def test_call_is_bound_method(self) -> None:
        """Normalize.__call__ must be a class method, not a module-level function."""
        import inspect

        normalize = Normalize()
        assert callable(normalize), "Normalize instance must be callable"
        assert inspect.ismethod(normalize.__call__), "__call__ must be a bound method"

    def test_call_image_only_returns_tensor_and_none(self) -> None:
        """Normalize(image, None) returns (tensor, None) without raising."""
        normalize = Normalize()
        image = torch.zeros(3, 64, 64)
        out_img, out_tgt = normalize(image, None)
        assert isinstance(out_img, torch.Tensor)
        assert out_tgt is None

    @pytest.mark.parametrize(
        "boxes,image_hw,expected_cxcywh_norm",
        [
            pytest.param(
                torch.tensor([[0.0, 0.0, 100.0, 50.0]]),
                (50, 100),
                torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
                id="full_image_box",
            ),
            pytest.param(
                torch.tensor([[10.0, 10.0, 30.0, 40.0]]),
                (100, 100),
                torch.tensor([[0.2, 0.25, 0.2, 0.3]]),
                id="non_square_box",
            ),
        ],
    )
    def test_call_normalizes_boxes(
        self,
        boxes: torch.Tensor,
        image_hw: tuple[int, int],
        expected_cxcywh_norm: torch.Tensor,
    ) -> None:
        """Normalize.__call__ converts boxes from xyxy pixel coords to normalized cxcywh."""
        height, width = image_hw
        normalize = Normalize()
        image = torch.zeros(3, height, width)
        target = {"boxes": boxes.clone()}
        _, out_tgt = normalize(image, target)
        torch.testing.assert_close(out_tgt["boxes"], expected_cxcywh_norm, atol=1e-4, rtol=0.0)

    def test_call_normalizes_keypoints_to_unit_range(self) -> None:
        """Normalize.__call__ divides keypoint x by width and y by height."""
        normalize = Normalize()
        height, width = 100, 200
        image = torch.zeros(3, height, width)
        kp = torch.tensor([[[100.0, 50.0, 2.0]]])  # x=100 of 200w, y=50 of 100h
        target = {"boxes": torch.zeros(1, 4), "keypoints": kp}
        _, out_tgt = normalize(image, target)
        assert out_tgt["keypoints"][0, 0, 0].item() == pytest.approx(0.5, abs=1e-5)
        assert out_tgt["keypoints"][0, 0, 1].item() == pytest.approx(0.5, abs=1e-5)
        assert out_tgt["keypoints"][0, 0, 2].item() == pytest.approx(2.0)

    def test_call_does_not_mutate_original_target(self) -> None:
        """Normalize.__call__ must not mutate the caller's target dict."""
        normalize = Normalize()
        image = torch.zeros(3, 50, 100)
        boxes_original = torch.tensor([[0.0, 0.0, 100.0, 50.0]])
        target = {"boxes": boxes_original.clone()}
        normalize(image, target)
        torch.testing.assert_close(target["boxes"], boxes_original, rtol=0.0, atol=0.0)


@_requires_albumentations
class TestReplayHorizontalFlipParity:
    """Unit tests for AlbumentationsWrapper._replay_horizontal_flip_parity using fixture dicts."""

    @pytest.mark.parametrize("num_flips", [0, 1, 2, 3])
    @pytest.mark.parametrize("axis_key", ["axis", "d"])
    def test_legacy_flip_parity(self, num_flips: int, axis_key: str) -> None:
        """Legacy Flip replay parameters contribute to horizontal-flip parity."""
        replay = {
            "transforms": [
                {"__class_fullname__": "Flip", "applied": True, "params": {axis_key: 1}} for _ in range(num_flips)
            ]
        }

        assert AlbumentationsWrapper._replay_horizontal_flip_parity(replay) == bool(num_flips % 2)

    @pytest.mark.parametrize(
        "replay,expected",
        [
            pytest.param(
                {"__class_fullname__": "HorizontalFlip", "applied": True, "params": {}},
                True,
                id="horizontal-flip-applied",
            ),
            pytest.param(
                {"__class_fullname__": "HorizontalFlip", "applied": False, "params": {}},
                False,
                id="horizontal-flip-not-applied",
            ),
            pytest.param(
                {"__class_fullname__": "TimeReverse", "applied": True, "params": {}},
                True,
                id="time-reverse-applied",
            ),
            pytest.param(
                {"__class_fullname__": "Flip", "applied": True, "params": {"axis": 1}},
                True,
                id="flip-horizontal-axis",
            ),
            pytest.param(
                {"__class_fullname__": "Flip", "applied": True, "params": {"axis": 0}},
                False,
                id="flip-vertical-axis",
            ),
            pytest.param(
                {"__class_fullname__": "Flip", "applied": False, "params": {"axis": 1}},
                False,
                id="flip-not-applied",
            ),
            pytest.param(
                {
                    "__class_fullname__": "D4",
                    "applied": True,
                    "params": {"group_element": "h"},
                },
                True,
                id="d4-horizontal-element",
            ),
            pytest.param(
                {
                    "__class_fullname__": "D4",
                    "applied": True,
                    "params": {"group_element": "r90"},
                },
                False,
                id="d4-rotation-element",
            ),
            pytest.param(
                {
                    "__class_fullname__": "D4",
                    "applied": False,
                    "params": {"group_element": "h"},
                },
                False,
                id="d4-not-applied",
            ),
            pytest.param(
                {
                    "__class_fullname__": "SquareSymmetry",
                    "applied": True,
                    "params": {"group_element": "h"},
                },
                True,
                id="square-symmetry-horizontal",
            ),
            pytest.param(
                {
                    "__class_fullname__": "SquareSymmetry",
                    "applied": True,
                    "params": {"group_element": "r90"},
                },
                False,
                id="square-symmetry-rotation",
            ),
            pytest.param(
                None,
                False,
                id="none-replay",
            ),
            pytest.param(
                "not-a-dict",
                False,
                id="non-dict-replay",
            ),
            pytest.param(
                {
                    "transforms": [
                        {"__class_fullname__": "HorizontalFlip", "applied": True, "params": {}},
                    ]
                },
                True,
                id="nested-horizontal-flip",
            ),
            pytest.param(
                {
                    "transforms": [
                        {"__class_fullname__": "HorizontalFlip", "applied": False, "params": {}},
                    ]
                },
                False,
                id="nested-horizontal-flip-not-applied",
            ),
        ],
    )
    def test_replay_horizontal_flip_parity(self, replay: object, expected: bool) -> None:
        """Fixture replay dicts should be correctly classified as horizontal flip or not."""
        assert AlbumentationsWrapper._replay_horizontal_flip_parity(replay) == expected
