# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Albumentations augmentations inside the COCO transform pipelines and the training data flow.

Also covers keypoint handling: horizontal-flip filtering, flip parity and keypoint scaling across resolutions.
"""

from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from rfdetr.datasets._aug_utils import filter_keypoint_hflip_augmentations
from rfdetr.datasets._develop import _SimpleDataset
from rfdetr.datasets._torchvision import RandomHorizontalFlip
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.utilities import collate_fn
from tests._markers import _requires_albumentations


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
        aug_wrappers = wrappers[1:]  # skip resize wrapper

        assert len(aug_wrappers) == 1
        assert aug_wrappers[0].transform.transforms[0].__class__.__name__ == "HorizontalFlip"

    @pytest.mark.parametrize("split", ["val", "val_speed", "test"])
    @pytest.mark.parametrize("make_transforms", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_aug_config_ignored_outside_train(self, make_transforms, split):
        """aug_config is ignored for val, val_speed and test splits, which keep the torchvision resize."""
        pipeline = make_transforms(split, 640, aug_config={"HorizontalFlip": {"p": 1.0}})

        assert not any(isinstance(t, AlbumentationsWrapper) for t in pipeline.transforms)


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
