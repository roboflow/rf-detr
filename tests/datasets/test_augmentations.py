# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Albumentations augmentation wrappers."""

from unittest import mock

import albumentations as alb
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from rfdetr.datasets._aug_utils import filter_keypoint_hflip_augmentations
from rfdetr.datasets._develop import _SimpleDataset
from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE, AUG_CONFIG
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize, _build_albu_transform
from rfdetr.utilities import collate_fn


class _FakeRandomSizedCropV2:
    """Test double for Albumentations 2.x-style RandomSizedCrop API."""

    def __init__(self, *, min_max_height, size, p=1.0):
        self.min_max_height = min_max_height
        self.size = size
        self.p = p


class _FakeRandomSizedCropV1:
    """Test double for Albumentations 1.x-style RandomSizedCrop API."""

    def __init__(self, *, min_max_height, height, width, p=1.0):
        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.p = p


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


class TestAlbumentationsWrapper:
    """Tests for AlbumentationsWrapper class."""

    @pytest.mark.parametrize(
        "transform_class,params,box_in,box_out",
        [
            (alb.HorizontalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [70.0, 20.0, 90.0, 40.0]),
            (alb.VerticalFlip, {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [10.0, 60.0, 30.0, 80.0]),
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

    def test_resize_transforms_keypoint_coordinates(self):
        """Resize scales keypoint coordinates and preserves visibility values."""
        wrapper = AlbumentationsWrapper(alb.Resize(height=100, width=200, p=1.0))
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[15.0, 10.0, 2.0], [0.0, 0.0, 0.0]]]),
        }

        _, transformed = wrapper(image, target)

        torch.testing.assert_close(
            transformed["boxes"],
            torch.tensor([[20.0, 10.0, 60.0, 50.0]]),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[30.0, 20.0, 2.0], [0.0, 0.0, 0.0]]]),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_horizontal_flip_transforms_keypoint_coordinates(self):
        """Horizontal flip mirrors keypoint coordinates using Albumentations geometry."""
        wrapper = AlbumentationsWrapper(alb.HorizontalFlip(p=1.0))
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 5.0, 2.0], [30.0, 25.0, 2.0], [0.0, 0.0, 0.0]]]),
        }

        _, transformed = wrapper(image, target)

        torch.testing.assert_close(
            transformed["boxes"],
            torch.tensor([[70.0, 5.0, 90.0, 25.0]]),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[89.0, 5.0, 2.0], [69.0, 25.0, 2.0], [0.0, 0.0, 0.0]]]),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "num_instances",
        [
            pytest.param(0, id="zero_instances"),
            pytest.param(1, id="one_instance"),
            pytest.param(2, id="two_instances"),
        ],
    )
    def test_horizontal_flip_with_keypoint_flip_pairs_handles_ndarray_bboxes(self, num_instances):
        """Regression test for #1125.

        Albumentations 2.x returns ``bboxes`` as a NumPy ndarray of shape (N, 4); 1.x returned a list of tuples. The
        horizontal-flip swap path used to inspect ``bboxes`` with list-style truthiness, which raised ``ValueError: The
        truth value of an array with more than one element is ambiguous`` on any ndarray with more than one element.
        This test exercises that path across multi/single/empty instance counts.
        """
        wrapper = AlbumentationsWrapper(
            alb.HorizontalFlip(p=1.0),
            keypoint_flip_pairs=[0, 1],
        )

        image = Image.new("RGB", (100, 50))

        if num_instances == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
                "keypoints": torch.zeros((0, 2, 3), dtype=torch.float32),
            }
        else:
            boxes = []
            keypoints = []
            for i in range(num_instances):
                x = 10.0 + i * 30.0
                boxes.append([x, 5.0, x + 20.0, 25.0])
                keypoints.append([[x + 5.0, 10.0, 2.0], [x + 15.0, 20.0, 2.0]])
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor([1] * num_instances, dtype=torch.long),
                "keypoints": torch.tensor(keypoints, dtype=torch.float32),
            }

        # Pre-fix: this call raised ValueError for num_instances >= 1 under Albumentations 2.x.
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape[0] == num_instances
        assert aug_target["keypoints"].shape[0] == num_instances

    def test_horizontal_flip_swaps_paired_keypoints(self):
        """HFlip with keypoint_flip_pairs exchanges keypoint slots for the configured pair."""
        wrapper = AlbumentationsWrapper(
            alb.HorizontalFlip(p=1.0),
            keypoint_flip_pairs=[0, 1],
        )
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 95.0, 45.0]]),
            "labels": torch.tensor([1]),
            # kp0 at x=10 (left), kp1 at x=80 (right)
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
        }

        _, transformed = wrapper(image, target)

        kp = transformed["keypoints"][0]  # shape [2, 3]
        # After HFlip (W=100): kp0→x=89, kp1→x=19. After swap: slot0 gets kp1's flipped x=19,
        # slot1 gets kp0's flipped x=89. Without swap the ordering would be inverted (89 > 19).
        torch.testing.assert_close(kp[0, 0], torch.tensor(19.0), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(kp[1, 0], torch.tensor(89.0), rtol=1e-4, atol=1e-6)

    def test_nested_horizontal_flip_swaps_slots_after_all_geometry(self):
        """Nested HFlip+VFlip should mirror coordinates once, then swap only the left/right slots."""
        wrapper = AlbumentationsWrapper(
            alb.Sequential([alb.HorizontalFlip(p=1.0), alb.VerticalFlip(p=1.0)], p=1.0),
            keypoint_flip_pairs=[0, 1],
        )
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 95.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor(
                [
                    [
                        [10.0, 10.0, 2.0],
                        [80.0, 30.0, 2.0],
                        [50.0, 20.0, 1.0],
                    ]
                ]
            ),
        }

        _, transformed = wrapper(image, target)

        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[19.0, 19.0, 2.0], [89.0, 39.0, 2.0], [49.0, 29.0, 1.0]]]),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "transform,expected_keypoints",
        [
            pytest.param(
                alb.HorizontalFlip(p=0.0),
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="disabled-horizontal-flip",
            ),
            pytest.param(
                alb.VerticalFlip(p=1.0),
                torch.tensor([[[10.0, 39.0, 2.0], [80.0, 19.0, 2.0]]]),
                id="vertical-flip",
            ),
            pytest.param(
                alb.Resize(height=50, width=100, p=1.0),
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="resize",
            ),
            pytest.param(
                alb.Crop(x_min=0, y_min=0, x_max=100, y_max=50, p=1.0),
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="full-image-crop",
            ),
        ],
    )
    def test_non_horizontal_geometry_does_not_swap_paired_keypoints(self, transform, expected_keypoints):
        """Configured HFlip pairs should not swap slots when no horizontal flip applied."""
        wrapper = AlbumentationsWrapper(transform, keypoint_flip_pairs=[0, 1])
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 95.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
        }

        _, transformed = wrapper(image, target)

        torch.testing.assert_close(transformed["keypoints"], expected_keypoints, rtol=1e-4, atol=1e-6)

    def test_crop_filters_keypoints_with_removed_boxes(self):
        """When a crop removes a box, its keypoints are removed with the same instance."""
        wrapper = AlbumentationsWrapper(alb.Crop(x_min=0, y_min=0, x_max=50, y_max=50, p=1.0))
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0], [60.0, 5.0, 80.0, 25.0]]),
            "labels": torch.tensor([1, 1]),
            "area": torch.tensor([400.0, 400.0]),
            "keypoints": torch.tensor(
                [
                    [[15.0, 10.0, 2.0], [25.0, 20.0, 2.0]],
                    [[65.0, 10.0, 2.0], [75.0, 20.0, 2.0]],
                ]
            ),
        }

        _, transformed = wrapper(image, target)

        assert transformed["boxes"].shape == (1, 4)
        assert transformed["labels"].tolist() == [1]
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[15.0, 10.0, 2.0], [25.0, 20.0, 2.0]]]),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_non_geometric_transform_preserves_boxes(self):
        """Test that non-geometric transforms preserve bounding boxes."""
        transform = alb.GaussianBlur(blur_limit=3, p=1.0)
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
        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)

    def test_multiple_boxes(self):
        """Test wrapper handles multiple bounding boxes."""
        transform = alb.HorizontalFlip(p=1.0)
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
        transform = alb.Resize(height=64, width=64)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        aug_image, aug_target = wrapper(image, None)

        assert isinstance(aug_image, Image.Image)
        assert aug_image.size == (64, 64)
        assert aug_target is None

    def test_invalid_target_type(self):
        """Test wrapper raises error for invalid target type."""
        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))

        with pytest.raises(TypeError, match="target must be a dictionary"):
            wrapper(image, "invalid_target")

    def test_missing_labels_key(self):
        """Test wrapper raises error when labels key is missing."""
        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]])}

        with pytest.raises(KeyError, match="target must contain 'labels' key"):
            wrapper(image, target)

    def test_invalid_boxes_shape(self):
        """Test wrapper raises error for invalid boxes shape."""
        transform = alb.HorizontalFlip(p=1.0)
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

        Regression test for bug where orig_size (shape [2] for [h, w]) was incorrectly treated as a per-instance field
        when there were exactly 2 boxes, causing orig_size to be filtered/indexed incorrectly and leading to
        inconsistent tensor shapes in batches.
        """
        transform = alb.HorizontalFlip(p=1.0)
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

        Critical regression test: With 2 boxes, both orig_size and masks have first dimension = 2, but they must be
        treated differently:
        - orig_size (shape [2]): global field, should NOT be filtered
        - masks (shape [2, H, W]): per-instance field, SHOULD be transformed
        """
        transform = alb.HorizontalFlip(p=1.0)
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
            (alb.HorizontalFlip, {"p": 1.0}),
            (alb.VerticalFlip, {"p": 1.0}),
            (alb.Rotate, {"limit": 45, "p": 1.0}),
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
        transform = alb.HorizontalFlip(p=1.0)
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
            (alb.HorizontalFlip, {"p": 1.0}),
            (alb.VerticalFlip, {"p": 1.0}),
            (alb.Rotate, {"limit": 15, "p": 1.0}),  # Small angle to avoid boxes going out
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
            (alb.GaussianBlur, {"blur_limit": 3, "p": 1.0}),
            (alb.RandomBrightnessContrast, {"p": 1.0}),
            (alb.GaussNoise, {"p": 1.0}),
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
        transform = alb.HorizontalFlip(p=1.0)
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
        transform = alb.HorizontalFlip(p=1.0)
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

        Regression test for: when a prior crop removes all annotations, target["masks"] has shape (0, H, W). Passing an
        empty list to albumentations raises ValueError: masks cannot be empty.
        """
        transform = alb.HorizontalFlip(p=1.0)
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
        transform = alb.GaussianBlur(blur_limit=3, p=1.0)
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
        transform = alb.HorizontalFlip(p=1.0)
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
        transform = alb.HorizontalFlip(p=1.0)
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
        transform = alb.Crop(x_min=0, y_min=0, x_max=50, y_max=50, p=1.0)
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

    def test_degenerate_bbox_at_image_boundary_is_silently_dropped(self):
        """Degenerate boxes (x_min == x_max or y_min == y_max) must not raise ValueError.

        Regression test: COCO annotations sometimes place a box exactly on the image boundary so that both x coordinates
        equal the image width (normalized: 1.0). Albumentations' check_bboxes rejects these with "x_max is less than or
        equal to x_min", crashing the DataLoader worker.
        """
        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        width, height = 100, 100
        image = Image.new("RGB", (width, height))

        target = {
            # box 0: valid                box 1: x_min==x_max (right edge)
            # box 2: y_min==y_max (bottom edge)
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 50.0, 60.0],  # valid — should survive
                    [100.0, 14.0, 100.0, 17.0],  # degenerate: x_min == x_max
                    [10.0, 100.0, 50.0, 100.0],  # degenerate: y_min == y_max
                ]
            ),
            "labels": torch.tensor([1, 2, 3]),
            "area": torch.tensor([1600.0, 0.0, 0.0]),
        }

        # Must not raise ValueError
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Only the valid box survives
        assert aug_target["boxes"].shape[0] == 1, f"Expected 1 valid box, got {aug_target['boxes'].shape[0]}"
        assert aug_target["labels"].tolist() == [1]
        assert aug_target["area"].shape[0] == 1

    def test_degenerate_bbox_mixed_with_masks(self):
        """Degenerate boxes are dropped together with their corresponding masks."""
        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        width, height = 100, 100
        image = Image.new("RGB", (width, height))

        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 20:60, 10:50] = 1  # valid mask

        target = {
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 50.0, 60.0],  # valid
                    [100.0, 14.0, 100.0, 17.0],  # degenerate: x_min == x_max
                ]
            ),
            "labels": torch.tensor([1, 2]),
            "masks": masks,
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_target["boxes"].shape[0] == 1
        assert aug_target["labels"].tolist() == [1]
        assert aug_target["masks"].shape[0] == 1


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
            "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.1, 0.1), "p": 0.3},
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


class TestRandomSizedCropCompat:
    """Tests for RandomSizedCrop cross-version parameter normalization edge cases."""

    @pytest.mark.parametrize(
        "params, expected_missing",
        [
            pytest.param(
                {"min_max_height": [100, 200], "height": 256},
                "width",
                id="height_without_width",
            ),
            pytest.param(
                {"min_max_height": [100, 200], "width": 256},
                "height",
                id="width_without_height",
            ),
        ],
    )
    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
    def test_errors_on_partial_hw_with_v2_api(self, params, expected_missing):
        with pytest.raises(ValueError, match=f"missing '{expected_missing}'"):
            _build_albu_transform("RandomSizedCrop", params)

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                {"min_max_height": [100, 200], "size": (256, 256), "height": 256},
                id="size_and_height",
            ),
            pytest.param(
                {"min_max_height": [100, 200], "size": (256, 256), "width": 256},
                id="size_and_width",
            ),
            pytest.param(
                {
                    "min_max_height": [100, 200],
                    "size": (256, 256),
                    "height": 256,
                    "width": 256,
                },
                id="size_and_height_and_width",
            ),
        ],
    )
    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
    def test_size_takes_precedence_over_hw_on_v2_api(self, params):
        # No TypeError means height/width were correctly dropped before instantiation
        transform = _build_albu_transform("RandomSizedCrop", params)
        assert transform.size == (256, 256)

    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV1)
    def test_scalar_size_passes_through_on_v1_legacy_path(self):
        # Scalar size=640 does not match isinstance(size, Sequence), so the v1
        # legacy branch leaves it in the params dict. FakeV1 does not accept
        # ``size`` so this should raise a TypeError from the constructor — our
        # normalization code does NOT raise a ValueError for this case.
        with pytest.raises(TypeError):
            _build_albu_transform(
                "RandomSizedCrop",
                {"min_max_height": [100, 200], "size": 640},
            )

    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
    def test_adapts_height_width_for_v2_api(self):
        """RandomSizedCrop config with height/width is adapted to the Albumentations 2.x size API."""
        transform = _build_albu_transform(
            "RandomSizedCrop",
            {"min_max_height": [384, 600], "height": 640, "width": 640},
        )

        assert isinstance(transform, _FakeRandomSizedCropV2)
        assert transform.min_max_height == [384, 600]
        assert transform.size == (640, 640)

    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV1)
    def test_adapts_size_for_v1_api(self):
        """RandomSizedCrop config with size is adapted to the Albumentations 1.x height/width API."""
        transform = _build_albu_transform(
            "RandomSizedCrop",
            {"min_max_height": [384, 600], "size": (640, 640)},
        )

        assert isinstance(transform, _FakeRandomSizedCropV1)
        assert transform.min_max_height == [384, 600]
        assert transform.height == 640
        assert transform.width == 640

    @mock.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
    def test_from_config_partial_height_is_silently_skipped(self):
        """from_config swallows the ValueError for partial height-only config and skips the transform.

        This documents the intentional silent-skip behavior: from_config wraps _build_albu_transform in a broad except
        clause so bad configs produce a warning rather than an exception.
        """
        config = {
            "HorizontalFlip": {"p": 0.5},
            "RandomSizedCrop": {"min_max_height": [100, 200], "height": 256},
        }

        transforms = AlbumentationsWrapper.from_config(config)

        # The invalid RandomSizedCrop is silently dropped; only HorizontalFlip survives.
        assert len(transforms) == 1
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == ["HorizontalFlip"]


class TestAlbumentationsWrapperNestedConfig:
    """Tests for nested container (OneOf, SomeOf, Sequential) support in from_config."""

    def test_one_of_geometric_detection(self):
        """OneOf containing a geometric transform is treated as geometric."""
        wrapper = AlbumentationsWrapper(alb.OneOf([alb.HorizontalFlip(p=1.0), alb.GaussianBlur(p=1.0)]))
        assert wrapper._is_geometric is True

    def test_one_of_pixel_detection(self):
        """OneOf containing only pixel transforms is treated as pixel-level."""
        wrapper = AlbumentationsWrapper(alb.OneOf([alb.GaussianBlur(p=1.0), alb.Blur(p=1.0)]))
        assert wrapper._is_geometric is False

    def test_sequential_geometric_detection(self):
        """Sequential containing a geometric transform is treated as geometric."""
        wrapper = AlbumentationsWrapper(alb.Sequential([alb.Rotate(limit=45, p=1.0), alb.GaussianBlur(p=1.0)]))
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
        assert isinstance(inner, alb.OneOf)
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
        assert isinstance(inner, alb.Sequential)
        assert isinstance(inner.transforms[0], alb.OneOf)

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
        assert isinstance(inner, alb.OneOf)
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
        assert isinstance(inner, alb.Sequential)
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
        assert isinstance(transforms[1].transform.transforms[0], alb.OneOf)

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
        assert isinstance(inner, alb.OneOf)
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
        assert isinstance(inner, alb.Sequential)
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

        assert isinstance(inner, alb.SomeOf)
        assert inner.p == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "hflip_name",
        [
            pytest.param("HorizontalFlip", id="HorizontalFlip"),
            pytest.param("Flip", id="Flip"),
            pytest.param("D4", id="D4"),
        ],
    )
    def test_hflip_disabled_for_keypoint_pipeline(self, hflip_name: str) -> None:
        """HFlip-type transforms are skipped when keypoint_flip_pairs is provided."""
        config = {hflip_name: {"p": 0.5}, "GaussianBlur": {"p": 0.5}}

        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[])

        names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert hflip_name not in names
        assert "GaussianBlur" in names

    def test_hflip_disable_logs_warning(self) -> None:
        """from_config logs a warning when HorizontalFlip is disabled for a keypoint pipeline."""
        config = {"HorizontalFlip": {"p": 0.5}}

        with mock.patch("rfdetr.datasets.transforms.logger") as mock_log:
            AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[])

        assert mock_log.warning.called
        call_args = mock_log.warning.call_args[0]
        assert "HorizontalFlip" in str(call_args)

    def test_hflip_included_when_no_keypoints(self) -> None:
        """HorizontalFlip is NOT skipped when keypoint_flip_pairs is None (detection pipeline)."""
        config = {"HorizontalFlip": {"p": 0.5}}

        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=None)

        names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert "HorizontalFlip" in names

    def test_hflip_included_when_keypoint_flip_pairs_are_configured(self) -> None:
        """HorizontalFlip is safe for keypoint pipelines when semantic flip pairs are configured."""
        config = {"HorizontalFlip": {"p": 0.5}}

        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[0, 1])

        names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert "HorizontalFlip" in names


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

        This is a critical integration test that simulates actual training conditions where multiple samples with
        different numbers of boxes are batched together. It specifically tests that orig_size remains consistent across
        the batch.
        """
        # Create augmentations
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
        "transform_class,transform_kwargs",
        [
            pytest.param(alb.HorizontalFlip, {"p": 1.0}, id="horizontal_flip"),
            pytest.param(alb.VerticalFlip, {"p": 1.0}, id="vertical_flip"),
            pytest.param(alb.RandomRotate90, {"p": 1.0}, id="random_rotate_90"),
        ],
    )
    @pytest.mark.parametrize(
        "include_masks", [pytest.param(False, id="detection"), pytest.param(True, id="segmentation")]
    )
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

    @pytest.mark.parametrize(
        "make_transforms,expected_resize_wrappers",
        [
            # make_coco_transforms test: SmallestMaxSize + LongestMaxSize = 2 wrappers
            pytest.param(make_coco_transforms, 2, id="make_coco_transforms"),
            # make_coco_transforms_square_div_64 test: Resize = 1 wrapper
            pytest.param(make_coco_transforms_square_div_64, 1, id="make_coco_transforms_square_div_64"),
        ],
    )
    def test_aug_config_not_applied_on_test(self, make_transforms, expected_resize_wrappers):
        """aug_config is ignored for test splits — only resize wrappers are present."""
        pipeline = make_transforms("test", 640, aug_config={"HorizontalFlip": {"p": 1.0}})
        wrappers = [t for t in pipeline.transforms if isinstance(t, AlbumentationsWrapper)]
        assert len(wrappers) == expected_resize_wrappers


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

    def test_nonsquare_val_does_not_pass_original_dimensions(self) -> None:
        """Non-square val transform must not emit the original 1920x1080 dimensions — the core regression."""
        transform = make_coco_transforms("val", self._RESOLUTION)
        tensor, _ = transform(self._make_image(), None)
        height, width = tensor.shape[-2], tensor.shape[-1]
        assert (height, width) != (self._INPUT_H, self._INPUT_W), (
            f"Transform emitted original {self._INPUT_H}x{self._INPUT_W} — resize was not applied"
        )

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


class TestAugPresets:
    """Regression tests for built-in augmentation presets."""

    def test_aug_aggressive_translate_percent_is_bidirectional(self) -> None:
        """AUG_AGGRESSIVE translate_percent must allow both positive and negative translations.

        (0.1, 0.1) is a degenerate range that only shifts right/down; the correct range is (-0.1, 0.1).
        """
        translate = AUG_AGGRESSIVE["Affine"]["translate_percent"]
        lo, hi = translate
        assert lo < 0, (
            f"AUG_AGGRESSIVE translate_percent lower bound must be negative to allow "
            f"left/up translation; got {translate!r}"
        )
        assert hi > 0, (
            f"AUG_AGGRESSIVE translate_percent upper bound must be positive to allow "
            f"right/down translation; got {translate!r}"
        )
        assert lo < hi, f"AUG_AGGRESSIVE translate_percent must be a non-degenerate range; got {translate!r}"


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

    @pytest.mark.parametrize(
        "resolution",
        [
            pytest.param(576, id="default_576"),
            pytest.param(640, id="larger_640"),
            pytest.param(768, id="larger_768"),
            pytest.param(960, id="larger_960"),
        ],
    )
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

    @pytest.mark.parametrize(
        "resolution",
        [
            pytest.param(576, id="default_576"),
            pytest.param(640, id="larger_640"),
            pytest.param(768, id="larger_768"),
            pytest.param(960, id="larger_960"),
        ],
    )
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

    @pytest.mark.parametrize(
        "resolution",
        [
            pytest.param(576, id="default_576"),
            pytest.param(640, id="larger_640"),
            pytest.param(768, id="larger_768"),
            pytest.param(960, id="larger_960"),
        ],
    )
    def test_train_pipeline_visible_keypoints_normalised_in_unit_range(self, resolution: int) -> None:
        """Train transform (random resize/crop) must produce normalised keypoints in [0, 1]."""
        transform = make_coco_transforms_square_div_64("train", resolution, aug_config={})
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        _, transformed = transform(image, self._make_target())

        kp = transformed["keypoints"]  # shape (N, K, 3); x and y already normalised by Normalize
        visible_mask = kp[:, :, 2] > 0  # (N, K) bool
        if visible_mask.any():
            visible_x = kp[:, :, 0][visible_mask]
            visible_y = kp[:, :, 1][visible_mask]
            assert (visible_x >= 0).all() and (visible_x <= 1.0).all(), (
                f"normalised keypoint x out of [0, 1] at resolution={resolution}: {visible_x}"
            )
            assert (visible_y >= 0).all() and (visible_y <= 1.0).all(), (
                f"normalised keypoint y out of [0, 1] at resolution={resolution}: {visible_y}"
            )

    @pytest.mark.parametrize(
        "resolution",
        [
            pytest.param(576, id="default_576"),
            pytest.param(640, id="larger_640"),
            pytest.param(768, id="larger_768"),
            pytest.param(960, id="larger_960"),
        ],
    )
    def test_output_tensor_shape_matches_resolution(self, resolution: int) -> None:
        """Transform output image tensor must be (C, resolution, resolution) for all resolutions."""
        transform = make_coco_transforms_square_div_64("val", resolution, aug_config={})
        image = Image.new("RGB", (self._INPUT_W, self._INPUT_H))

        tensor, _ = transform(image, None)

        assert tensor.shape[-2:] == (resolution, resolution), (
            f"expected (C, {resolution}, {resolution}), got {tuple(tensor.shape)}"
        )


class TestNormalize:
    """Unit tests for Normalize.__call__."""

    def test_normalize_call_is_bound_method(self) -> None:
        """Normalize.__call__ must be a class method, not a module-level function."""
        import inspect

        normalize = Normalize()
        assert callable(normalize), "Normalize instance must be callable"
        assert inspect.ismethod(normalize.__call__), "__call__ must be a bound method"

    def test_normalize_call_image_only_returns_normalized_tensor(self) -> None:
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
    def test_normalize_call_normalizes_boxes(
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

    def test_normalize_call_normalizes_keypoints_to_unit_range(self) -> None:
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

    def test_normalize_call_does_not_mutate_original_target(self) -> None:
        """Normalize.__call__ must not mutate the caller's target dict."""
        normalize = Normalize()
        image = torch.zeros(3, 50, 100)
        boxes_original = torch.tensor([[0.0, 0.0, 100.0, 50.0]])
        target = {"boxes": boxes_original.clone()}
        normalize(image, target)
        torch.testing.assert_close(target["boxes"], boxes_original, rtol=0.0, atol=0.0)


class TestReplayContainsHorizontalFlip:
    """Unit tests for AlbumentationsWrapper._replay_contains_horizontal_flip using fixture dicts."""

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
    def test_replay_contains_horizontal_flip(self, replay: object, expected: bool) -> None:
        """Fixture replay dicts should be correctly classified as horizontal flip or not."""
        assert AlbumentationsWrapper._replay_contains_horizontal_flip(replay) == expected


class TestFromConfigStrict:
    """AlbumentationsWrapper.from_config strict mode raises on required-transform failures instead of skipping."""

    def test_strict_false_skips_unbuildable_transform(self):
        """Lenient mode (default) logs and skips a transform that cannot be built."""
        result = AlbumentationsWrapper.from_config([{"NotARealTransform": {"p": 1.0}}])

        assert result == []

    def test_strict_true_raises_on_unbuildable_transform(self):
        """Strict mode surfaces a RuntimeError so a corrupt required pipeline fails loudly."""
        with pytest.raises(RuntimeError, match="NotARealTransform"):
            AlbumentationsWrapper.from_config([{"NotARealTransform": {"p": 1.0}}], strict=True)

    def test_strict_true_builds_valid_config(self):
        """Strict mode still returns wrappers when every transform builds successfully."""
        result = AlbumentationsWrapper.from_config([{"HorizontalFlip": {"p": 0.5}}], strict=True)

        assert len(result) == 1
