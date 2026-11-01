# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Albumentations augmentation wrappers."""

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
from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize, _build_albu_transform
from rfdetr.utilities import collate_fn
from rfdetr.utilities.imports import _IS_ALBUMENTATIONS_INSTALLED

albumentations_only = pytest.mark.skipif(
    not _IS_ALBUMENTATIONS_INSTALLED, reason="albumentations not installed; skip Albumentations wrapper tests"
)
pytestmark = albumentations_only

__doctest_requires__ = {"_has_albumentations_transform": ["albumentations"]}


def _has_albumentations_transform(name: str) -> bool:
    """Report whether the installed Albumentations exports a transform.

    Albumentations is imported inside the call, not at module scope. The call still runs at collection time, because
    ``skipif`` conditions and parametrize arguments evaluate it, so an Albumentations that is installed but fails to
    import errors during collection instead of skipping.

    Args:
        name: Albumentations attribute name, e.g. ``"SquareSymmetry"``.

    Returns:
        ``True`` when Albumentations is installed and exposes ``name``.

    Examples:
        >>> _has_albumentations_transform("HorizontalFlip")
        True
        >>> _has_albumentations_transform("NoSuchTransform")
        False
    """
    if not _IS_ALBUMENTATIONS_INSTALLED:
        return False
    import albumentations as alb

    return hasattr(alb, name)


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
        "transform_name,params,box_in,box_out",
        [
            pytest.param(
                "HorizontalFlip",
                {"p": 1.0},
                [10.0, 20.0, 30.0, 40.0],
                [70.0, 20.0, 90.0, 40.0],
                id="horizontal-flip",
            ),
            pytest.param(
                "TimeReverse",
                {"p": 1.0},
                [10.0, 20.0, 30.0, 40.0],
                [70.0, 20.0, 90.0, 40.0],
                id="time-reverse",
            ),
            pytest.param(
                "VerticalFlip",
                {"p": 1.0},
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 60.0, 30.0, 80.0],
                id="vertical-flip",
            ),
        ],
    )
    def test_flip_transforms_with_boxes(self, transform_name, params, box_in, box_out):
        """Test flip transforms correctly transform bounding boxes."""
        import albumentations as alb

        transform = getattr(alb, transform_name)(**params)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([box_in]), "labels": torch.tensor([1])}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert torch.allclose(aug_target["boxes"], torch.tensor([box_out]), atol=1.0)
        assert torch.equal(aug_target["labels"], target["labels"])

    def test_resize_transforms_keypoint_coordinates(self):
        """Resize scales keypoint coordinates and preserves visibility values."""
        import albumentations as alb

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
        import albumentations as alb

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

    @pytest.mark.parametrize("num_instances", [0, 1, 2])
    def test_horizontal_flip_with_keypoint_flip_pairs_handles_ndarray_bboxes(self, num_instances):
        """Regression test for #1125.

        Albumentations 2.x returns ``bboxes`` as a NumPy ndarray of shape (N, 4); 1.x returned a list of tuples. The
        horizontal-flip swap path used to inspect ``bboxes`` with list-style truthiness, which raised ``ValueError: The
        truth value of an array with more than one element is ambiguous`` on any ndarray with more than one element.
        This test exercises that path across multi/single/empty instance counts.
        """
        import albumentations as alb

        wrapper = AlbumentationsWrapper(
            alb.HorizontalFlip(p=1.0),
            keypoint_flip_pairs=[0, 1],
        )

        image = Image.new("RGB", (100, 50))

        xs = [10.0 + i * 30.0 for i in range(num_instances)]
        target = {
            "boxes": torch.tensor([[x, 5.0, x + 20.0, 25.0] for x in xs], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([1] * num_instances, dtype=torch.long),
            "keypoints": torch.tensor(
                [[[x + 5.0, 10.0, 2.0], [x + 15.0, 20.0, 2.0]] for x in xs], dtype=torch.float32
            ).reshape(-1, 2, 3),
        }

        # Pre-fix: this call raised ValueError for num_instances >= 1 under Albumentations 2.x.
        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        assert aug_target["boxes"].shape[0] == num_instances
        assert aug_target["keypoints"].shape[0] == num_instances

    @pytest.mark.parametrize("transform_name", ["HorizontalFlip", "TimeReverse"])
    def test_horizontal_flip_swaps_paired_keypoints(self, transform_name):
        """HFlip-type transforms, including the TimeReverse alias, exchange keypoint slots for the configured pair."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(
            getattr(alb, transform_name)(p=1.0),
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

    @pytest.mark.skipif(
        not _has_albumentations_transform("SquareSymmetry"),
        reason="SquareSymmetry is unavailable in this Albumentations version",
    )
    def test_square_symmetry_explicit_pairs_keeps_instances_aligned(self, monkeypatch):
        """SquareSymmetry's horizontal element keeps image annotations and keypoint slots aligned."""
        import albumentations as alb

        square_symmetry = alb.SquareSymmetry(p=1.0)
        monkeypatch.setattr(square_symmetry, "get_params", lambda: {"group_element": "h"})
        if hasattr(square_symmetry, "get_params_dependent_on_data"):
            monkeypatch.setattr(
                square_symmetry,
                "get_params_dependent_on_data",
                lambda params, data: {"group_element": "h"},
            )

        wrapper = AlbumentationsWrapper(square_symmetry, keypoint_flip_pairs=[0, 1])
        height, width = 50, 100
        image_array = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[0, 5:25, 10:30] = 1
        masks[1, 10:40, 60:90] = 1
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0], [60.0, 10.0, 90.0, 40.0]]),
            "labels": torch.tensor([3, 7]),
            "area": torch.tensor([400.0, 900.0]),
            "iscrowd": torch.tensor([0, 1]),
            "masks": masks,
            "keypoints": torch.tensor(
                [
                    [[12.0, 10.0, 2.0], [28.0, 20.0, 2.0], [20.0, 15.0, 0.0]],
                    [[65.0, 15.0, 2.0], [85.0, 35.0, 2.0], [75.0, 25.0, 0.0]],
                ]
            ),
        }

        transformed_image, transformed = wrapper(Image.fromarray(image_array), target)

        assert np.array_equal(np.asarray(transformed_image), np.fliplr(image_array))
        torch.testing.assert_close(
            transformed["boxes"],
            torch.tensor([[70.0, 5.0, 90.0, 25.0], [10.0, 10.0, 40.0, 40.0]]),
        )
        assert transformed["labels"].tolist() == [3, 7]
        torch.testing.assert_close(transformed["area"], torch.tensor([400.0, 900.0]))
        assert transformed["iscrowd"].tolist() == [0, 1]
        assert transformed["masks"].dtype == torch.bool
        assert transformed["masks"].shape == (2, height, width)
        torch.testing.assert_close(transformed["masks"], torch.flip(masks.bool(), dims=[2]))
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor(
                [
                    [[71.0, 20.0, 2.0], [87.0, 10.0, 2.0], [0.0, 0.0, 0.0]],
                    [[14.0, 35.0, 2.0], [34.0, 15.0, 2.0], [0.0, 0.0, 0.0]],
                ]
            ),
        )

    @pytest.mark.parametrize("num_flips", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "transform_name",
        # SquareSymmetry is not exported by the minimum supported Albumentations version.
        ["HorizontalFlip", "TimeReverse", "D4"]
        + (["SquareSymmetry"] if _has_albumentations_transform("SquareSymmetry") else []),
    )
    @pytest.mark.parametrize("container_name", ["Sequential", "OneOf", "SomeOf", "SomeOf-replacement"])
    def test_repeated_horizontal_flips_preserve_keypoint_identities(
        self, num_flips: int, transform_name: str, container_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swap joint identities and visibility only for an odd number of applied flips."""
        import albumentations as alb

        num_transforms = 1 if container_name == "SomeOf-replacement" else num_flips
        transforms = [getattr(alb, transform_name)(p=1.0) for _ in range(num_transforms)]
        if transform_name in {"D4", "SquareSymmetry"}:
            for transform in transforms:
                monkeypatch.setattr(transform, "get_params", mock.Mock(return_value={"group_element": "h"}))
        if container_name == "SomeOf-replacement":
            transforms = [alb.SomeOf(transforms, n=num_flips, replace=True, p=1.0)]
        transforms.extend([alb.HorizontalFlip(p=0.0), alb.Sequential([alb.HorizontalFlip(p=1.0)], p=0.0)])
        if container_name != "Sequential":
            transforms = [
                alb.Sequential(transforms[:1], p=1.0),
                alb.Sequential(transforms[1:], p=1.0),
            ]
            if container_name == "OneOf":
                transforms = [alb.OneOf([alb.Sequential(transforms, p=1.0)], p=1.0)]
            else:
                transforms = [alb.SomeOf(transforms, n=2, replace=False, p=1.0)]
        wrapper = AlbumentationsWrapper(alb.Sequential(transforms, p=1.0), keypoint_flip_pairs=[0, 1, 2, 3])
        image_array = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor(
                [[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0], [20.0, 15.0, 2.0], [0.0, 0.0, 0.0], [50.0, 20.0, 2.0]]]
            ),
        }

        transformed_image, transformed = wrapper(Image.fromarray(image_array), target)

        if num_flips % 2:
            expected_image = np.fliplr(image_array)
            expected_boxes = torch.tensor([[10.0, 5.0, 95.0, 45.0]])
            expected_keypoints = torch.tensor(
                [[[19.0, 30.0, 1.0], [89.0, 10.0, 2.0], [0.0, 0.0, 0.0], [79.0, 15.0, 2.0], [49.0, 20.0, 2.0]]]
            )
        else:
            expected_image = image_array
            expected_boxes = target["boxes"]
            expected_keypoints = target["keypoints"]
        np.testing.assert_array_equal(np.asarray(transformed_image), expected_image)
        torch.testing.assert_close(transformed["boxes"], expected_boxes)
        torch.testing.assert_close(transformed["labels"], target["labels"])
        torch.testing.assert_close(transformed["keypoints"], expected_keypoints, rtol=0, atol=0)

    @pytest.mark.parametrize("scenario", ["skip-between-flips", "skip-last-flip", "vary-d4-elements"])
    def test_repeated_transform_uses_each_execution(self, scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated transforms count only applied horizontal mirrors with their own parameters."""
        import albumentations as alb

        if scenario == "vary-d4-elements":
            transform = alb.D4(p=1.0)
            monkeypatch.setattr(
                transform,
                "get_params",
                mock.Mock(side_effect=[{"group_element": "h"}, {"group_element": "v"}, {"group_element": "h"}]),
            )
        else:
            transform = alb.HorizontalFlip(p=1.0)
            applied = [True, False, True] if scenario == "skip-between-flips" else [True, True, False]
            monkeypatch.setattr(transform, "should_apply", mock.Mock(side_effect=applied))
        wrapper = AlbumentationsWrapper(alb.SomeOf([transform], n=3, replace=True, p=1.0), keypoint_flip_pairs=[0, 1])
        image_array = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0], [0.0, 0.0, 0.0]]]),
        }

        transformed_image, transformed = wrapper(Image.fromarray(image_array), target)

        expected_image = np.flipud(image_array) if scenario == "vary-d4-elements" else image_array
        expected_keypoints = (
            torch.tensor([[[10.0, 39.0, 2.0], [80.0, 19.0, 1.0], [0.0, 0.0, 0.0]]])
            if scenario == "vary-d4-elements"
            else target["keypoints"]
        )
        np.testing.assert_array_equal(np.asarray(transformed_image), expected_image)
        torch.testing.assert_close(transformed["boxes"], target["boxes"])
        torch.testing.assert_close(transformed["labels"], target["labels"])
        torch.testing.assert_close(transformed["keypoints"], expected_keypoints, rtol=0, atol=0)

    @pytest.mark.parametrize("outer_container", ["Compose", "RandomOrder"])
    def test_container_allowlist_nesting_uses_execution_log(self, outer_container: str) -> None:
        """A repeating SomeOf(replace=True) nested under Compose or RandomOrder still logs each applied occurrence.

        Existing repeated-flip coverage always wraps containers in an outer Sequential; this exercises the other two
        names on the execution-log allowlist (Compose, RandomOrder) as the outer container, proving parity is derived
        from the execution log rather than silently falling back to ReplayCompose metadata.
        """
        import albumentations as alb

        transform = alb.HorizontalFlip(p=1.0)
        some_of = alb.SomeOf([transform], n=3, replace=True, p=1.0)
        outer = alb.Compose([some_of]) if outer_container == "Compose" else alb.RandomOrder([some_of], n=1, p=1.0)
        wrapper = AlbumentationsWrapper(outer, keypoint_flip_pairs=[0, 1])

        assert wrapper._uses_execution_log is True

        image_array = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0]]]),
        }

        transformed_image, transformed = wrapper(Image.fromarray(image_array), target)

        # 3 real applications of HorizontalFlip(p=1.0) inside SomeOf(replace=True) is an odd count, so the net
        # effect is a single flip -- the same parity a correctly-functioning execution log must derive even
        # though ReplayCompose metadata would only record one entry for the repeated nested occurrence.
        np.testing.assert_array_equal(np.asarray(transformed_image), np.fliplr(image_array))
        torch.testing.assert_close(transformed["boxes"], torch.tensor([[10.0, 5.0, 95.0, 45.0]]))
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[19.0, 30.0, 1.0], [89.0, 10.0, 2.0]]]),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("pipeline", ["direct", "nested", "mixed"])
    def test_one_or_other_preserves_flip_parity(self, pipeline: str) -> None:
        """Containers without execution logging still preserve paired keypoint identities."""
        import albumentations as alb

        transform = alb.OneOrOther(alb.HorizontalFlip(p=1.0), alb.VerticalFlip(p=1.0), p=1.0)
        if pipeline == "nested":
            transform = alb.Sequential([alb.OneOf([transform], p=1.0)], p=1.0)
        elif pipeline == "mixed":
            transform = alb.Sequential([transform, alb.HorizontalFlip(p=1.0)], p=1.0)
        wrapper = AlbumentationsWrapper(transform, keypoint_flip_pairs=[0, 1])
        image_array = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0], [0.0, 0.0, 0.0]]]),
        }

        transformed_image, transformed = wrapper(Image.fromarray(image_array), target)

        if pipeline == "mixed":
            expected_image = image_array
            expected_boxes = target["boxes"]
            expected_keypoints = target["keypoints"]
        else:
            expected_image = np.fliplr(image_array)
            expected_boxes = torch.tensor([[10.0, 5.0, 95.0, 45.0]])
            expected_keypoints = torch.tensor([[[19.0, 30.0, 1.0], [89.0, 10.0, 2.0], [0.0, 0.0, 0.0]]])
        np.testing.assert_array_equal(np.asarray(transformed_image), expected_image)
        torch.testing.assert_close(transformed["boxes"], expected_boxes)
        torch.testing.assert_close(transformed["labels"], target["labels"])
        torch.testing.assert_close(transformed["keypoints"], expected_keypoints, rtol=0, atol=0)

    def test_nested_horizontal_flip_swaps_slots_after_all_geometry(self):
        """Nested HFlip+VFlip should mirror coordinates once, then swap only the left/right slots."""
        import albumentations as alb

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
        "transform_name,transform_kwargs,expected_keypoints",
        [
            pytest.param(
                "HorizontalFlip",
                {"p": 0.0},
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="disabled-horizontal-flip",
            ),
            pytest.param(
                "VerticalFlip",
                {"p": 1.0},
                torch.tensor([[[10.0, 39.0, 2.0], [80.0, 19.0, 2.0]]]),
                id="vertical-flip",
            ),
            pytest.param(
                "Resize",
                {"height": 50, "width": 100, "p": 1.0},
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="resize",
            ),
            pytest.param(
                "Crop",
                {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 50, "p": 1.0},
                torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 2.0]]]),
                id="full-image-crop",
            ),
        ],
    )
    def test_non_horizontal_geometry_does_not_swap_paired_keypoints(
        self, transform_name, transform_kwargs, expected_keypoints
    ):
        """Configured HFlip pairs should not swap slots when no horizontal flip applied."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(getattr(alb, transform_name)(**transform_kwargs), keypoint_flip_pairs=[0, 1])
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
        import albumentations as alb

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
        import albumentations as alb

        transform = alb.GaussianBlur(blur_limit=3, p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "labels": torch.tensor([1])}

        aug_image, aug_target = wrapper(image, target)

        assert isinstance(aug_image, Image.Image)
        # Boxes should be unchanged
        assert torch.equal(aug_target["boxes"], target["boxes"])
        assert torch.equal(aug_target["labels"], target["labels"])

    def test_empty_boxes_pass_through(self):
        """Empty boxes and labels (no masks key) come back with the same empty shapes."""
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

        transform = alb.Resize(height=64, width=64)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        aug_image, aug_target = wrapper(image, None)

        assert isinstance(aug_image, Image.Image)
        assert aug_image.size == (64, 64)
        assert aug_target is None

    def test_invalid_target_type(self):
        """Test wrapper raises error for invalid target type."""
        import albumentations as alb

        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))

        with pytest.raises(TypeError) as exc_info:
            wrapper(image, "invalid_target")

        assert "target must be a dictionary" in str(exc_info.value)

    def test_missing_labels_key(self):
        """Test wrapper raises error when labels key is missing."""
        import albumentations as alb

        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]])}

        with pytest.raises(KeyError) as exc_info:
            wrapper(image, target)

        assert "target must contain 'labels' key" in str(exc_info.value)

    def test_invalid_boxes_shape(self):
        """Test wrapper raises error for invalid boxes shape."""
        import albumentations as alb

        transform = alb.HorizontalFlip(p=1.0)
        wrapper = AlbumentationsWrapper(transform)

        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([10.0, 20.0, 30.0]),  # Invalid shape
            "labels": torch.tensor([1]),
        }

        with pytest.raises(ValueError) as exc_info:
            wrapper(image, target)

        assert "boxes must have shape" in str(exc_info.value)

    def test_orig_size_preserved_with_two_boxes(self):
        """Test that orig_size is not treated as per-instance field when num_boxes=2.

        Regression test for bug where orig_size (shape [2] for [h, w]) was incorrectly treated as a per-instance field
        when there were exactly 2 boxes, causing orig_size to be filtered/indexed incorrectly and leading to
        inconsistent tensor shapes in batches.
        """
        import albumentations as alb

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
        import albumentations as alb

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
        "transform_name,params",
        [
            ("HorizontalFlip", {"p": 1.0}),
            ("VerticalFlip", {"p": 1.0}),
            ("Rotate", {"limit": 45, "p": 1.0}),
        ],
    )
    def test_various_geometric_transforms(self, transform_name, params):
        """Test various geometric transforms work correctly."""
        import albumentations as alb

        transform = getattr(alb, transform_name)(**params)
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
        import albumentations as alb

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
        "transform_name,params",
        [
            ("HorizontalFlip", {"p": 1.0}),
            ("VerticalFlip", {"p": 1.0}),
            ("Rotate", {"limit": 15, "p": 1.0}),  # Small angle to avoid boxes going out
        ],
    )
    def test_various_geometric_transforms_with_masks(self, transform_name, params):
        """Test various geometric transforms correctly transform masks."""
        import albumentations as alb

        transform = getattr(alb, transform_name)(**params)
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
        # Rotation can remove boxes that leave the image; any surviving mask must keep content.
        assert aug_target["masks"].shape[0] == aug_target["boxes"].shape[0]
        assert aug_target["boxes"].shape[0] == 0 or aug_target["masks"].any()

    @pytest.mark.parametrize(
        "transform_name,params",
        [
            ("GaussianBlur", {"blur_limit": 3, "p": 1.0}),
            ("RandomBrightnessContrast", {"p": 1.0}),
            ("GaussNoise", {"p": 1.0}),
        ],
    )
    def test_pixel_transforms_preserve_masks(self, transform_name, params):
        """Test pixel-level transforms preserve masks unchanged."""
        import albumentations as alb

        transform = getattr(alb, transform_name)(**params)
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
        import albumentations as alb

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

    def test_geometric_transform_with_empty_masks_tensor(self):
        """Test that a geometric transform does not crash when masks tensor is empty (0 instances).

        Regression test for: when a prior crop removes all annotations, target["masks"] has shape (0, H, W). Passing an
        empty list to albumentations raises ValueError: masks cannot be empty.
        """
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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

        with pytest.raises(ValueError) as exc_info:
            wrapper(image, target)

        assert "masks must have shape" in str(exc_info.value)

    @pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.float32])
    def test_mask_dtype_handling(self, mask_dtype):
        """Test wrapper handles different mask dtypes correctly (uint8, float32)."""
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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

    @pytest.mark.parametrize("rotate_method", ["largest_box", "ellipse"])
    def test_geometric_transform_recomputes_boxes_from_masks(self, rotate_method):
        """Geometric transforms recompute tight boxes from transformed masks, per instance (#1636)."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Affine(rotate=(45, 45), rotate_method=rotate_method, p=1.0))

        masks = np.zeros((3, 200, 200), dtype=bool)
        masks[0, 90:110, 20:100] = True
        masks[1, 20:100, 150:160] = True
        masks[2, 150:160, 100:180] = True
        boxes = [[20.0, 90.0, 100.0, 110.0], [150.0, 20.0, 160.0, 100.0], [100.0, 150.0, 180.0, 160.0]]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            "masks": torch.as_tensor(masks, dtype=torch.bool),
            "area": torch.tensor([1600.0, 800.0, 800.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0, 1, 0]),
        }
        _, aug_target = wrapper(Image.new("RGB", (200, 200)), target)

        out_masks = aug_target["masks"].numpy()
        assert out_masks.shape[0] == len(aug_target["labels"]) == len(aug_target["iscrowd"]) >= 1
        for mask, box in zip(out_masks, aug_target["boxes"].tolist()):
            ys, xs = np.where(mask)
            assert box == [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        expected_area = (aug_target["boxes"][:, 2] - aug_target["boxes"][:, 0]) * (
            aug_target["boxes"][:, 3] - aug_target["boxes"][:, 1]
        )
        torch.testing.assert_close(aug_target["area"], expected_area)

    def test_geometric_transform_drops_ghost_box_with_empty_mask(self):
        """A partly visible box whose mask left the canvas is dropped; the other instance stays aligned (#1636)."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Affine(translate_px={"x": 15, "y": 0}, p=1.0))

        masks = torch.zeros((3, 100, 100), dtype=torch.uint8)
        masks[0, 10:20, 88:95] = 1  # ghost: box stays partly inside, mask is shifted off the canvas
        masks[1, 50:70, 10:30] = 1
        masks[2, 70:90, 40:60] = 1
        target = {
            "boxes": torch.tensor(
                [[60.0, 10.0, 95.0, 40.0], [10.0, 50.0, 30.0, 70.0], [40.0, 70.0, 60.0, 90.0]], dtype=torch.float32
            ),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            "masks": masks,
            "area": torch.tensor([1050.0, 400.0, 400.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0, 1, 0]),
        }
        _, aug_target = wrapper(Image.new("RGB", (100, 100)), target)

        assert aug_target["labels"].tolist() == [2, 3]
        assert aug_target["iscrowd"].tolist() == [1, 0]
        assert aug_target["masks"].shape == (2, 100, 100)
        assert aug_target["boxes"].tolist() == [[25.0, 50.0, 45.0, 70.0], [55.0, 70.0, 75.0, 90.0]]
        assert aug_target["area"].tolist() == [400.0, 400.0]

    def test_geometric_transform_keeps_box_only_instance(self):
        """An instance without any mask foreground (box-only annotation) keeps its transformed box (#1558)."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.HorizontalFlip(p=1.0))

        masks = torch.zeros((2, 100, 100), dtype=torch.uint8)
        masks[0, 20:40, 10:30] = 1
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [60.0, 60.0, 80.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "masks": masks,
        }
        _, aug_target = wrapper(Image.new("RGB", (100, 100)), target)

        assert aug_target["labels"].tolist() == [1, 2]
        torch.testing.assert_close(aug_target["boxes"][1], torch.tensor([20.0, 60.0, 40.0, 90.0]))
        assert not aug_target["masks"][1].any()

    def test_geometric_transform_drops_mask_lost_to_downscale(self):
        """A thin object whose mask vanishes under nearest-neighbour downscale is dropped, not kept as a ghost."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Resize(25, 25, p=1.0))

        masks = torch.zeros((2, 100, 100), dtype=torch.uint8)
        masks[0, 50, 10:90] = 1  # 1 px high line: erased by 4x nearest downscale
        masks[1, 20:60, 20:60] = 1
        target = {
            "boxes": torch.tensor([[10.0, 50.0, 90.0, 51.0], [20.0, 20.0, 60.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "masks": masks,
        }
        _, aug_target = wrapper(Image.new("RGB", (100, 100)), target)

        assert aug_target["labels"].tolist() == [2]
        assert aug_target["masks"].shape == (1, 25, 25)

    def test_geometric_transform_reflected_copies_keep_albumentations_boxes(self):
        """Instances duplicated by a reflecting border mode keep their own boxes, not one union box."""
        import albumentations as alb
        import cv2

        wrapper = AlbumentationsWrapper(
            alb.Affine(translate_px={"x": 30, "y": 0}, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
        )

        masks = torch.zeros((1, 100, 100), dtype=torch.uint8)
        masks[0, 10:20, 40:50] = 1
        target = {
            "boxes": torch.tensor([[40.0, 10.0, 50.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "masks": masks,
        }
        _, aug_target = wrapper(Image.new("RGB", (100, 100)), target)

        widths = (aug_target["boxes"][:, 2] - aug_target["boxes"][:, 0]).tolist()
        assert all(width <= 10.0 for width in widths)

    def test_geometric_transform_box_only_instance_keeps_alignment_with_masked_instance(self):
        """A box-only instance beside a masked one keeps its Albumentations box and per-instance fields (#1558)."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Affine(translate_px={"x": 5, "y": 5}, p=1.0))

        height, width = 100, 100
        masks = torch.zeros((2, height, width), dtype=torch.uint8)
        masks[1, 50:70, 50:70] = 1
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 70.0, 70.0]], dtype=torch.float32),
            "labels": torch.tensor([10, 20], dtype=torch.long),
            "masks": masks,
            "area": torch.tensor([400.0, 400.0], dtype=torch.float32),
        }
        _, aug_target = wrapper(Image.new("RGB", (width, height)), target)

        assert aug_target["labels"].tolist() == [10, 20]
        torch.testing.assert_close(
            aug_target["boxes"], torch.tensor([[15.0, 15.0, 35.0, 35.0], [55.0, 55.0, 75.0, 75.0]])
        )
        assert aug_target["masks"].shape == (2, height, width)
        assert not aug_target["masks"][0].any()
        assert aug_target["masks"][1].any()
        assert aug_target["area"].tolist() == [400.0, 400.0]

    def test_geometric_transform_all_empty_masks_clears_target(self):
        """When every mask leaves the canvas while a partly visible box remains, the target is cleared (#1636)."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Affine(translate_px={"x": 15, "y": 0}, p=1.0))

        masks = torch.zeros((1, 100, 100), dtype=torch.uint8)
        masks[0, 10:20, 88:95] = 1
        target = {
            "boxes": torch.tensor([[60.0, 10.0, 95.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "masks": masks,
            "area": torch.tensor([1050.0], dtype=torch.float32),
        }
        _, aug_target = wrapper(Image.new("RGB", (100, 100)), target)

        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)
        assert aug_target["masks"].shape == (0, 100, 100)
        assert aug_target["area"].shape == (0,)


class TestAlbumentationsWrapperFromConfig:
    """Tests for AlbumentationsWrapper.from_config() static method."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"HorizontalFlip": {"p": 0.5}, "VerticalFlip": {"p": 0.3}}, id="valid"),
            pytest.param({"HorizontalFlip": {"p": 1.0}, "GaussianBlur": {"p": 1.0}}, id="geometric-and-pixel"),
            pytest.param(
                {
                    "Rotate": {"limit": (90, 90), "p": 0.5},
                    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.1, 0.1), "p": 0.3},
                },
                id="complex-params",
            ),
        ],
    )
    def test_builds_one_wrapper_per_entry_in_config_order(self, config):
        """Every config entry becomes one AlbumentationsWrapper, in config order."""
        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 2
        assert all(isinstance(t, AlbumentationsWrapper) for t in transforms)
        transform_names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert transform_names == list(config.keys())

    def test_build_from_empty_config(self):
        """Test building from empty config returns empty list."""
        config = {}

        transforms = AlbumentationsWrapper.from_config(config)

        assert len(transforms) == 0

    @pytest.mark.parametrize(
        "skipped_entry",
        [
            pytest.param({"NonExistentTransform": {"p": 0.5}}, id="unknown-transform"),
            pytest.param({"InvalidTransform": "not_a_dict"}, id="non-dict-params"),
        ],
    )
    def test_unbuildable_entry_is_skipped(self, skipped_entry):
        """An unknown transform name or non-dict params is skipped, leaving only the valid transform."""
        config = {"HorizontalFlip": {"p": 0.5}, **skipped_entry}

        transforms = AlbumentationsWrapper.from_config(config)

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
        with pytest.raises(TypeError) as exc_info:
            AlbumentationsWrapper.from_config("invalid")

        assert "config_dict must be a dictionary or list" in str(exc_info.value)

    def test_from_config_some_of_replacement_uses_execution_log(self):
        """from_config-built SomeOf(replace=True) parity matches the equivalent direct-constructor wrapper.

        Existing from_config coverage only builds OneOf/Sequential containers with default probabilities; this builds a
        repeating SomeOf via from_config and checks it behaves identically to the same container built directly with
        AlbumentationsWrapper, proving the execution-log path also works end-to-end through config-driven construction.
        """
        import albumentations as alb

        config = {
            "SomeOf": {
                "transforms": [{"HorizontalFlip": {"p": 1.0}}],
                "n": 2,
                "replace": True,
                "p": 1.0,
            }
        }
        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[0, 1])
        wrapper = transforms[0]
        direct = AlbumentationsWrapper(
            alb.SomeOf([alb.HorizontalFlip(p=1.0)], n=2, replace=True, p=1.0),
            keypoint_flip_pairs=[0, 1],
        )

        assert wrapper._uses_execution_log is True
        assert direct._uses_execution_log is True

        image_array = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
        config_target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0]]]),
        }
        direct_target = {
            "boxes": torch.tensor([[5.0, 5.0, 90.0, 45.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [80.0, 30.0, 1.0]]]),
        }

        config_image, config_out = wrapper(Image.fromarray(image_array), config_target)
        direct_image, direct_out = direct(Image.fromarray(image_array), direct_target)

        np.testing.assert_array_equal(np.asarray(config_image), np.asarray(direct_image))
        torch.testing.assert_close(config_out["boxes"], direct_out["boxes"])
        torch.testing.assert_close(config_out["keypoints"], direct_out["keypoints"], rtol=0, atol=0)
        # Two real applications of a p=1.0 HorizontalFlip is an even count, so the net geometric effect --
        # and the keypoint-slot parity -- must match the untransformed input exactly.
        np.testing.assert_array_equal(np.asarray(config_image), image_array)
        torch.testing.assert_close(config_out["keypoints"], config_target["keypoints"], rtol=0, atol=0)


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
    def test_errors_on_partial_hw_with_v2_api(self, mocker, params, expected_missing):
        """Passing only one of height/width with the v2 API raises naming the missing dimension."""
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
        with pytest.raises(ValueError) as exc_info:
            _build_albu_transform("RandomSizedCrop", params)

        assert f"missing '{expected_missing}'" in str(exc_info.value)

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
    def test_size_takes_precedence_over_hw_on_v2_api(self, mocker, params):
        """On the v2 API an explicit ``size`` wins and conflicting height/width are dropped before instantiation."""
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
        # No TypeError means height/width were correctly dropped before instantiation
        transform = _build_albu_transform("RandomSizedCrop", params)
        assert transform.size == (256, 256)

    def test_scalar_size_passes_through_on_v1_legacy_path(self, mocker):
        """A scalar ``size`` is left in the params on the v1 path, so the v1 constructor rejects it with TypeError.

        A scalar is not a ``Sequence``, so the normalizer cannot split it into height/width; it forwards ``size``
        unchanged and raises no ValueError of its own, which the height-only v2 errors do.
        """
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV1)

        with pytest.raises(TypeError):
            _build_albu_transform("RandomSizedCrop", {"min_max_height": [100, 200], "size": 640})

    def test_adapts_height_width_for_v2_api(self, mocker):
        """RandomSizedCrop config with height/width is adapted to the Albumentations 2.x size API."""
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
        transform = _build_albu_transform(
            "RandomSizedCrop",
            {"min_max_height": [384, 600], "height": 640, "width": 640},
        )

        assert isinstance(transform, _FakeRandomSizedCropV2)
        assert transform.min_max_height == [384, 600]
        assert transform.size == (640, 640)

    def test_adapts_size_for_v1_api(self, mocker):
        """RandomSizedCrop config with size is adapted to the Albumentations 1.x height/width API."""
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV1)
        transform = _build_albu_transform(
            "RandomSizedCrop",
            {"min_max_height": [384, 600], "size": (640, 640)},
        )

        assert isinstance(transform, _FakeRandomSizedCropV1)
        assert transform.min_max_height == [384, 600]
        assert transform.height == 640
        assert transform.width == 640

    def test_from_config_partial_height_is_silently_skipped(self, mocker):
        """from_config swallows the ValueError for partial height-only config and skips the transform.

        This documents the intentional silent-skip behavior: from_config wraps _build_albu_transform in a broad except
        clause so bad configs produce a warning rather than an exception.
        """
        mocker.patch("rfdetr.datasets.transforms.alb.RandomSizedCrop", new=_FakeRandomSizedCropV2)
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

    @pytest.mark.parametrize(
        ("container", "members", "expected_geometric"),
        [
            pytest.param("OneOf", ["HorizontalFlip", "GaussianBlur"], True, id="one-of-with-geometric"),
            pytest.param("OneOf", ["GaussianBlur", "Blur"], False, id="one-of-pixel-only"),
            pytest.param("Sequential", ["Rotate", "GaussianBlur"], True, id="sequential-with-geometric"),
        ],
    )
    def test_container_geometric_detection(self, container, members, expected_geometric):
        """A container is geometric when any member is geometric, and pixel-level when all members are pixel-level."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(getattr(alb, container)([getattr(alb, name)(p=1.0) for name in members]))

        assert wrapper._is_geometric is expected_geometric

    def test_from_config_nested_one_of(self):
        """from_config builds a OneOf wrapper from nested config."""
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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
        import albumentations as alb

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

    def test_from_config_one_of_applies_geometric(self):
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

    def test_from_config_one_of_applies_pixel(self):
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

    @pytest.mark.parametrize("container", ["OneOf", "Sequential"])
    @pytest.mark.parametrize(
        ("extra", "expected_p"),
        [pytest.param({"p": 0.0}, 0.0, id="explicit-p-is-respected"), pytest.param({}, 1.0, id="default-p-is-one")],
    )
    def test_container_p_from_config(self, container, extra, expected_p):
        """A container keeps the p given in config and defaults to p=1.0 when none is supplied."""
        import albumentations as alb

        config = {container: {"transforms": [{"HorizontalFlip": {"p": 1.0}}], **extra}}
        transforms = AlbumentationsWrapper.from_config(config)
        inner = transforms[0].transform.transforms[0]
        assert isinstance(inner, getattr(alb, container))
        assert inner.p == pytest.approx(expected_p)

    def test_one_of_empty_transforms_raises(self):
        """OneOf with no transforms raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _build_albu_transform("OneOf", {"transforms": []})

        assert "at least one" in str(exc_info.value)

    def test_some_of_single_p(self):
        """SomeOf with a plain p (block probability) still works without probs."""
        import albumentations as alb

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

    @pytest.mark.parametrize("hflip_name", ["HorizontalFlip", "TimeReverse", "Flip", "D4", "SquareSymmetry"])
    def test_hflip_disabled_for_keypoint_pipeline(self, hflip_name: str) -> None:
        """HFlip-type transforms are skipped when keypoint_flip_pairs is empty (no left/right pairs configured)."""
        config = {hflip_name: {"p": 0.5}, "GaussianBlur": {"p": 0.5}}

        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[])

        names = [t.transform.transforms[0].__class__.__name__ for t in transforms]
        assert hflip_name not in names
        assert "GaussianBlur" in names

    def test_hflip_disable_logs_warning(self, mocker) -> None:
        """from_config logs a warning when HorizontalFlip is disabled for a keypoint pipeline."""
        config = {"HorizontalFlip": {"p": 0.5}}

        mock_log = mocker.patch("rfdetr.datasets.transforms.logger")
        AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=[])

        assert mock_log.warning.called
        call_args = mock_log.warning.call_args[0]
        assert "HorizontalFlip" in str(call_args)

    @pytest.mark.parametrize(
        "keypoint_flip_pairs",
        [pytest.param(None, id="detection-pipeline"), pytest.param([0, 1], id="semantic-pairs-configured")],
    )
    def test_hflip_included_unless_pairs_are_empty(self, keypoint_flip_pairs) -> None:
        """HorizontalFlip is kept for a detection pipeline (pairs None) and when semantic flip pairs are configured."""
        config = {"HorizontalFlip": {"p": 0.5}}

        transforms = AlbumentationsWrapper.from_config(config, keypoint_flip_pairs=keypoint_flip_pairs)

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


class TestFirstUnloggedContainer:
    """Unit tests for AlbumentationsWrapper._first_unlogged_container using fixture container trees."""

    def test_empty_container_is_fully_logged(self) -> None:
        """A container with zero children reports no offending container."""
        import albumentations as alb

        assert AlbumentationsWrapper._first_unlogged_container(alb.Compose([])) is None

    def test_deeply_nested_allowlisted_containers_are_fully_logged(self) -> None:
        """SomeOf(replace=True) nested inside SomeOf(replace=True) inside Sequential is still fully logged.

        Exercises depth > 2 nesting where every container along the path is on the execution-log allowlist, confirming
        the recursive walk does not mistake depth alone for an unlogged container.
        """
        import albumentations as alb

        tree = alb.Sequential(
            [
                alb.SomeOf(
                    [alb.SomeOf([alb.HorizontalFlip(p=1.0)], n=1, replace=True, p=1.0)],
                    n=1,
                    replace=True,
                    p=1.0,
                )
            ],
            p=1.0,
        )

        assert AlbumentationsWrapper._first_unlogged_container(tree) is None

    def test_deeply_nested_offending_container_is_reported(self) -> None:
        """A non-allowlisted container nested below allowlisted ones is still identified as the offender."""
        import albumentations as alb

        tree = alb.Sequential(
            [
                alb.SomeOf(
                    [alb.OneOrOther(alb.HorizontalFlip(p=1.0), alb.VerticalFlip(p=1.0), p=1.0)],
                    n=1,
                    replace=True,
                    p=1.0,
                )
            ],
            p=1.0,
        )

        assert AlbumentationsWrapper._first_unlogged_container(tree) == "OneOrOther"


class TestFromConfigStrict:
    """AlbumentationsWrapper.from_config strict mode raises on required-transform failures instead of skipping."""

    def test_strict_false_skips_unbuildable_transform(self):
        """Lenient mode (default) logs and skips a transform that cannot be built."""
        result = AlbumentationsWrapper.from_config([{"NotARealTransform": {"p": 1.0}}])

        assert result == []

    def test_strict_true_raises_on_unbuildable_transform(self):
        """Strict mode surfaces a RuntimeError so a corrupt required pipeline fails loudly."""
        with pytest.raises(RuntimeError) as exc_info:
            AlbumentationsWrapper.from_config([{"NotARealTransform": {"p": 1.0}}], strict=True)

        assert "NotARealTransform" in str(exc_info.value)

    def test_strict_true_builds_valid_config(self):
        """Strict mode still returns wrappers when every transform builds successfully."""
        result = AlbumentationsWrapper.from_config([{"HorizontalFlip": {"p": 0.5}}], strict=True)

        assert len(result) == 1
