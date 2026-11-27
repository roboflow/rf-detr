# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``AlbumentationsWrapper``, the Albumentations augmentation wrapper, applied to images and their targets."""

from functools import partial
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.utilities.imports import _IS_ALBUMENTATIONS_INSTALLED
from tests._markers import _requires_albumentations

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


def _keep_channels(image: np.ndarray, count: int, **params: Any) -> np.ndarray:
    """Keep the first *count* channels of an ``(H, W, C)`` image, as a channel-changing transform would.

    Used through ``functools.partial`` as the image function of ``albumentations.Lambda``, which passes the
    transform parameters as keyword arguments.

    Args:
        image: ``(H, W, C)`` image array.
        count: Number of leading channels to keep; ``1`` returns a 2-D grayscale array.
        **params: Albumentations transform parameters, ignored.

    Returns:
        Contiguous array with the first *count* channels of *image*.

    Examples:
        >>> _keep_channels(np.zeros((8, 10, 4), dtype=np.uint8), 3).shape
        (8, 10, 3)
        >>> _keep_channels(np.zeros((8, 10, 3), dtype=np.uint8), 1).shape
        (8, 10)
    """
    kept = image[..., 0] if count == 1 else image[..., :count]
    return np.ascontiguousarray(kept)


#: Skip a test node that needs Albumentations' ``SquareSymmetry``, which older releases lack.
_requires_square_symmetry = pytest.mark.skipif(
    not _has_albumentations_transform("SquareSymmetry"),
    reason="SquareSymmetry is unavailable in this Albumentations version",
)


@_requires_albumentations
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

    @_requires_square_symmetry
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


@_requires_albumentations
class TestAlbumentationsWrapperChannelCount:
    """AlbumentationsWrapper refuses a transform that changes the channel count of the image."""

    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    @pytest.mark.parametrize("transform_name", ["HorizontalFlip", "RandomBrightnessContrast"])
    def test_channel_count_is_kept(self, mode: str, transform_name: str) -> None:
        """Grayscale and four-channel images keep their mode through geometric and pixel transforms."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(getattr(alb, transform_name)(p=1.0))
        target = {"boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]), "labels": torch.tensor([1])}

        aug_image, _ = wrapper(Image.new(mode, (10, 8)), target)

        assert aug_image.mode == mode

    @pytest.mark.parametrize("geometric", [False, True])
    @pytest.mark.parametrize("with_target", [False, True])
    def test_dropped_channel_raises(self, geometric: bool, with_target: bool) -> None:
        """Dropping the fourth channel fails on the pixel, geometric and target-free paths."""
        import albumentations as alb

        dropper = alb.Lambda(image=partial(_keep_channels, count=3), p=1.0)
        transform = alb.Sequential([alb.HorizontalFlip(p=1.0), dropper], p=1.0) if geometric else dropper
        target = {"boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]), "labels": torch.tensor([1])} if with_target else None

        with pytest.raises(ValueError, match="changed the image from 4 to 3 channels"):
            AlbumentationsWrapper(transform)(Image.new("RGBA", (10, 8)), target)

    def test_rgb_to_grayscale_raises(self) -> None:
        """Reducing an RGB image to one channel fails instead of reaching the model as a grayscale image."""
        import albumentations as alb

        wrapper = AlbumentationsWrapper(alb.Lambda(image=partial(_keep_channels, count=1), p=1.0))
        target = {"labels": torch.tensor([1])}

        with pytest.raises(ValueError, match="changed the image from 3 to 1 channels"):
            wrapper(Image.new("RGB", (10, 8)), target)
