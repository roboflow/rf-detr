# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for building Albumentations augmentations from config dicts, including nested containers and presets.

Also covers strict mode, the first-unlogged-container lookup and ``RandomSizedCrop`` compatibility across Albumentations
1.x and 2.x.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE
from rfdetr.datasets.transforms import AlbumentationsWrapper, _build_albu_transform
from tests._markers import _requires_albumentations


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


@_requires_albumentations
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


@_requires_albumentations
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


@_requires_albumentations
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


@_requires_albumentations
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


@_requires_albumentations
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


@_requires_albumentations
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
