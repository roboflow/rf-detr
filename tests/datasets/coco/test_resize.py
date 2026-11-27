# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests for the training resize builders of both augmentation backends.

``_build_train_resize_config`` builds the Albumentations resize config and ``_build_train_resize_transforms`` the
torchvision-native resize pipeline. The two backends are meant to express the same augmentation recipe, so the
torchvision non-square crop branch is asserted against the same shape the config tests pin for Albumentations.
"""

import json

import pytest
import torch
from PIL import Image

from rfdetr.datasets._torchvision import RandomChoice, RandomResize, RandomSizedCrop
from rfdetr.datasets.coco import _build_train_resize_config, _build_train_resize_transforms

#: Every scale-count / squareness combination the builder distinguishes.
_SCALE_SQUARE_CASES = [
    pytest.param([640], True, id="square-single"),
    pytest.param([480, 640], True, id="square-multi"),
    pytest.param([640], False, id="nonsquare-single"),
    pytest.param([480, 640], False, id="nonsquare-multi"),
]


_SQUARE_SCALES = [pytest.param([640], id="square-single"), pytest.param([480, 640], id="square-multi")]
_NONSQUARE_SCALES = [pytest.param([640], id="nonsquare-single"), pytest.param([480, 640], id="nonsquare-multi")]


class TestBuildTrainResizeConfigStructure:
    """Top-level structure is always a single-element list wrapping a OneOf."""

    @pytest.mark.parametrize("scales,square", _SCALE_SQUARE_CASES)
    def test_returns_single_element_list(self, scales, square):
        """The builder returns a one-element list for every scale and squareness combination."""
        result = _build_train_resize_config(scales, square=square)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.parametrize("scales,square", _SCALE_SQUARE_CASES)
    def test_top_level_is_oneof_with_two_branches(self, scales, square):
        """The single entry is a OneOf holding exactly the direct-resize and crop branches."""
        result = _build_train_resize_config(scales, square=square)
        entry = result[0]
        assert "OneOf" in entry
        oneof = entry["OneOf"]
        assert len(oneof["transforms"]) == 2


class TestBuildTrainResizeConfigOptionB:
    """Option B is SmallestMaxSize jitter followed by one RandomSizedCrop per target scale, for every combination."""

    @pytest.mark.parametrize("scales,square", _SCALE_SQUARE_CASES)
    def test_option_b_is_sequential_with_oneof_crop(self, scales, square):
        """Option B resizes each crop directly to its target scale, with no extra longest-side cap step."""
        result = _build_train_resize_config(scales, square=square)
        option_b = result[0]["OneOf"]["transforms"][1]
        crops = [
            {"RandomSizedCrop": {"min_max_height": [384, 600], "height": scale, "width": scale}} for scale in scales
        ]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {"OneOf": {"transforms": crops}},
                ]
            }
        }


class TestBuildTrainResizeConfigSquareOptionA:
    """Square=True — Option A is a OneOf of plain square Resize transforms, one per scale."""

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="single-640"),
            pytest.param([480], id="single-480"),
            pytest.param([480, 640], id="two-scales"),
            pytest.param([384, 512, 640], id="three-scales"),
        ],
    )
    def test_option_a_is_oneof_of_resizes(self, scales):
        """Option A holds one square Resize per requested scale."""
        result = _build_train_resize_config(scales, square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {"OneOf": {"transforms": [{"Resize": {"height": s, "width": s}} for s in scales]}}


class TestBuildTrainResizeConfigNonSquareOptionA:
    """Square=False — Option A is SmallestMaxSize (scalar for one scale, list for several) then a 1333 long-side cap."""

    @pytest.mark.parametrize(
        "scales,expected_size",
        [
            pytest.param([640], 640, id="single-scale-uses-scalar"),
            pytest.param([480, 640], [480, 640], id="multi-scale-uses-list"),
        ],
    )
    def test_option_a_size_form(self, scales, expected_size):
        """Option A passes a scalar size for one scale and the list itself for several, with the default cap."""
        result = _build_train_resize_config(scales, square=False)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": expected_size}},
                    {"CappedLongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_custom_max_size(self):
        """A custom ``max_size`` replaces the default 1333 long-side cap in Option A."""
        result = _build_train_resize_config([640], square=False, max_size=800)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["Sequential"]["transforms"][1] == {"CappedLongestMaxSize": {"max_size": 800}}

    def test_custom_max_size_applies_to_option_a_only(self):
        """max_size caps option_a's long side; option_b now resizes the crop directly to the target (no cap step)."""
        result = _build_train_resize_config([480, 640], square=False, max_size=1000)
        option_a = result[0]["OneOf"]["transforms"][0]
        option_b_steps = result[0]["OneOf"]["transforms"][1]["Sequential"]["transforms"]
        assert option_a["Sequential"]["transforms"][1] == {"CappedLongestMaxSize": {"max_size": 1000}}
        assert not any("LongestMaxSize" in step for step in option_b_steps)


class TestBuildTrainResizeConfigNonSquareScaleJitter:
    """Non-square option_b must keep RandomSizedCrop (scale jitter), not a fixed RandomCrop.

    Regression tests for https://github.com/roboflow/rf-detr/issues/1018 — PR #752 replaced RandomSizeCrop(384, 600)
    with a fixed RandomCrop(384, 384), silently removing scale jitter from the non-square training pipeline.

    The ``fix-resize-crop`` branch keeps RandomSizedCrop and removes the wasteful fixed-384 intermediate hop: the crop
    now resizes directly to the target scale (per-scale ``OneOf``, mirroring the square path). ``min_max_height`` uses
    ``[384, 600]`` to match the full SmallestMaxSize range — when the image short side is 400, albumentations clamps
    the crop to the image height (a full-image crop), which is the original DETR recipe behaviour and preserves
    zoom-out diversity across the SmallestMaxSize range.
    """

    @pytest.mark.parametrize(
        "scales", [pytest.param([640], id="nonsquare-single"), pytest.param([480, 640], id="nonsquare-multi")]
    )
    def test_option_b_crop_step_uses_random_sized_crop(self, scales):
        """Non-square option_b crop must use RandomSizedCrop, never fixed RandomCrop (issue #1018)."""
        result = _build_train_resize_config(scales, square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        crop_step = option_b["Sequential"]["transforms"][1]
        crop_variants = crop_step["OneOf"]["transforms"]
        assert crop_variants and all(
            "RandomSizedCrop" in entry and "RandomCrop" not in entry for entry in crop_variants
        )

    @pytest.mark.parametrize(
        "scales", [pytest.param([640], id="nonsquare-single"), pytest.param([480, 640], id="nonsquare-multi")]
    )
    def test_option_b_crop_uses_full_scale_jitter_range(self, scales):
        """RandomSizedCrop min_max_height matches SmallestMaxSize range [384, 600] for full zoom-out diversity."""
        result = _build_train_resize_config(scales, square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        crop_variants = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        assert all(entry["RandomSizedCrop"]["min_max_height"] == [384, 600] for entry in crop_variants)

    @pytest.mark.parametrize(
        "scales", [pytest.param([640], id="square-single"), pytest.param([480, 640], id="square-multi")]
    )
    def test_square_option_b_unchanged(self, scales):
        """Square path must still use RandomSizedCrop parameterized by scale."""
        result = _build_train_resize_config(scales, square=True)
        option_b = result[0]["OneOf"]["transforms"][1]
        inner_transforms = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        for entry in inner_transforms:
            assert "RandomSizedCrop" in entry
            assert entry["RandomSizedCrop"]["min_max_height"] == [384, 600]


class TestBuildTrainResizeConfigScaleJitter:
    """scale_jitter=False drops Option B so only the direct-resize branch (Option A) runs."""

    @pytest.mark.parametrize("square", [True, False])
    def test_scale_jitter_false_drops_crop_branch(self, square):
        """No RandomSizedCrop anywhere in result when scale jitter is disabled."""
        result = _build_train_resize_config([480, 640], square=square, scale_jitter=False)
        assert len(result) == 1
        assert "RandomSizedCrop" not in json.dumps(result)

    @pytest.mark.parametrize("square", [True, False])
    def test_scale_jitter_true_produces_one_of_with_crop(self, square):
        """Default (scale_jitter=True) outer entry is a two-branch OneOf wrapping option_b with crop."""
        result = _build_train_resize_config([480, 640], square=square, scale_jitter=True)
        assert len(result) == 1
        assert "OneOf" in result[0]
        assert "RandomSizedCrop" in json.dumps(result)


class TestCappedLongestMaxSizeRuntimeBehavior:
    """CappedLongestMaxSize only shrinks -- never upscales -- unlike plain LongestMaxSize.

    Regression tests: chaining SmallestMaxSize(resolution) -> LongestMaxSize(cap) previously forced every
    non-square image's longest side up to `cap` (e.g. 1333) regardless of the requested `resolution`, since
    Albumentations' LongestMaxSize always resizes to exactly `max_size`, up or down. CappedLongestMaxSize clamps
    the resolved scale to <= 1.0 so it behaves as a true cap, matching torchvision RandomResize's conditional
    max_size semantics.
    """

    @staticmethod
    def _resize_output_size(image_hw, resolution, max_size, scale_jitter=False):
        """Run the wrapped Albumentations resize config and return ``(width, height)``.

        Example:
            >>> size = TestCappedLongestMaxSizeRuntimeBehavior._resize_output_size(  # doctest: +ELLIPSIS
            ...     (90, 120), 640, 1333
            ... )
            [...] [INFO] rf-detr - Built 1 Albumentations transforms from config
            >>> isinstance(size, tuple) and len(size) == 2
            True
        """
        import numpy as np
        import torch
        from PIL import Image

        from rfdetr.datasets.coco import _build_train_resize_config
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        config = _build_train_resize_config([resolution], square=False, max_size=max_size, scale_jitter=scale_jitter)
        wrapper = AlbumentationsWrapper.from_config(config)[0]
        image = Image.fromarray(np.zeros((*image_hw, 3), dtype=np.uint8))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        out_image, _ = wrapper(image, target)
        return out_image.size  # (width, height)

    def test_does_not_upscale_when_already_within_cap(self):
        """A resize that already fits under max_size is left alone, not forced up to max_size."""
        width, height = self._resize_output_size((90, 120), resolution=640, max_size=1333)
        assert max(width, height) < 1333, "longest side must not be inflated to the cap"
        assert (width, height) == (853, 640), "SmallestMaxSize(640) result must pass through unchanged"

    def test_still_caps_when_resize_would_exceed_max_size(self):
        """An extreme aspect ratio that would exceed max_size after SmallestMaxSize is still capped."""
        width, height = self._resize_output_size((100, 3000), resolution=640, max_size=1000)
        assert max(width, height) == 1000, "longest side must be capped at max_size when it would be exceeded"


class TestNonSquareCropBranch:
    """Non-square Option B resizes each crop straight to a target scale, with no fixed-384 intermediate."""

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
    )
    def test_crop_branch_resamples_once(self, scales):
        """The crop branch is short-side resize then crop — two stages, not three.

        The third stage used to be a second ``RandomResize`` undoing a fixed 384x384 crop output: every image on this
        branch was resampled down to 384 and then back up to the target scale. That is both wasted work and a real loss
        of detail, and it is the hop the Albumentations backend already removed.
        """
        pipeline = _build_train_resize_transforms(scales, square=False)

        stages = pipeline.transform2.transforms

        assert [type(stage) for stage in stages] == [RandomResize, RandomChoice]

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
    )
    def test_crop_outputs_one_variant_per_target_scale(self, scales):
        """Each crop variant resizes directly to one requested scale, mirroring the square path.

        The former fixed 384x384 output followed by another resize needlessly resampled each crop twice.
        """
        pipeline = _build_train_resize_transforms(scales, square=False)

        crops = pipeline.transform2.transforms[1].transforms

        assert [crop.size for crop in crops] == [(scale, scale) for scale in scales]

    def test_crop_keeps_the_full_scale_jitter_range(self):
        """Crop heights are still sampled from [384, 600], matching the short-side resize range above them."""
        pipeline = _build_train_resize_transforms([480, 640], square=False)

        crops = pipeline.transform2.transforms[1].transforms

        assert all(crop.min_max_height == (384, 600) for crop in crops)

    def test_crop_branch_honors_the_non_square_max_size(self) -> None:
        """A direct crop output above the cap keeps paired image and target sizes within the cap.

        This forces Option B directly because ``RandomSelect`` otherwise hides the crop branch behind random selection.
        The prior final ``RandomResize`` capped this branch; direct crop output must retain that bound without adding a
        second image resample.
        """
        max_size = 1024
        pipeline = _build_train_resize_transforms([1536], square=False, max_size=max_size)
        image = Image.new("RGB", (800, 600))
        target = {"boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]), "labels": torch.tensor([1])}

        transformed_image, transformed_target = pipeline.transform2(image, target)

        assert transformed_image.size == (max_size, max_size)
        assert transformed_target["size"].tolist() == [max_size, max_size]


class TestSquareCropBranchUnchanged:
    """The square path already resized each crop directly to its target scale and must stay that way."""

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="square-single"),
            pytest.param([480, 640], id="square-multi"),
        ],
    )
    def test_crop_outputs_one_variant_per_target_scale(self, scales):
        """Square Option B keeps one RandomSizedCrop per scale, resizing straight to it."""
        pipeline = _build_train_resize_transforms(scales, square=True)

        crops = pipeline.transform2.transforms[1].transforms

        assert all(isinstance(crop, RandomSizedCrop) for crop in crops)
        assert [crop.size for crop in crops] == [(scale, scale) for scale in scales]


class TestDirectResizeBranch:
    """Option A is untouched by the crop-branch change and still differs by ``square``."""

    def test_non_square_resizes_short_side_under_a_long_side_cap(self):
        """Non-square Option A is a single capped RandomResize, so the aspect ratio survives."""
        pipeline = _build_train_resize_transforms([480, 640], square=False)

        resize_a = pipeline.transform1

        assert isinstance(resize_a, RandomResize)
        assert resize_a.sizes == [480, 640]

    def test_scale_jitter_disabled_returns_the_direct_resize_alone(self):
        """``scale_jitter=False`` drops Option B entirely rather than returning a two-branch selector."""
        pipeline = _build_train_resize_transforms([480, 640], square=False, scale_jitter=False)

        assert isinstance(pipeline, RandomResize)
