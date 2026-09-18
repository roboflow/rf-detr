# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests for _build_train_resize_transforms, the torchvision-native resize pipeline.

The Albumentations backend's equivalent builder is covered by test_coco_resize_config.py. The two backends are meant to
express the same augmentation recipe, so the non-square crop branch is asserted here against the same shape that file
already pins for Albumentations.
"""

import pytest
import torch
from PIL import Image

from rfdetr.datasets._torchvision import RandomChoice, RandomResize, RandomSizedCrop
from rfdetr.datasets.coco import _build_train_resize_transforms


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
