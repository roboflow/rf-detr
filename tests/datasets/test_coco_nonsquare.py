# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Non-square ``(height, width)`` resolutions in the COCO transform builders."""

import warnings
from typing import Any

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import (
    _build_train_resize_config,
    compute_multi_scale_scales,
    draft_size_for_transforms,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from tests._markers import _requires_albumentations

_BACKENDS = [
    pytest.param(None, id="torchvision"),
    pytest.param({"HorizontalFlip": {"p": 0.5}}, id="albumentations", marks=_requires_albumentations),
]


def _sample() -> tuple[Image.Image, dict[str, Any]]:
    """Return a 200x120 image with one box."""
    target = {
        "boxes": torch.tensor([[20.0, 30.0, 80.0, 90.0]]),
        "labels": torch.tensor([1]),
        "size": torch.tensor([120, 200]),
        "orig_size": torch.tensor([120, 200]),
    }
    return Image.new("RGB", (200, 120)), target


def _build(factory: Any, image_set: str, resolution: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factory(image_set, resolution, **kwargs)


class TestMultiScaleScales:
    """``compute_multi_scale_scales`` keeps side lengths for an int and returns pairs for a pair."""

    def test_int_resolution_keeps_side_lengths(self) -> None:
        """An int resolution returns the same side lengths as before."""
        assert compute_multi_scale_scales(512) == [320, 384, 448, 512, 576, 640, 704, 768]

    @pytest.mark.parametrize("resolution", [(384, 768), (768, 384), (512, 512)])
    def test_pair_resolution_yields_block_aligned_pairs(self, resolution: tuple[int, int]) -> None:
        """Every scale of a pair is block-aligned and keeps the base orientation and aspect ratio."""
        scales = compute_multi_scale_scales(resolution, patch_size=16, num_windows=4)

        assert resolution in scales
        for height, width in scales:
            assert height % 64 == 0 and width % 64 == 0
            assert (height <= width) == (resolution[0] <= resolution[1])
            assert height / width == pytest.approx(resolution[0] / resolution[1], rel=0.2)

    def test_pair_resolution_steps_the_short_side_like_a_side_length(self) -> None:
        """The shorter side of a pair takes the steps a side length of that size takes."""
        scales = compute_multi_scale_scales((384, 768), patch_size=16, num_windows=4)

        assert [height for height, _ in scales] == compute_multi_scale_scales(384, patch_size=16, num_windows=4)

    def test_pair_resolution_drops_too_small_scales(self) -> None:
        """Scales whose shorter side is under two blocks are dropped."""
        scales = compute_multi_scale_scales((128, 256), patch_size=16, num_windows=4)

        assert min(min(scale) for scale in scales) == 128

    @pytest.mark.parametrize(
        ("resolution", "multi_scale", "expected"),
        [
            pytest.param((384, 768), False, 768, id="pair"),
            pytest.param((768, 384), False, 768, id="transposed-pair"),
            pytest.param(512, True, 768, id="int-multi-scale"),
            pytest.param((384, 768), True, 1280, id="pair-multi-scale"),
        ],
    )
    def test_draft_size_uses_longer_side(self, resolution: Any, multi_scale: bool, expected: int) -> None:
        """The decode box keeps the longer side of the largest scale."""
        assert draft_size_for_transforms("train", resolution, multi_scale=multi_scale) == expected


class TestTrainResizeConfig:
    """``_build_train_resize_config`` takes ``(height, width)`` scales."""

    def test_square_branch_resizes_to_height_width(self) -> None:
        """The exact-resize branch resizes to the pair's height and width."""
        (entry,) = _build_train_resize_config([(64, 96)], square=True, scale_jitter=False)

        assert entry == {"OneOf": {"transforms": [{"Resize": {"height": 64, "width": 96}}]}}

    def test_aspect_preserving_branch_uses_short_side(self) -> None:
        """The aspect-preserving branch resizes the shorter side to the pair's shorter side."""
        (entry,) = _build_train_resize_config([(64, 96), (128, 192)], square=False, scale_jitter=False)

        assert entry["Sequential"]["transforms"][0] == {"SmallestMaxSize": {"max_size": [64, 128]}}

    @pytest.mark.parametrize("scale_jitter", [True, False])
    def test_aspect_preserving_config_treats_a_pair_as_its_short_side(self, scale_jitter: bool) -> None:
        """Without exact resize, a pair builds the config of its shorter side, crop branch included."""
        from_pairs = _build_train_resize_config([(64, 96), (192, 128)], square=False, scale_jitter=scale_jitter)
        from_short_sides = _build_train_resize_config([64, 128], square=False, scale_jitter=scale_jitter)

        assert from_pairs == from_short_sides

    def test_int_scales_build_the_same_config_as_square_pairs(self) -> None:
        """A side length and the square pair of that side build one config."""
        from_ints = _build_train_resize_config([64, 128], square=True)
        from_pairs = _build_train_resize_config([(64, 64), (128, 128)], square=True)

        assert from_ints == from_pairs


class TestNonSquareTransforms:
    """The transform builders resize to a ``(height, width)`` resolution on both backends."""

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    @pytest.mark.parametrize("scale_jitter", [True, False])
    def test_square_div_64_resizes_to_exact_height_width(
        self, image_set: str, aug_config: Any, scale_jitter: bool
    ) -> None:
        """The exact-resize pipeline outputs ``(height, width)`` for every split."""
        transforms = _build(
            make_coco_transforms_square_div_64, image_set, (64, 96), aug_config=aug_config, scale_jitter=scale_jitter
        )
        image, target = transforms(*_sample())

        assert image.shape == (3, 64, 96)
        assert target["size"].tolist() == [64, 96]

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    def test_multi_scale_square_div_64_draws_from_pair_scales(self, aug_config: Any) -> None:
        """Multi-scale training draws each output size from the pair scales."""
        transforms = _build(
            make_coco_transforms_square_div_64,
            "train",
            (128, 192),
            multi_scale=True,
            patch_size=16,
            num_windows=2,
            aug_config=aug_config,
        )
        scales = set(compute_multi_scale_scales((128, 192), patch_size=16, num_windows=2))
        for _ in range(8):
            image, _ = transforms(*_sample())
            assert tuple(image.shape[-2:]) in scales

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    @pytest.mark.parametrize("image_set", ["train", "val", "test"])
    def test_aspect_preserving_pipeline_uses_short_side(self, image_set: str, aug_config: Any) -> None:
        """The aspect-preserving pipeline keeps the source aspect, with its short side at ``min(height, width)``."""
        transforms = _build(make_coco_transforms, image_set, (60, 96), aug_config=aug_config, scale_jitter=False)
        image, _ = transforms(*_sample())

        assert tuple(image.shape[-2:]) == (60, 100)

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    @pytest.mark.parametrize("resolution", [(60, 96), (96, 60)])
    def test_aspect_preserving_scale_jitter_never_stretches_to_the_pair(
        self, aug_config: Any, resolution: tuple[int, int]
    ) -> None:
        """With scale jitter, both branches act on the shorter side: a 60x100 resize or a 60x60 crop, as for 60."""
        transforms = _build(make_coco_transforms, "train", resolution, aug_config=aug_config, scale_jitter=True)

        shapes = {tuple(transforms(*_sample())[0].shape[-2:]) for _ in range(40)}

        assert shapes == {(60, 100), (60, 60)}

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    def test_val_speed_resizes_to_exact_height_width(self, aug_config: Any) -> None:
        """``val_speed`` resizes to exactly ``(height, width)``."""
        transforms = _build(make_coco_transforms, "val_speed", (64, 96), aug_config=aug_config)
        image, _ = transforms(*_sample())

        assert image.shape == (3, 64, 96)
