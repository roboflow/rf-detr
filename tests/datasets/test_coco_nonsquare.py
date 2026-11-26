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

from rfdetr.datasets import _torchvision
from rfdetr.datasets._torchvision import RandomSizedCrop
from rfdetr.datasets.coco import (
    _build_train_resize_config,
    _random_sized_crop_config,
    compute_multi_scale_scales,
    draft_size_for_transforms,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.datasets.transforms import AlbumentationsWrapper
from tests._markers import _requires_albumentations

_BACKENDS = [
    pytest.param(None, id="torchvision"),
    pytest.param({"HorizontalFlip": {"p": 0.5}}, id="albumentations", marks=_requires_albumentations),
]


def _sample() -> tuple[Image.Image, dict[str, Any]]:
    """Return a 200x120 image with one box.

    Examples:
        >>> image, target = _sample()
        >>> image.size, target["boxes"].tolist()
        ((200, 120), [[20.0, 30.0, 80.0, 90.0]])
    """
    target = {
        "boxes": torch.tensor([[20.0, 30.0, 80.0, 90.0]]),
        "labels": torch.tensor([1]),
        "size": torch.tensor([120, 200]),
        "orig_size": torch.tensor([120, 200]),
    }
    return Image.new("RGB", (200, 120)), target


def _grid_sample(width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
    """Return a ``width`` x ``height`` image covered by a grid of 40x40 boxes.

    Examples:
        >>> image, target = _grid_sample(200, 120)
        >>> image.size, target["boxes"].tolist()
        ((200, 120), [[10.0, 10.0, 50.0, 50.0], [70.0, 10.0, 110.0, 50.0], [130.0, 10.0, 170.0, 50.0]])
    """
    lefts, tops = torch.meshgrid(torch.arange(10.0, width - 50, 60), torch.arange(10.0, height - 50, 60), indexing="xy")
    lefts, tops = lefts.flatten(), tops.flatten()
    count = len(lefts)
    target = {
        "boxes": torch.stack([lefts, tops, lefts + 40, tops + 40], dim=1),
        "labels": torch.ones(count, dtype=torch.int64),
        "area": torch.full((count,), 1600.0),
        "iscrowd": torch.zeros(count, dtype=torch.int64),
        "size": torch.tensor([height, width]),
        "orig_size": torch.tensor([height, width]),
    }
    return Image.new("RGB", (width, height)), target


def _box_aspects(image: torch.Tensor, target: dict[str, Any]) -> torch.Tensor:
    """Return the pixel width / height of each normalized ``cxcywh`` box that lies clear of the image border.

    Examples:
        >>> _box_aspects(torch.zeros(3, 64, 128), {"boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.5, 0.1, 1, 0.2]])})
        tensor([2.])
    """
    height, width = image.shape[-2:]
    center_x, center_y, box_width, box_height = (target["boxes"] * torch.tensor([width, height, width, height])).T
    inside = (
        (center_x - box_width / 2 > 1)
        & (center_y - box_height / 2 > 1)
        & (center_x + box_width / 2 < width - 1)
        & (center_y + box_height / 2 < height - 1)
    )
    return (box_width / box_height)[inside]


def _build(factory: Any, image_set: str, resolution: Any, **kwargs: Any) -> Any:
    """Return ``factory(image_set, resolution, **kwargs)`` with its ``UserWarning`` messages silenced.

    Examples:
        >>> _build(lambda image_set, resolution, **kwargs: (image_set, resolution, kwargs), "val", (64, 96), p=1)
        ('val', (64, 96), {'p': 1})
    """
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

    @pytest.mark.parametrize(
        ("pair", "min_max_height", "w2h_ratio"),
        [
            pytest.param((384, 672), [384, 600], 1.75, id="wide"),
            pytest.param((672, 384), [672, 1050], 384 / 672, id="tall"),
        ],
    )
    def test_exact_resize_crop_has_the_aspect_ratio_of_the_pair(
        self, pair: tuple[int, int], min_max_height: list[int], w2h_ratio: float
    ) -> None:
        """The scale-jitter crop of a pair has the pair's aspect ratio, its shorter side within 384-600."""
        (entry,) = _build_train_resize_config([pair], square=True)

        (crop,) = entry["OneOf"]["transforms"][1]["Sequential"]["transforms"][1]["OneOf"]["transforms"]

        assert crop == {
            "AspectRandomSizedCrop": {
                "min_max_height": min_max_height,
                "height": pair[0],
                "width": pair[1],
                "w2h_ratio": pytest.approx(w2h_ratio),
            }
        }

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
    @pytest.mark.parametrize(
        ("resolution", "source"),
        [
            pytest.param((64, 112), (1050, 600), id="wide"),
            pytest.param((112, 64), (600, 1050), id="tall"),
            pytest.param(64, (600, 600), id="square"),
        ],
    )
    def test_scale_jitter_keeps_object_aspect_when_source_matches_resolution(
        self, aug_config: Any, resolution: Any, source: tuple[int, int]
    ) -> None:
        """Neither branch stretches a source that has the resolution's aspect ratio: square boxes stay square."""
        transforms = _build(
            make_coco_transforms_square_div_64, "train", resolution, aug_config=aug_config, scale_jitter=True
        )
        torch.manual_seed(0)

        aspects = torch.cat([_box_aspects(*transforms(*_grid_sample(*source))) for _ in range(40)])

        assert len(aspects) > 40
        assert aspects.min() > 0.97 and aspects.max() < 1.03

    @pytest.mark.parametrize("output_size", [(64, 112), (112, 64)])
    @pytest.mark.parametrize("source", [(800, 600), (600, 800), (300, 280)])
    def test_torchvision_crop_shrinks_both_sides_to_fit_the_image(
        self, output_size: tuple[int, int], source: tuple[int, int]
    ) -> None:
        """A crop that does not fit the image shrinks on both sides, so it keeps the output's aspect ratio."""
        cropper = RandomSizedCrop((384, 600), output_size)
        torch.manual_seed(0)

        for _ in range(20):
            image, target = cropper(*_grid_sample(*source))
            boxes = target["boxes"]
            inside = (boxes[:, :2] > 1).all(dim=1) & (boxes[:, 2] < image.width - 1) & (boxes[:, 3] < image.height - 1)
            aspects = ((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))[inside]

            assert image.size == (output_size[1], output_size[0])
            assert len(aspects) > 0
            assert aspects.min() > 0.97 and aspects.max() < 1.03

    @_requires_albumentations
    @pytest.mark.parametrize("output_size", [(64, 112), (112, 64)])
    @pytest.mark.parametrize("source", [(800, 600), (600, 800), (300, 280)])
    def test_albumentations_crop_shrinks_both_sides_to_fit_the_image(
        self, output_size: tuple[int, int], source: tuple[int, int]
    ) -> None:
        """The Albumentations crop of a pair shrinks on both sides to fit an image of another aspect ratio."""
        (cropper,) = AlbumentationsWrapper.from_config([_random_sized_crop_config(*output_size)], strict=True)

        for _ in range(20):
            image, target = cropper(*_grid_sample(*source))
            boxes = target["boxes"]
            inside = (boxes[:, :2] > 1).all(dim=1) & (boxes[:, 2] < image.width - 1) & (boxes[:, 3] < image.height - 1)
            aspects = ((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))[inside]

            assert image.size == (output_size[1], output_size[0])
            assert len(aspects) > 0
            assert aspects.min() > 0.97 and aspects.max() < 1.03

    @pytest.mark.parametrize("source", [(800, 600), (450, 700), (300, 280)])
    def test_torchvision_square_crop_draws_as_before(
        self, source: tuple[int, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A square output keeps the historical crop box and random draws, so seeded runs do not change."""
        width, height = source
        boxes: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            _torchvision, "crop", lambda image, target, *box: boxes.append(box) or (image, target), raising=True
        )

        torch.manual_seed(0)
        RandomSizedCrop((384, 600), (64, 64))(Image.new("RGB", source))
        next_draw = torch.rand(1)

        torch.manual_seed(0)
        max_side = min(600, height, width)
        min_side = min(384, max_side)
        side = int(torch.randint(min_side, max_side + 1, ()).item()) if max_side > min_side else max_side
        top = int(torch.randint(0, height - side + 1, ()).item()) if height > side else 0
        left = int(torch.randint(0, width - side + 1, ()).item()) if width > side else 0
        assert boxes == [(top, left, side, side)]
        assert torch.equal(next_draw, torch.rand(1))

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
