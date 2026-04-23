# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterization tests for _build_train_resize_config."""

import albumentations as alb
import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import _build_train_resize_config, make_coco_transforms
from rfdetr.datasets.transforms import AlbumentationsWrapper


class TestBuildTrainResizeConfigStructure:
    """Top-level structure is always a single-element list wrapping a OneOf."""

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
            pytest.param([640], False, id="nonsquare-single"),
            pytest.param([480, 640], False, id="nonsquare-multi"),
        ],
    )
    def test_returns_single_element_list(self, scales, square):
        result = _build_train_resize_config(scales, square=square)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
            pytest.param([640], False, id="nonsquare-single"),
            pytest.param([480, 640], False, id="nonsquare-multi"),
        ],
    )
    def test_top_level_is_oneof_with_two_branches(self, scales, square):
        result = _build_train_resize_config(scales, square=square)
        entry = result[0]
        assert "OneOf" in entry
        oneof = entry["OneOf"]
        assert len(oneof["transforms"]) == 2


class TestBuildTrainResizeConfigSquareSingleScale:
    """square=True, single scale — OneOf[Resize] + Sequential[..., OneOf[RandomSizedCrop]]."""

    def test_option_a_is_oneof_wrapping_single_resize(self):
        result = _build_train_resize_config([640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [{"Resize": {"height": 640, "width": 640}}],
            }
        }

    def test_option_b_is_sequential_with_oneof_crop(self):
        result = _build_train_resize_config([640], square=True)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 640, "width": 640}},
                            ],
                        }
                    },
                ]
            }
        }

    def test_uses_correct_scale_value(self):
        result = _build_train_resize_config([480], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [{"Resize": {"height": 480, "width": 480}}],
            }
        }


class TestBuildTrainResizeConfigSquareMultiScale:
    """square=True, multiple scales — OneOf[Resize] + Sequential[..., OneOf[RandomSizedCrop]]."""

    def test_option_a_is_oneof_of_resizes(self):
        result = _build_train_resize_config([480, 640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [
                    {"Resize": {"height": 480, "width": 480}},
                    {"Resize": {"height": 640, "width": 640}},
                ],
            }
        }

    def test_option_b_is_sequential_with_oneof_crop(self):
        result = _build_train_resize_config([480, 640], square=True)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 480, "width": 480}},
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 640, "width": 640}},
                            ],
                        }
                    },
                ]
            }
        }

    def test_three_scales_produce_three_resize_options(self):
        result = _build_train_resize_config([384, 512, 640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert len(option_a["OneOf"]["transforms"]) == 3


class TestBuildTrainResizeConfigNonSquareSingleScale:
    """square=False, single scale — SmallestMaxSize uses scalar, default cap 1333."""

    def test_option_a_uses_scalar_size(self):
        result = _build_train_resize_config([640], square=False)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": 640}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_option_b_uses_scalar_size(self):
        result = _build_train_resize_config([640], square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {"RandomCrop": {"height": 384, "width": 384}},
                    {"SmallestMaxSize": {"max_size": 640}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_custom_max_size(self):
        result = _build_train_resize_config([640], square=False, max_size=800)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["Sequential"]["transforms"][1] == {"LongestMaxSize": {"max_size": 800}}


class TestBuildTrainResizeConfigNonSquareMultiScale:
    """square=False, multiple scales — SmallestMaxSize uses list directly."""

    def test_option_a_uses_list_size(self):
        result = _build_train_resize_config([480, 640], square=False)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [480, 640]}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_option_b_uses_list_size(self):
        result = _build_train_resize_config([480, 640], square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {"RandomCrop": {"height": 384, "width": 384}},
                    {"SmallestMaxSize": {"max_size": [480, 640]}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_custom_max_size_propagates_to_both_options(self):
        result = _build_train_resize_config([480, 640], square=False, max_size=1000)
        option_a = result[0]["OneOf"]["transforms"][0]
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_a["Sequential"]["transforms"][1] == {"LongestMaxSize": {"max_size": 1000}}
        assert option_b["Sequential"]["transforms"][3] == {"LongestMaxSize": {"max_size": 1000}}


class TestBuildTrainResizeConfigDivisor:
    """divisor param appends PadIfNeeded so outputs are divisible by the backbone block size."""

    def _expected_pad(self, divisor: int) -> dict:
        return {
            "PadIfNeeded": {
                "min_height": None,
                "min_width": None,
                "pad_height_divisor": divisor,
                "pad_width_divisor": divisor,
                "border_mode": 0,
                "fill": 0,
            }
        }

    def test_nonsquare_single_scale_appends_pad_to_option_a(self):
        result = _build_train_resize_config([896], square=False, max_size=1333, divisor=32)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["Sequential"]["transforms"][-1] == self._expected_pad(32)

    def test_nonsquare_single_scale_appends_pad_to_option_b(self):
        result = _build_train_resize_config([896], square=False, max_size=1333, divisor=32)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b["Sequential"]["transforms"][-1] == self._expected_pad(32)

    def test_nonsquare_multiscale_appends_pad_to_both_options(self):
        result = _build_train_resize_config([480, 640], square=False, divisor=56)
        option_a = result[0]["OneOf"]["transforms"][0]
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_a["Sequential"]["transforms"][-1] == self._expected_pad(56)
        assert option_b["Sequential"]["transforms"][-1] == self._expected_pad(56)

    def test_divisor_none_preserves_old_behavior(self):
        """Default divisor=None produces the same config as before the fix."""
        result = _build_train_resize_config([896], square=False, max_size=1333)
        option_a = result[0]["OneOf"]["transforms"][0]
        for t in option_a["Sequential"]["transforms"]:
            assert "PadIfNeeded" not in t

    def test_square_branch_ignores_divisor(self):
        """Square outputs are already fixed-size; no PadIfNeeded needed even if divisor is set."""
        result = _build_train_resize_config([640], square=True, divisor=32)
        option_b = result[0]["OneOf"]["transforms"][1]
        for t in option_b["Sequential"]["transforms"]:
            assert "PadIfNeeded" not in t


class TestNonSquareResizeDivisibilityRegression:
    """Regression test for issue #983.

    When ``square_resize_div_64=False``, the training resize pipeline must produce
    outputs with dimensions divisible by ``patch_size * num_windows``; otherwise
    the windowed-attention backbone rejects the input.
    """

    @pytest.fixture(autouse=True)
    def reset_rng(self):
        """Reset global RNG state before each test to prevent cross-test leakage."""
        np.random.seed(42)
        torch.manual_seed(42)
        yield

    @pytest.mark.parametrize(
        "img_h,img_w",
        [
            pytest.param(1200, 800, id="portrait-3:2"),
            pytest.param(800, 1200, id="landscape-3:2"),
            pytest.param(1024, 768, id="4:3"),
            pytest.param(1500, 500, id="wide-3:1"),
            pytest.param(500, 1500, id="tall-1:3"),
            pytest.param(777, 613, id="arbitrary-non-aligned"),
        ],
    )
    def test_output_dims_divisible_by_block_size(self, img_h: int, img_w: int) -> None:
        """For many random trials (covering both OneOf branches), output H/W are divisible by 32."""
        patch_size, num_windows = 16, 2
        divisor = patch_size * num_windows  # 32

        wrappers = AlbumentationsWrapper.from_config(
            _build_train_resize_config([896], square=False, max_size=1333, divisor=divisor)
        )
        assert len(wrappers) == 1
        wrapper = wrappers[0]

        image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (img_h, img_w, 3), dtype=np.uint8))
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        for seed in range(40):
            np.random.seed(seed)
            torch.manual_seed(seed)
            aug_image, _ = wrapper(image, target)
            w, h = aug_image.size
            assert h % divisor == 0, f"seed={seed} input=({img_h},{img_w}) output_h={h} not divisible by {divisor}"
            assert w % divisor == 0, f"seed={seed} input=({img_h},{img_w}) output_w={w} not divisible by {divisor}"

    def test_make_coco_transforms_train_passes_divisor(self) -> None:
        """End-to-end: make_coco_transforms(train) with patch_size=16, num_windows=2 produces dims divisible by 32."""
        patch_size, num_windows = 16, 2
        divisor = patch_size * num_windows

        transform = make_coco_transforms(
            image_set="train",
            resolution=896,
            multi_scale=False,
            patch_size=patch_size,
            num_windows=num_windows,
        )

        image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (1024, 768, 3), dtype=np.uint8))
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "orig_size": torch.tensor([1024, 768]),
            "size": torch.tensor([1024, 768]),
        }

        for seed in range(20):
            np.random.seed(seed)
            torch.manual_seed(seed)
            out_image, _ = transform(image, target)
            # After ToImage + ToDtype the tensor is CHW
            _, h, w = out_image.shape
            assert h % divisor == 0, f"seed={seed} output_h={h} not divisible by {divisor}"
            assert w % divisor == 0, f"seed={seed} output_w={w} not divisible by {divisor}"

    def test_make_coco_transforms_val_produces_divisible_dims(self) -> None:
        """Val pipeline must also produce dims divisible by patch_size * num_windows."""
        patch_size, num_windows = 16, 2
        divisor = patch_size * num_windows

        transform = make_coco_transforms(
            image_set="val",
            resolution=896,
            multi_scale=False,
            patch_size=patch_size,
            num_windows=num_windows,
        )

        image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (1024, 768, 3), dtype=np.uint8))
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "orig_size": torch.tensor([1024, 768]),
            "size": torch.tensor([1024, 768]),
        }

        out_image, _ = transform(image, target)
        _, h, w = out_image.shape
        assert h % divisor == 0, f"val output_h={h} not divisible by {divisor}"
        assert w % divisor == 0, f"val output_w={w} not divisible by {divisor}"

    def test_val_pipeline_structural_contains_pad_if_needed(self) -> None:
        """Val/test resize pipeline third wrapper must be PadIfNeeded with correct divisor."""

        patch_size, num_windows = 16, 2
        block_size = patch_size * num_windows

        compose = make_coco_transforms(
            image_set="val",
            resolution=896,
            multi_scale=False,
            patch_size=patch_size,
            num_windows=num_windows,
        )

        # Third transform in the Compose must be the divisor-padding wrapper
        wrapper = compose.transforms[2]
        assert isinstance(wrapper, AlbumentationsWrapper), (
            f"Expected AlbumentationsWrapper at index 2, got {type(wrapper).__name__}"
        )
        inner = wrapper.transform.transforms[0]
        assert isinstance(inner, alb.PadIfNeeded), f"Expected PadIfNeeded inside wrapper, got {type(inner).__name__}"
        assert inner.pad_height_divisor == block_size
        assert inner.pad_width_divisor == block_size
