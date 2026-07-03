# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests for _build_train_resize_config."""

import pytest

from rfdetr.datasets.coco import _build_train_resize_config


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
    """Square=True, single scale — OneOf[Resize] + Sequential[..., OneOf[RandomSizedCrop]]."""

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
    """Square=True, multiple scales — OneOf[Resize] + Sequential[..., OneOf[RandomSizedCrop]]."""

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
    """Square=False, single scale — SmallestMaxSize uses scalar, default cap 1333."""

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
                    {
                        "OneOf": {
                            "transforms": [
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 640, "width": 640}},
                            ]
                        }
                    },
                ]
            }
        }

    def test_custom_max_size(self):
        result = _build_train_resize_config([640], square=False, max_size=800)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["Sequential"]["transforms"][1] == {"LongestMaxSize": {"max_size": 800}}


class TestBuildTrainResizeConfigNonSquareMultiScale:
    """Square=False, multiple scales — SmallestMaxSize uses list directly."""

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
                    {
                        "OneOf": {
                            "transforms": [
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 480, "width": 480}},
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 640, "width": 640}},
                            ]
                        }
                    },
                ]
            }
        }

    def test_custom_max_size_applies_to_option_a_only(self):
        """max_size caps option_a's long side; option_b now resizes the crop directly to the target (no cap step)."""
        result = _build_train_resize_config([480, 640], square=False, max_size=1000)
        option_a = result[0]["OneOf"]["transforms"][0]
        option_b_steps = result[0]["OneOf"]["transforms"][1]["Sequential"]["transforms"]
        assert option_a["Sequential"]["transforms"][1] == {"LongestMaxSize": {"max_size": 1000}}
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
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
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
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
    )
    def test_option_b_crop_uses_full_scale_jitter_range(self, scales):
        """RandomSizedCrop min_max_height matches SmallestMaxSize range [384, 600] for full zoom-out diversity."""
        result = _build_train_resize_config(scales, square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        crop_variants = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        assert all(entry["RandomSizedCrop"]["min_max_height"] == [384, 600] for entry in crop_variants)

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
        ],
    )
    def test_square_option_b_unchanged(self, scales, square):
        """Square path must still use RandomSizedCrop parameterized by scale."""
        result = _build_train_resize_config(scales, square=square)
        option_b = result[0]["OneOf"]["transforms"][1]
        inner_transforms = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        for entry in inner_transforms:
            assert "RandomSizedCrop" in entry
            assert entry["RandomSizedCrop"]["min_max_height"] == [384, 600]
