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
    def test_top_level_is_oneof_with_equal_branch_probs(self, scales, square):
        result = _build_train_resize_config(scales, square=square)
        entry = result[0]
        assert "OneOf" in entry
        oneof = entry["OneOf"]
        assert oneof["probs"] == [0.5, 0.5]
        assert oneof["p"] == 1.0
        assert len(oneof["transforms"]) == 2


class TestBuildTrainResizeConfigSquareSingleScale:
    """square=True, single scale — OneOf[Resize] + Sequential[..., OneOf[RandomSizedCrop]]."""

    def test_option_a_is_oneof_wrapping_single_resize(self):
        result = _build_train_resize_config([640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [{"Resize": {"height": 640, "width": 640}}],
                "probs": [1.0],
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
                            "probs": [1.0],
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
                "probs": [1.0],
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
                "probs": [0.5, 0.5],
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
                            "probs": [0.5, 0.5],
                        }
                    },
                ]
            }
        }

    def test_equal_probs_for_three_scales(self):
        result = _build_train_resize_config([384, 512, 640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["OneOf"]["probs"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


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
