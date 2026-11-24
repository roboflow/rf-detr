# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests that ``RFDETR`` normalizes each input channel with the statistics training uses."""

import pytest
from pytest_mock import MockerFixture

from rfdetr import RFDETRNano
from rfdetr.datasets._aug_utils import cycle_channel_stats
from rfdetr.datasets.kornia_transforms import IMAGENET_MEAN, IMAGENET_STD


class TestModelChannelStats:
    """``RFDETR.means`` and ``RFDETR.stds`` repeat the RGB statistics over the model's channels."""

    @pytest.mark.parametrize(
        ("num_channels", "expected_means", "expected_stds"),
        [
            pytest.param(1, [0.485], [0.229], id="grayscale"),
            pytest.param(3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], id="rgb"),
            pytest.param(4, [0.485, 0.456, 0.406, 0.485], [0.229, 0.224, 0.225, 0.229], id="four-channel"),
        ],
    )
    def test_stats_cover_each_channel(
        self,
        num_channels: int,
        expected_means: list[float],
        expected_stds: list[float],
        mocker: MockerFixture,
    ) -> None:
        """Extra channels reuse the RGB values in order, and the statistics stay lists."""
        # The statistics are set in ``__init__`` and do not depend on the network, so skip building it.
        mocker.patch.object(RFDETRNano, "get_model")

        model = RFDETRNano(num_channels=num_channels, pretrain_weights=None, device="cpu")

        # A tuple of the same values compares unequal to a list, so this also pins the list type.
        assert (model.means, model.stds) == (expected_means, expected_stds)

    @pytest.mark.parametrize("num_channels", [1, 3, 4])
    def test_stats_match_training_normalization(self, num_channels: int, mocker: MockerFixture) -> None:
        """Prediction normalizes every channel with the values the training pipelines compute."""
        mocker.patch.object(RFDETRNano, "get_model")

        model = RFDETRNano(num_channels=num_channels, pretrain_weights=None, device="cpu")

        training_stats = (
            list(cycle_channel_stats(IMAGENET_MEAN, num_channels)),
            list(cycle_channel_stats(IMAGENET_STD, num_channels)),
        )
        assert (model.means, model.stds) == training_stats
