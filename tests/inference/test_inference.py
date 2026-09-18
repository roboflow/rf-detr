# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr.inference weight-adaptation helpers."""

import pytest
import torch

from rfdetr.inference import _adapt_input_conv


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure reproducible random state for every test in this module."""
    torch.manual_seed(0)


class TestAdaptInputConv:
    @pytest.mark.parametrize(
        ("num_channels", "expected_shape", "expected_builder"),
        [
            pytest.param(3, (8, 3, 3, 3), lambda weight: weight, id="identity_3ch"),
            pytest.param(1, (8, 1, 3, 3), lambda weight: weight.mean(dim=1, keepdim=True), id="mean_1ch"),
            pytest.param(
                4,
                (8, 4, 3, 3),
                lambda weight: torch.cat([weight, weight], dim=1)[:, :4] * (3.0 / 4.0),
                id="tile_4ch",
            ),
            pytest.param(
                6,
                (8, 6, 3, 3),
                lambda weight: torch.cat([weight, weight], dim=1)[:, :6] * (3.0 / 6.0),
                id="tile_6ch",
            ),
            pytest.param(
                2,
                (8, 2, 3, 3),
                lambda weight: weight[:, :2] * (3.0 / 2.0),
                id="tile_2ch",
            ),
        ],
    )
    def test_adapt_input_conv(self, num_channels, expected_shape, expected_builder):
        """Verify shape and values for each _adapt_input_conv branch."""
        conv_weight = torch.randn(8, 3, 3, 3)

        adapted_weight = _adapt_input_conv(num_channels, conv_weight)
        expected_weight = expected_builder(conv_weight)

        assert adapted_weight.shape == expected_shape
        torch.testing.assert_close(adapted_weight, expected_weight)
