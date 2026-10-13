# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr.inference weight-adaptation helpers."""

import pytest
import torch

from rfdetr.config import RFDETRNanoConfig
from rfdetr.inference import _adapt_input_conv, _build_model_context
from rfdetr.utilities.package import is_installed


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


@pytest.mark.skipif(not is_installed("peft"), reason="backbone_lora requires the optional peft package")
class TestBuildModelContextLoraChannels:
    """``_build_model_context`` adapts DINOv2's patch embedding even when LoRA has wrapped the encoder."""

    def test_lora_wrapped_encoder_accepts_extra_channels(self) -> None:
        """A LoRA-wrapped DINOv2 encoder gets a 4-channel patch embedding instead of a ValueError.

        ``backbone_lora=True`` swaps the DinoV2 encoder for a PEFT wrapper before channel adaptation runs, so a type
        check on the encoder object rejected a supported configuration; the encoder name decides instead.
        """
        from peft import PeftModel  # optional dependency, guarded by the class skipif

        config = RFDETRNanoConfig(num_channels=4, backbone_lora=True, pretrain_weights=None, device="cpu")

        encoder = _build_model_context(config).model.backbone[0].encoder

        assert isinstance(encoder, PeftModel)
        assert encoder.encoder.embeddings.patch_embeddings.projection.in_channels == 4
