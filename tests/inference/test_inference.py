# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for ``rfdetr.inference`` model building."""

import pytest
import torch

from rfdetr.config import RFDETRNanoConfig
from rfdetr.inference import _build_model_context
from tests._markers import _requires_peft


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure reproducible random state for every test in this module."""
    torch.manual_seed(0)


@_requires_peft
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
