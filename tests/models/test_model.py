# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr import RFDETRBase, RFDETRLarge


def _get_patch_embed_projection(model: torch.nn.Module) -> torch.nn.Conv2d:
    """Return the patch-embedding projection layer for an RF-DETR model.

    Args:
        model: Instantiated RF-DETR model.

    Returns:
        The convolution used to project image channels into patch embeddings.

    Raises:
        AssertionError: If the patch-embedding projection cannot be located.
    """
    for attr_chain in (
        ("model", "backbone", "patch_embed", "proj"),
        ("backbone", "patch_embed", "proj"),
        ("patch_embed", "proj"),
    ):
        current = model
        for attr in attr_chain:
            if not hasattr(current, attr):
                break
            current = getattr(current, attr)
        else:
            if isinstance(current, torch.nn.Conv2d):
                return current

    for name, module in model.named_modules():
        if name.endswith("patch_embed.proj") and isinstance(module, torch.nn.Conv2d):
            return module

    msg = "Could not find patch embedding projection on model"
    raise AssertionError(msg)


@pytest.mark.parametrize("model_class", [RFDETRBase, RFDETRLarge])
@pytest.mark.parametrize("channels", [1, 4])
def test_multispectral_support(model_class, channels: int) -> None:
    model = model_class(
        num_channels=channels,
        device="cpu",
        pretrain_weights=None,
    )

    patch_embed_projection = _get_patch_embed_projection(model)

    assert patch_embed_projection.in_channels == channels
