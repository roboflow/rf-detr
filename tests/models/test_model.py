# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr import RFDETRBase, RFDETRLarge


def _get_patch_embed_projection(model) -> torch.nn.Conv2d:
    """Return the patch-embedding projection layer for an RF-DETR model.

    RFDETR wrappers are not nn.Module; the underlying PyTorch module lives at
    ``model.model.model``.  Walk named_modules() on that object.

    Args:
        model: Instantiated RF-DETR wrapper (RFDETRBase / RFDETRLarge).

    Returns:
        The convolution used to project image channels into patch embeddings.

    Raises:
        AssertionError: If the patch-embedding projection cannot be located.
    """
    # model.model → model context; model.model.model → nn.Module
    nn_model = model.model.model
    proj = nn_model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection
    if isinstance(proj, torch.nn.Conv2d):
        return proj

    # Fallback: scan named_modules on the underlying nn.Module
    for name, module in nn_model.named_modules():
        if "patch_embeddings" in name and "projection" in name and isinstance(module, torch.nn.Conv2d):
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
