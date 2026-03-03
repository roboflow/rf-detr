# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.main import Model
from rfdetr.models.backbone.dinov2_with_windowed_attn import Dinov2WithRegistersDropPath
from rfdetr.models.lwdetr import LWDETR


@pytest.fixture
def model_with_drop_path() -> Model:
    """Create RF-DETR Nano model with drop_path enabled."""
    return Model(
        encoder="dinov2_windowed_small",
        num_classes=3,
        device="cpu",
        pretrain_weights=None,
        drop_path=0.1,
        resolution=384,
        vit_encoder_num_layers=12,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        out_feature_indexes=[2, 5, 8, 11],
        projector_scale=["P4"],
        hidden_dim=256,
        dec_layers=3,
        segmentation_head=False,
    )


@pytest.fixture
def model_without_drop_path() -> Model:
    """Create RF-DETR Nano model without drop_path."""
    return Model(
        encoder="dinov2_windowed_small",
        num_classes=3,
        device="cpu",
        pretrain_weights=None,
        drop_path=0.0,
        resolution=384,
        vit_encoder_num_layers=12,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        out_feature_indexes=[2, 5, 8, 11],
        projector_scale=["P4"],
        hidden_dim=256,
        dec_layers=3,
        segmentation_head=False,
    )


def test_get_backbone_encoder_layers_dinov2(model_with_drop_path: Model) -> None:
    """Verify _get_backbone_encoder_layers() returns encoder.encoder.layer for DinoV2."""
    model: LWDETR = model_with_drop_path.model

    layers = model._get_backbone_encoder_layers()
    assert layers is not None

    enc = model.backbone[0].encoder
    assert hasattr(enc, "encoder"), "DinoV2 encoder should have encoder attribute"
    assert hasattr(enc.encoder, "encoder"), "DinoV2 encoder.encoder should have encoder attribute"
    assert hasattr(enc.encoder.encoder, "layer"), "DinoV2 encoder.encoder.encoder should have layer attribute"
    assert layers is enc.encoder.encoder.layer, "Should return encoder.encoder.encoder.layer"

    assert len(layers) > 0, "Should have at least one layer"
    for layer in layers:
        assert hasattr(layer, "drop_path"), "Each layer should have drop_path attribute"


def test_update_drop_path_dinov2(model_with_drop_path: Model) -> None:
    """Verify update_drop_path() sets drop_prob values correctly with linear schedule."""
    model: LWDETR = model_with_drop_path.model

    layers = model._get_backbone_encoder_layers()
    assert layers is not None

    num_layers = len(layers)
    drop_path_rate = 0.1

    model.update_drop_path(drop_path_rate, num_layers)

    # All layers must be Dinov2WithRegistersDropPath (drop_path_rate=0.1 > 0 at model build time).
    expected_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
    for i, layer in enumerate(layers):
        assert isinstance(layer.drop_path, Dinov2WithRegistersDropPath), (
            f"Layer {i} drop_path should be Dinov2WithRegistersDropPath, got {type(layer.drop_path)}"
        )
        actual_prob = layer.drop_path.drop_prob
        assert abs(actual_prob - expected_rates[i]) < 1e-6, (
            f"Layer {i} drop_prob should be {expected_rates[i]}, got {actual_prob}"
        )

    assert abs(layers[0].drop_path.drop_prob - 0.0) < 1e-6, "First layer should have drop_prob = 0"
    assert abs(layers[-1].drop_path.drop_prob - drop_path_rate) < 1e-6, (
        f"Last layer should have drop_prob = {drop_path_rate}"
    )


def test_drop_path_initialization(model_with_drop_path: Model, model_without_drop_path: Model) -> None:
    """Verify drop_path initialization: Dinov2WithRegistersDropPath vs Identity based on rate."""
    model_with_dp: LWDETR = model_with_drop_path.model
    model_without_dp: LWDETR = model_without_drop_path.model

    layers_with_dp = model_with_dp._get_backbone_encoder_layers()
    layers_without_dp = model_without_dp._get_backbone_encoder_layers()

    assert layers_with_dp is not None
    assert layers_without_dp is not None

    # drop_path_rate=0.1 → every layer initialised as Dinov2WithRegistersDropPath
    for i, layer in enumerate(layers_with_dp):
        assert hasattr(layer, "drop_path"), "Layer should have drop_path attribute"
        assert isinstance(layer.drop_path, Dinov2WithRegistersDropPath), (
            f"Layer {i}: expected Dinov2WithRegistersDropPath, got {type(layer.drop_path)}"
        )

    # drop_path_rate=0.0 → every layer initialised as nn.Identity
    for i, layer in enumerate(layers_without_dp):
        assert hasattr(layer, "drop_path"), "Layer should have drop_path attribute"
        assert isinstance(layer.drop_path, torch.nn.Identity), (
            f"Layer {i}: expected nn.Identity for zero drop_path, got {type(layer.drop_path)}"
        )


def test_update_drop_path_handles_missing_layers(model_with_drop_path: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify update_drop_path() handles models without recognizable layer structure gracefully."""
    model: LWDETR = model_with_drop_path.model

    monkeypatch.setattr(model, "_get_backbone_encoder_layers", lambda: None)

    # Should not raise an error, just return early
    model.update_drop_path(0.1, 12)
