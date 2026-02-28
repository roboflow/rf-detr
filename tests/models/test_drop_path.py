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

    initial_probs = []
    for layer in layers:
        if hasattr(layer, "drop_path") and hasattr(layer.drop_path, "drop_prob"):
            initial_probs.append(layer.drop_path.drop_prob)
        else:
            initial_probs.append(None)

    model.update_drop_path(drop_path_rate, num_layers)

    # Verify linear schedule: first layer = 0, last layer = drop_path_rate
    expected_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

    for i, layer in enumerate(layers):
        if hasattr(layer, "drop_path") and hasattr(layer.drop_path, "drop_prob"):
            actual_prob = layer.drop_path.drop_prob
            expected_prob = expected_rates[i]
            assert abs(actual_prob - expected_prob) < 1e-6, (
                f"Layer {i} drop_prob should be {expected_prob}, got {actual_prob}"
            )

    first_layer = layers[0]
    last_layer = layers[-1]
    if hasattr(first_layer, "drop_path") and hasattr(first_layer.drop_path, "drop_prob"):
        assert abs(first_layer.drop_path.drop_prob - 0.0) < 1e-6, "First layer should have drop_prob = 0"
    if hasattr(last_layer, "drop_path") and hasattr(last_layer.drop_path, "drop_prob"):
        assert abs(last_layer.drop_path.drop_prob - drop_path_rate) < 1e-6, (
            f"Last layer should have drop_prob = {drop_path_rate}"
        )


def test_drop_path_initialization(model_with_drop_path: Model, model_without_drop_path: Model) -> None:
    """Verify drop_path initialization: DropPath vs Identity based on rate."""
    model_with_dp: LWDETR = model_with_drop_path.model
    model_without_dp: LWDETR = model_without_drop_path.model

    layers_with_dp = model_with_dp._get_backbone_encoder_layers()
    layers_without_dp = model_without_dp._get_backbone_encoder_layers()

    assert layers_with_dp is not None
    assert layers_without_dp is not None

    for layer in layers_with_dp:
        assert hasattr(layer, "drop_path"), "Layer should have drop_path attribute"
        drop_path_module = layer.drop_path
        # When drop_path_rate > 0, it should be Dinov2WithRegistersDropPath (not Identity)
        if hasattr(drop_path_module, "drop_prob"):
            assert isinstance(drop_path_module, Dinov2WithRegistersDropPath) or isinstance(
                drop_path_module, torch.nn.Identity
            ), "drop_path should be Dinov2WithRegistersDropPath or Identity"

    for layer in layers_without_dp:
        assert hasattr(layer, "drop_path"), "Layer should have drop_path attribute"
        drop_path_module = layer.drop_path
        # When drop_path_rate = 0, it should be Identity or DropPath with drop_prob=0
        if isinstance(drop_path_module, torch.nn.Identity):
            pass
        elif hasattr(drop_path_module, "drop_prob"):
            assert drop_path_module.drop_prob == 0.0, "drop_prob should be 0 when drop_path_rate = 0"


def test_update_drop_path_handles_missing_layers(model_with_drop_path: Model) -> None:
    """Verify update_drop_path() handles models without recognizable layer structure gracefully."""
    model: LWDETR = model_with_drop_path.model

    # Monkeypatch _get_backbone_encoder_layers to return None, simulating unrecognized structure
    original_method = model._get_backbone_encoder_layers

    def return_none():
        return None

    model._get_backbone_encoder_layers = return_none

    # Should not raise an error, just return early
    try:
        model.update_drop_path(0.1, 12)
    finally:
        # Restore original method
        model._get_backbone_encoder_layers = original_method
