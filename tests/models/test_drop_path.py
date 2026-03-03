# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import rfdetr.models.backbone.dinov2 as dinov2_module
from rfdetr.main import Model
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    Dinov2WithRegistersDropPath,
    WindowedDinov2WithRegistersBackbone,
)
from rfdetr.models.lwdetr import LWDETR


@pytest.fixture
def model_with_drop_path(monkeypatch: pytest.MonkeyPatch) -> Model:
    """Create RF-DETR Nano model with drop_path enabled."""
    monkeypatch.setattr(
        WindowedDinov2WithRegistersBackbone,
        "from_pretrained",
        classmethod(lambda cls, name, config: cls(config)),
    )
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
def model_without_drop_path(monkeypatch: pytest.MonkeyPatch) -> Model:
    """Create RF-DETR Nano model without drop_path."""
    monkeypatch.setattr(
        WindowedDinov2WithRegistersBackbone,
        "from_pretrained",
        classmethod(lambda cls, name, config: cls(config)),
    )
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


def test_update_drop_path_partial_layers(model_with_drop_path: Model) -> None:
    """Verify min() guard prevents IndexError when vit_encoder_num_layers > len(layers)."""
    model: LWDETR = model_with_drop_path.model

    layers = model._get_backbone_encoder_layers()
    assert layers is not None
    actual_num_layers = len(layers)

    # Request more layers than exist in the backbone
    requested_num_layers = actual_num_layers + 4
    drop_path_rate = 0.2

    # Should not raise IndexError
    model.update_drop_path(drop_path_rate, requested_num_layers)

    # Each updated layer gets a rate from 0 to drop_path_rate (shorter, capped linspace)
    expected_rates = [x.item() for x in torch.linspace(0, drop_path_rate, actual_num_layers)]
    for i in range(actual_num_layers):
        actual_prob = layers[i].drop_path.drop_prob
        assert abs(actual_prob - expected_rates[i]) < 1e-6, (
            f"Layer {i} drop_prob should be {expected_rates[i]}, got {actual_prob}"
        )


def test_non_windowed_drop_path_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a warning is emitted when drop_path_rate > 0 with non-windowed backbone."""
    mock_backbone = MagicMock()
    monkeypatch.setattr(dinov2_module, "AutoBackbone", MagicMock(from_pretrained=MagicMock(return_value=mock_backbone)))

    # The rf-detr logger sets propagate=False, so intercept warning() directly.
    warning_messages: list[str] = []
    rf_detr_logger = logging.getLogger("rf-detr")
    monkeypatch.setattr(rf_detr_logger, "warning", lambda msg, *args, **kwargs: warning_messages.append(msg))

    DinoV2(size="base", use_windowed_attn=False, drop_path_rate=0.1)

    assert any("drop_path_rate" in msg and "ignored" in msg for msg in warning_messages), (
        "Expected warning about drop_path_rate being ignored for non-windowed backbone"
    )


def test_get_backbone_encoder_layers_blocks_path() -> None:
    """Verify _get_backbone_encoder_layers() returns enc.blocks for standard ViT backbones."""
    mock_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
    # SimpleNamespace gives only the attributes we define, so hasattr checks work correctly.
    mock_encoder = SimpleNamespace(blocks=mock_blocks)
    mock_self = SimpleNamespace(backbone=[SimpleNamespace(encoder=mock_encoder)])

    result = LWDETR._get_backbone_encoder_layers(mock_self)  # type: ignore[arg-type]
    assert result is mock_blocks, "Should return enc.blocks for standard ViT backbone"


def test_get_backbone_encoder_layers_trunk_blocks_path() -> None:
    """Verify _get_backbone_encoder_layers() returns enc.trunk.blocks for aimv2 backbones."""
    mock_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
    mock_trunk = SimpleNamespace(blocks=mock_blocks)
    mock_encoder = SimpleNamespace(trunk=mock_trunk)  # no 'blocks' at top level
    mock_self = SimpleNamespace(backbone=[SimpleNamespace(encoder=mock_encoder)])

    result = LWDETR._get_backbone_encoder_layers(mock_self)  # type: ignore[arg-type]
    assert result is mock_blocks, "Should return enc.trunk.blocks for aimv2 backbone"
