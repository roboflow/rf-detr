# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the encoder registry that lets extension packages plug a non-DINOv2 encoder into ``build_backbone``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from rfdetr.models import backbone as backbone_pkg
from rfdetr.models.backbone import Joiner, build_backbone, register_backbone
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.utilities.tensors import NestedTensor


class _ToyEncoder(nn.Module):
    """Stand-in encoder: one conv whose output is repeated once per requested feature level."""

    def __init__(self, num_levels: int, channels: int = 8, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, channels, kernel_size=patch_size, stride=patch_size)
        self.num_levels = num_levels
        self._out_feature_channels = [channels] * num_levels

    def forward(self, x: Tensor) -> list[Tensor]:
        feat = self.proj(x)
        return [feat] * self.num_levels


class _ToyBackbone(Backbone):
    """Backbone subclass that swaps in ``_ToyEncoder`` and records the kwargs it was built with."""

    received: dict[str, Any] = {}

    def _build_encoder(self, name: str, **kwargs: Any) -> nn.Module:
        type(self).received = {"name": name, **kwargs}
        return _ToyEncoder(num_levels=len(kwargs["out_feature_indexes"]), patch_size=kwargs["patch_size"])


def _build(encoder: str, **overrides: Any) -> Joiner:
    kwargs: dict[str, Any] = dict(
        encoder=encoder,
        vit_encoder_num_layers=12,
        pretrained_encoder=None,
        window_block_indexes=None,
        drop_path=0.0,
        out_channels=16,
        out_feature_indexes=[2, 5, 8, 11],
        projector_scale=["P4"],
        use_cls_token=False,
        hidden_dim=16,
        position_embedding="sine",
        freeze_encoder=False,
        layer_norm=True,
        target_shape=(64, 64),
        rms_norm=False,
        backbone_lora=False,
        force_no_pretrain=False,
        gradient_checkpointing=False,
        load_dinov2_weights=False,
        patch_size=4,
        num_windows=2,
        positional_encoding_size=16,
    )
    kwargs.update(overrides)
    return build_backbone(**kwargs)


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Restore the module-level registry after every test so registrations never leak."""
    saved = dict(backbone_pkg._BACKBONE_REGISTRY)
    try:
        yield
    finally:
        backbone_pkg._BACKBONE_REGISTRY.clear()
        backbone_pkg._BACKBONE_REGISTRY.update(saved)


class TestRegisterBackbone:
    """``register_backbone`` validation and idempotence."""

    def test_registered_class_builds_the_backbone(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        joiner = _build("toy_encoder")

        assert isinstance(joiner, Joiner)
        assert type(joiner[0]) is _ToyBackbone
        assert isinstance(joiner[0].encoder, _ToyEncoder)

    def test_encoder_receives_the_build_arguments(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        _build("toy_encoder", window_block_indexes=[0, 1], drop_path=0.1, gradient_checkpointing=True)

        assert _ToyBackbone.received == {
            "name": "toy_encoder",
            "out_feature_indexes": [2, 5, 8, 11],
            "target_shape": (64, 64),
            "gradient_checkpointing": True,
            "load_pretrained_weights": False,
            "patch_size": 4,
            "num_windows": 2,
            "positional_encoding_size": 16,
            "drop_path": 0.1,
            "window_block_indexes": [0, 1],
        }

    def test_registered_backbone_keeps_core_projector_and_masks(self) -> None:
        """The subclass only replaces the encoder; the projector and padding-mask plumbing stay core's."""
        register_backbone("toy_encoder", _ToyBackbone)
        joiner = _build("toy_encoder")
        images = torch.randn(2, 3, 64, 64)
        mask = torch.zeros(2, 64, 64, dtype=torch.bool)
        mask[1, :, 32:] = True

        features, poss, cross = joiner(NestedTensor(images, mask))

        assert cross is None
        assert [tuple(f.tensors.shape) for f in features] == [(2, 16, 16, 16)]
        assert features[0].mask[1, :, 8:].all() and not features[0].mask[0].any()
        assert tuple(poss[0].shape) == (2, 16, 16, 16)

    def test_re_registering_the_same_class_is_a_no_op(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)
        register_backbone("toy_encoder", _ToyBackbone)

        assert backbone_pkg._BACKBONE_REGISTRY["toy_encoder"] is _ToyBackbone

    def test_conflicting_registration_raises(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        class _OtherBackbone(_ToyBackbone):
            pass

        with pytest.raises(ValueError, match="already registered"):
            register_backbone("toy_encoder", _OtherBackbone)

    @pytest.mark.parametrize(
        "backbone_cls",
        [
            pytest.param(nn.Module, id="not-a-backbone-subclass"),
            pytest.param(_ToyEncoder(num_levels=1), id="instance-not-class"),
        ],
    )
    def test_non_backbone_class_raises(self, backbone_cls: Any) -> None:
        with pytest.raises(TypeError, match="Backbone subclass"):
            register_backbone("toy_encoder", backbone_cls)

    @pytest.mark.parametrize("encoder", ["dinov2_windowed_small", "dinov2_windowed_base", "dinov2_base"])
    def test_dinov2_encoder_names_cannot_be_registered(self, encoder: str) -> None:
        """Every name Backbone would parse as DINOv2 stays DINOv2, including ones outside EncoderName."""
        with pytest.raises(ValueError, match="DINOv2"):
            register_backbone(encoder, _ToyBackbone)


class TestUnregisteredEncoders:
    """Encoders without a registration keep the DINOv2 path unchanged."""

    def test_dinov2_name_builds_core_backbone(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        joiner = _build(
            "dinov2_windowed_small", out_feature_indexes=[12], projector_scale=["P3"], patch_size=16, num_windows=1
        )

        assert type(joiner[0]) is Backbone
        assert isinstance(joiner[0].encoder, DinoV2)

    def test_unknown_non_dinov2_name_names_the_registered_encoders(self) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        with pytest.raises(ValueError, match=r"Unknown encoder 'not_a_registered_encoder'.*registered: toy_encoder"):
            _build("not_a_registered_encoder")

    def test_dinov2_encoder_receives_the_backbone_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pins Backbone -> DinoV2 argument forwarding, which moved into Backbone._build_encoder."""
        from unittest.mock import MagicMock

        import rfdetr.models.backbone.backbone as backbone_module

        encoder = MagicMock(_out_feature_channels=[384])
        dinov2 = MagicMock(return_value=encoder)
        monkeypatch.setattr(backbone_module, "DinoV2", dinov2)

        _build(
            "dinov2_registers_windowed_small",
            out_feature_indexes=[12],
            window_block_indexes=[0, 1],
            drop_path=0.2,
            gradient_checkpointing=True,
            load_dinov2_weights=True,
            patch_size=14,
            num_windows=4,
            positional_encoding_size=37,
            target_shape=(518, 518),
        )

        dinov2.assert_called_once_with(
            size="small",
            out_feature_indexes=[12],
            shape=(518, 518),
            use_registers=True,
            use_windowed_attn=True,
            gradient_checkpointing=True,
            load_dinov2_weights=True,
            patch_size=14,
            num_windows=4,
            positional_encoding_size=37,
            drop_path_rate=0.2,
            window_block_indexes=[0, 1],
        )


class TestForceNoPretrain:
    """``force_no_pretrain`` stops every encoder, registered or DINOv2, from fetching upstream weights."""

    @pytest.mark.parametrize(
        ("force", "expected"), [pytest.param(False, True, id="default"), pytest.param(True, False, id="forced")]
    )
    def test_force_no_pretrain_overrides_load_dinov2_weights(self, force: bool, expected: bool) -> None:
        register_backbone("toy_encoder", _ToyBackbone)

        _build("toy_encoder", load_dinov2_weights=True, force_no_pretrain=force)

        assert _ToyBackbone.received["load_pretrained_weights"] is expected
