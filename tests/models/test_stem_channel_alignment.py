# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests of the stem helpers behind ``num_channels``, on a toy module with the DINOv2 attribute path.

``tests/models/test_input_channels.py`` covers the same helpers on real models; these cases are the ones a real model
makes slow or cannot reach: every channel-count combination, a frozen stem and a LoRA-wrapped checkpoint key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rfdetr.models.weights import _widen_stem_for_checkpoint, adapt_input_channels

#: State-dict key of the DINOv2 patch-embedding projection weight.
_PROJ_KEY = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
_ENCODER = "dinov2_windowed_small"


def _config(num_channels: int) -> SimpleNamespace:
    """Return the two ``ModelConfig`` fields ``_widen_stem_for_checkpoint`` reads.

    Examples:
        >>> _config(4).num_channels
        4
    """
    return SimpleNamespace(num_channels=num_channels, encoder=_ENCODER)


def _fake_model(num_channels: int = 3, embed_dim: int = 8, patch: int = 2) -> torch.nn.Module:
    """Build a module exposing only the attribute path ``adapt_input_channels`` walks.

    Args:
        num_channels: Input channels of the patch-embedding projection.
        embed_dim: Output channels of the projection.
        patch: Kernel size and stride of the projection.

    Returns:
        A module whose ``backbone[0].encoder.encoder.embeddings.patch_embeddings.projection`` is a ``Conv2d``.

    Examples:
        >>> _PROJ_KEY in _fake_model().state_dict()
        True
        >>> _stem(_fake_model(4)).in_channels
        4
    """
    patch_embeddings = torch.nn.Module()
    patch_embeddings.num_channels = num_channels
    patch_embeddings.projection = torch.nn.Conv2d(num_channels, embed_dim, kernel_size=patch, stride=patch)
    model = torch.nn.Module()
    outer = model
    for name in ("backbone", "encoder", "encoder", "embeddings"):
        inner = torch.nn.Module()
        setattr(outer, name, torch.nn.ModuleList([inner]) if name == "backbone" else inner)
        outer = inner
    outer.patch_embeddings = patch_embeddings
    return model


def _stem(model: torch.nn.Module) -> torch.nn.Conv2d:
    """Return the patch-embedding projection of a model built by :func:`_fake_model`.

    Examples:
        >>> _stem(_fake_model(3)).weight.shape
        torch.Size([8, 3, 2, 2])
    """
    return model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection


class TestAdaptInputChannels:
    """``adapt_input_channels`` widens a 3-channel projection and leaves a 3-channel model alone."""

    @pytest.mark.parametrize("num_channels", [1, 2, 4, 5])
    def test_adapts_a_three_channel_stem(self, num_channels: int) -> None:
        """The projection and the patch embedding's recorded channel count both follow ``num_channels``."""
        model = _fake_model(3)

        adapt_input_channels(model, num_channels, _ENCODER)

        assert _stem(model).weight.shape[1] == num_channels
        assert _stem(model).in_channels == num_channels
        assert model.backbone[0].encoder.encoder.embeddings.patch_embeddings.num_channels == num_channels

    def test_three_channel_target_is_a_noop_for_any_encoder(self) -> None:
        """A 3-channel model keeps its projection object and weights, and its encoder name is not checked."""
        model = _fake_model(3)
        projection = _stem(model)
        before = projection.weight.detach().clone()

        adapt_input_channels(model, 3, "toy")

        assert _stem(model) is projection
        assert torch.equal(projection.weight.detach(), before)

    def test_keeps_requires_grad_of_a_frozen_stem(self) -> None:
        """A frozen projection stays frozen after it is adapted."""
        model = _fake_model(3)
        _stem(model).weight.requires_grad_(False)

        adapt_input_channels(model, 4, _ENCODER)

        assert not _stem(model).weight.requires_grad

    def test_refused_encoder_leaves_the_stem_untouched(self) -> None:
        """A non-DINOv2 encoder is rejected before the model is changed."""
        model = _fake_model(3)

        with pytest.raises(ValueError, match="num_channels=4 is supported for DINOv2 encoders only, not encoder='toy'"):
            adapt_input_channels(model, 4, "toy")

        assert _stem(model).weight.shape[1] == 3


class TestWidenStemForCheckpoint:
    """The pre-load alignment covers each combination of checkpoint and configured channel count."""

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param(_PROJ_KEY, id="plain"),
            pytest.param(
                "backbone.0.encoder.base_model.model.encoder.embeddings.patch_embeddings.projection.weight",
                id="lora-wrapped",
            ),
        ],
    )
    def test_four_channel_checkpoint_widens_the_stem_before_load(self, key: str) -> None:
        """A 4-channel checkpoint widens a freshly built 3-channel projection, also under PEFT's key prefix."""
        model = _fake_model(3)

        _widen_stem_for_checkpoint(model, {key: torch.zeros(8, 4, 2, 2)}, _config(4))

        assert _stem(model).weight.shape[1] == 4

    @pytest.mark.parametrize(
        "state",
        [
            pytest.param({_PROJ_KEY: torch.zeros(8, 3, 2, 2)}, id="three-channel-stem"),
            pytest.param({"head.weight": torch.zeros(2)}, id="no-stem"),
        ],
    )
    def test_stem_is_left_for_adaptation_after_load(self, state: dict[str, torch.Tensor]) -> None:
        """A 3-channel checkpoint, or one without a stem, loads into the 3-channel projection."""
        model = _fake_model(3)

        _widen_stem_for_checkpoint(model, state, _config(4))

        assert _stem(model).weight.shape[1] == 3

    @pytest.mark.parametrize(("checkpoint_channels", "num_channels"), [(4, 3), (4, 1), (1, 3), (1, 4), (5, 4)])
    def test_mismatched_non_rgb_checkpoint_raises(self, checkpoint_channels: int, num_channels: int) -> None:
        """A non-RGB checkpoint cannot be reconciled with a different configured channel count."""
        model = _fake_model(3)
        state = {_PROJ_KEY: torch.zeros(8, checkpoint_channels, 2, 2)}

        with pytest.raises(ValueError, match=f"saved from a model with num_channels={checkpoint_channels}, but"):
            _widen_stem_for_checkpoint(model, state, _config(num_channels))

        assert _stem(model).weight.shape[1] == 3

    def test_state_dict_loads_after_alignment_and_survives_adaptation(self) -> None:
        """In the builders' call order (align, load, adapt) the loaded weights come out unchanged."""
        model = _fake_model(3)
        state = _fake_model(4).state_dict()

        _widen_stem_for_checkpoint(model, state, _config(4))
        incompatible = model.load_state_dict(state, strict=False)
        adapt_input_channels(model, 4, _ENCODER)

        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        assert torch.equal(_stem(model).weight.detach(), state[_PROJ_KEY])
