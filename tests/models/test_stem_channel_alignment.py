# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The patch-embedding projection follows ``num_channels`` when a model is built and when a checkpoint is loaded.

``adapt_input_channels`` derives a ``num_channels`` projection from a 3-channel one and runs after
``load_pretrain_weights``. A checkpoint saved from a model with ``num_channels != 3`` already holds a projection of that
width, so ``_align_stem_to_checkpoint`` widens the model before the state dict is loaded and the later
``adapt_input_channels`` call must leave the loaded weights alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.inference import _build_model_context
from rfdetr.models.weights import _align_stem_to_checkpoint, _checkpoint_stem_channels, adapt_input_channels
from rfdetr.training.module_model import RFDETRModelModule

#: State-dict key of the DINOv2 patch-embedding projection weight.
_PROJ_KEY = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"


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
    """Return the patch-embedding projection of a model built by :func:`_fake_model` or by ``rfdetr``.

    Examples:
        >>> _stem(_fake_model(3)).weight.shape
        torch.Size([8, 3, 2, 2])
    """
    return model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection


def _save_four_channel_checkpoint(path: Path) -> torch.Tensor:
    """Save the state dict of an untrained 4-channel ``RFDETRNano`` and return its projection weight.

    The projection is overwritten with random values first: an untrained 4-channel projection is the tiled 3-channel
    initialization, which a reload that re-derives the projection from 3 channels could reproduce by accident.

    Args:
        path: File the ``{"model": state_dict}`` checkpoint is written to.

    Returns:
        The projection weight stored in the checkpoint.

    Examples:
        >>> import tempfile
        >>> _save_four_channel_checkpoint(Path(tempfile.mkdtemp()) / "four_channel.pth").shape[1]
        4
    """
    model = _build_model_context(RFDETRNanoConfig(pretrain_weights=None, num_channels=4, device="cpu")).model
    with torch.no_grad():
        _stem(model).weight.copy_(torch.randn_like(_stem(model).weight))
    state = model.state_dict()
    torch.save({"model": state}, path)
    return state[_PROJ_KEY].clone()


class TestCheckpointStemChannels:
    """The checkpoint's input channel count is read off its state dict."""

    def test_reads_channel_count_from_projection_weight(self) -> None:
        """The second dimension of the projection weight is the channel count."""
        assert _checkpoint_stem_channels({_PROJ_KEY: torch.zeros(8, 4, 2, 2)}) == 4

    def test_reads_channel_count_from_lora_wrapped_key(self) -> None:
        """A LoRA checkpoint nests the projection under PEFT's ``base_model.model`` prefix."""
        key = "backbone.0.encoder.base_model.model.encoder.embeddings.patch_embeddings.projection.weight"
        assert _checkpoint_stem_channels({key: torch.zeros(8, 4, 2, 2)}) == 4

    def test_returns_none_when_no_projection_present(self) -> None:
        """A state dict without a patch-embedding projection has no channel count."""
        assert _checkpoint_stem_channels({"some.other.weight": torch.zeros(3)}) is None

    def test_ignores_non_4d_tensors_on_a_matching_key(self) -> None:
        """Only a convolution weight carries the channel count."""
        assert _checkpoint_stem_channels({_PROJ_KEY: torch.zeros(8)}) is None


class TestAdaptInputChannels:
    """``adapt_input_channels`` widens a 3-channel projection once and rejects non-DINOv2 encoders."""

    @pytest.mark.parametrize("num_channels", [1, 2, 4, 5])
    def test_adapts_a_three_channel_stem(self, num_channels: int) -> None:
        """The projection and the patch embedding's recorded channel count both follow ``num_channels``."""
        model = _fake_model(3)

        adapt_input_channels(model, num_channels, encoder_name="dinov2_windowed_small")

        assert _stem(model).weight.shape[1] == num_channels
        assert _stem(model).in_channels == num_channels
        assert model.backbone[0].encoder.encoder.embeddings.patch_embeddings.num_channels == num_channels

    def test_second_call_leaves_weights_untouched(self) -> None:
        """Adapting again would rescale the weights by ``3 / num_channels``."""
        model = _fake_model(3)
        adapt_input_channels(model, 4)
        after_first = _stem(model).weight.detach().clone()

        adapt_input_channels(model, 4)

        assert torch.equal(_stem(model).weight.detach(), after_first)

    def test_three_channel_target_is_a_noop(self) -> None:
        """A 3-channel model keeps its projection object and weights."""
        model = _fake_model(3)
        projection = _stem(model)
        before = projection.weight.detach().clone()

        adapt_input_channels(model, 3)

        assert _stem(model) is projection
        assert torch.equal(projection.weight.detach(), before)

    def test_keeps_requires_grad_of_a_frozen_stem(self) -> None:
        """A frozen projection stays frozen after it is adapted."""
        model = _fake_model(3)
        _stem(model).weight.requires_grad_(False)

        adapt_input_channels(model, 4)

        assert not _stem(model).weight.requires_grad

    def test_non_dinov2_encoder_name_raises(self) -> None:
        """A named non-DINOv2 encoder is rejected before the model is touched."""
        model = _fake_model(3)

        with pytest.raises(ValueError, match="num_channels=4 is supported for DINOv2 encoders only, not encoder='toy'"):
            adapt_input_channels(model, 4, encoder_name="toy")

        assert _stem(model).weight.shape[1] == 3

    def test_encoder_without_patch_embedding_raises(self) -> None:
        """Without an encoder name, an encoder that has no DINOv2 patch embedding is rejected."""
        model = torch.nn.Module()
        outer = torch.nn.Module()
        outer.encoder = torch.nn.Linear(2, 2)
        model.backbone = torch.nn.ModuleList([outer])

        with pytest.raises(ValueError, match="supported for DINOv2 encoders only, not encoder='Linear'"):
            adapt_input_channels(model, 4)

    def test_three_channels_skips_the_encoder_check(self) -> None:
        """A 3-channel model of any encoder passes through."""
        adapt_input_channels(_fake_model(3), 3, encoder_name="toy")


class TestAlignStemToCheckpoint:
    """The pre-load alignment covers each combination of checkpoint and configured channel count."""

    def test_four_channel_checkpoint_widens_the_stem_before_load(self) -> None:
        """A 4-channel checkpoint widens a freshly built 3-channel projection."""
        model = _fake_model(3)

        _align_stem_to_checkpoint(model, {_PROJ_KEY: torch.zeros(8, 4, 2, 2)}, num_channels=4)

        assert _stem(model).weight.shape[1] == 4

    def test_three_channel_checkpoint_leaves_the_stem_alone(self) -> None:
        """A 3-channel checkpoint loads into the 3-channel projection, which is adapted afterwards."""
        model = _fake_model(3)

        _align_stem_to_checkpoint(model, {_PROJ_KEY: torch.zeros(8, 3, 2, 2)}, num_channels=4)

        assert _stem(model).weight.shape[1] == 3

    def test_checkpoint_without_a_stem_is_ignored(self) -> None:
        """A checkpoint without a projection changes nothing."""
        model = _fake_model(3)

        _align_stem_to_checkpoint(model, {"head.weight": torch.zeros(2)}, num_channels=4)

        assert _stem(model).weight.shape[1] == 3

    @pytest.mark.parametrize(("checkpoint_channels", "num_channels"), [(4, 3), (4, 1), (1, 3), (1, 4), (5, 4)])
    def test_mismatched_non_rgb_checkpoint_raises(self, checkpoint_channels: int, num_channels: int) -> None:
        """A non-RGB checkpoint cannot be reconciled with a different configured channel count."""
        model = _fake_model(3)
        state = {_PROJ_KEY: torch.zeros(8, checkpoint_channels, 2, 2)}

        with pytest.raises(ValueError, match=f"saved from a model with num_channels={checkpoint_channels}, but"):
            _align_stem_to_checkpoint(model, state, num_channels=num_channels)

        assert _stem(model).weight.shape[1] == 3

    def test_state_dict_loads_after_alignment_and_survives_adaptation(self) -> None:
        """The call order of the builders: align, load, adapt.

        The loaded weights come out unchanged.
        """
        model = _fake_model(3)
        state = _fake_model(4).state_dict()

        _align_stem_to_checkpoint(model, state, num_channels=4)
        incompatible = model.load_state_dict(state, strict=False)
        adapt_input_channels(model, 4)

        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        assert torch.equal(_stem(model).weight.detach(), state[_PROJ_KEY])


class TestFourChannelCheckpointRoundTrip:
    """A 4-channel checkpoint loads back into the inference model and into the training module."""

    def test_inference_model_reloads_stem_exactly(self, tmp_path: Path) -> None:
        """The inference builder loads the checkpoint's projection instead of re-deriving it from 3 channels."""
        checkpoint_path = tmp_path / "four_channel.pth"
        saved_stem = _save_four_channel_checkpoint(checkpoint_path)

        config = RFDETRNanoConfig(pretrain_weights=str(checkpoint_path), num_channels=4, device="cpu")
        reloaded = _build_model_context(config).model

        assert torch.equal(reloaded.state_dict()[_PROJ_KEY], saved_stem)

    def test_training_module_reloads_stem_exactly(self, tmp_path: Path) -> None:
        """The training module loads the checkpoint's projection, so fine-tuning starts from the saved weights."""
        checkpoint_path = tmp_path / "four_channel.pth"
        saved_stem = _save_four_channel_checkpoint(checkpoint_path)

        model_config = RFDETRNanoConfig(pretrain_weights=str(checkpoint_path), num_channels=4, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)
        module = RFDETRModelModule(model_config, train_config)

        assert torch.equal(module.model.state_dict()[_PROJ_KEY], saved_stem)

    @pytest.mark.parametrize("num_channels", [3, 1])
    def test_four_channel_checkpoint_into_other_channel_count_raises(self, tmp_path: Path, num_channels: int) -> None:
        """Loading a 4-channel checkpoint into a model of another channel count names both counts."""
        checkpoint_path = tmp_path / "four_channel.pth"
        _save_four_channel_checkpoint(checkpoint_path)
        config = RFDETRNanoConfig(pretrain_weights=str(checkpoint_path), num_channels=num_channels, device="cpu")

        expected = f"num_channels=4, but this model is configured with num_channels={num_channels}"
        with pytest.raises(ValueError, match=expected):
            _build_model_context(config)


class TestTrainingModuleInputChannels:
    """``RFDETRModelModule`` builds its model with the configured input channel count."""

    @pytest.mark.parametrize("num_channels", [1, 3, 4])
    def test_module_stem_matches_num_channels(self, tmp_path: Path, num_channels: int) -> None:
        """The training model's patch embedding takes ``num_channels`` input channels and runs a forward pass."""
        model_config = RFDETRNanoConfig(pretrain_weights=None, num_channels=num_channels, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)

        module = RFDETRModelModule(model_config, train_config)

        assert _stem(module.model).in_channels == num_channels
        resolution = model_config.resolution
        with torch.no_grad():
            outputs = module.model.eval()(torch.rand(1, num_channels, resolution, resolution))
        assert outputs["pred_boxes"].shape[0] == 1
