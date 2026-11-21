# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests that ``num_channels`` reaches the stem of the prediction model, the training model and checkpoint loads."""

from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.inference import _build_model_context
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.weights import adapt_input_channels
from rfdetr.training.module_model import RFDETRModelModule


def _stem(model: LWDETR) -> torch.nn.Conv2d:
    """Return the patch-embedding convolution of a DINOv2-backed model.

    Examples:
        >>> config = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
        >>> _stem(_build_model_context(config).model).in_channels
        3
    """
    return model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection


def _save_checkpoint(tmp_path: Path, num_channels: int) -> tuple[str, torch.Tensor]:
    """Save a checkpoint of a Nano model built with *num_channels* and return its path and stem weight.

    Examples:
        >>> import tempfile
        >>> path, stem = _save_checkpoint(Path(tempfile.mkdtemp()), 4)
        >>> stem.shape[1]
        4
    """
    config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=None, device="cpu")
    model = _build_model_context(config).model
    with torch.no_grad():
        _stem(model).weight.normal_()
    path = tmp_path / f"checkpoint_{num_channels}ch.pth"
    torch.save({"model": model.state_dict()}, path)
    return str(path), _stem(model).weight.detach().clone()


class TestAdaptInputChannels:
    """``adapt_input_channels`` rebuilds the stem once, and only for DINOv2 encoders."""

    def test_second_call_leaves_weights_unchanged(self) -> None:
        """A stem that already has the requested width is not rescaled again."""
        model = _build_model_context(RFDETRNanoConfig(num_channels=4, pretrain_weights=None, device="cpu")).model
        widened = _stem(model).weight.detach().clone()

        adapt_input_channels(model, 4, "dinov2_windowed_small")

        torch.testing.assert_close(_stem(model).weight, widened)

    def test_non_dinov2_encoder_is_refused(self) -> None:
        """Another encoder family raises instead of being left with a 3-channel stem."""
        model = _build_model_context(RFDETRNanoConfig(pretrain_weights=None, device="cpu")).model

        with pytest.raises(ValueError, match="DINOv2 encoders only"):
            adapt_input_channels(model, 4, "pe_core_tiny")

    @pytest.mark.parametrize(("freeze_encoder", "warnings"), [(True, 1), (False, 0)])
    def test_frozen_stem_is_reported(self, freeze_encoder: bool, warnings: int, mocker: MockerFixture) -> None:
        """A frozen patch embedding cannot learn the new channels, so widening it warns."""
        warning = mocker.patch("rfdetr.models.weights.logger.warning")

        _build_model_context(
            RFDETRNanoConfig(num_channels=4, freeze_encoder=freeze_encoder, pretrain_weights=None, device="cpu")
        )

        frozen = [call for call in warning.call_args_list if "patch embedding is frozen" in call.args[0]]
        assert len(frozen) == warnings


class TestTrainingModelChannels:
    """The model that ``train()`` optimizes takes ``num_channels`` inputs, like the prediction model."""

    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_training_module_stem_matches_config(self, num_channels: int, tmp_path: Path) -> None:
        """``RFDETRModelModule`` widens its own model, which is built separately from the prediction one."""
        model_config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=None, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)

        module = RFDETRModelModule(model_config, train_config)

        assert _stem(module.model).in_channels == num_channels


class TestCheckpointStemAlignment:
    """A checkpoint saved from a widened model loads back, and a mismatched one is refused."""

    def test_widened_checkpoint_loads_its_own_stem(self, tmp_path: Path) -> None:
        """Fine-tuning from a 4-channel run keeps the trained filters of the fourth channel."""
        path, saved_stem = _save_checkpoint(tmp_path, 4)
        config = RFDETRNanoConfig(num_channels=4, pretrain_weights=path, device="cpu")

        model = _build_model_context(config).model

        torch.testing.assert_close(_stem(model).weight, saved_stem)

    def test_training_module_loads_widened_checkpoint(self, tmp_path: Path) -> None:
        """The training build loads the same checkpoint through the same alignment."""
        path, saved_stem = _save_checkpoint(tmp_path, 4)
        model_config = RFDETRNanoConfig(num_channels=4, pretrain_weights=path, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)

        module = RFDETRModelModule(model_config, train_config)

        torch.testing.assert_close(_stem(module.model).weight, saved_stem)

    @pytest.mark.parametrize("num_channels", [1, 3])
    def test_checkpoint_of_another_width_is_refused(self, num_channels: int, tmp_path: Path) -> None:
        """Only a 3-channel checkpoint can be adapted, so a 4-channel one needs ``num_channels=4``."""
        path, _ = _save_checkpoint(tmp_path, 4)
        config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=path, device="cpu")

        with pytest.raises(ValueError, match="pass num_channels=4 to load this checkpoint"):
            _build_model_context(config)
