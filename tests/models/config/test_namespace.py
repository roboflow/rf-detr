# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for _namespace_from_configs() config forwarding."""

import dataclasses
from typing import Any

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, RFDETRSegNanoConfig, RFDETRSmallConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.models import build_model
from rfdetr.models._defaults import MODEL_DEFAULTS
from rfdetr.models._types import BuilderArgs


def _make_ns(model_config: Any = None, train_config: Any = None, **train_kwargs: Any) -> Any:
    """Build the builder namespace from a Nano model config and a minimal ``TrainConfig`` unless overridden.

    Args:
        model_config: Model config to use instead of ``RFDETRNanoConfig(num_classes=80)``.
        train_config: Train config to use instead of ``TrainConfig(dataset_dir="/tmp", **train_kwargs)``.
        **train_kwargs: ``TrainConfig`` fields for the default train config.

    Examples:
        >>> _make_ns(use_ema=False).use_ema
        False
    """
    model_config = model_config or RFDETRNanoConfig(num_classes=80)
    train_config = train_config or TrainConfig(dataset_dir="/tmp", **train_kwargs)
    return _namespace_from_configs(model_config, train_config)


class TestNamespaceForwarding:
    """_namespace_from_configs() forwards TrainConfig fields that were previously hardcoded to wrong defaults."""

    def test_aug_config_forwarded(self) -> None:
        """A set ``aug_config`` reaches the namespace unchanged."""
        aug = {"hsv_h": 0.015, "hsv_s": 0.7}
        assert _make_ns(aug_config=aug).aug_config == aug

    def test_aug_config_defaults_to_none(self) -> None:
        """Without an explicit ``aug_config`` the namespace carries None."""
        assert _make_ns().aug_config is None

    @pytest.mark.parametrize("value", [True, False])
    def test_use_ema_forwarded(self, value: bool) -> None:
        """``TrainConfig.use_ema`` reaches the namespace for both values."""
        assert _make_ns(use_ema=value).use_ema is value

    @pytest.mark.parametrize("value", [True, False])
    def test_early_stopping_use_ema_forwarded(self, value: bool) -> None:
        """``TrainConfig.early_stopping_use_ema`` reaches the namespace for both values."""
        assert _make_ns(early_stopping_use_ema=value).early_stopping_use_ema is value


class TestNamespaceProtocol:
    """_namespace_from_configs() output must satisfy the BuilderArgs Protocol."""

    def test_namespace_satisfies_builderargs_protocol(self) -> None:
        """The namespace passes ``isinstance(..., BuilderArgs)`` on every supported Python version.

        On Python 3.10/3.11 this is a structural no-op (no method members to check); on 3.12+ it verifies data-attribute
        presence.
        """
        assert isinstance(_make_ns(), BuilderArgs)


class TestNamespaceFieldOwnership:
    """The namespace reads each field from the authoritative owner."""

    def test_cls_loss_coef_from_train_config(self) -> None:
        """ns.cls_loss_coef reflects TrainConfig.cls_loss_coef, not ModelConfig."""
        ns = _make_ns(cls_loss_coef=2.5)
        assert ns.cls_loss_coef == pytest.approx(2.5)

    def test_cls_loss_coef_segmentation_default_matches_pre_1_7_effective_value(self) -> None:
        """SegmentationTrainConfig default preserves the pre-1.7 effective loss_ce weight."""
        ns = _make_ns(RFDETRSegNanoConfig(), SegmentationTrainConfig(dataset_dir="/tmp"))
        assert ns.cls_loss_coef == pytest.approx(1.0)

    def test_cls_loss_coef_segmentation_explicit_value_wins(self) -> None:
        """An explicit SegmentationTrainConfig.cls_loss_coef propagates to the namespace."""
        ns = _make_ns(RFDETRSegNanoConfig(), SegmentationTrainConfig(dataset_dir="/tmp", cls_loss_coef=5.0))
        assert ns.cls_loss_coef == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ("config_class", "expected_num_select"),
        [
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
            pytest.param(RFDETRSmallConfig, 300, id="small"),
        ],
    )
    def test_num_select_from_model_config(self, config_class: type, expected_num_select: int) -> None:
        """ns.num_select equals the model config's num_select, regardless of TrainConfig's own default (300)."""
        ns = _make_ns(config_class())
        assert ns.num_select == expected_num_select


class TestDimFeedforward:
    """``ModelConfig.dim_feedforward`` reaches the builder namespace and the decoder layers it sizes."""

    def test_default_matches_legacy_hardcoded_width(self) -> None:
        """Without an override the namespace keeps the legacy width of 2048."""
        assert _make_ns(RFDETRNanoConfig()).dim_feedforward == 2048

    def test_override_is_forwarded(self) -> None:
        """A configured width reaches the namespace."""
        assert _make_ns(RFDETRNanoConfig(dim_feedforward=1024)).dim_feedforward == 1024

    def test_model_config_wins_over_defaults(self) -> None:
        """A ``ModelConfig`` width overrides ``ModelDefaults.dim_feedforward``, which is only a fallback shadow."""
        defaults = dataclasses.replace(MODEL_DEFAULTS, dim_feedforward=512)

        ns = _namespace_from_configs(
            RFDETRNanoConfig(dim_feedforward=1024), TrainConfig(dataset_dir="/tmp"), defaults=defaults
        )

        assert ns.dim_feedforward == 1024

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_width_is_rejected(self, value: int) -> None:
        """Zero and negative widths fail config validation, naming the field."""
        with pytest.raises(ValueError) as exc_info:
            RFDETRNanoConfig(dim_feedforward=value)
        assert "dim_feedforward" in str(exc_info.value)

    def test_decoder_layers_use_configured_width(self) -> None:
        """Every decoder layer of the built model uses the configured feed-forward width."""
        ns = _make_ns(RFDETRNanoConfig(dim_feedforward=1024, pretrain_weights=None))

        model = build_model(ns)

        assert {layer.linear1.out_features for layer in model.transformer.decoder.layers} == {1024}
