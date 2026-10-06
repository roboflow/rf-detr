# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for _namespace_from_configs() config forwarding."""

import dataclasses
import sys
from typing import Any

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, RFDETRSegNanoConfig, RFDETRSmallConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.models._defaults import MODEL_DEFAULTS
from rfdetr.models._types import BuilderArgs


class TestNamespaceForwarding:
    """Verify that _namespace_from_configs() forwards TrainConfig fields that were previously hardcoded to wrong
    defaults."""

    def _make_ns(self: "TestNamespaceForwarding", **tc_kwargs: Any) -> Any:
        """Build a namespace for tests with minimal default TrainConfig values."""
        mc = RFDETRNanoConfig(num_classes=80)
        tc_kwargs.setdefault("dataset_dir", "/tmp")
        tc = TrainConfig(**tc_kwargs)
        return _namespace_from_configs(mc, tc)

    def test_aug_config_forwarded_when_set(self: "TestNamespaceForwarding") -> None:
        aug = {"hsv_h": 0.015, "hsv_s": 0.7}
        ns = self._make_ns(aug_config=aug)
        assert ns.aug_config == aug

    def test_aug_config_none_by_default(self: "TestNamespaceForwarding") -> None:
        ns = self._make_ns()
        assert ns.aug_config is None

    def test_use_ema_forwarded_true(self: "TestNamespaceForwarding") -> None:
        ns = self._make_ns(use_ema=True)
        assert ns.use_ema is True

    def test_use_ema_forwarded_false(self: "TestNamespaceForwarding") -> None:
        ns = self._make_ns(use_ema=False)
        assert ns.use_ema is False

    def test_early_stopping_use_ema_forwarded_true(self: "TestNamespaceForwarding") -> None:
        ns = self._make_ns(early_stopping_use_ema=True)
        assert ns.early_stopping_use_ema is True

    def test_early_stopping_use_ema_forwarded_false(self: "TestNamespaceForwarding") -> None:
        ns = self._make_ns(early_stopping_use_ema=False)
        assert ns.early_stopping_use_ema is False


class TestNamespaceProtocol:
    """_namespace_from_configs() output must satisfy the BuilderArgs Protocol."""

    def _make_ns(self, mc=None, tc=None):
        mc = mc or RFDETRNanoConfig(num_classes=80)
        tc = tc or TrainConfig(dataset_dir="/tmp")
        return _namespace_from_configs(mc, tc)

    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="Runtime Protocol attribute checks require Python 3.12+",
    )
    def test_namespace_satisfies_builderargs_protocol_py312(self) -> None:
        """On Python 3.12+, isinstance() verifies data-attribute presence."""
        ns = self._make_ns()
        assert isinstance(ns, BuilderArgs)

    def test_namespace_is_builderargs_instance(self) -> None:
        """Isinstance() check passes on all supported Python versions.

        On Python 3.10/3.11 this is a structural no-op (no method members to check).  On 3.12+ it verifies attribute
        presence.  The test documents the intent regardless of Python version.
        """
        ns = self._make_ns()
        assert isinstance(ns, BuilderArgs)


class TestNamespaceFieldOwnership:
    """Verify that the namespace reads each field from the authoritative owner."""

    def _make_ns(self, mc=None, tc=None):
        mc = mc or RFDETRNanoConfig(num_classes=80)
        tc = tc or TrainConfig(dataset_dir="/tmp")
        return _namespace_from_configs(mc, tc)

    # --- cls_loss_coef must come from TrainConfig ---

    def test_cls_loss_coef_from_train_config(self) -> None:
        """ns.cls_loss_coef must reflect TrainConfig.cls_loss_coef, not ModelConfig."""
        mc = RFDETRNanoConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp", cls_loss_coef=2.5)
        ns = _namespace_from_configs(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(2.5)

    def test_cls_loss_coef_segmentation_default_matches_pre_1_7_effective_value(self) -> None:
        """SegmentationTrainConfig default must preserve the pre-1.7 effective loss_ce weight."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        ns = _namespace_from_configs(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(1.0)

    def test_cls_loss_coef_segmentation_explicit_train_config_value_wins(self) -> None:
        """Explicit SegmentationTrainConfig.cls_loss_coef values must propagate to namespace."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp", cls_loss_coef=5.0)
        ns = _namespace_from_configs(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(5.0)

    # --- num_select must come from ModelConfig unconditionally ---

    def test_num_select_from_model_config(self) -> None:
        """ns.num_select must equal mc.num_select regardless of tc.num_select."""
        mc = RFDETRSegNanoConfig()  # num_select=100
        tc = TrainConfig(dataset_dir="/tmp")  # num_select=300 (default — was the bug)
        ns = _namespace_from_configs(mc, tc)
        assert ns.num_select == 100

    @pytest.mark.parametrize(
        "config_class, expected_num_select",
        [
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
            pytest.param(RFDETRSmallConfig, 300, id="small"),
        ],
    )
    def test_num_select_matches_model_config_variant(self, config_class, expected_num_select) -> None:
        """ns.num_select must equal the model config's num_select for each variant."""
        mc = config_class()
        tc = TrainConfig(dataset_dir="/tmp")
        ns = _namespace_from_configs(mc, tc)
        assert ns.num_select == expected_num_select


class TestDimFeedforward:
    """``ModelConfig.dim_feedforward`` reaches the builder namespace and the decoder layers it sizes."""

    def test_default_matches_the_legacy_hardcoded_width(self) -> None:
        ns = _namespace_from_configs(RFDETRNanoConfig(), TrainConfig(dataset_dir="/tmp"))

        assert ns.dim_feedforward == 2048

    def test_override_is_forwarded(self) -> None:
        ns = _namespace_from_configs(RFDETRNanoConfig(dim_feedforward=1024), TrainConfig(dataset_dir="/tmp"))

        assert ns.dim_feedforward == 1024

    def test_model_config_wins_over_defaults(self) -> None:
        """A ``ModelConfig`` width overrides ``ModelDefaults.dim_feedforward``, which is only a fallback shadow."""
        defaults = dataclasses.replace(MODEL_DEFAULTS, dim_feedforward=512)

        ns = _namespace_from_configs(
            RFDETRNanoConfig(dim_feedforward=1024), TrainConfig(dataset_dir="/tmp"), defaults=defaults
        )

        assert ns.dim_feedforward == 1024

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_width_is_rejected(self, value: int) -> None:
        with pytest.raises(ValueError, match="dim_feedforward"):
            RFDETRNanoConfig(dim_feedforward=value)

    def test_decoder_layers_use_the_configured_width(self) -> None:
        from rfdetr.models import build_model

        ns = _namespace_from_configs(
            RFDETRNanoConfig(dim_feedforward=1024, pretrain_weights=None), TrainConfig(dataset_dir="/tmp")
        )
        model = build_model(ns)

        widths = {layer.linear1.out_features for layer in model.transformer.decoder.layers}
        assert widths == {1024}
