# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for build_namespace() config forwarding."""

import sys
from typing import Any

import pytest

from rfdetr._namespace import build_namespace
from rfdetr.config import RFDETRBaseConfig, RFDETRSegNanoConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.models._types import BuilderArgs


class TestBuildNamespaceForwarding:
    """Verify that build_namespace() forwards TrainConfig fields that were
    previously hardcoded to wrong defaults."""

    def _make_ns(self: "TestBuildNamespaceForwarding", **tc_kwargs: Any) -> Any:
        """Build a namespace for tests with minimal default TrainConfig values."""
        mc = RFDETRBaseConfig(num_classes=80)
        tc_kwargs.setdefault("dataset_dir", "/tmp")
        tc = TrainConfig(**tc_kwargs)
        return build_namespace(mc, tc)

    def test_aug_config_forwarded_when_set(self: "TestBuildNamespaceForwarding") -> None:
        aug = {"hsv_h": 0.015, "hsv_s": 0.7}
        ns = self._make_ns(aug_config=aug)
        assert ns.aug_config == aug

    def test_aug_config_none_by_default(self: "TestBuildNamespaceForwarding") -> None:
        ns = self._make_ns()
        assert ns.aug_config is None

    def test_use_ema_forwarded_true(self: "TestBuildNamespaceForwarding") -> None:
        ns = self._make_ns(use_ema=True)
        assert ns.use_ema is True

    def test_use_ema_forwarded_false(self: "TestBuildNamespaceForwarding") -> None:
        ns = self._make_ns(use_ema=False)
        assert ns.use_ema is False

    def test_early_stopping_use_ema_forwarded_true(self: "TestBuildNamespaceForwarding") -> None:
        ns = self._make_ns(early_stopping_use_ema=True)
        assert ns.early_stopping_use_ema is True

    def test_early_stopping_use_ema_forwarded_false(self: "TestBuildNamespaceForwarding") -> None:
        ns = self._make_ns(early_stopping_use_ema=False)
        assert ns.early_stopping_use_ema is False


class TestBuildNamespaceProtocol:
    """build_namespace() output must satisfy the BuilderArgs Protocol."""

    def _make_ns(self, mc=None, tc=None):
        mc = mc or RFDETRBaseConfig(num_classes=80)
        tc = tc or TrainConfig(dataset_dir="/tmp")
        return build_namespace(mc, tc)

    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="Runtime Protocol attribute checks require Python 3.12+",
    )
    def test_namespace_satisfies_builderargs_protocol_py312(self) -> None:
        """On Python 3.12+, isinstance() verifies data-attribute presence."""
        ns = self._make_ns()
        assert isinstance(ns, BuilderArgs)

    def test_namespace_is_builderargs_instance(self) -> None:
        """isinstance() check passes on all supported Python versions.

        On Python 3.10/3.11 this is a structural no-op (no method members to
        check).  On 3.12+ it verifies attribute presence.  The test documents
        the intent regardless of Python version.
        """
        ns = self._make_ns()
        assert isinstance(ns, BuilderArgs)


class TestBuildNamespaceFieldOwnership:
    """Verify that the namespace reads each field from the authoritative owner."""

    def _make_ns(self, mc=None, tc=None):
        mc = mc or RFDETRBaseConfig(num_classes=80)
        tc = tc or TrainConfig(dataset_dir="/tmp")
        return build_namespace(mc, tc)

    # --- cls_loss_coef must come from TrainConfig ---

    def test_cls_loss_coef_from_train_config(self) -> None:
        """ns.cls_loss_coef must reflect TrainConfig.cls_loss_coef, not ModelConfig."""
        mc = RFDETRBaseConfig(num_classes=80)  # cls_loss_coef=1.0 (ModelConfig default)
        tc = TrainConfig(dataset_dir="/tmp", cls_loss_coef=2.5)
        ns = build_namespace(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(2.5)

    def test_cls_loss_coef_segmentation_uses_train_config_value(self) -> None:
        """SegmentationTrainConfig.cls_loss_coef=5.0 must propagate to namespace."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")  # cls_loss_coef=5.0
        ns = build_namespace(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(5.0)

    def test_cls_loss_coef_train_config_wins_over_explicit_model_config(self) -> None:
        """When both are explicitly set, TrainConfig.cls_loss_coef takes precedence."""
        with pytest.warns(DeprecationWarning, match="ModelConfig\\.cls_loss_coef is deprecated"):
            mc = RFDETRBaseConfig(num_classes=80, cls_loss_coef=0.5)
        tc = TrainConfig(dataset_dir="/tmp", cls_loss_coef=3.0)
        ns = build_namespace(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(3.0)

    def test_cls_loss_coef_model_config_explicit_is_preserved_during_deprecation(self) -> None:
        """Explicit ModelConfig.cls_loss_coef remains effective until removal."""
        with pytest.warns(DeprecationWarning, match="ModelConfig\\.cls_loss_coef is deprecated"):
            mc = RFDETRBaseConfig(num_classes=80, cls_loss_coef=2.5)
        tc = TrainConfig(dataset_dir="/tmp")
        ns = build_namespace(mc, tc)
        assert ns.cls_loss_coef == pytest.approx(2.5)

    # --- num_select must come from ModelConfig unconditionally ---

    def test_num_select_from_model_config(self) -> None:
        """ns.num_select must equal mc.num_select regardless of tc.num_select."""
        mc = RFDETRSegNanoConfig()  # num_select=100
        tc = TrainConfig(dataset_dir="/tmp")  # num_select=300 (default — was the bug)
        ns = build_namespace(mc, tc)
        assert ns.num_select == 100

    @pytest.mark.parametrize(
        "config_class, expected_num_select",
        [
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
            pytest.param(RFDETRBaseConfig, 300, id="base"),
        ],
    )
    def test_num_select_matches_model_config_variant(self, config_class, expected_num_select) -> None:
        """ns.num_select must equal the model config's num_select for each variant."""
        mc = config_class()
        tc = TrainConfig(dataset_dir="/tmp")
        ns = build_namespace(mc, tc)
        assert ns.num_select == expected_num_select
