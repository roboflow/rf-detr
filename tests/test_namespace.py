# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for build_namespace() config forwarding."""

from typing import Any

from rfdetr._namespace import build_namespace
from rfdetr.config import RFDETRBaseConfig, TrainConfig


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
