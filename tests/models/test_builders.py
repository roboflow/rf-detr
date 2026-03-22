# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterization tests for config-native builder functions.

These tests validate build_model_from_config() and build_criterion_from_config()
which accept Pydantic config objects directly instead of requiring a pre-built
SimpleNamespace.  Until the functions are implemented, all tests skip via the
module-level pytestmark.
"""

import pytest

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRSegNanoConfig,
    SegmentationTrainConfig,
    TrainConfig,
)

try:
    from rfdetr.models import build_model_from_config, build_criterion_from_config

    HAS_CONFIG_BUILDERS = True
except ImportError:
    HAS_CONFIG_BUILDERS = False

pytestmark = pytest.mark.skipif(
    not HAS_CONFIG_BUILDERS,
    reason="build_model_from_config not yet implemented",
)


class TestBuildModelFromConfig:
    """Tests for build_model_from_config(model_config, defaults=MODEL_DEFAULTS)."""

    def test_returns_lwdetr_for_base_config(self) -> None:
        """build_model_from_config with RFDETRBaseConfig returns an LWDETR instance."""
        from rfdetr.models.lwdetr import LWDETR

        mc = RFDETRBaseConfig(num_classes=80)
        model = build_model_from_config(mc)
        assert isinstance(model, LWDETR), (
            f"Expected LWDETR instance, got {type(model).__name__}"
        )

    def test_num_classes_correct(self) -> None:
        """num_classes=5 in config should produce class_embed with out_features=6.

        build_model adds +1 to num_classes (background class convention).
        """
        mc = RFDETRBaseConfig(num_classes=5)
        model = build_model_from_config(mc)
        assert model.class_embed.out_features == 6, (
            f"Expected class_embed.out_features=6 (num_classes+1), "
            f"got {model.class_embed.out_features}"
        )

    def test_parity_with_build_model_via_namespace(self) -> None:
        """Parameter count must match between config-native and namespace paths."""
        from rfdetr._namespace import build_namespace
        from rfdetr.models.lwdetr import build_model

        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")

        model_config_native = build_model_from_config(mc)
        ns = build_namespace(mc, tc)
        model_namespace = build_model(ns)

        params_native = sum(p.numel() for p in model_config_native.parameters())
        params_namespace = sum(p.numel() for p in model_namespace.parameters())
        assert params_native == params_namespace, (
            f"Parameter count mismatch: "
            f"config-native={params_native}, namespace={params_namespace}"
        )

    def test_segmentation_head_created_when_true(self) -> None:
        """RFDETRSegNanoConfig has segmentation_head=True; model must have it."""
        mc = RFDETRSegNanoConfig()
        model = build_model_from_config(mc)
        assert model.segmentation_head is not None, (
            "Expected segmentation_head to be created for RFDETRSegNanoConfig"
        )


class TestBuildCriterionFromConfig:
    """Tests for build_criterion_from_config(model_config, train_config, defaults)."""

    def test_returns_tuple(self) -> None:
        """build_criterion_from_config must return a 2-tuple (SetCriterion, PostProcess)."""
        from rfdetr.models.criterion import SetCriterion
        from rfdetr.models.postprocess import PostProcess

        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")
        result = build_criterion_from_config(mc, tc)
        assert isinstance(result, tuple), (
            f"Expected tuple, got {type(result).__name__}"
        )
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        criterion, postprocess = result
        assert isinstance(criterion, SetCriterion), (
            f"Expected SetCriterion, got {type(criterion).__name__}"
        )
        assert isinstance(postprocess, PostProcess), (
            f"Expected PostProcess, got {type(postprocess).__name__}"
        )

    def test_num_select_postprocess(self) -> None:
        """RFDETRSegNanoConfig has num_select=100; PostProcess must reflect it."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        _, postprocess = build_criterion_from_config(mc, tc)
        assert postprocess.num_select == 100, (
            f"Expected PostProcess.num_select=100, got {postprocess.num_select}"
        )

    def test_segmentation_losses_included(self) -> None:
        """With segmentation config, 'masks' must be in criterion.losses."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        criterion, _ = build_criterion_from_config(mc, tc)
        assert "masks" in criterion.losses, (
            f"Expected 'masks' in criterion.losses, got {criterion.losses}"
        )
