# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterization tests for config-native builder functions.

These tests validate build_model_from_config() and build_criterion_from_config()
which accept Pydantic config objects directly instead of requiring a pre-built
SimpleNamespace. If these functions cannot be imported, all tests skip via the
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
    from rfdetr.models import build_criterion_from_config, build_model_from_config

    HAS_CONFIG_BUILDERS = True
except ImportError:
    HAS_CONFIG_BUILDERS = False

pytestmark = pytest.mark.skipif(
    not HAS_CONFIG_BUILDERS,
    reason="config-native builder functions are not importable",
)


class TestBuildModelFromConfig:
    """Tests for build_model_from_config(model_config, defaults=MODEL_DEFAULTS)."""

    def test_returns_lwdetr_for_base_config(self) -> None:
        """build_model_from_config with RFDETRBaseConfig returns an LWDETR instance."""
        from rfdetr.models.lwdetr import LWDETR

        mc = RFDETRBaseConfig(num_classes=80)
        model = build_model_from_config(mc)
        assert isinstance(model, LWDETR), f"Expected LWDETR instance, got {type(model).__name__}"

    def test_num_classes_correct(self) -> None:
        """num_classes=5 in config should produce class_embed with out_features=6.

        build_model adds +1 to num_classes (background class convention).
        """
        mc = RFDETRBaseConfig(num_classes=5)
        model = build_model_from_config(mc)
        assert model.class_embed.out_features == 6, (
            f"Expected class_embed.out_features=6 (num_classes+1), got {model.class_embed.out_features}"
        )

    def test_parity_with_build_model_via_namespace(self) -> None:
        """Parameter count must match between config-native and namespace paths."""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.models.lwdetr import build_model

        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")

        model_config_native = build_model_from_config(mc, tc)
        ns = _namespace_from_configs(mc, tc)
        model_namespace = build_model(ns)

        params_native = sum(p.numel() for p in model_config_native.parameters())
        params_namespace = sum(p.numel() for p in model_namespace.parameters())
        assert params_native == params_namespace, (
            f"Parameter count mismatch: config-native={params_native}, namespace={params_namespace}"
        )

    def test_segmentation_head_created_when_true(self) -> None:
        """RFDETRSegNanoConfig has segmentation_head=True; model must have it."""
        mc = RFDETRSegNanoConfig()
        model = build_model_from_config(mc)
        assert model.segmentation_head is not None, "Expected segmentation_head to be created for RFDETRSegNanoConfig"

    def test_drop_path_uses_train_config_value(self) -> None:
        """Non-default TrainConfig.drop_path must reach the model builder path."""
        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp", drop_path=0.2)
        model = build_model_from_config(mc, tc)

        layers = model._get_backbone_encoder_layers()
        assert layers is not None
        assert hasattr(layers[-1], "drop_path")
        assert layers[-1].drop_path.drop_prob == pytest.approx(0.2)

    def test_rejects_encoder_only_defaults(self) -> None:
        """The config-native builder guarantees an LWDETR return value."""
        from dataclasses import replace

        from rfdetr.models import MODEL_DEFAULTS

        mc = RFDETRBaseConfig(num_classes=80)

        with pytest.raises(ValueError, match="encoder_only=False"):
            build_model_from_config(mc, defaults=replace(MODEL_DEFAULTS, encoder_only=True))

    def test_rejects_backbone_only_defaults(self) -> None:
        """backbone_only=True in defaults must also raise ValueError."""
        from dataclasses import replace

        from rfdetr.models import MODEL_DEFAULTS

        mc = RFDETRBaseConfig(num_classes=80)

        with pytest.raises(ValueError, match="backbone_only=False"):
            build_model_from_config(mc, defaults=replace(MODEL_DEFAULTS, backbone_only=True))

    def test_none_train_config_uses_dummy(self) -> None:
        """build_model_from_config with train_config=None must not raise."""
        mc = RFDETRBaseConfig(num_classes=80)
        model = build_model_from_config(mc, train_config=None)
        assert model is not None, "Expected a model, got None"


class TestBuildCriterionFromConfig:
    """Tests for build_criterion_from_config(model_config, train_config, defaults)."""

    def test_returns_tuple(self) -> None:
        """build_criterion_from_config must return a 2-tuple (SetCriterion, PostProcess)."""
        from rfdetr.models.criterion import SetCriterion
        from rfdetr.models.postprocess import PostProcess

        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")
        result = build_criterion_from_config(mc, tc)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        criterion, postprocess = result
        assert isinstance(criterion, SetCriterion), f"Expected SetCriterion, got {type(criterion).__name__}"
        assert isinstance(postprocess, PostProcess), f"Expected PostProcess, got {type(postprocess).__name__}"

    def test_num_select_postprocess(self) -> None:
        """RFDETRSegNanoConfig has num_select=100; PostProcess must reflect it."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        _, postprocess = build_criterion_from_config(mc, tc)
        assert postprocess.num_select == 100, f"Expected PostProcess.num_select=100, got {postprocess.num_select}"

    def test_segmentation_losses_included(self) -> None:
        """With segmentation config, 'masks' must be in criterion.losses."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        criterion, _ = build_criterion_from_config(mc, tc)
        assert "masks" in criterion.losses, f"Expected 'masks' in criterion.losses, got {criterion.losses}"

    def test_custom_defaults_focal_alpha_applied(self) -> None:
        """Custom focal_alpha in ModelDefaults must reach SetCriterion."""
        from dataclasses import replace

        from rfdetr.models import MODEL_DEFAULTS

        mc = RFDETRBaseConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")
        custom_defaults = replace(MODEL_DEFAULTS, focal_alpha=0.5)
        criterion, _ = build_criterion_from_config(mc, tc, defaults=custom_defaults)
        assert criterion.focal_alpha == pytest.approx(0.5), f"Expected focal_alpha=0.5, got {criterion.focal_alpha}"
