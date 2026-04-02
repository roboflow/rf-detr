# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterization tests for build_model() and build_criterion_and_postprocessors().

These tests pin the current behavior of the legacy namespace-based builder
functions. They serve as a safety net during the config-native builder
refactoring: any change that alters these outputs is a regression.

All tests in this file must pass against the CURRENT codebase.
"""

import pytest
import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR, build_criterion_and_postprocessors, build_model
from rfdetr.models.postprocess import PostProcess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ns(mc=None, tc=None):
    """Build a namespace suitable for builder functions."""
    mc = mc or RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
    tc = tc or TrainConfig(dataset_dir="/tmp")
    return _namespace_from_configs(mc, tc)


# ---------------------------------------------------------------------------
# build_model characterization
# ---------------------------------------------------------------------------


class TestBuildModelCharacterization:
    """Pin current build_model() behaviour for the standard code path."""

    def test_returns_lwdetr_instance(self) -> None:
        ns = _make_ns()
        model = build_model(ns)
        assert isinstance(model, LWDETR)

    def test_num_classes_plus_one(self) -> None:
        """build_model applies the +1 background class convention."""
        mc = RFDETRBaseConfig(num_classes=5, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        assert model.class_embed.out_features == 6

    def test_num_queries_forwarded(self) -> None:
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        assert model.num_queries == mc.num_queries

    @pytest.mark.parametrize(
        "config_class, expected_queries",
        [
            pytest.param(RFDETRBaseConfig, 300, id="base"),
            pytest.param(RFDETRNanoConfig, 300, id="nano"),
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
        ],
    )
    def test_num_queries_per_config_variant(self, config_class, expected_queries) -> None:
        mc = config_class(pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        assert model.num_queries == expected_queries

    def test_segmentation_head_none_for_detection(self) -> None:
        ns = _make_ns()
        model = build_model(ns)
        assert model.segmentation_head is None

    def test_segmentation_head_present_for_seg_config(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        assert model.segmentation_head is not None

    def test_aux_loss_enabled_by_default(self) -> None:
        ns = _make_ns()
        model = build_model(ns)
        assert model.aux_loss is True

    def test_group_detr_forwarded(self) -> None:
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        assert model.group_detr == mc.group_detr

    def test_num_feature_levels_set_on_args(self) -> None:
        """build_model mutates args.num_feature_levels = len(projector_scale)."""
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        build_model(ns)
        assert ns.num_feature_levels == len(mc.projector_scale)

    @pytest.mark.parametrize(
        "config_class, expected_param_count_range",
        [
            pytest.param(RFDETRBaseConfig, (25_000_000, 40_000_000), id="base"),
            pytest.param(RFDETRNanoConfig, (25_000_000, 40_000_000), id="nano"),
        ],
    )
    def test_param_count_in_expected_range(self, config_class, expected_param_count_range) -> None:
        """Sanity check that the model has a plausible number of parameters."""
        mc = config_class(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        model = build_model(ns)
        total = sum(p.numel() for p in model.parameters())
        low, high = expected_param_count_range
        assert low <= total <= high, f"Expected param count in [{low}, {high}], got {total}"

    def test_encoder_only_returns_triple(self) -> None:
        """When encoder_only=True, build_model returns (encoder, None, None)."""
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        ns.encoder_only = True
        result = build_model(ns)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3
        encoder, second, third = result
        assert second is None
        assert third is None
        assert encoder is not None

    def test_backbone_only_returns_triple(self) -> None:
        """When backbone_only=True, build_model returns (backbone, None, None)."""
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        ns.backbone_only = True
        result = build_model(ns)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3
        backbone, second, third = result
        assert second is None
        assert third is None
        assert backbone is not None


# ---------------------------------------------------------------------------
# build_criterion_and_postprocessors characterization
# ---------------------------------------------------------------------------


class TestBuildCriterionCharacterization:
    """Pin current build_criterion_and_postprocessors() behaviour."""

    def test_returns_criterion_and_postprocess(self) -> None:
        ns = _make_ns()
        criterion, postprocess = build_criterion_and_postprocessors(ns)
        assert isinstance(criterion, SetCriterion)
        assert isinstance(postprocess, PostProcess)

    def test_detection_losses_list(self) -> None:
        """Detection-only config has exactly ['labels', 'boxes', 'cardinality']."""
        ns = _make_ns()
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.losses == ["labels", "boxes", "cardinality"]

    def test_segmentation_losses_include_masks(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        ns = _make_ns(mc=mc, tc=tc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert "masks" in criterion.losses

    def test_num_select_forwarded_to_postprocess(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        _, postprocess = build_criterion_and_postprocessors(ns)
        assert postprocess.num_select == 100

    def test_num_select_default_for_base(self) -> None:
        ns = _make_ns()
        _, postprocess = build_criterion_and_postprocessors(ns)
        assert postprocess.num_select == 300

    def test_weight_dict_contains_base_losses(self) -> None:
        ns = _make_ns()
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert "loss_ce" in criterion.weight_dict
        assert "loss_bbox" in criterion.weight_dict
        assert "loss_giou" in criterion.weight_dict

    def test_weight_dict_values_match_namespace(self) -> None:
        ns = _make_ns()
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.weight_dict["loss_ce"] == ns.cls_loss_coef
        assert criterion.weight_dict["loss_bbox"] == ns.bbox_loss_coef
        assert criterion.weight_dict["loss_giou"] == ns.giou_loss_coef

    def test_segmentation_weight_dict_contains_mask_losses(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        ns = _make_ns(mc=mc, tc=tc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert "loss_mask_ce" in criterion.weight_dict
        assert "loss_mask_dice" in criterion.weight_dict

    def test_aux_loss_expands_weight_dict(self) -> None:
        """With aux_loss=True and 3 dec_layers, weight_dict has aux entries _0 and _1."""
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        assert ns.aux_loss is True
        criterion, _ = build_criterion_and_postprocessors(ns)
        # dec_layers=3 -> 2 aux layers (0 and 1)
        assert "loss_ce_0" in criterion.weight_dict
        assert "loss_ce_1" in criterion.weight_dict

    def test_two_stage_adds_enc_losses(self) -> None:
        """With two_stage=True, weight_dict has '_enc' suffix entries."""
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        assert ns.two_stage is True
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert "loss_ce_enc" in criterion.weight_dict
        assert "loss_bbox_enc" in criterion.weight_dict
        assert "loss_giou_enc" in criterion.weight_dict

    def test_criterion_num_classes_plus_one(self) -> None:
        mc = RFDETRBaseConfig(num_classes=5, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.num_classes == 6

    def test_focal_alpha_forwarded(self) -> None:
        ns = _make_ns()
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.focal_alpha == pytest.approx(0.25)

    def test_group_detr_forwarded_to_criterion(self) -> None:
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.group_detr == mc.group_detr

    def test_segmentation_criterion_has_mask_point_sample_ratio(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        ns = _make_ns(mc=mc, tc=tc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.mask_point_sample_ratio == 16

    def test_ia_bce_loss_forwarded(self) -> None:
        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert criterion.ia_bce_loss == mc.ia_bce_loss


# ---------------------------------------------------------------------------
# _build_model_context characterization
# ---------------------------------------------------------------------------


class TestBuildModelContextCharacterization:
    """Pin current _build_model_context() behaviour.

    _build_model_context is the inference-path factory used by RFDETR.get_model().
    It has zero test coverage today.
    """

    def test_returns_model_context(self) -> None:
        from rfdetr.detr import ModelContext, _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert isinstance(ctx, ModelContext)

    def test_model_is_lwdetr(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert isinstance(ctx.model, LWDETR)

    def test_postprocess_is_postprocess(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert isinstance(ctx.postprocess, PostProcess)

    def test_resolution_from_config(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert ctx.resolution == mc.resolution

    def test_device_from_config(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert ctx.device == torch.device("cpu")

    def test_torch_device_cpu_from_config(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device=torch.device("cpu"))
        ctx = _build_model_context(mc)
        assert ctx.device == torch.device("cpu")

    def test_class_names_none_without_pretrain(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert ctx.class_names is None

    def test_num_select_on_postprocess(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert ctx.postprocess.num_select == 100

    def test_args_namespace_attached(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert hasattr(ctx.args, "num_classes")
        assert hasattr(ctx.args, "num_select")

    def test_inference_model_initially_none(self) -> None:
        from rfdetr.detr import _build_model_context

        mc = RFDETRBaseConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ctx = _build_model_context(mc)
        assert ctx.inference_model is None


# ---------------------------------------------------------------------------
# RFDETRModelModule.__init__ characterization
# ---------------------------------------------------------------------------


class TestRFDETRModelModuleInitCharacterization:
    """Pin RFDETRModelModule.__init__() structural outputs.

    The existing test_module_model.py tests the init via mocked build_model and
    build_namespace. These tests exercise the REAL init path (no mocks) to
    characterize what a freshly built module looks like.
    """

    def _make_module(self, mc=None, tc=None):
        from rfdetr.training.module_model import RFDETRModelModule

        mc = mc or RFDETRBaseConfig(num_classes=5, pretrain_weights=None, device="cpu")
        tc = tc or TrainConfig(dataset_dir="/tmp")
        return RFDETRModelModule(mc, tc)

    def test_model_attribute_is_lwdetr(self) -> None:
        module = self._make_module()
        # model could be wrapped by torch.compile, so check the underlying type
        underlying = getattr(module.model, "_orig_mod", module.model)
        assert isinstance(underlying, LWDETR)

    def test_criterion_is_set_criterion(self) -> None:
        module = self._make_module()
        assert isinstance(module.criterion, SetCriterion)

    def test_postprocess_is_postprocess(self) -> None:
        module = self._make_module()
        assert isinstance(module.postprocess, PostProcess)

    def test_strict_loading_false(self) -> None:
        """strict_loading=False allows partial state-dict loading."""
        module = self._make_module()
        assert module.strict_loading is False

    def test_configs_stored(self) -> None:
        mc = RFDETRBaseConfig(num_classes=5, pretrain_weights=None, device="cpu")
        tc = TrainConfig(dataset_dir="/tmp")
        module = self._make_module(mc=mc, tc=tc)
        assert module.model_config is mc
        assert module.train_config is tc

    def test_criterion_num_classes_matches_model(self) -> None:
        """Criterion and model must agree on num_classes (both use +1 convention)."""
        mc = RFDETRBaseConfig(num_classes=5, pretrain_weights=None, device="cpu")
        module = self._make_module(mc=mc)
        underlying = getattr(module.model, "_orig_mod", module.model)
        assert module.criterion.num_classes == underlying.class_embed.out_features

    def test_postprocess_num_select_matches_config(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        module = self._make_module(mc=mc, tc=tc)
        assert module.postprocess.num_select == mc.num_select

    def test_segmentation_criterion_with_seg_config(self) -> None:
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        module = self._make_module(mc=mc, tc=tc)
        assert "masks" in module.criterion.losses
