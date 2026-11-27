# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests for build_model() and build_criterion_and_postprocessors().

These tests pin the current behavior of the legacy namespace-based builder functions. They serve as a safety net during
the config-native builder refactoring: any change that alters these outputs is a regression.

All tests in this file must pass against the CURRENT codebase.
"""

import pytest
import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    RFDETRKeypointPreviewConfig,
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.detr import ModelContext, _build_model_context
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR, build_criterion_and_postprocessors, build_model
from rfdetr.models.postprocess import PostProcess
from rfdetr.training.module_model import RFDETRModelModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ns(mc=None, tc=None):
    """Build a namespace suitable for builder functions.

    Examples:
        >>> ns = _make_ns()
        >>> ns.num_classes
        80
    """
    mc = mc or RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
    tc = tc or TrainConfig(dataset_dir="/tmp")
    return _namespace_from_configs(mc, tc)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    """Pin current build_model() behaviour for the standard code path."""

    def test_returns_lwdetr_instance(self) -> None:
        """The default namespace builds an ``LWDETR`` model."""
        model = build_model(_make_ns())
        assert isinstance(model, LWDETR)

    def test_num_classes_plus_one(self) -> None:
        """build_model applies the +1 background class convention."""
        mc = RFDETRNanoConfig(num_classes=5, pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        assert model.class_embed.out_features == 6

    def test_num_queries_forwarded(self) -> None:
        """The config's ``num_queries`` reaches the model."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        assert model.num_queries == mc.num_queries

    @pytest.mark.parametrize(
        "config_class, expected_queries",
        [
            pytest.param(RFDETRSmallConfig, 300, id="small"),
            pytest.param(RFDETRNanoConfig, 300, id="nano"),
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
        ],
    )
    def test_num_queries_per_config_variant(self, config_class, expected_queries) -> None:
        """Each variant config builds a model with its own default query count."""
        mc = config_class(pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        assert model.num_queries == expected_queries

    def test_segmentation_head_none_for_detection(self) -> None:
        """A detection config builds no segmentation head."""
        model = build_model(_make_ns())
        assert model.segmentation_head is None

    def test_segmentation_head_present_for_seg_config(self) -> None:
        """A segmentation config builds a segmentation head."""
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        assert model.segmentation_head is not None

    def test_aux_loss_enabled_by_default(self) -> None:
        """Auxiliary decoder losses are on by default."""
        model = build_model(_make_ns())
        assert model.aux_loss is True

    def test_group_detr_forwarded(self) -> None:
        """The config's ``group_detr`` reaches the model."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        assert model.group_detr == mc.group_detr

    def test_num_feature_levels_set_on_args(self) -> None:
        """build_model mutates args.num_feature_levels = len(projector_scale)."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        build_model(ns)
        assert ns.num_feature_levels == len(mc.projector_scale)

    @pytest.mark.parametrize(
        "config_class, expected_param_count_range",
        [
            pytest.param(RFDETRSmallConfig, (25_000_000, 40_000_000), id="small"),
            pytest.param(RFDETRNanoConfig, (25_000_000, 40_000_000), id="nano"),
        ],
    )
    def test_param_count_in_expected_range(self, config_class, expected_param_count_range) -> None:
        """Sanity check that the model has a plausible number of parameters."""
        mc = config_class(num_classes=80, pretrain_weights=None, device="cpu")
        model = build_model(_make_ns(mc=mc))
        total = sum(p.numel() for p in model.parameters())
        low, high = expected_param_count_range
        assert low <= total <= high, f"Expected param count in [{low}, {high}], got {total}"

    @pytest.mark.parametrize("flag", ["encoder_only", "backbone_only"])
    def test_encoder_or_backbone_only_returns_triple(self, flag: str) -> None:
        """When encoder_only/backbone_only is set, build_model returns (module, None, None)."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        setattr(ns, flag, True)
        result = build_model(ns)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3
        module, second, third = result
        assert second is None
        assert third is None
        assert module is not None


# ---------------------------------------------------------------------------
# build_criterion_and_postprocessors
# ---------------------------------------------------------------------------


@pytest.fixture
def seg_criterion() -> SetCriterion:
    """Criterion built from the segmentation Nano config and its matching train config."""
    mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
    tc = SegmentationTrainConfig(dataset_dir="/tmp")
    criterion, _ = build_criterion_and_postprocessors(_make_ns(mc=mc, tc=tc))
    return criterion


class TestBuildCriterion:
    """Pin current build_criterion_and_postprocessors() behaviour."""

    def test_returns_criterion_and_postprocess(self) -> None:
        """The builder returns a ``(SetCriterion, PostProcess)`` pair."""
        criterion, postprocess = build_criterion_and_postprocessors(_make_ns())
        assert isinstance(criterion, SetCriterion)
        assert isinstance(postprocess, PostProcess)

    def test_detection_losses_list(self) -> None:
        """Detection-only config has exactly ['labels', 'boxes', 'cardinality']."""
        criterion, _ = build_criterion_and_postprocessors(_make_ns())
        assert criterion.losses == ["labels", "boxes", "cardinality"]

    def test_segmentation_losses_include_masks(self, seg_criterion: SetCriterion) -> None:
        """A segmentation config adds the 'masks' loss."""
        assert "masks" in seg_criterion.losses

    @pytest.mark.parametrize(
        "config_class, expected_num_select",
        [
            pytest.param(RFDETRSegNanoConfig, 100, id="seg_nano"),
            pytest.param(RFDETRNanoConfig, 300, id="nano"),
        ],
    )
    def test_num_select_forwarded_to_postprocess(self, config_class, expected_num_select) -> None:
        """The variant's ``num_select`` default reaches PostProcess."""
        mc = config_class(pretrain_weights=None, device="cpu")
        _, postprocess = build_criterion_and_postprocessors(_make_ns(mc=mc))
        assert postprocess.num_select == expected_num_select

    @pytest.mark.parametrize(
        "loss_key, coef_attr",
        [
            pytest.param("loss_ce", "cls_loss_coef", id="ce"),
            pytest.param("loss_bbox", "bbox_loss_coef", id="bbox"),
            pytest.param("loss_giou", "giou_loss_coef", id="giou"),
        ],
    )
    def test_weight_dict_holds_namespace_coefficient(self, loss_key: str, coef_attr: str) -> None:
        """Each base loss is present in weight_dict with the coefficient from the namespace."""
        ns = _make_ns()
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert loss_key in criterion.weight_dict
        assert criterion.weight_dict[loss_key] == getattr(ns, coef_attr)

    @pytest.mark.parametrize("loss_key", ["loss_mask_ce", "loss_mask_dice"])
    def test_segmentation_weight_dict_contains_mask_losses(self, seg_criterion: SetCriterion, loss_key: str) -> None:
        """A segmentation config adds the mask losses to weight_dict."""
        assert loss_key in seg_criterion.weight_dict

    @pytest.mark.parametrize("loss_key", ["loss_ce_0", "loss_ce_1"])
    def test_aux_loss_expands_weight_dict(self, loss_key: str) -> None:
        """With aux_loss=True and 3 dec_layers, weight_dict has aux entries _0 and _1.

        Uses ``RFDETRSmallConfig`` rather than the nano default because the assertion counts aux entries, which is
        ``dec_layers - 1``; nano's ``dec_layers=2`` would yield only ``_0``.
        """
        mc = RFDETRSmallConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        assert ns.aux_loss is True
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert loss_key in criterion.weight_dict

    @pytest.mark.parametrize("loss_key", ["loss_ce_enc", "loss_bbox_enc", "loss_giou_enc"])
    def test_two_stage_adds_enc_losses(self, loss_key: str) -> None:
        """With two_stage=True, weight_dict has '_enc' suffix entries."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        ns = _make_ns(mc=mc)
        assert ns.two_stage is True
        criterion, _ = build_criterion_and_postprocessors(ns)
        assert loss_key in criterion.weight_dict

    def test_criterion_num_classes_plus_one(self) -> None:
        """The criterion applies the +1 background class convention."""
        mc = RFDETRNanoConfig(num_classes=5, pretrain_weights=None, device="cpu")
        criterion, _ = build_criterion_and_postprocessors(_make_ns(mc=mc))
        assert criterion.num_classes == 6

    def test_focal_alpha_forwarded(self) -> None:
        """The default focal alpha (0.25) reaches the criterion."""
        criterion, _ = build_criterion_and_postprocessors(_make_ns())
        assert criterion.focal_alpha == pytest.approx(0.25)

    def test_group_detr_forwarded_to_criterion(self) -> None:
        """The config's ``group_detr`` reaches the criterion."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        criterion, _ = build_criterion_and_postprocessors(_make_ns(mc=mc))
        assert criterion.group_detr == mc.group_detr

    def test_segmentation_criterion_has_mask_point_sample_ratio(self, seg_criterion: SetCriterion) -> None:
        """A segmentation criterion samples mask points at ratio 16."""
        assert seg_criterion.mask_point_sample_ratio == 16

    def test_ia_bce_loss_forwarded(self) -> None:
        """The config's ``ia_bce_loss`` reaches the criterion."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        criterion, _ = build_criterion_and_postprocessors(_make_ns(mc=mc))
        assert criterion.ia_bce_loss == mc.ia_bce_loss


# ---------------------------------------------------------------------------
# _build_model_context
# ---------------------------------------------------------------------------


@pytest.fixture
def nano_ctx() -> ModelContext:
    """Model context built from a CPU Nano config without pretrained weights."""
    return _build_model_context(RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu"))


@pytest.fixture
def seg_nano_ctx() -> ModelContext:
    """Model context built from a CPU segmentation Nano config without pretrained weights."""
    return _build_model_context(RFDETRSegNanoConfig(pretrain_weights=None, device="cpu"))


class TestBuildModelContext:
    """Pin current _build_model_context() behaviour.

    _build_model_context is the inference-path factory used by RFDETR.get_model().
    """

    def test_returns_model_context(self, nano_ctx: ModelContext) -> None:
        """The factory returns a ``ModelContext``."""
        assert isinstance(nano_ctx, ModelContext)

    def test_model_is_lwdetr(self, nano_ctx: ModelContext) -> None:
        """The context holds an ``LWDETR`` model."""
        assert isinstance(nano_ctx.model, LWDETR)

    def test_postprocess_is_postprocess(self, nano_ctx: ModelContext) -> None:
        """The context holds a ``PostProcess``."""
        assert isinstance(nano_ctx.postprocess, PostProcess)

    def test_resolution_from_config(self) -> None:
        """The context resolution equals the config resolution."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu")
        assert _build_model_context(mc).resolution == mc.resolution

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(torch.device("cpu"), id="torch-device"),
        ],
    )
    def test_device_from_config(self, device) -> None:
        """A device given as a string or a ``torch.device`` resolves to ``torch.device("cpu")``."""
        mc = RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device=device)
        assert _build_model_context(mc).device == torch.device("cpu")

    def test_class_names_none_without_pretrain(self, nano_ctx: ModelContext) -> None:
        """Without pretrained weights there are no class names."""
        assert nano_ctx.class_names is None

    def test_num_select_on_postprocess(self, seg_nano_ctx: ModelContext) -> None:
        """The segmentation variant's ``num_select`` reaches PostProcess."""
        assert seg_nano_ctx.postprocess.num_select == 100

    def test_keypoint_preview_postprocess_has_keypoint_schema(self) -> None:
        """The keypoint preview config yields a PostProcess with the person keypoint schema."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
        assert _build_model_context(mc).postprocess.num_keypoints_per_class == [17]

    def test_args_namespace_attached(self, nano_ctx: ModelContext) -> None:
        """The context carries the built args namespace."""
        assert hasattr(nano_ctx.args, "num_classes")
        assert hasattr(nano_ctx.args, "num_select")

    def test_inference_model_initially_none(self, nano_ctx: ModelContext) -> None:
        """No optimized inference model exists until one is requested."""
        assert nano_ctx.inference_model is None

    def test_args_dataset_dir_does_not_leak_cwd(self, nano_ctx: ModelContext) -> None:
        """The serialized namespace must not embed the caller's realpathed CWD as dataset_dir."""
        assert nano_ctx.args.dataset_dir is None

    def test_args_output_dir_does_not_leak_cwd(self, nano_ctx: ModelContext) -> None:
        """The serialized namespace must keep a relative output_dir, not the caller's realpathed CWD."""
        assert nano_ctx.args.output_dir == "output"

    def test_predict_postprocess_always_upsamples_masks_regardless_of_eval_flag(
        self, seg_nano_ctx: ModelContext
    ) -> None:
        """RFDETR.predict's PostProcess must always upsample masks to full image resolution.

        TrainConfig.eval_masks_head_resolution is a validation-only cost-saving knob; it must never leak into the
        inference/predict path via ``_build_model_context``'s internal dummy ``TrainConfig`` — deployment predictions
        always report at native image resolution regardless of how the checkpoint was trained.
        """
        assert seg_nano_ctx.postprocess.upsample_masks_to_image_size is True


# ---------------------------------------------------------------------------
# RFDETRModelModule.__init__
# ---------------------------------------------------------------------------


def _make_module(mc=None, tc=None) -> RFDETRModelModule:
    """Build a real ``RFDETRModelModule`` (Nano, 5 classes, CPU, no pretrained weights) unless configs are given.

    Examples:
        >>> _make_module().criterion.num_classes
        6
    """
    mc = mc or RFDETRNanoConfig(num_classes=5, pretrain_weights=None, device="cpu")
    tc = tc or TrainConfig(dataset_dir="/tmp")
    return RFDETRModelModule(mc, tc)


class TestRFDETRModelModuleInit:
    """Pin RFDETRModelModule.__init__() structural outputs.

    The existing test_module_model.py tests the init via mocked build_model and build_namespace. These tests exercise
    the REAL init path (no mocks) to characterize what a freshly built module looks like.
    """

    def test_model_attribute_is_lwdetr(self) -> None:
        """The module's model is an ``LWDETR`` (possibly wrapped by torch.compile)."""
        module = _make_module()
        underlying = getattr(module.model, "_orig_mod", module.model)
        assert isinstance(underlying, LWDETR)

    def test_criterion_is_set_criterion(self) -> None:
        """The module builds a ``SetCriterion``."""
        assert isinstance(_make_module().criterion, SetCriterion)

    def test_postprocess_is_postprocess(self) -> None:
        """The module builds a ``PostProcess``."""
        assert isinstance(_make_module().postprocess, PostProcess)

    def test_strict_loading_false(self) -> None:
        """strict_loading=False allows partial state-dict loading."""
        assert _make_module().strict_loading is False

    def test_configs_stored(self) -> None:
        """The module keeps the exact model and train config objects it was given."""
        mc = RFDETRNanoConfig(num_classes=5, pretrain_weights=None, device="cpu")
        tc = TrainConfig(dataset_dir="/tmp")
        module = _make_module(mc=mc, tc=tc)
        assert module.model_config is mc
        assert module.train_config is tc

    def test_criterion_num_classes_matches_model(self) -> None:
        """Criterion and model must agree on num_classes (both use +1 convention)."""
        module = _make_module()
        underlying = getattr(module.model, "_orig_mod", module.model)
        assert module.criterion.num_classes == underlying.class_embed.out_features

    def test_postprocess_num_select_matches_config(self) -> None:
        """PostProcess ``num_select`` follows the model config."""
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        module = _make_module(mc=mc, tc=tc)
        assert module.postprocess.num_select == mc.num_select

    def test_segmentation_criterion_with_seg_config(self) -> None:
        """A segmentation config adds the 'masks' loss to the module's criterion."""
        mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu")
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        module = _make_module(mc=mc, tc=tc)
        assert "masks" in module.criterion.losses
