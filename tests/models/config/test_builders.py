# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterization tests for build_model() and build_criterion_and_postprocessors().

These tests pin the current behavior of the legacy namespace-based builder functions. They serve as a safety net during
the config-native builder refactoring: any change that alters these outputs is a regression.

All tests in this file must pass against the CURRENT codebase.

Characterization tests for config-native builder functions.

These tests validate build_model_from_config() and build_criterion_from_config() which accept Pydantic config objects
directly instead of requiring a pre-built SimpleNamespace. If these functions cannot be imported, all tests skip via the
module-level pytestmark.

Tests that ``num_channels`` reaches the stem of the prediction model, the training model and checkpoint loads.

Tests for the optional plus-model import block in ``rfdetr.platform.models``.
"""

import importlib.util
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

import rfdetr.platform.models as platform_models_module
from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    RFDETRKeypointPreviewConfig,
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.datasets._aug_utils import cycle_channel_stats
from rfdetr.datasets.kornia_transforms import IMAGENET_MEAN, IMAGENET_STD, build_normalize
from rfdetr.detr import ModelContext, _build_model_context
from rfdetr.models import MODEL_DEFAULTS
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR, build_criterion_and_postprocessors, build_model
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.weights import adapt_input_channels
from rfdetr.platform import _PLUS_EXPORTS
from rfdetr.training.module_model import RFDETRModelModule
from tests._markers import _requires_kornia


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

    The existing tests in ``tests/training/model/`` test the init via mocked build_model and build_namespace. These
    tests exercise the REAL init path (no mocks) to characterize what a freshly built module looks like.
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


try:
    from rfdetr.models import build_criterion_from_config, build_model_from_config

    HAS_CONFIG_BUILDERS = True
except ImportError:
    HAS_CONFIG_BUILDERS = False


#: Skip every test here when the config-native builder functions cannot be imported.
_requires_config_builders = pytest.mark.skipif(
    not HAS_CONFIG_BUILDERS,
    reason="config-native builder functions are not importable",
)


@_requires_config_builders
class TestBuildModelFromConfig:
    """Tests for build_model_from_config(model_config, defaults=MODEL_DEFAULTS)."""

    def test_returns_lwdetr_for_base_config(self) -> None:
        """build_model_from_config with RFDETRNanoConfig returns an LWDETR instance."""
        mc = RFDETRNanoConfig(num_classes=80)
        model = build_model_from_config(mc)
        assert isinstance(model, LWDETR), f"Expected LWDETR instance, got {type(model).__name__}"

    def test_num_classes_includes_background_class(self) -> None:
        """num_classes=5 in config should produce class_embed with out_features=6.

        build_model adds +1 to num_classes (background class convention).
        """
        mc = RFDETRNanoConfig(num_classes=5)
        model = build_model_from_config(mc)
        assert model.class_embed.out_features == 6, (
            f"Expected class_embed.out_features=6 (num_classes+1), got {model.class_embed.out_features}"
        )

    def test_parity_with_build_model_via_namespace(self) -> None:
        """Parameter count must match between config-native and namespace paths."""
        mc = RFDETRNanoConfig(num_classes=80)
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
        mc = RFDETRNanoConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp", drop_path=0.2)
        model = build_model_from_config(mc, tc)

        layers = model._get_backbone_encoder_layers()
        assert layers is not None
        assert hasattr(layers[-1], "drop_path")
        assert layers[-1].drop_path.drop_prob == pytest.approx(0.2)

    @pytest.mark.parametrize("flag", ["encoder_only", "backbone_only"])
    def test_rejects_non_model_defaults(self, flag: str) -> None:
        """The config-native builder guarantees an LWDETR return value, so encoder_only/backbone_only raise
        ValueError."""
        mc = RFDETRNanoConfig(num_classes=80)

        with pytest.raises(ValueError, match=f"{flag}=False"):
            build_model_from_config(mc, defaults=replace(MODEL_DEFAULTS, **{flag: True}))

    def test_none_train_config_builds_model(self) -> None:
        """build_model_from_config with train_config=None falls back to a default train config and builds a model."""
        mc = RFDETRNanoConfig(num_classes=80)
        model = build_model_from_config(mc, train_config=None)
        assert model is not None, "Expected a model, got None"


@_requires_config_builders
class TestBuildCriterionFromConfig:
    """Tests for build_criterion_from_config(model_config, train_config, defaults)."""

    def test_returns_criterion_and_postprocess_tuple(self) -> None:
        """build_criterion_from_config must return a 2-tuple (SetCriterion, PostProcess)."""
        mc = RFDETRNanoConfig(num_classes=80)
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
        mc = RFDETRNanoConfig(num_classes=80)
        tc = TrainConfig(dataset_dir="/tmp")
        custom_defaults = replace(MODEL_DEFAULTS, focal_alpha=0.5)
        criterion, _ = build_criterion_from_config(mc, tc, defaults=custom_defaults)
        assert criterion.focal_alpha == pytest.approx(0.5), f"Expected focal_alpha=0.5, got {criterion.focal_alpha}"

    @pytest.mark.parametrize("eval_masks_head_resolution", [True, False])
    def test_eval_masks_head_resolution_controls_postprocess_upsample(self, eval_masks_head_resolution: bool) -> None:
        """TrainConfig.eval_masks_head_resolution disables PostProcess mask upsampling; the False default keeps it
        on."""
        mc = RFDETRSegNanoConfig()
        tc = SegmentationTrainConfig(dataset_dir="/tmp", eval_masks_head_resolution=eval_masks_head_resolution)
        _, postprocess = build_criterion_from_config(mc, tc)
        assert postprocess.upsample_masks_to_image_size is (not eval_masks_head_resolution)


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


class TestTrainingModelChannels:
    """The model that ``train()`` optimizes takes ``num_channels`` inputs, like the prediction model."""

    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_training_module_stem_matches_config(self, num_channels: int, tmp_path: Path) -> None:
        """``RFDETRModelModule`` widens its own model, which is built separately from the prediction one."""
        model_config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=None, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)

        module = RFDETRModelModule(model_config, train_config)

        assert _stem(module.model).in_channels == num_channels

    @pytest.mark.parametrize(("freeze_encoder", "warnings"), [(True, 1), (False, 0)])
    def test_training_module_reports_frozen_stem(
        self, freeze_encoder: bool, warnings: int, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """A frozen stem of the training model warns, since its filters for the extra channels are not trained."""
        warning = mocker.patch("rfdetr.training.module_model.logger.warning")
        model_config = RFDETRNanoConfig(
            num_channels=4, freeze_encoder=freeze_encoder, pretrain_weights=None, device="cpu"
        )
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)

        RFDETRModelModule(model_config, train_config)

        frozen = [call for call in warning.call_args_list if "patch embedding is frozen" in call.args[0]]
        assert len(frozen) == warnings


class TestPretrainedCheckpointAdaptation:
    """A 3-channel pretrained checkpoint is loaded as it is, then its stem is adapted to ``num_channels``."""

    @pytest.mark.parametrize(
        ("num_channels", "expected_stem"),
        [
            pytest.param(1, lambda saved: saved.mean(dim=1, keepdim=True), id="mean-over-rgb"),
            pytest.param(4, lambda saved: torch.cat([saved, saved[:, :1]], dim=1) * 3.0 / 4.0, id="tiled-and-scaled"),
        ],
    )
    def test_three_channel_checkpoint_is_adapted_after_load(
        self, num_channels: int, expected_stem: Callable[[torch.Tensor], torch.Tensor], tmp_path: Path
    ) -> None:
        """Fine-tuning from the released 3-channel weights derives the stem filters from the saved RGB ones."""
        path, saved_stem = _save_checkpoint(tmp_path, 3)
        config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=path, device="cpu")

        model = _build_model_context(config).model

        torch.testing.assert_close(_stem(model).weight, expected_stem(saved_stem))


@_requires_kornia
class TestKorniaNormalizeChannels:
    """The Kornia normalizer of the training pipeline accepts tensors with the model's channel count."""

    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_normalizes_every_channel_with_cycled_imagenet_statistics(self, num_channels: int) -> None:
        """A 4-channel batch cycles the ImageNet statistics, a 1-channel one uses the red statistics."""
        normalize = build_normalize(
            mean=cycle_channel_stats(IMAGENET_MEAN, num_channels), std=cycle_channel_stats(IMAGENET_STD, num_channels)
        )
        batch = torch.full((2, num_channels, 8, 8), 0.5)

        normalized = normalize(batch)

        mean = torch.tensor([IMAGENET_MEAN[index % 3] for index in range(num_channels)]).view(1, -1, 1, 1)
        std = torch.tensor([IMAGENET_STD[index % 3] for index in range(num_channels)]).view(1, -1, 1, 1)
        torch.testing.assert_close(normalized, ((batch - mean) / std))


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


_XLARGE_SYMBOLS = frozenset({"RFDETR2XLarge", "RFDETRXLarge"})


_PE_SYMBOLS = frozenset({"RFDETRAtto", "RFDETRFemto", "RFDETRPico"})


def _load_platform_models(
    monkeypatch: pytest.MonkeyPatch,
    plus_symbols: frozenset[str],
    pe_error: ImportError | None = None,
) -> types.ModuleType:
    """Execute a fresh copy of ``rfdetr.platform.models`` against a fake ``rfdetr_plus``.

    The shipped module decides what to export while it is imported, so each case needs its own copy; the fake
    package stands in for whichever ``rfdetr_plus`` release the case describes.

    Args:
        monkeypatch: Patches ``sys.modules`` and the plus-availability flag for the duration of the test.
        plus_symbols: Model names the fake ``rfdetr_plus.models`` provides.
        pe_error: When given, the fake ``rfdetr_plus.models`` raises it on any PE-model attribute lookup, like a
            plus release whose PE import fails on a broken dependency.

    Returns:
        The freshly executed module, not registered in ``sys.modules``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as patch:
        ...     module = _load_platform_models(patch, frozenset({"RFDETRXLarge", "RFDETR2XLarge"}))
        ...     sorted(module.__all__)
        ['RFDETR2XLarge', 'RFDETRXLarge']
    """
    fake_models = types.ModuleType("rfdetr_plus.models")
    for symbol in plus_symbols:
        setattr(fake_models, symbol, type(symbol, (), {}))
    if pe_error is not None:

        def _raise_for_pe(name: str) -> object:
            if name in _PE_SYMBOLS:
                raise pe_error
            raise AttributeError(name)

        fake_models.__getattr__ = _raise_for_pe  # type: ignore[attr-defined]
    fake_package = types.ModuleType("rfdetr_plus")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rfdetr_plus", fake_package)
    monkeypatch.setitem(sys.modules, "rfdetr_plus.models", fake_models)
    monkeypatch.setattr("rfdetr.platform._IS_RFDETR_PLUS_AVAILABLE", True)

    spec = importlib.util.spec_from_file_location(
        "_rfdetr_platform_models_under_test", Path(platform_models_module.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPlusImportBlock:
    """What ``rfdetr.platform.models`` exports for each rfdetr_plus release shape."""

    def test_current_rfdetr_plus_exports_every_plus_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A current rfdetr_plus exposes every name in ``_PLUS_EXPORTS``."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS | _PE_SYMBOLS)

        assert set(module.__all__) == _PLUS_EXPORTS

    def test_rfdetr_plus_predating_pe_models_keeps_xlarge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A release that ships XLarge but not the PE models still exports XLarge.

        The ``rfdetr[plus]`` floor admits such releases, so the PE import failing must not hide the XLarge models.
        """
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS)

        assert set(module.__all__) == _XLARGE_SYMBOLS
        assert module.RFDETRXLarge.__name__ == "RFDETRXLarge"

    def test_failed_pe_import_keeps_xlarge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A PE import that fails for another reason also leaves the XLarge exports intact."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS, pe_error=ImportError("broken dependency"))

        assert set(module.__all__) == _XLARGE_SYMBOLS


class TestMissingPeModelAccess:
    """Accessing a PE model the installed rfdetr_plus lacks."""

    @pytest.mark.parametrize("symbol", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_old_rfdetr_plus_raises_upgrade_hint(self, monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
        """Asking an old rfdetr_plus for a PE model raises the upgrade hint, not a bare AttributeError."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS)

        with pytest.raises(ImportError) as raised:
            getattr(module, symbol)

        assert "predates it" in str(raised.value)

    def test_upgrade_hint_chains_the_failed_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The upgrade hint carries the swallowed import error as its cause.

        A PE import that fails on a broken dependency reads like an outdated install; the chained cause is what tells
        the two apart in the traceback.
        """
        cause = ImportError("broken dependency")
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS, pe_error=cause)

        with pytest.raises(ImportError) as raised:
            module.RFDETRAtto

        assert "predates it" in str(raised.value)
        assert raised.value.__cause__ is cause
