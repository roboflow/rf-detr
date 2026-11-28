# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Direct unit tests for the hardcoded architectural constants in ``rfdetr.models._defaults``.

``ModelDefaults`` and ``MODEL_DEFAULTS`` are otherwise only exercised indirectly, through model construction and config
default resolution elsewhere in the suite. These tests pin the contract of the module itself: its default field values,
the ``MODEL_DEFAULTS`` singleton, and the frozen/slotted immutability that keeps those constants from being mutated at
runtime.

Regression tests for _namespace_from_configs() config forwarding.

Regression tests for keypoint config defaults and namespace forwarding.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    KeypointTrainConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.models import build_model
from rfdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults
from rfdetr.models._types import BuilderArgs

#: Expected default value for every ``ModelDefaults`` field, keyed by name. Mirrors ``rfdetr/models/_defaults.py``.
_EXPECTED_DEFAULTS: dict[str, object] = {
    "drop_mode": "standard",
    "drop_schedule": "constant",
    "cutoff_epoch": 0,
    "pretrained_encoder": None,
    "pretrain_exclude_keys": None,
    "pretrain_keys_modify_to_load": None,
    "pretrained_distiller": None,
    "vit_encoder_num_layers": 12,
    "window_block_indexes": None,
    "position_embedding": "sine",
    "rms_norm": False,
    "force_no_pretrain": False,
    "dim_feedforward": 2048,
    "decoder_norm": "LN",
    "freeze_batch_norm": False,
    "use_cls_token": False,
    "encoder_only": False,
    "backbone_only": False,
    "aux_loss": True,
    "focal_alpha": 0.25,
    "set_cost_class": 2.0,
    "set_cost_bbox": 5.0,
    "set_cost_giou": 2.0,
    "bbox_loss_coef": 5.0,
    "giou_loss_coef": 2.0,
    "sum_group_losses": False,
    "use_varifocal_loss": False,
    "use_position_supervised_loss": False,
    "print_freq": 10,
    "do_benchmark": False,
    "dropout": 0.0,
    "coco_path": None,
    "dont_save_weights": False,
    "start_epoch": 0,
    "eval": False,
    "world_size": 1,
    "dist_url": "env://",
    "lr_scheduler": "step",
    "lr_min_factor": 0.0,
    "subcommand": None,
}


class TestModelDefaultsFieldValues:
    """Contract tests for ``ModelDefaults``' declared field defaults."""

    def test_default_field_values_match_expected_constants(self) -> None:
        """A freshly constructed ``ModelDefaults`` matches the full expected default mapping.

        Guards against silent drift: a renamed field, a changed default, or a newly added field left out of
        ``_EXPECTED_DEFAULTS`` all fail this single dict comparison, instead of surfacing later as an unrelated
        model-construction test failure far from the actual cause.
        """
        defaults = ModelDefaults()

        assert dataclasses.asdict(defaults) == _EXPECTED_DEFAULTS

    def test_declared_fields_match_expected_default_keys(self) -> None:
        """The dataclass declares exactly the fields ``_EXPECTED_DEFAULTS`` enumerates, no more, no less.

        Catches a field added to or removed from ``ModelDefaults`` without a matching update to this test file's
        expectations, which the value-equality test above cannot distinguish from a field whose default simply changed.
        """
        field_names = {field.name for field in dataclasses.fields(ModelDefaults)}

        assert field_names == set(_EXPECTED_DEFAULTS)


class TestModelDefaultsSingleton:
    """Contract tests for the module-level ``MODEL_DEFAULTS`` singleton."""

    def test_singleton_equals_a_freshly_constructed_instance(self) -> None:
        """``MODEL_DEFAULTS`` is a plain default-valued ``ModelDefaults``, not a customized instance.

        Callers such as ``build_namespace()`` and ``RFDETR.__init__`` rely on ``MODEL_DEFAULTS`` as the implicit default
        for the ``defaults`` parameter; this pins that it carries no overrides.
        """
        assert MODEL_DEFAULTS == ModelDefaults()
        assert isinstance(MODEL_DEFAULTS, ModelDefaults)


class TestModelDefaultsImmutability:
    """Contract tests for the frozen, slotted nature of ``ModelDefaults``."""

    def test_setting_a_declared_field_raises_frozen_instance_error(self) -> None:
        """Reassigning an existing field on a constructed instance is rejected.

        ``ModelDefaults`` is ``frozen=True`` specifically so the shared ``MODEL_DEFAULTS`` singleton cannot be mutated
        by one caller and silently affect every other caller that reads it.
        """
        defaults = ModelDefaults()

        with pytest.raises(dataclasses.FrozenInstanceError):
            defaults.drop_mode = "custom"

    def test_setting_an_undeclared_attribute_raises_type_error(self) -> None:
        """Assigning an attribute the dataclass never declared is also rejected, not silently accepted.

        The exception is CPython-version-dependent: before 3.13 a stale-``cls`` bug in the frozen dataclass's generated
        ``__setattr__`` raises plain ``TypeError``; 3.13+ raises ``FrozenInstanceError`` as for a declared field. Both
        are accepted so the test passes on Python 3.10-3.14; either way, no attribute may be silently added.
        """
        defaults = ModelDefaults()

        with pytest.raises((TypeError, dataclasses.FrozenInstanceError)):
            defaults.not_a_declared_field = "value"  # type: ignore[attr-defined]


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


def test_keypoint_config_defaults() -> None:
    """Default model/train keypoint configuration values should match the preview contract.

    ``keypoint_nll_loss_coef`` defaults to 1.0 (a 0.5 default dampening OKS@75 oscillation was reverted to align with
    the other keypoint loss terms), so this guards against silently reintroducing it.
    """
    model = RFDETRKeypointPreviewConfig()
    train = KeypointTrainConfig(dataset_dir="/tmp")

    assert model.use_grouppose_keypoints is True
    assert model.dual_projector is True
    assert model.dual_projector_kp_only is True
    assert model.num_keypoints_per_class == [17]
    assert model.positional_encoding_size == 576 // 12

    assert train.keypoint_l1_loss_coef == pytest.approx(1.0)
    assert train.keypoint_findable_loss_coef == pytest.approx(1.0)
    assert train.keypoint_visible_loss_coef == pytest.approx(1.0)
    assert train.keypoint_nll_loss_coef == pytest.approx(1.0)
    assert train.cls_loss_coef == pytest.approx(2.0)


def test_keypoint_preview_config_person_schema() -> None:
    """Person-keypoint preview config must expose a person-only schema."""
    model = RFDETRKeypointPreviewConfig()

    assert model.num_keypoints_per_class == [17]
    assert sum(model.num_keypoints_per_class) == 17
    assert model.out_feature_indexes == [3, 6, 9, 12]
    assert model.num_windows == 2
    assert model.dec_layers == 4
    assert model.patch_size == 12
    assert model.resolution == 576
    assert model.pretrain_weights == "rf-detr-keypoint-preview-xlarge.pth"


def test_keypoint_fields_propagate_to_namespace(tmp_path) -> None:
    """All keypoint config fields are forwarded through _namespace_from_configs."""
    model = RFDETRKeypointPreviewConfig()
    train = KeypointTrainConfig(
        dataset_dir=str(tmp_path),
        keypoint_flip_pairs=[0, 1, 2, 3],
        keypoint_l1_loss_coef=1.5,
        keypoint_findable_loss_coef=2.5,
        keypoint_visible_loss_coef=3.5,
        keypoint_nll_loss_coef=4.5,
    )

    namespace = _namespace_from_configs(model, train)

    assert namespace.use_grouppose_keypoints is True
    assert namespace.keypoint_cross_attn is True
    assert namespace.inter_instance_kp_attn is False
    assert namespace.grouppose_keypoint_dim_downscale == 1
    assert namespace.dual_projector is True
    assert namespace.dual_projector_kp_only is True
    assert namespace.num_keypoints_per_class == [17]
    assert namespace.keypoint_flip_pairs == [0, 1, 2, 3]
    assert namespace.keypoint_l1_loss_coef == pytest.approx(1.5)
    assert namespace.keypoint_findable_loss_coef == pytest.approx(2.5)
    assert namespace.keypoint_visible_loss_coef == pytest.approx(3.5)
    assert namespace.keypoint_nll_loss_coef == pytest.approx(4.5)


def test_segmentation_train_config_cls_loss_coef_default() -> None:
    """SegmentationTrainConfig.cls_loss_coef must default to 1.0, not the erroneous 5.0.

    The 5.0 value was always present in SegmentationTrainConfig but was dead code pre-v1.7 (namespace builder read from
    ModelConfig=1.0).  The v1.7 TrainConfig ownership migration silently activated it.  This test guards against re-
    introducing that regression.
    """
    tc = SegmentationTrainConfig(dataset_dir="/tmp")
    assert tc.cls_loss_coef == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("config_class", "kwargs"),
    [
        pytest.param(RFDETRBaseConfig, {"num_classes": 1, "keypoint_private_hidden_dim": 256}, id="model-config"),
        # KeypointTrainConfig uses extra="forbid": unknown kwargs raise instead of being silently dropped.
        pytest.param(
            KeypointTrainConfig,
            {"dataset_dir": "/tmp", "keypoint_private_hidden_dim": 256, "keypoint_private_loss_coef": 1.0},
            id="train-config",
        ),
    ],
)
def test_unknown_keypoint_fields_are_not_public_config_fields(config_class, kwargs) -> None:
    """Private keypoint implementation fields are not accepted as public model or train config."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        config_class(**kwargs)
