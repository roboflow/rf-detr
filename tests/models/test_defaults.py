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
"""

from __future__ import annotations

import dataclasses

import pytest

from rfdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults

#: Expected default value for every ``ModelDefaults`` field, keyed by field name. Mirrors the field
#: declarations in ``src/rfdetr/models/_defaults.py`` so a drift in either place fails this test.
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

        Guards against silent drift: a renamed field, a changed default, or a newly added field
        left out of ``_EXPECTED_DEFAULTS`` all fail this single dict comparison, instead of surfacing
        later as an unrelated model-construction test failure far from the actual cause.
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

        The frozen dataclass's generated ``__setattr__`` intercepts every assignment before ``slots=True`` would
        otherwise raise ``AttributeError`` for an undeclared name. Which exception surfaces for an *undeclared* name
        is CPython-version-dependent: pre-3.13 has a stale-``cls``-reference bug in the generated ``__setattr__`` that
        raises plain ``TypeError`` instead of the intended ``dataclasses.FrozenInstanceError``; 3.13+ fixes it, so the
        undeclared-name path raises ``FrozenInstanceError`` just like the declared-field path above. Both are accepted
        here so the test passes across this project's whole supported range (Python 3.10-3.14); either way, the
        assignment must not silently grow an instance dict on ``ModelDefaults``.
        """
        defaults = ModelDefaults()

        with pytest.raises((TypeError, dataclasses.FrozenInstanceError)):
            defaults.not_a_declared_field = "value"  # type: ignore[attr-defined]
