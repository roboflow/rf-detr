# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the legacy checkpoint generation utility.

Covers the compat shims, ``_get_state_dict``, ``_get_patch_size``, ``_build_model``, and an integration smoke-test for
``generate_checkpoint()``.
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import pytorch_utils

from tests.legacy.generate_checkpoint import (
    _build_model,
    _get_patch_size,
    _get_state_dict,
    _install_transformers_compat,
    generate_checkpoint,
)


def _get_transformers_backbone_utils() -> ModuleType:
    """Import deprecated transformers backbone module without test warnings.

    Returns:
        Imported ``transformers.utils.backbone_utils`` module.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing `Backbone.*` from `utils/backbone_utils.py` is deprecated.*",
            category=FutureWarning,
        )
        return importlib.import_module("transformers.utils.backbone_utils")


def test_transformers_compat_installs_missing_prune_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """RF-DETR 1.4 imports the old pruning helper from transformers.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            the historical public helper.
    """
    monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)

    _install_transformers_compat()

    helper: Callable[[set[int], int, int, set[int]], tuple[set[int], torch.LongTensor]] = getattr(
        pytorch_utils, "find_pruneable_heads_and_indices"
    )
    heads, index = helper({1}, 4, 3, set())

    assert heads == {1}
    assert index.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]


def test_transformers_compat_installs_missing_backbone_alignment_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """RF-DETR 1.4 imports the old backbone alignment helper from transformers.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            the historical public helper.
    """
    backbone_utils = _get_transformers_backbone_utils()
    monkeypatch.delattr(backbone_utils, "get_aligned_output_features_output_indices", raising=False)

    _install_transformers_compat()

    helper: Callable[[list[str] | None, list[int] | tuple[int, ...] | None, list[str]], tuple[list[str], list[int]]]
    helper = getattr(backbone_utils, "get_aligned_output_features_output_indices")

    features, indices = helper(None, (1, 2), ["stem", "layer1", "layer2"])

    assert features == ["layer1", "layer2"]
    assert indices == [1, 2]


def test_transformers_compat_installs_init_backbone_on_backbone_mixin(monkeypatch: pytest.MonkeyPatch) -> None:
    """RF-DETR 1.4 calls super()._init_backbone(config) on its DINOv2 backbone.

    Transformers v5 removed BackboneMixin._init_backbone; the shim must restore
    it and populate stage_names, _out_features, _out_indices on the instance.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            _init_backbone from BackboneMixin.
    """
    backbone_module = importlib.import_module("transformers.backbone_utils")
    backbone_mixin_cls = getattr(backbone_module, "BackboneMixin", None)
    if backbone_mixin_cls is None:
        pytest.skip("BackboneMixin not available in installed transformers")

    monkeypatch.delattr(backbone_mixin_cls, "_init_backbone", raising=False)
    assert not hasattr(backbone_mixin_cls, "_init_backbone"), "precondition: method removed"

    _install_transformers_compat()

    assert hasattr(backbone_mixin_cls, "_init_backbone"), "shim must add _init_backbone to BackboneMixin"

    stage_names = ["stem", "stage1", "stage2"]
    config = SimpleNamespace(
        stage_names=stage_names,
        out_features=["stage1", "stage2"],
        out_indices=[1, 2],
    )
    obj: Any = object.__new__(backbone_mixin_cls)
    backbone_mixin_cls._init_backbone(obj, config)

    assert obj.stage_names == stage_names
    assert obj._out_features == ["stage1", "stage2"]
    assert obj._out_indices == [1, 2]


# ---------------------------------------------------------------------------
# _get_state_dict
# ---------------------------------------------------------------------------


class TestGetStateDict:
    """Unit tests for _get_state_dict — extracts state dict from rfdetr facade."""

    def test_returns_state_dict_from_current_three_level_layout(self) -> None:
        """Returns dict from model.model.model when present and non-empty.

        Current layout: RFDETR -> .model (Model) -> .model (nn.Module).
        """
        expected = {"weight": torch.zeros(3)}
        inner = MagicMock()
        inner.model.state_dict.return_value = expected
        model = SimpleNamespace(model=inner)

        assert _get_state_dict(model) == expected

    def test_falls_back_to_legacy_two_level_layout(self) -> None:
        """Falls back to model.model.state_dict() when three-level returns empty.

        Legacy layout: RFDETR -> .model (nn.Module directly).
        """
        expected = {"weight": torch.zeros(3)}
        inner = MagicMock()
        inner.model.state_dict.return_value = {}  # three-level empty -> fallback
        inner.state_dict.return_value = expected
        model = SimpleNamespace(model=inner)

        assert _get_state_dict(model) == expected

    def test_raises_runtime_error_when_no_layout_yields_state_dict(self) -> None:
        """RuntimeError raised when neither layout produces a non-empty state dict."""
        with pytest.raises(RuntimeError, match="Cannot extract state_dict"):
            _get_state_dict(object())


# ---------------------------------------------------------------------------
# _get_patch_size
# ---------------------------------------------------------------------------


class TestGetPatchSize:
    """Unit tests for _get_patch_size — reads patch_size from rfdetr facade."""

    def test_reads_model_config_patch_size(self) -> None:
        """Returns patch_size from model.model_config.patch_size when present."""
        model = SimpleNamespace(model_config=SimpleNamespace(patch_size=32))
        assert _get_patch_size(model) == 32

    def test_falls_back_to_config_patch_size(self) -> None:
        """Falls back to model.config.patch_size when model_config is absent."""
        model = SimpleNamespace(config=SimpleNamespace(patch_size=14))
        assert _get_patch_size(model) == 14

    def test_defaults_to_16_when_no_patch_size_found(self) -> None:
        """Returns the hard-coded default of 16 when no attribute path resolves."""
        assert _get_patch_size(object()) == 16


# ---------------------------------------------------------------------------
# _build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    """Unit tests for _build_model — instantiates rfdetr model with fallback."""

    def test_falls_through_to_rfdetr_base_when_preferred_missing(self) -> None:
        """Falls back to RFDETRBase when the preferred class is absent.

        Simulates a release that ships RFDETRBase but not the caller's preferred class (e.g. RFDETRSmall added later).
        """
        fake_rfdetr = types.ModuleType("rfdetr")
        fake_instance = object()
        mock_base = MagicMock(return_value=fake_instance)
        fake_rfdetr.RFDETRBase = mock_base  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            result = _build_model("NonExistentClass", num_classes=2, device="cpu")

        mock_base.assert_called_once_with(pretrain_weights=None, num_classes=2, device="cpu")
        assert result is fake_instance

    def test_raises_when_all_candidate_classes_missing(self) -> None:
        """RuntimeError when neither preferred class, RFDETRBase, nor RFDETR available."""
        fake_rfdetr = types.ModuleType("rfdetr")

        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            with pytest.raises(RuntimeError, match="Could not instantiate"):
                _build_model("NonExistentClass", num_classes=2, device="cpu")


# ---------------------------------------------------------------------------
# generate_checkpoint (integration)
# ---------------------------------------------------------------------------


class TestGenerateCheckpoint:
    """Integration tests for generate_checkpoint() against the installed rfdetr."""

    def test_output_file_has_required_keys(self, tmp_path: Path) -> None:
        """generate_checkpoint writes a .pth with model/args/epoch/rfdetr_version keys.

        Uses the currently installed rfdetr (dev source) with pretrain_weights=None
        so no remote download is performed.

        Args:
            tmp_path: Pytest-provided temporary directory.
        """
        out = tmp_path / "checkpoint_test.pth"
        generate_checkpoint(str(out))

        assert out.is_file(), "checkpoint file must be created at the specified path"

        ckpt: dict = torch.load(out, map_location="cpu", weights_only=False)

        assert "model" in ckpt, f"'model' key missing; present keys: {list(ckpt)}"
        assert isinstance(ckpt["model"], dict), "'model' value must be a state-dict (dict)"
        assert ckpt["model"], "'model' state dict must not be empty"
        assert "args" in ckpt, f"'args' key missing; present keys: {list(ckpt)}"
        assert "epoch" in ckpt, f"'epoch' key missing; present keys: {list(ckpt)}"
        assert "rfdetr_version" in ckpt, f"'rfdetr_version' key missing; present keys: {list(ckpt)}"
