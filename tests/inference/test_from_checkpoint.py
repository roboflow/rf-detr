# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETR.from_checkpoint classmethod.

Most tests isolate the inference logic by patching ``torch.load`` and the target model class inside ``rfdetr.variants``
(or ``rfdetr.platform.models`` for plus models). The round-trip cases in ``TestFromCheckpointStrippedBestTotal`` instead
build a real, small CPU model (Nano or keypoint preview) and run it through a real ``strip_checkpoint`` +
``from_checkpoint`` cycle, to prove a working model actually comes out the other end. Either way, no model weights are
downloaded and no GPU memory is allocated.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rfdetr.config import PretrainWeightsCompatibilityWarning
from rfdetr.detr import _CHECKPOINT_PLUS_MODEL_MAP_ENTRIES, _CHECKPOINT_PLUS_STEM_ENTRIES, RFDETR
from rfdetr.detr import logger as detr_logger
from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE, _PLUS_EXPORTS
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.utilities.state_dict import strip_checkpoint
from rfdetr.variants import RFDETRKeypointPreview, RFDETRNano, RFDETRSmall


class _CustomObj:
    """Module-level class that weights_only=True rejects (not in safe globals).

    Must be at module scope so pickle can resolve the fully-qualified name during torch.save.  Local/nested classes
    cannot be pickled.
    """


def _ns(pretrain_weights: str, num_classes: int = 80) -> dict:
    """Fake legacy checkpoint with argparse.Namespace args.

    Examples:
        >>> ckpt = _ns("rf-detr-small.pth", num_classes=3)
        >>> ckpt["args"].num_classes
        3
        >>> ckpt["args"].pretrain_weights
        'rf-detr-small.pth'
    """
    return {"args": argparse.Namespace(pretrain_weights=pretrain_weights, num_classes=num_classes)}


def _dict(pretrain_weights: str, num_classes: int = 80) -> dict:
    """Fake PTL-style checkpoint with dict args.

    Examples:
        >>> ckpt = _dict("rf-detr-small.pth", num_classes=5)
        >>> ckpt["args"]["num_classes"]
        5
        >>> ckpt["args"]["pretrain_weights"]
        'rf-detr-small.pth'
    """
    return {"args": {"pretrain_weights": pretrain_weights, "num_classes": num_classes}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_from_checkpoint(ckpt: dict, path: Path, cls_patch_target: str, **kwargs):
    """Invoke RFDETR.from_checkpoint with torch.load mocked to return *ckpt* and the model class at *cls_patch_target*
    replaced by a MagicMock.

    Returns:
        Tuple of (result, mock_class).

    Examples:
        This helper patches ``torch.load`` and a model class — it cannot be run without a real
        ``Path`` argument or live imports, so the example below is illustrative only.

        callable(_call_from_checkpoint)
        # True
    """
    mock_instance = MagicMock()
    with (
        patch("rfdetr.detr.torch.load", return_value=ckpt),
        patch(cls_patch_target) as mock_cls,
    ):
        mock_cls.return_value = mock_instance
        result = RFDETR.from_checkpoint(path, **kwargs)
    return result, mock_cls


# ---------------------------------------------------------------------------
# Namespace args (legacy .pth checkpoints)
# ---------------------------------------------------------------------------


class TestFromCheckpointNamespaceArgs:
    """from_checkpoint with argparse.Namespace args (legacy engine.py format)."""

    @pytest.mark.parametrize(
        ("pretrain_weights, patch_target"),
        [
            ("rf-detr-nano.pth", "RFDETRNano"),
            ("rf-detr-small.pth", "RFDETRSmall"),
            ("rf-detr-medium.pth", "RFDETRMedium"),
            ("rf-detr-large.pth", "RFDETRLarge"),
            ("rf-detr-keypoint-preview-xlarge.pth", "RFDETRKeypointPreview"),
            ("rf-detr-base.pth", "RFDETRBase"),
            ("rf-detr-seg-nano.pt", "RFDETRSegNano"),
            ("rf-detr-seg-small.pt", "RFDETRSegSmall"),
            ("rf-detr-seg-medium.pt", "RFDETRSegMedium"),
            ("rf-detr-seg-large.pt", "RFDETRSegLarge"),
            ("rf-detr-seg-xlarge.pt", "RFDETRSegXLarge"),
            ("rf-detr-seg-xxlarge.pt", "RFDETRSeg2XLarge"),
            ("rf-detr-seg-preview.pt", "RFDETRSegPreview"),
        ],
    )
    def test_characterization_infers_correct_class_namespace(
        self,
        tmp_path: Path,
        pretrain_weights: str,
        patch_target: str,
    ) -> None:
        """Namespace-style args: correct subclass is called for each model size."""
        result, mock_cls = _call_from_checkpoint(
            _ns(pretrain_weights), tmp_path / "ckpt.pth", f"rfdetr.variants.{patch_target}"
        )

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("num_classes") == 80
        assert call_kwargs.get("pretrain_weights") == str(tmp_path / "ckpt.pth")
        assert result is mock_cls.return_value

    @pytest.mark.parametrize(
        "missing_value",
        [
            pytest.param("none", id="bare-none"),
            pytest.param("null", id="bare-null"),
            pytest.param("", id="empty"),
            pytest.param("  None  ", id="whitespace-None"),
            pytest.param("  ", id="whitespace-only"),
            pytest.param(" null ", id="whitespace-null"),
            pytest.param(None, id="python-None"),
        ],
    )
    def test_namespace_args_falls_back_to_checkpoint_filename_when_pretrain_weights_missing(
        self, tmp_path: Path, missing_value: str | None
    ) -> None:
        """Namespace args: filename fallback fires when pretrain_weights is unset-like."""
        ckpt = _ns(missing_value)  # type: ignore[arg-type]
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "rf-detr-small.pth", "rfdetr.variants.RFDETRSmall")
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["num_classes"] == 80


# ---------------------------------------------------------------------------
# Dict args (PTL / converted checkpoints)
# ---------------------------------------------------------------------------


class TestFromCheckpointDictArgs:
    """from_checkpoint with dict-style args (PTL or convert_legacy_checkpoint output)."""

    @pytest.mark.parametrize(
        ("pretrain_weights, patch_target"),
        [
            ("rf-detr-small.pth", "RFDETRSmall"),
            ("rf-detr-base.pth", "RFDETRBase"),
        ],
    )
    def test_characterization_infers_correct_class_dict(
        self,
        tmp_path: Path,
        pretrain_weights: str,
        patch_target: str,
    ) -> None:
        """Dict-style args: correct subclass is called without AttributeError."""
        _, mock_cls = _call_from_checkpoint(
            _dict(pretrain_weights), tmp_path / "ckpt.pth", f"rfdetr.variants.{patch_target}"
        )

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("num_classes") == 80

    def test_characterization_dict_args_missing_num_classes_uses_default(self, tmp_path: Path) -> None:
        """Dict args without num_classes: constructor is called without num_classes kwarg."""
        ckpt = {"args": {"pretrain_weights": "rf-detr-small.pth"}}
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRSmall")

        call_kwargs = mock_cls.call_args.kwargs
        assert "num_classes" not in call_kwargs


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFromCheckpointEdgeCases:
    """Edge-case handling in from_checkpoint."""

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """from_checkpoint raises FileNotFoundError when path does not exist."""
        with pytest.raises(FileNotFoundError):
            RFDETR.from_checkpoint(tmp_path / "nope.pth")

    def test_directory_path_raises_os_error(self, tmp_path: Path) -> None:
        """from_checkpoint raises OSError when path is a directory, not a file."""
        with pytest.raises((OSError, IsADirectoryError)):
            RFDETR.from_checkpoint(tmp_path)

    def test_characterization_unknown_pretrain_weights_raises_value_error(self, tmp_path: Path) -> None:
        """Unrecognised pretrain_weights name raises a descriptive ValueError."""
        ckpt = _ns("/my/custom/finetuned.pth")
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="Could not infer model class"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_filename_fallback_unrecognized_name_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError fires via filename-fallback path when filename has no known model token."""
        ckpt = {"args": {"pretrain_weights": "none", "num_classes": 80}}
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="Could not infer model class"):
                RFDETR.from_checkpoint(tmp_path / "finetuned.pth")

    @pytest.mark.skipif(_IS_RFDETR_PLUS_AVAILABLE, reason="rfdetr_plus is installed — guard not active")
    def test_filename_fallback_xlarge_without_plus_raises_import_error(self, tmp_path: Path) -> None:
        """ImportError fires via filename-fallback path when rfdetr_plus is absent."""
        ckpt = {"args": {"pretrain_weights": "none", "num_classes": 80}}
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError):
                RFDETR.from_checkpoint(tmp_path / "rf-detr-xlarge-starter.pth")

    def test_missing_args_key_names_where_the_settings_are(self, tmp_path: Path) -> None:
        """A checkpoint recording no ``args`` is refused with the same guidance whatever its extension.

        The refusal used to require a ``pytorch-lightning_version`` key as well, which every rfdetr ``.pth`` writes and
        no converted ``.ckpt`` does, so files in between fell through to a bare ``KeyError: 'args'``.
        """
        ckpt = {"model": {}}
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="training_config.json"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_converted_legacy_checkpoint_is_refused_with_guidance(self, tmp_path: Path) -> None:
        """``convert_legacy_checkpoint`` output stores args under ``hyper_parameters``, so it names no model.

        Real files, no mocked loader: the converted ``.ckpt`` carries neither ``args`` nor ``pytorch-
        lightning_version``, which is exactly the combination that used to escape the check.
        """
        source = tmp_path / "legacy.pth"
        torch.save({"model": {"class_embed.weight": torch.zeros(4, 8)}, "args": {"num_classes": 3}}, source)
        converted = tmp_path / "converted.ckpt"
        convert_legacy_checkpoint(str(source), str(converted))

        with pytest.raises(ValueError, match="training_config.json"):
            RFDETR.from_checkpoint(converted)

    def test_lightning_ckpt_without_args_points_at_training_config(self, tmp_path: Path) -> None:
        """A ``.ckpt`` from 1.11.0 or earlier records no model; the error names where its settings are (#1552)."""
        ckpt = {"state_dict": {"model.class_embed.weight": torch.zeros(4, 8)}, "pytorch-lightning_version": "2.6.6"}
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="training_config.json"):
                RFDETR.from_checkpoint(tmp_path / "last.ckpt")

    def test_characterization_callable_on_subclass(self, tmp_path: Path) -> None:
        """from_checkpoint can be called on a concrete subclass (RFDETRSmall)."""
        mock_instance = MagicMock()
        with (
            patch("rfdetr.detr.torch.load", return_value=_ns("rf-detr-small.pth")),
            patch("rfdetr.variants.RFDETRSmall") as mock_cls,
        ):
            mock_cls.return_value = mock_instance
            result = RFDETRSmall.from_checkpoint(tmp_path / "ckpt.pth")

        assert result is mock_instance
        mock_cls.assert_called_once()

    def test_characterization_extra_kwargs_forwarded(self, tmp_path: Path) -> None:
        """Extra **kwargs are forwarded to the model constructor."""
        _, mock_cls = _call_from_checkpoint(
            _ns("rf-detr-small.pth"),
            tmp_path / "ckpt.pth",
            "rfdetr.variants.RFDETRSmall",
            resolution=640,
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("resolution") == 640

    def test_trust_checkpoint_true_forwarded_to_constructor(self, tmp_path: Path) -> None:
        """trust_checkpoint=True reaches the model constructor, not just the metadata read.

        Regression coverage: the constructor reloads the same checkpoint file via
        ``load_pretrain_weights`` — without forwarding ``trust_checkpoint`` into
        ``constructor_kwargs``, that reload always used the unsafe-load default, making
        ``trust_checkpoint=True`` inert for checkpoints that actually need it.
        """
        _, mock_cls = _call_from_checkpoint(
            _ns("rf-detr-small.pth"),
            tmp_path / "ckpt.pth",
            "rfdetr.variants.RFDETRSmall",
            trust_checkpoint=True,
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["trust_checkpoint"] is True

    def test_trust_checkpoint_defaults_to_false_in_constructor(self, tmp_path: Path) -> None:
        """trust_checkpoint defaults to False in the forwarded constructor kwargs."""
        _, mock_cls = _call_from_checkpoint(
            _ns("rf-detr-small.pth"),
            tmp_path / "ckpt.pth",
            "rfdetr.variants.RFDETRSmall",
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["trust_checkpoint"] is False

    def test_characterization_pretrain_weights_in_kwargs_is_overridden(self, tmp_path: Path) -> None:
        """pretrain_weights passed in **kwargs is silently overridden by the checkpoint path."""
        _, mock_cls = _call_from_checkpoint(
            _ns("rf-detr-small.pth"),
            tmp_path / "ckpt.pth",
            "rfdetr.variants.RFDETRSmall",
            pretrain_weights="/should/be/overridden.pth",
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["pretrain_weights"] == str(tmp_path / "ckpt.pth")

    def test_characterization_caller_num_classes_overrides_checkpoint(self, tmp_path: Path) -> None:
        """Caller-supplied num_classes takes precedence over the checkpoint's stored value."""
        _, mock_cls = _call_from_checkpoint(
            _ns("rf-detr-small.pth", num_classes=80),
            tmp_path / "ckpt.pth",
            "rfdetr.variants.RFDETRSmall",
            num_classes=5,
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["num_classes"] == 5

    def test_checkpoint_model_config_forwarded_to_constructor(self, tmp_path: Path) -> None:
        """Reload should preserve schema-dependent model config from PTL ``.pth`` checkpoints."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth", "num_classes": 1},
            "model_name": "RFDETRKeypointPreview",
            "model_config": {
                "num_keypoints_per_class": [0, 17],
                "use_grouppose_keypoints": True,
                "dual_projector": True,
                "pretrain_weights": "/old/path.pth",
            },
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt,
            tmp_path / "checkpoint_best_total.pth",
            "rfdetr.variants.RFDETRKeypointPreview",
        )

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["num_keypoints_per_class"] == [0, 17]
        assert call_kwargs["use_grouppose_keypoints"] is True
        assert call_kwargs["dual_projector"] is True
        assert call_kwargs["num_classes"] == 1
        assert call_kwargs["pretrain_weights"] == str(tmp_path / "checkpoint_best_total.pth")

    @pytest.mark.skipif(_IS_RFDETR_PLUS_AVAILABLE, reason="rfdetr_plus is installed — guard not active")
    def test_characterization_xlarge_without_plus_raises_import_error(self, tmp_path: Path) -> None:
        """Xlarge checkpoint without rfdetr_plus raises ImportError instead of wrong class."""
        for weights in ("rf-detr-xlarge.pth", "rf-detr-xxlarge.pth"):
            ckpt = _ns(weights)
            with patch("rfdetr.detr.torch.load", return_value=ckpt):
                with pytest.raises(ImportError):
                    RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_trust_gate_rejects_custom_class_by_default(self, tmp_path: Path) -> None:
        """from_checkpoint raises RuntimeError for custom-class checkpoints without trust_checkpoint=True.

        Scenario: user calls from_checkpoint on a file containing an unrecognised Python class.
        The default trust_checkpoint=False must reject it to prevent arbitrary code execution.
        """
        ckpt_path = tmp_path / "custom_obj.pth"
        torch.save({"model": {}, "args": _CustomObj(), "model_name": "RFDETRSmall"}, ckpt_path)

        with pytest.raises(RuntimeError, match="trust_checkpoint=True"):
            RFDETR.from_checkpoint(ckpt_path)


# ---------------------------------------------------------------------------
# Deprecated class instantiation
# ---------------------------------------------------------------------------


class TestDeprecatedClassInstantiation:
    """Deprecated model classes emit deprecation warnings on instantiation."""

    @pytest.mark.parametrize(
        ("cls_name, import_path"),
        [
            ("RFDETRBase", "rfdetr.variants.RFDETRBase"),
            ("RFDETRLargeDeprecated", "rfdetr.variants.RFDETRLargeDeprecated"),
            ("RFDETRSegPreview", "rfdetr.variants.RFDETRSegPreview"),
        ],
    )
    @patch("rfdetr.detr.RFDETR.__init__", return_value=None)
    def test_direct_instantiation_is_allowed(self, _mock_0, cls_name: str, import_path: str) -> None:
        """Direct instantiation of a deprecated class does not raise RuntimeError."""
        import importlib

        module_path, attr = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        model = cls()
        assert model.__class__.__name__ == cls_name

    @pytest.mark.parametrize("pretrain_weights", ["rf-detr-base.pth", "rf-detr-seg-preview.pt"])
    def test_from_checkpoint_resolves_deprecated_class(
        self,
        tmp_path: Path,
        pretrain_weights: str,
    ) -> None:
        """from_checkpoint still resolves deprecated classes without KeyError on minimal mocked checkpoints."""
        ckpt = _ns(pretrain_weights)
        with (
            patch("rfdetr.detr.torch.load", return_value=ckpt),
            patch("rfdetr.detr.RFDETR.__init__", return_value=None),
        ):
            model = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")
        assert model.__class__.__name__ in {"RFDETRBase", "RFDETRSegPreview"}


# ---------------------------------------------------------------------------
# model_name in checkpoint (#887)
# ---------------------------------------------------------------------------


def _ckpt_with_model_name(model_name: str, num_classes: int = 80) -> dict:
    """Fake checkpoint with model_name key (new format).

    Examples:
        >>> ckpt = _ckpt_with_model_name("RFDETRSmall", num_classes=2)
        >>> ckpt["model_name"]
        'RFDETRSmall'
        >>> ckpt["args"]["num_classes"]
        2
    """
    return {
        "args": {"pretrain_weights": "rf-detr-small.pth", "num_classes": num_classes},
        "model_name": model_name,
    }


class TestFromCheckpointModelName:
    """from_checkpoint uses model_name when present in checkpoint."""

    @pytest.mark.parametrize(
        ("model_name, patch_target"),
        [
            ("RFDETRNano", "RFDETRNano"),
            ("RFDETRSmall", "RFDETRSmall"),
            ("RFDETRMedium", "RFDETRMedium"),
            ("RFDETRLarge", "RFDETRLarge"),
            ("RFDETRKeypointPreview", "RFDETRKeypointPreview"),
            ("RFDETRBase", "RFDETRBase"),
            ("RFDETRSegNano", "RFDETRSegNano"),
            ("RFDETRSegPreview", "RFDETRSegPreview"),
            ("RFDETRSegSmall", "RFDETRSegSmall"),
            ("RFDETRSegMedium", "RFDETRSegMedium"),
            ("RFDETRSegLarge", "RFDETRSegLarge"),
            ("RFDETRSegXLarge", "RFDETRSegXLarge"),
            ("RFDETRSeg2XLarge", "RFDETRSeg2XLarge"),
        ],
    )
    def test_model_name_resolves_correct_class(self, tmp_path: Path, model_name: str, patch_target: str) -> None:
        """model_name in checkpoint maps directly to the correct subclass."""
        result, mock_cls = _call_from_checkpoint(
            _ckpt_with_model_name(model_name), tmp_path / "ckpt.pth", f"rfdetr.variants.{patch_target}"
        )
        mock_cls.assert_called_once()
        assert result is mock_cls.return_value

    def test_model_name_takes_priority_over_pretrain_weights(self, tmp_path: Path) -> None:
        """model_name is used even when pretrain_weights points to a different size."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-nano.pth", "num_classes": 80},
            "model_name": "RFDETRLarge",
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRLarge")
        mock_cls.assert_called_once()

    def test_falls_back_to_pretrain_weights_without_model_name(self, tmp_path: Path) -> None:
        """Old checkpoints without model_name still work via pretrain_weights parsing."""
        ckpt = _dict("rf-detr-small.pth")
        assert "model_name" not in ckpt
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRSmall")
        mock_cls.assert_called_once()

    @pytest.mark.parametrize(
        "missing_value",
        [
            pytest.param("none", id="bare-none"),
            pytest.param("null", id="bare-null"),
            pytest.param("", id="empty"),
            pytest.param("  None  ", id="whitespace-None"),
            pytest.param("  ", id="whitespace-only"),
            pytest.param(" null ", id="whitespace-null"),
            pytest.param(None, id="python-None"),
        ],
    )
    def test_falls_back_to_checkpoint_filename_when_pretrain_weights_missing(
        self, tmp_path: Path, missing_value: str | None
    ) -> None:
        """When pretrain_weights is missing-like, from_checkpoint infers class from checkpoint filename."""
        ckpt = {"args": {"pretrain_weights": missing_value, "num_classes": 80}}
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "rf-detr-small.pth", "rfdetr.variants.RFDETRSmall")
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["num_classes"] == 80

    def test_unknown_model_name_falls_back_to_pretrain_weights(self, tmp_path: Path) -> None:
        """Unrecognised model_name falls back to pretrain_weights parsing."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-small.pth", "num_classes": 80},
            "model_name": "UnknownModel",
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRSmall")
        mock_cls.assert_called_once()

    def test_model_name_with_whitespace_is_stripped(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace in model_name is stripped before class resolution."""
        ckpt = _ckpt_with_model_name("  RFDETRSmall  ")
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRSmall")
        mock_cls.assert_called_once()

    @pytest.mark.parametrize(
        "model_name, expected_class",
        [
            ("RFDETRBase", "RFDETRBase"),
            ("RFDETRSegPreview", "RFDETRSegPreview"),
        ],
    )
    def test_model_name_deprecated_class_resolves_and_instantiates(
        self, tmp_path: Path, model_name: str, expected_class: str
    ) -> None:
        """from_checkpoint resolves deprecated model_name values and instantiates the resolved class."""
        ckpt = _ckpt_with_model_name(model_name)
        with (
            patch("rfdetr.detr.torch.load", return_value=ckpt),
            patch("rfdetr.detr.RFDETR.__init__", return_value=None),
        ):
            model = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")
        assert model.__class__.__name__ == expected_class

    def test_large_deprecated_model_name_resolves_to_deprecated_class(self, tmp_path: Path) -> None:
        """Checkpoints saved with model_name='RFDETRLargeDeprecated' must load as RFDETRLargeDeprecated.

        Before the fix, RFDETRLargeDeprecated was absent from _name_map; the substring matcher would pick RFDETRLarge,
        which fails with a pydantic literal_error when the saved model_config carries encoder='dinov2_windowed_base'
        (only valid for the deprecated Large configuration).
        """
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-large.pth", "num_classes": 80},
            "model_name": "RFDETRLargeDeprecated",
            "model_config": {
                "encoder": "dinov2_windowed_base",
                "projector_scale": "P4",
            },
        }
        result, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRLargeDeprecated")
        mock_cls.assert_called_once()
        assert result is mock_cls.return_value

    @pytest.mark.skipif(_IS_RFDETR_PLUS_AVAILABLE, reason="rfdetr_plus is installed — guard not active")
    @pytest.mark.parametrize("model_name", ["RFDETRXLarge", "RFDETR2XLarge"])
    def test_plus_model_name_without_plus_raises_import_error(self, tmp_path: Path, model_name: str) -> None:
        """Plus checkpoints using model_name raise install guidance without rfdetr_plus."""
        ckpt = {
            "args": {"pretrain_weights": "", "num_classes": 80},
            "model_name": model_name,
        }
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError, match="rfdetr_plus package"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")


# ---------------------------------------------------------------------------
# num_classes provenance (fine-tuning a from_checkpoint model on a new dataset)
# ---------------------------------------------------------------------------


@pytest.fixture
def args_only_checkpoint(tmp_path: Path) -> Path:
    """Minimal checkpoint with num_classes in args only; model_config carries no num_classes key.

    Covers the legacy checkpoint format where num_classes is embedded in the args dict rather
    than in model_config.  from_checkpoint extracts it via the args path (detr.py:454-457) and
    injects it into constructor_kwargs — this fixture verifies that path also clears the
    Pydantic provenance marker.  Only exercises the args-injection path; model_config path
    covered by ``two_class_checkpoint``.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the saved checkpoint file.
    """
    path = tmp_path / "small_two_class_args_only.pth"
    torch.save(
        {
            "model": {"class_embed.bias": torch.zeros(3)},
            "model_name": "RFDETRSmall",
            "model_config": {},
            "args": {"class_names": ["cat", "dog"], "num_classes": 2},
        },
        path,
    )
    return path


@pytest.fixture
def two_class_checkpoint(tmp_path: Path) -> Path:
    """Save a minimal synthetic 2-class checkpoint to disk (no downloads, no real weights).

    Follows the lightweight checkpoint pattern used elsewhere in the suite (``test_detr_shim``,
    ``test_load_pretrain_weights``): write only what ``from_checkpoint``/``load_pretrain_weights`` actually inspect —
    the ``class_embed.bias`` tensor sized for 2 classes + background, plus the metadata used to resolve the model
    (``model_name``) and the class count (``model_config`` carrying ``num_classes=2``).  A *non-default*
    ``num_classes`` is what trips the user-override guards, so it is written explicitly rather than relying on a
    published checkpoint (whose default 90 would not trip them).  ``from_checkpoint`` still builds a real model from
    this, which is what the provenance and head-shape assertions exercise.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the saved checkpoint file.
    """
    path = tmp_path / "small_two_class.pth"
    torch.save(
        {
            "model": {"class_embed.bias": torch.zeros(3)},
            "model_name": "RFDETRSmall",
            "model_config": {"num_classes": 2},
            "args": {"class_names": ["cat", "dog"]},
        },
        path,
    )
    return path


class TestFromCheckpointNumClassesProvenance:
    """Checkpoint-derived num_classes must not be treated as a user override.

    Regression tests for https://github.com/roboflow/rf-detr/issues/1092: ``from_checkpoint`` copies ``num_classes``
    out of the checkpoint into the constructor kwargs, which used to mark the field as explicitly user-set.  Both
    provenance guards (``RFDETR._align_num_classes_from_dataset`` and the head re-init logic in
    ``rfdetr.models.weights.load_pretrain_weights``) then refused to adapt the detection head to a new dataset's
    class count, breaking fine-tuning from a checkpoint.
    """

    def test_checkpoint_num_classes_is_not_marked_user_set(self, two_class_checkpoint: Path) -> None:
        """from_checkpoint adopts the checkpoint class count without warning about pretrained weights."""
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=PretrainWeightsCompatibilityWarning)
            model = RFDETR.from_checkpoint(two_class_checkpoint)

        assert model.model_config.num_classes == 2
        assert model.model.model.class_embed.bias.shape[0] == 3, "Head must match checkpoint (2 classes + background)."
        assert "num_classes" not in model.model_config.model_fields_set, (
            "Checkpoint-derived num_classes must not be recorded as explicitly user-set; "
            "otherwise train() refuses to align the head to a new dataset's class count."
        )

    def test_train_alignment_adapts_head_to_new_dataset(
        self, two_class_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fine-tuning a from_checkpoint model on a dataset with a different class count adapts the head."""
        model = RFDETR.from_checkpoint(two_class_checkpoint)
        monkeypatch.setattr(RFDETR, "_detect_num_classes_for_training", staticmethod(lambda *a, **k: 5))

        model._align_num_classes_from_dataset("<five-class-dataset>")

        assert model.model_config.num_classes == 5
        assert model.model.args.num_classes == 5
        # train() rebuilds the model from model_config (inside RFDETRModelModule), reloading the checkpoint
        # weights with the aligned class count; the rebuilt head must adopt the dataset class count.
        rebuilt = model.get_model(model.model_config)
        assert rebuilt.model.class_embed.bias.shape[0] == 6, "Rebuilt head must have 5 classes + background."

    def test_explicit_num_classes_kwarg_still_wins(
        self,
        two_class_checkpoint: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An explicit num_classes kwarg to from_checkpoint stays authoritative over the dataset."""
        model = RFDETR.from_checkpoint(two_class_checkpoint, num_classes=7)

        assert model.model_config.num_classes == 7
        assert "num_classes" in model.model_config.model_fields_set
        assert model.model.model.class_embed.bias.shape[0] == 8, "Head must expand to 7 classes + background."

        monkeypatch.setattr(RFDETR, "_detect_num_classes_for_training", staticmethod(lambda *a, **k: 5))
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            model._align_num_classes_from_dataset("<five-class-dataset>")

        assert model.model_config.num_classes == 7, "Explicit user num_classes must be preserved."
        assert any("Using the model's configured value" in record.message for record in caplog.records)

    def test_checkpoint_num_classes_from_args_not_marked_user_set(self, args_only_checkpoint: Path) -> None:
        """num_classes injected from checkpoint args (not model_config) is cleared from model_fields_set."""
        model = RFDETR.from_checkpoint(args_only_checkpoint)

        assert model.model_config.num_classes == 2
        assert "num_classes" not in model.model_config.model_fields_set, (
            "num_classes from checkpoint args must not be recorded as explicitly user-set; "
            "otherwise train() refuses to adapt the head to a new dataset's class count."
        )

    def test_explicit_default_num_classes_pins_head(
        self,
        two_class_checkpoint: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Passing num_classes equal to the ModelConfig default still pins the detection head.

        An explicit num_classes is honored regardless of whether it equals the class default:
        ``_align_num_classes_from_dataset`` keys off whether the field was set, not whether the
        value differs from the default, so the dataset count cannot silently override it.  This
        guards against re-introducing the ``value != default`` clause, whose asymmetric behavior
        (default silently aligned, non-default preserved) was the bug this test now pins.
        """
        model = RFDETR.from_checkpoint(two_class_checkpoint)
        default_nc = type(model.model_config).model_fields["num_classes"].default
        # Simulate calling from_checkpoint(path, num_classes=<default>):
        # assigning the field adds "num_classes" to model_fields_set automatically (Pydantic v2).
        model.model_config.num_classes = default_nc

        assert "num_classes" in model.model_config.model_fields_set
        monkeypatch.setattr(RFDETR, "_detect_num_classes_for_training", staticmethod(lambda *a, **k: 5))
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            model._align_num_classes_from_dataset("<five-class-dataset>")

        assert model.model_config.num_classes == default_nc, (
            "Explicitly passing the ModelConfig default for num_classes must pin the head; "
            "the dataset class count must not silently override an explicit user setting."
        )
        assert any("Using the model's configured value" in record.message for record in caplog.records)

    def test_explicit_default_num_classes_via_from_checkpoint_integrated(
        self,
        two_class_checkpoint: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """from_checkpoint(path, num_classes=<default>) pins head via the integrated code path.

        Unlike test_explicit_default_num_classes_pins_head which simulates the explicit-default scenario via post-
        construction assignment, this test calls from_checkpoint directly with num_classes=default_nc.  A regression in
        how from_checkpoint passes num_classes into the constructor would be caught here but not by the proxy-based
        test.
        """
        default_nc = RFDETRSmall._model_config_class.model_fields["num_classes"].default
        model = RFDETR.from_checkpoint(two_class_checkpoint, num_classes=default_nc)

        assert model.model_config.num_classes == default_nc
        assert "num_classes" in model.model_config.model_fields_set, (
            "from_checkpoint with explicit num_classes must keep it in model_fields_set; "
            "only checkpoint-derived num_classes should be cleared."
        )

        monkeypatch.setattr(RFDETR, "_detect_num_classes_for_training", staticmethod(lambda *a, **k: 5))
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            model._align_num_classes_from_dataset("<five-class-dataset>")

        assert model.model_config.num_classes == default_nc, (
            "Head must remain pinned at default_nc after alignment; "
            "from_checkpoint-supplied num_classes must not be silently overridden."
        )
        assert any("Using the model's configured value" in record.message for record in caplog.records)

    def test_equal_class_count_does_not_rebuild_head(
        self, two_class_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpoint and dataset sharing the same class count leaves the head unchanged."""
        model = RFDETR.from_checkpoint(two_class_checkpoint)
        original_bias_shape = model.model.model.class_embed.bias.shape
        monkeypatch.setattr(RFDETR, "_detect_num_classes_for_training", staticmethod(lambda *a, **k: 2))

        model._align_num_classes_from_dataset("<two-class-dataset>")

        assert model.model_config.num_classes == 2
        assert model.model.model.class_embed.bias.shape == original_bias_shape, (
            "Head must not be rebuilt when dataset class count matches checkpoint class count."
        )


# ---------------------------------------------------------------------------
# Weight-based schema inference
# ---------------------------------------------------------------------------


def _make_kp_active_mask(schema: list[int]) -> torch.Tensor:
    """Build a bool _kp_active_mask tensor encoding *schema* (mirrors LwDetr._create_kp_active_mask).

    Args:
        schema: Keypoints-per-class list, e.g. ``[0, 33]`` for background + 33-kp class.

    Returns:
        Bool tensor of shape ``[len(schema), max(schema)]`` with True in active keypoint slots.

    Examples:
        >>> mask = _make_kp_active_mask([0, 3])
        >>> mask.shape
        torch.Size([2, 3])
        >>> mask[0].tolist()
        [False, False, False]
        >>> mask[1].tolist()
        [True, True, True]
    """
    if not schema or max(schema) == 0:
        return torch.zeros(0, 0, dtype=torch.bool)
    max_kp = max(schema)
    mask = torch.zeros(len(schema), max_kp, dtype=torch.bool)
    for idx, n_kp in enumerate(schema):
        mask[idx, :n_kp] = True
    return mask


class TestFromCheckpointWeightInference:
    """from_checkpoint infers schema from checkpoint weights when model_config is absent or stale.

    Regression tests for the bug where a fine-tuned 33-kp keypoint model loaded with the COCO default [0, 17] schema
    because model_config["num_keypoints_per_class"] was never updated from the default before the checkpoint was saved.
    The authoritative schema is embedded in the checkpoint weights via the _kp_active_mask buffer; from_checkpoint now
    reads it directly.
    """

    def test_infers_keypoint_schema_from_kp_active_mask(self, tmp_path: Path) -> None:
        """Stale model_config kp schema [0, 17] is overridden by weight-inferred [0, 33]."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth"},
            "model_name": "RFDETRKeypointPreview",
            "model_config": {"num_keypoints_per_class": [0, 17]},
            "model": {"_kp_active_mask": _make_kp_active_mask([0, 33])},
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRKeypointPreview"
        )

        assert mock_cls.call_args.kwargs["num_keypoints_per_class"] == [0, 33]

    def test_infers_keypoint_schema_when_model_config_absent(self, tmp_path: Path) -> None:
        """num_keypoints_per_class is inferred from _kp_active_mask when model_config is missing."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth"},
            "model_name": "RFDETRKeypointPreview",
            "model": {"_kp_active_mask": _make_kp_active_mask([0, 33])},
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRKeypointPreview"
        )

        assert mock_cls.call_args.kwargs["num_keypoints_per_class"] == [0, 33]

    def test_user_kwarg_wins_over_weight_inferred_keypoint_schema(self, tmp_path: Path) -> None:
        """Explicit num_keypoints_per_class kwarg overrides weight-inferred [0, 33] schema."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth"},
            "model_name": "RFDETRKeypointPreview",
            "model": {"_kp_active_mask": _make_kp_active_mask([0, 33])},
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt,
            tmp_path / "checkpoint_best_total.pth",
            "rfdetr.variants.RFDETRKeypointPreview",
            num_keypoints_per_class=[0, 17],
        )

        assert mock_cls.call_args.kwargs["num_keypoints_per_class"] == [0, 17]

    def test_infers_num_classes_from_class_embed_weight(self, tmp_path: Path) -> None:
        """Stale model_config num_classes=90 is overridden by class_embed.weight shape inference."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-small.pth"},
            "model_name": "RFDETRSmall",
            "model_config": {"num_classes": 90},
            "model": {"class_embed.weight": torch.zeros(3, 256)},
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRSmall")

        assert mock_cls.call_args.kwargs["num_classes"] == 2

    def test_user_kwarg_wins_over_weight_inferred_num_classes(self, tmp_path: Path) -> None:
        """Explicit num_classes kwarg overrides weight-inferred value from class_embed.weight."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-small.pth"},
            "model_name": "RFDETRSmall",
            "model": {"class_embed.weight": torch.zeros(3, 256)},
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt,
            tmp_path / "checkpoint_best_total.pth",
            "rfdetr.variants.RFDETRSmall",
            num_classes=90,
        )

        assert mock_cls.call_args.kwargs["num_classes"] == 90

    def test_infers_schema_from_ptl_ckpt_state_dict_format(self, tmp_path: Path) -> None:
        """Weight inference works for PTL-native .ckpt format (state_dict with model.

        prefix).
        """
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth"},
            "model_name": "RFDETRKeypointPreview",
            "state_dict": {
                "model._kp_active_mask": _make_kp_active_mask([0, 33]),
                "model.class_embed.weight": torch.zeros(3, 256),
            },
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "checkpoint.ckpt", "rfdetr.variants.RFDETRKeypointPreview")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["num_keypoints_per_class"] == [0, 33]
        assert call_kwargs["num_classes"] == 2

    def test_consistent_checkpoint_produces_no_override(self, tmp_path: Path) -> None:
        """When model_config and weights agree, weight inference leaves constructor_kwargs unchanged."""
        ckpt = {
            "args": {"pretrain_weights": "rf-detr-keypoint-preview-xlarge.pth"},
            "model_name": "RFDETRKeypointPreview",
            "model_config": {"num_keypoints_per_class": [0, 33], "num_classes": 2},
            "model": {
                "_kp_active_mask": _make_kp_active_mask([0, 33]),
                "class_embed.weight": torch.zeros(3, 256),
            },
        }
        _, mock_cls = _call_from_checkpoint(
            ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRKeypointPreview"
        )

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["num_keypoints_per_class"] == [0, 33]
        assert call_kwargs["num_classes"] == 2


# ---------------------------------------------------------------------------
# checkpoint_best_total.pth after strip_checkpoint
# ---------------------------------------------------------------------------


class TestFromCheckpointStrippedBestTotal:
    """``checkpoint_best_total.pth`` goes through ``strip_checkpoint``; reloading it must keep the architecture."""

    def test_stripped_checkpoint_restores_trained_resolution(self, tmp_path: Path) -> None:
        """A model trained at a non-default resolution reloads at that resolution and predicts the same boxes."""
        torch.manual_seed(0)
        model = RFDETRNano(pretrain_weights=None, device="cpu", num_classes=3, resolution=224)
        path = tmp_path / "checkpoint_best_total.pth"
        # The payload BestModelCallback writes before on_fit_end strips it into checkpoint_best_total.pth.
        torch.save(
            {
                "model": model.model.model.state_dict(),
                "args": {"class_names": ["a", "b", "c"]},
                "model_name": "RFDETRNano",
                "model_config": model.model_config.model_dump(),
                "optimizer_states": [],
            },
            path,
        )
        strip_checkpoint(path, extra_metadata={"best_total_source": "ema"})

        loaded = RFDETR.from_checkpoint(path, device="cpu")

        assert loaded.model_config.resolution == 224, "resolution must survive strip_checkpoint"
        image = torch.rand(3, 160, 200, generator=torch.Generator().manual_seed(0))
        expected = model.predict(image, threshold=0.0)
        actual = loaded.predict(image, threshold=0.0)
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=1e-4, err_msg="boxes differ after reload")

    def test_stripped_checkpoint_does_not_forward_training_host_device(self, tmp_path: Path) -> None:
        """The training host's device is never restored, even through a real strip_checkpoint round trip.

        A ``device="cpu"`` build with a plain ``device="cpu"`` reload cannot fail whether or not the skip clause in
        ``from_checkpoint`` fires, since the checkpoint value and the explicit kwarg already agree. This test instead
        simulates a checkpoint written on a GPU training host (``model_config["device"] = "cuda"``) and reloads with
        no ``device=`` override, so only the host-policy skip clause — not kwarg precedence — can make it pass.
        """
        model = RFDETRNano(pretrain_weights=None, device="cpu", num_classes=3, resolution=224)
        model_config_dict = model.model_config.model_dump()
        model_config_dict["device"] = "cuda"  # simulate a checkpoint written on a GPU training host
        path = tmp_path / "checkpoint_best_total.pth"
        torch.save(
            {
                "model": model.model.model.state_dict(),
                "args": {"class_names": ["a", "b", "c"]},
                "model_name": "RFDETRNano",
                "model_config": model_config_dict,
                "optimizer_states": [],
            },
            path,
        )
        strip_checkpoint(path, extra_metadata={"best_total_source": "ema"})

        loaded = RFDETR.from_checkpoint(path)

        assert loaded.model_config.device != "cuda", "the training host's device must not be restored"

    def test_stripped_checkpoint_restores_keypoint_model_resolution(self, tmp_path: Path) -> None:
        """A real (unmocked) keypoint model trained at a non-default resolution reloads and predicts at it.

        The existing mocked coverage (``test_checkpoint_model_config_forwarded_to_constructor``) only proves the
        restored fields are forwarded as kwargs to a ``MagicMock`` constructor, never that a real keypoint model
        (a non-default schema, unlike plain detection) actually builds and runs from them — the exact gap the
        CHANGELOG's own keypoint-model repro describes.
        """
        torch.manual_seed(0)
        model = RFDETRKeypointPreview(
            pretrain_weights=None,
            device="cpu",
            resolution=96,
            num_queries=4,
            num_classes=2,
            num_keypoints_per_class=[3, 3],
        )
        path = tmp_path / "checkpoint_best_total.pth"
        torch.save(
            {
                "model": model.model.model.state_dict(),
                "args": {"class_names": ["a", "b"]},
                "model_name": "RFDETRKeypointPreview",
                "model_config": model.model_config.model_dump(),
                "optimizer_states": [],
            },
            path,
        )
        strip_checkpoint(path, extra_metadata={"best_total_source": "ema"})

        loaded = RFDETR.from_checkpoint(path, device="cpu")

        assert loaded.model_config.resolution == 96, "resolution must survive strip_checkpoint for keypoint models"
        image = torch.rand(3, 120, 160, generator=torch.Generator().manual_seed(0))
        result = loaded.predict(image, threshold=0.0)
        assert hasattr(result, "xy"), "restored keypoint model must actually predict at the restored resolution"

    @pytest.mark.parametrize(
        ("extra", "kwargs", "expected_file"),
        [
            pytest.param({"best_total_source": "ema"}, {}, "checkpoint_best_ema.pth", id="old-best-total-ema"),
            pytest.param(
                {"best_total_source": "regular"}, {}, "checkpoint_best_regular.pth", id="old-best-total-regular"
            ),
            pytest.param(
                {"best_total_source": "ema", "model_config": {"resolution": 224}}, {}, None, id="model-config-present"
            ),
            pytest.param({}, {}, None, id="no-best-total-source"),
            pytest.param(
                {"best_total_source": "ema"},
                {"resolution": 224},
                "checkpoint_best_ema.pth",
                id="caller-passes-only-resolution",
            ),
            pytest.param(
                {"best_total_source": "ema"},
                {"resolution": 224, "num_select": 50, "dec_layers": 3},
                None,
                id="caller-passes-every-silent-field",
            ),
            pytest.param(
                {"best_total_source": "ema"},
                {"resolution": 224, "num_select": 50, "dec_layers": 3, "segmentation_head": True},
                "checkpoint_best_ema.pth",
                id="segmentation-still-missing-mask-downsample-ratio",
            ),
        ],
    )
    def test_missing_model_config_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        extra: dict,
        kwargs: dict,
        expected_file: str | None,
    ) -> None:
        """An old stripped best-total file warns and names its source until the caller passes every setting it lost."""
        ckpt = {"model": {}, "args": {"class_names": ["a"]}, "model_name": "RFDETRNano", **extra}
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _call_from_checkpoint(ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano", **kwargs)

        warnings_about_config = [record.message for record in caplog.records if "no model_config" in record.message]
        if expected_file is None:
            assert not warnings_about_config, f"unexpected warning: {warnings_about_config}"
        else:
            assert any(expected_file in message for message in warnings_about_config), (
                f"expected a warning naming {expected_file}, got {warnings_about_config}"
            )

    def test_missing_model_config_warning_names_the_settings_still_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Passing some settings keeps the warning, which lists only the settings still falling back to defaults."""
        ckpt = {"model": {}, "args": {"class_names": ["a"]}, "model_name": "RFDETRNano", "best_total_source": "ema"}
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _call_from_checkpoint(
                ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano", resolution=224
            )

        messages = [record.message for record in caplog.records if "no model_config" in record.message]
        assert messages, "expected the missing model_config warning"
        assert "defaults: num_select, dec_layers." in messages[0], f"missing settings not listed: {messages[0]}"
        assert "resolution" not in messages[0], f"resolution was passed but is still listed: {messages[0]}"

    def test_strip_checkpoint_without_model_config_key_still_warns_on_reload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """strip_checkpoint on a file with no model_config key adds no phantom key, and the reload still warns.

        Every other test in this module supplies a model_config key (present, present-but-empty, or absent only through
        mocked construction); this exercises a real strip_checkpoint round trip on a file that never had the key at all,
        which must not invent one, and must still round-trip into the best-total-source warn path.
        """
        path = tmp_path / "checkpoint_best_total.pth"
        torch.save({"model": {}, "args": {"class_names": ["a"]}, "model_name": "RFDETRNano"}, path)

        strip_checkpoint(path, extra_metadata={"best_total_source": "ema"})

        stripped = torch.load(path, weights_only=False)
        assert "model_config" not in stripped, "strip_checkpoint must not invent a model_config key"

        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _call_from_checkpoint(stripped, path, "rfdetr.variants.RFDETRNano")

        messages = [record.message for record in caplog.records if "no model_config" in record.message]
        assert any("checkpoint_best_ema.pth" in message for message in messages), (
            f"expected the best-total warning naming the ema sibling, got {messages}"
        )

    def test_missing_model_config_warning_empty_dict_restores_nothing_and_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An empty model_config dict is still a dict, so it takes the restore branch and skips the warning branch.

        Documents current behavior at the boundary: ``model_config: {}`` passes ``isinstance(value, dict)``, so
        ``from_checkpoint`` never falls through to the ``best_total_source`` warning check even though the empty
        dict restores zero fields — the caller silently gets class defaults with no warning either way.
        """
        ckpt = {
            "model": {},
            "args": {"class_names": ["a"]},
            "model_name": "RFDETRNano",
            "model_config": {},
            "best_total_source": "ema",
        }
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _, mock_cls = _call_from_checkpoint(
                ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano"
            )

        messages = [record.message for record in caplog.records if "no model_config" in record.message]
        assert not messages, f"unexpected warning despite the model_config key being present: {messages}"
        call_kwargs = mock_cls.call_args.kwargs
        assert not {"resolution", "num_select", "dec_layers"} & call_kwargs.keys(), (
            f"an empty model_config unexpectedly restored fields: {call_kwargs}"
        )

    def test_missing_model_config_warning_fires_for_rfdetr_version_without_best_total_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A resolvable rfdetr_version with no best_total_source still warns, citing the version.

        This is the 1.7.0-1.8.x shape the union discriminator (``"rfdetr_version" in ckpt or "best_total_source" in
        ckpt``) exists to catch: model_config persistence started in 1.7.0 but best_total_source was only added in
        1.9.0, so a checkpoint from that window has no best_total_source to key off yet still lost model_config.
        """
        ckpt = {
            "model": {},
            "args": {"class_names": ["a"]},
            "model_name": "RFDETRNano",
            "rfdetr_version": "1.7.0",
        }
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _call_from_checkpoint(ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano")

        messages = [record.message for record in caplog.records if "no model_config" in record.message]
        assert messages, "expected the missing model_config warning to fire from rfdetr_version alone"
        assert "written by rfdetr 1.7.0" in messages[0], f"warning should cite the rfdetr_version: {messages[0]}"

    def test_training_host_device_is_not_restored(self, tmp_path: Path) -> None:
        """A checkpoint trained on a GPU host must not force ``device="cuda"`` on the loading host."""
        ckpt = {
            "model": {},
            "args": {"class_names": ["a"]},
            "model_name": "RFDETRNano",
            "model_config": {"device": "cuda", "resolution": 224},
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano")

        assert "device" not in mock_cls.call_args.kwargs, "the training host's device must not be forwarded"

    def test_training_host_optimization_flags_are_restored(self, tmp_path: Path) -> None:
        """Unlike device, compile/cuda_graphs/gradient_checkpointing/freeze_encoder ARE restored from model_config.

        The host-policy skip list in ``from_checkpoint`` only excludes ``pretrain_weights`` and ``device``; every other
        ``model_config`` field, including these training-time performance flags, is forwarded to the constructor like
        any other schema field.
        """
        ckpt = {
            "model": {},
            "args": {"class_names": ["a"]},
            "model_name": "RFDETRNano",
            "model_config": {
                "device": "cuda",
                "resolution": 224,
                "compile": True,
                "cuda_graphs": True,
                "gradient_checkpointing": True,
                "freeze_encoder": True,
            },
        }
        _, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "checkpoint_best_total.pth", "rfdetr.variants.RFDETRNano")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["compile"] is True
        assert call_kwargs["cuda_graphs"] is True
        assert call_kwargs["gradient_checkpointing"] is True
        assert call_kwargs["freeze_encoder"] is True
        assert "device" not in call_kwargs, "device stays host policy, unlike the other flags"
        assert mock_cls.call_args.kwargs["resolution"] == 224, "other model_config fields must still be restored"


# ---------------------------------------------------------------------------
# PE-Core-T plus models (RFDETRAtto / RFDETRFemto / RFDETRPico)
# ---------------------------------------------------------------------------


class TestFromCheckpointPEPlusModels:
    """Resolution of the PE-Core-T plus models, independent of which rfdetr_plus (if any) is installed.

    ``rfdetr.platform.models`` is patched into one of three states: a plus release that ships the PE models, an older
    plus release without them (only the XLarge models), and no plus at all.
    """

    _PE_SYMBOLS = ("RFDETRAtto", "RFDETRFemto", "RFDETRPico")

    @pytest.fixture
    def platform_models(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        """Patch ``rfdetr.platform.models`` so every plus symbol is controlled by the test.

        Examples:
            Fixture execution is managed by pytest, so this example cannot run standalone.
            >>> platform_models(monkeypatch)  # doctest: +SKIP
        """
        import rfdetr.platform
        import rfdetr.platform.models as platform_models

        monkeypatch.setattr(rfdetr.platform, "_IS_RFDETR_PLUS_AVAILABLE", True)
        monkeypatch.setattr(platform_models, "_IS_RFDETR_PLUS_AVAILABLE", True)
        for symbol in (*self._PE_SYMBOLS, "RFDETRXLarge", "RFDETR2XLarge"):
            monkeypatch.delitem(platform_models.__dict__, symbol, raising=False)
        return platform_models

    def _install(
        self, monkeypatch: pytest.MonkeyPatch, platform_models: ModuleType, symbols: Iterable[str]
    ) -> dict[str, MagicMock]:
        """Expose a fake class for each of *symbols* on *platform_models*, as an installed rfdetr_plus would.

        Examples:
            >>> import rfdetr.platform.models as platform_models
            >>> monkeypatch = pytest.MonkeyPatch()
            >>> fakes = TestFromCheckpointPEPlusModels()._install(monkeypatch, platform_models, ["RFDETRAtto"])
            >>> platform_models.RFDETRAtto is fakes["RFDETRAtto"]
            True
            >>> monkeypatch.undo()
        """
        fakes = {}
        for symbol in symbols:
            fakes[symbol] = MagicMock(name=symbol)
            fakes[symbol].__name__ = symbol
            monkeypatch.setitem(platform_models.__dict__, symbol, fakes[symbol])
        return fakes

    @pytest.mark.parametrize("model_name", _PE_SYMBOLS)
    def test_model_name_resolves_the_plus_class(self, monkeypatch, platform_models, tmp_path: Path, model_name) -> None:
        fakes = self._install(monkeypatch, platform_models, (*self._PE_SYMBOLS, "RFDETRXLarge", "RFDETR2XLarge"))
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": model_name}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            result = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        fakes[model_name].assert_called_once()
        assert result is fakes[model_name].return_value

    @pytest.mark.parametrize(
        "pretrain_weights, expected",
        [
            ("rf-detr-atto.pth", "RFDETRAtto"),
            ("rf-detr-femto.pth", "RFDETRFemto"),
            ("/cache/rfdetr/rf-detr-pico.pth", "RFDETRPico"),
        ],
    )
    def test_pretrain_weights_name_resolves_the_plus_class(
        self, monkeypatch, platform_models, tmp_path: Path, pretrain_weights: str, expected: str
    ) -> None:
        fakes = self._install(monkeypatch, platform_models, (*self._PE_SYMBOLS, "RFDETRXLarge", "RFDETR2XLarge"))

        with patch("rfdetr.detr.torch.load", return_value=_ns(pretrain_weights)):
            result = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert result is fakes[expected].return_value

    @pytest.mark.parametrize(
        "pretrain_weights",
        [
            pytest.param("/data/epicode/rf-detr-nano.pth", id="pico-inside-directory"),
            pytest.param("/data/tomatto/rf-detr-nano.pth", id="atto-inside-directory"),
        ],
    )
    def test_size_words_inside_paths_do_not_match(
        self, monkeypatch, platform_models, tmp_path: Path, pretrain_weights: str
    ) -> None:
        """Only release-filename stems select a PE model; a bare size word elsewhere in the path does not."""
        self._install(monkeypatch, platform_models, (*self._PE_SYMBOLS, "RFDETRXLarge", "RFDETR2XLarge"))

        result, mock_cls = _call_from_checkpoint(
            _ns(pretrain_weights), tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRNano"
        )

        assert result is mock_cls.return_value

    @pytest.mark.parametrize(
        "pretrain_weights",
        [
            pytest.param("/models/rf-detr-pico/rf-detr-nano.pth", id="posix-path"),
            pytest.param(r"c:\models\rf-detr-pico\rf-detr-nano.pth", id="windows-path"),
        ],
    )
    @pytest.mark.parametrize("plus", ["current", "older", "missing"])
    def test_release_stem_in_a_directory_name_does_not_match(
        self, monkeypatch, platform_models, tmp_path: Path, plus: str, pretrain_weights: str
    ) -> None:
        """A Nano file inside an ``rf-detr-pico`` directory is a Nano checkpoint, whichever rfdetr_plus is installed.

        The checkpoint records the path it was trained from, so a Windows run stores backslash separators; the stem scan
        must strip those directories too, on every host.
        """
        if plus == "missing":
            import rfdetr.platform

            monkeypatch.setattr(rfdetr.platform, "_IS_RFDETR_PLUS_AVAILABLE", False)
        else:
            pe_symbols = self._PE_SYMBOLS if plus == "current" else ()
            self._install(monkeypatch, platform_models, (*pe_symbols, "RFDETRXLarge", "RFDETR2XLarge"))

        result, mock_cls = _call_from_checkpoint(
            _ns(pretrain_weights), tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRNano"
        )

        assert result is mock_cls.return_value

    @pytest.mark.parametrize("model_name", _PE_SYMBOLS)
    def test_older_plus_model_name_raises_upgrade_hint(
        self, monkeypatch, platform_models, tmp_path: Path, model_name: str
    ) -> None:
        self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))
        ckpt = {"args": {"pretrain_weights": "rf-detr-nano.pth", "num_classes": 80}, "model_name": model_name}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError, match="predates it"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_older_plus_pretrain_weights_name_raises_upgrade_hint(
        self, monkeypatch, platform_models, tmp_path: Path
    ) -> None:
        self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))

        with patch("rfdetr.detr.torch.load", return_value=_ns("rf-detr-femto.pth")):
            with pytest.raises(ImportError, match="predates it"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_older_plus_upgrade_hint_chains_the_missing_symbol_error(
        self, monkeypatch, platform_models, tmp_path: Path
    ) -> None:
        """The upgrade hint keeps the lookup error that explains why the PE model is missing."""
        self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))
        ckpt = {"args": {"pretrain_weights": "rf-detr-nano.pth", "num_classes": 80}, "model_name": "RFDETRAtto"}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError, match="predates it") as raised:
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert isinstance(raised.value.__cause__, ImportError)
        assert "RFDETRAtto" in str(raised.value.__cause__)

    def test_upgrade_hint_chains_a_broken_pe_import(self, monkeypatch, platform_models, tmp_path: Path) -> None:
        """A PE import that failed for another reason stays reachable through the upgrade hint's cause chain.

        Without it, a broken rfdetr_plus dependency reads as an outdated install and the real failure is lost.
        """
        broken = ImportError("broken dependency")
        monkeypatch.setattr(platform_models, "_PLUS_PE_IMPORT_ERROR", broken)
        self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))

        with patch("rfdetr.detr.torch.load", return_value=_ns("rf-detr-femto.pth")):
            with pytest.raises(ImportError, match="predates it") as raised:
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert raised.value.__cause__.__cause__ is broken

    def test_older_plus_still_resolves_xlarge(self, monkeypatch, platform_models, tmp_path: Path) -> None:
        fakes = self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": "RFDETRXLarge"}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            result = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert result is fakes["RFDETRXLarge"].return_value

    @pytest.mark.parametrize(
        "pretrain_weights, expected",
        [
            ("/data/xlarge_runs/rf-detr-pico.pth", "RFDETRPico"),
            ("/runs/seg-large-sweep/rf-detr-atto.pth", "RFDETRAtto"),
            ("rfdetr-femto.pth", "RFDETRFemto"),
        ],
    )
    def test_release_stems_win_over_other_names_in_the_path(
        self, monkeypatch, platform_models, tmp_path: Path, pretrain_weights: str, expected: str
    ) -> None:
        fakes = self._install(monkeypatch, platform_models, (*self._PE_SYMBOLS, "RFDETRXLarge", "RFDETR2XLarge"))

        with patch("rfdetr.detr.torch.load", return_value=_ns(pretrain_weights)):
            result = RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert result is fakes[expected].return_value

    def test_older_plus_stem_does_not_fall_back_to_a_core_size_word(
        self, monkeypatch, platform_models, tmp_path: Path
    ) -> None:
        """``rf-detr-pico-small-ft.pth`` contains "small"; with an old plus it must not resolve to RFDETRSmall."""
        self._install(monkeypatch, platform_models, ("RFDETRXLarge", "RFDETR2XLarge"))

        with patch("rfdetr.detr.torch.load", return_value=_ns("rf-detr-pico-small-ft.pth")):
            with pytest.raises(ImportError, match="RFDETRPico is not available"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    @pytest.fixture
    def broken_plus_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rfdetr_plus is installed, but importing it fails on one of its own dependencies.

        Examples:
            Fixture execution is managed by pytest, so this example cannot run standalone.
            >>> broken_plus_import(monkeypatch)  # doctest: +SKIP
        """
        import importlib.abc
        import sys

        import rfdetr.platform

        class _FailingFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> None:
                """Fail the ``rfdetr.platform.models`` import as a missing ``timm`` would; defer the rest."""
                if fullname == "rfdetr.platform.models":
                    raise ModuleNotFoundError("No module named 'timm'", name="timm")
                return None

        monkeypatch.setattr(rfdetr.platform, "_IS_RFDETR_PLUS_AVAILABLE", True)
        monkeypatch.delitem(sys.modules, "rfdetr.platform.models", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_FailingFinder(), *sys.meta_path])

    def test_broken_plus_import_does_not_block_core_checkpoints(self, broken_plus_import, tmp_path: Path) -> None:
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": "RFDETRNano"}

        result, mock_cls = _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRNano")

        assert result is mock_cls.return_value

    @pytest.mark.parametrize("model_name", ["RFDETRAtto", "RFDETRXLarge"])
    def test_broken_plus_import_is_raised_for_plus_checkpoints(
        self, broken_plus_import, tmp_path: Path, model_name: str
    ) -> None:
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": model_name}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError, match="failed to import") as excinfo:
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
        assert excinfo.value.__cause__.name == "timm"

    def test_broken_plus_import_warns_once(
        self, broken_plus_import, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Loading core checkpoints repeatedly with a broken rfdetr_plus logs the import failure only once.

        Each ``from_checkpoint`` call retries the plus import, so a plain warning would repeat on every load of a loop.
        """
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": "RFDETRNano"}
        monkeypatch.setattr(detr_logger, "propagate", True)
        monkeypatch.setattr(detr_logger, "_warned_once", set())

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRNano")
            _call_from_checkpoint(ckpt, tmp_path / "ckpt.pth", "rfdetr.variants.RFDETRNano")

        messages = [record.message for record in caplog.records if "failed to import" in record.message]
        assert len(messages) == 1, messages

    @pytest.mark.parametrize("model_name", _PE_SYMBOLS)
    def test_without_plus_raises_install_hint(self, monkeypatch, platform_models, tmp_path: Path, model_name) -> None:
        import rfdetr.platform

        monkeypatch.setattr(rfdetr.platform, "_IS_RFDETR_PLUS_AVAILABLE", False)
        ckpt = {"args": {"pretrain_weights": "", "num_classes": 80}, "model_name": model_name}

        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ImportError, match="rfdetr_plus package"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")


class TestPlusExportsSingleSource:
    """Every plus-model lookup reads the one ``rfdetr.platform._PLUS_EXPORTS`` set instead of a hand-synced copy."""

    def test_modules_share_the_platform_set(self) -> None:
        """``rfdetr``, ``rfdetr.platform.models`` and ``rfdetr.detr`` hold the very same set object.

        A module that redefines its own copy drifts the next time a plus model is added, which is how only two of the
        five plus symbols ended up covered by tests.
        """
        import rfdetr
        import rfdetr.detr
        import rfdetr.platform.models as platform_models

        assert rfdetr._PLUS_EXPORTS is _PLUS_EXPORTS
        assert platform_models._PLUS_EXPORTS is _PLUS_EXPORTS
        assert rfdetr.detr._PLUS_EXPORTS is _PLUS_EXPORTS

    @pytest.mark.parametrize(
        "class_symbol",
        sorted({symbol for _, symbol in (*_CHECKPOINT_PLUS_MODEL_MAP_ENTRIES, *_CHECKPOINT_PLUS_STEM_ENTRIES)}),
    )
    def test_checkpoint_tables_name_only_plus_exports(self, class_symbol: str) -> None:
        """Each class a plus checkpoint-name or release-stem entry resolves to is a listed plus export.

        ``from_checkpoint`` only fetches the symbols in the set from ``rfdetr.platform.models``, so an entry naming
        anything else could never resolve to its plus class.
        """
        assert class_symbol in _PLUS_EXPORTS


class TestPlatformModelsPEExports:
    """``rfdetr.platform.models`` / ``rfdetr`` surface for the PE-Core-T plus models."""

    @pytest.mark.parametrize("symbol", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_listed_as_plus_exports(self, symbol: str) -> None:
        import rfdetr
        import rfdetr.platform.models as platform_models

        assert symbol in rfdetr._PLUS_EXPORTS
        assert symbol in platform_models._PLUS_EXPORTS

    @pytest.mark.parametrize("symbol", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_older_plus_access_raises_upgrade_hint(self, monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
        import rfdetr.platform.models as platform_models

        monkeypatch.setattr(platform_models, "_IS_RFDETR_PLUS_AVAILABLE", True)
        monkeypatch.delitem(platform_models.__dict__, symbol, raising=False)

        with pytest.raises(ImportError, match="predates it"):
            getattr(platform_models, symbol)

    def test_top_level_access_with_older_plus_raises_upgrade_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rfdetr
        import rfdetr.platform.models as platform_models

        monkeypatch.setattr(platform_models, "_IS_RFDETR_PLUS_AVAILABLE", True)
        monkeypatch.delitem(platform_models.__dict__, "RFDETRAtto", raising=False)
        monkeypatch.delitem(rfdetr.__dict__, "RFDETRAtto", raising=False)

        with pytest.raises(ImportError, match="predates it"):
            rfdetr.RFDETRAtto

    @pytest.mark.parametrize("symbol", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_missing_plus_access_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
        import rfdetr.platform.models as platform_models

        monkeypatch.setattr(platform_models, "_IS_RFDETR_PLUS_AVAILABLE", False)
        monkeypatch.delitem(platform_models.__dict__, symbol, raising=False)

        with pytest.raises(ImportError, match="plus"):
            getattr(platform_models, symbol)
