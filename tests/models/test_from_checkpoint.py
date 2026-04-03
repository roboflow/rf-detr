# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for RFDETR.from_checkpoint classmethod.

The inference logic is isolated by patching ``torch.load`` and the target
model class inside ``rfdetr.variants`` (or ``rfdetr.platform.models`` for
plus models).  No model weights are downloaded or GPU memory allocated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rfdetr.detr import RFDETR
from rfdetr.variants import RFDETRSmall

try:
    import rfdetr.platform.models as _pm

    HAS_PLUS = _pm._PLUS_AVAILABLE
except ImportError:
    HAS_PLUS = False


def _ns(pretrain_weights: str, num_classes: int = 80) -> dict:
    """Fake legacy checkpoint with argparse.Namespace args."""
    return {"args": argparse.Namespace(pretrain_weights=pretrain_weights, num_classes=num_classes)}


def _dict(pretrain_weights: str, num_classes: int = 80) -> dict:
    """Fake PTL-style checkpoint with dict args."""
    return {"args": {"pretrain_weights": pretrain_weights, "num_classes": num_classes}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_from_checkpoint(ckpt: dict, path: Path, cls_patch_target: str, **kwargs):
    """
    Invoke RFDETR.from_checkpoint with torch.load mocked to return *ckpt* and
    the model class at *cls_patch_target* replaced by a MagicMock.

    Returns:
        Tuple of (result, mock_class).
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

    def test_characterization_unknown_pretrain_weights_raises_value_error(self, tmp_path: Path) -> None:
        """Unrecognised pretrain_weights name raises a descriptive ValueError."""
        ckpt = _ns("/my/custom/finetuned.pth")
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="Could not infer model class"):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

    def test_characterization_missing_args_key_raises_key_error(self, tmp_path: Path) -> None:
        """Checkpoint without 'args' key raises KeyError."""
        ckpt = {"model": {}}
        with patch("rfdetr.detr.torch.load", return_value=ckpt):
            with pytest.raises(KeyError):
                RFDETR.from_checkpoint(tmp_path / "ckpt.pth")

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

    @pytest.mark.skipif(HAS_PLUS, reason="rfdetr_plus is installed — guard not active")
    def test_characterization_xlarge_without_plus_raises_import_error(self, tmp_path: Path) -> None:
        """xlarge checkpoint without rfdetr_plus raises ImportError instead of wrong class."""
        for weights in ("rf-detr-xlarge.pth", "rf-detr-xxlarge.pth"):
            ckpt = _ns(weights)
            with patch("rfdetr.detr.torch.load", return_value=ckpt):
                with pytest.raises(ImportError):
                    RFDETR.from_checkpoint(tmp_path / "ckpt.pth")


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
    def test_direct_instantiation_is_allowed(self, cls_name: str, import_path: str) -> None:
        """Direct instantiation of a deprecated class does not raise RuntimeError."""
        import importlib

        module_path, attr = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        with patch("rfdetr.detr.RFDETR.__init__", return_value=None):
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
    """Fake checkpoint with model_name key (new format)."""
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

    @pytest.mark.skipif(HAS_PLUS, reason="rfdetr_plus is installed — guard not active")
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
