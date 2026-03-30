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
        "pretrain_weights, patch_target",
        [
            pytest.param("rf-detr-nano.pth", "rfdetr.variants.RFDETRNano", id="nano"),
            pytest.param("rf-detr-small.pth", "rfdetr.variants.RFDETRSmall", id="small"),
            pytest.param("rf-detr-medium.pth", "rfdetr.variants.RFDETRMedium", id="medium"),
            pytest.param("rf-detr-large.pth", "rfdetr.variants.RFDETRLarge", id="large"),
            pytest.param("rf-detr-base.pth", "rfdetr.variants.RFDETRBase", id="base"),
            pytest.param("rf-detr-seg-nano.pt", "rfdetr.variants.RFDETRSegNano", id="seg-nano"),
            pytest.param("rf-detr-seg-small.pt", "rfdetr.variants.RFDETRSegSmall", id="seg-small"),
            pytest.param("rf-detr-seg-medium.pt", "rfdetr.variants.RFDETRSegMedium", id="seg-medium"),
            pytest.param("rf-detr-seg-large.pt", "rfdetr.variants.RFDETRSegLarge", id="seg-large"),
            pytest.param("rf-detr-seg-xlarge.pt", "rfdetr.variants.RFDETRSegXLarge", id="seg-xlarge"),
            pytest.param("rf-detr-seg-xxlarge.pt", "rfdetr.variants.RFDETRSeg2XLarge", id="seg-2xlarge"),
            pytest.param("rf-detr-seg-preview.pt", "rfdetr.variants.RFDETRSegPreview", id="seg-preview"),
        ],
    )
    def test_characterization_infers_correct_class_namespace(
        self,
        tmp_path: Path,
        pretrain_weights: str,
        patch_target: str,
    ) -> None:
        """Namespace-style args: correct subclass is called for each model size."""
        result, mock_cls = _call_from_checkpoint(_ns(pretrain_weights), tmp_path / "ckpt.pth", patch_target)

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
        "pretrain_weights, patch_target",
        [
            pytest.param("rf-detr-small.pth", "rfdetr.variants.RFDETRSmall", id="small"),
            pytest.param("rf-detr-base.pth", "rfdetr.variants.RFDETRBase", id="base"),
        ],
    )
    def test_characterization_infers_correct_class_dict(
        self,
        tmp_path: Path,
        pretrain_weights: str,
        patch_target: str,
    ) -> None:
        """Dict-style args: correct subclass is called without AttributeError."""
        _, mock_cls = _call_from_checkpoint(_dict(pretrain_weights), tmp_path / "ckpt.pth", patch_target)

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
            with pytest.raises(ValueError, match="Could not infer model size"):
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
