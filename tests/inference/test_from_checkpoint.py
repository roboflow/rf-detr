# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETR.from_checkpoint classmethod.

The inference logic is isolated by patching ``torch.load`` and the target model class inside ``rfdetr.variants`` (or
``rfdetr.platform.models`` for plus models).  No model weights are downloaded or GPU memory allocated.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.detr import RFDETR
from rfdetr.detr import logger as detr_logger
from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE
from rfdetr.variants import RFDETRSmall


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
    """Invoke RFDETR.from_checkpoint with torch.load mocked to return *ckpt* and the model class at *cls_patch_target*
    replaced by a MagicMock.

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
        """from_checkpoint adopts the checkpoint class count without marking it as a user override."""
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
