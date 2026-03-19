# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for validate_checkpoint_compatibility in rfdetr.utilities.state_dict."""

import logging
from types import SimpleNamespace

import pytest
import torch

from rfdetr.utilities.state_dict import validate_checkpoint_compatibility


class TestValidateCheckpointCompatibility:
    """Direct unit tests for validate_checkpoint_compatibility."""

    # ------------------------------------------------------------------
    # Early-return / silent-skip cases
    # ------------------------------------------------------------------

    def test_no_args_key_returns_without_raising(self):
        """Checkpoint without 'args' key must return silently."""
        checkpoint = {"model": {}}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    def test_ckpt_has_segmentation_head_model_does_not_skips(self):
        """One-sided: ckpt has segmentation_head, model_args lacks it — skip, no error."""
        ckpt_args = SimpleNamespace(segmentation_head=True, patch_size=14)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(patch_size=14)  # no segmentation_head attribute
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    def test_ckpt_lacks_patch_size_model_has_it_skips(self):
        """One-sided: ckpt has no patch_size, model has it — skip that check, no error."""
        ckpt_args = SimpleNamespace(segmentation_head=False)  # no patch_size
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    def test_compatible_checkpoint_no_exception(self):
        """Checkpoint with matching segmentation_head and patch_size must not raise."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    def test_compatible_segmentation_checkpoint_no_exception(self):
        """Matching segmentation model (seg_head=True both sides) must not raise."""
        ckpt_args = SimpleNamespace(segmentation_head=True, patch_size=16)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=True, patch_size=16)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    # ------------------------------------------------------------------
    # segmentation_head mismatch
    # ------------------------------------------------------------------

    def test_seg_ckpt_into_detection_model_raises(self):
        """Segmentation checkpoint loaded into a detection model must raise ValueError."""
        ckpt_args = SimpleNamespace(segmentation_head=True, patch_size=14)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        with pytest.raises(ValueError, match="segmentation head"):
            validate_checkpoint_compatibility(checkpoint, model_args)

    def test_detection_ckpt_into_seg_model_raises(self):
        """Detection checkpoint loaded into a segmentation model must raise ValueError."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=True, patch_size=14)
        with pytest.raises(ValueError, match="segmentation head"):
            validate_checkpoint_compatibility(checkpoint, model_args)

    # ------------------------------------------------------------------
    # patch_size mismatch
    # ------------------------------------------------------------------

    def test_patch_size_mismatch_raises_with_both_sizes(self):
        """patch_size mismatch must raise ValueError and mention both sizes."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=12)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=16)
        with pytest.raises(ValueError, match=r"patch_size=12.*patch_size=16|patch_size=16.*patch_size=12"):
            validate_checkpoint_compatibility(checkpoint, model_args)

    # ------------------------------------------------------------------
    # class-count mismatch warnings
    # ------------------------------------------------------------------

    def test_class_count_mismatch_backbone_pretrain_warns(self, caplog):
        """Backbone pretrain scenario: checkpoint 91 classes, model 2 — warns about re-init."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {
            "args": ckpt_args,
            "model": {"class_embed.bias": torch.randn(91)},
        }
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=2)

        rf_detr_logger = logging.getLogger("rf-detr")
        prev_propagate = rf_detr_logger.propagate
        rf_detr_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="rf-detr"):
                validate_checkpoint_compatibility(checkpoint, model_args)
        finally:
            rf_detr_logger.propagate = prev_propagate

        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("re-initialized to 2 classes" in msg for msg in warning_msgs), (
            f"Expected 're-initialized to 2 classes' warning, got: {warning_msgs}"
        )

    def test_class_count_mismatch_finetune_checkpoint_warns(self, caplog):
        """Fine-tuned checkpoint scenario: checkpoint 3 classes, model 90 — warns with num_classes hint."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {
            "args": ckpt_args,
            "model": {"class_embed.bias": torch.randn(3)},
        }
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=90)

        rf_detr_logger = logging.getLogger("rf-detr")
        prev_propagate = rf_detr_logger.propagate
        rf_detr_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="rf-detr"):
                validate_checkpoint_compatibility(checkpoint, model_args)
        finally:
            rf_detr_logger.propagate = prev_propagate

        warning_msgs = [r.getMessage() for r in caplog.records if r.name == "rf-detr" and r.levelno >= logging.WARNING]
        assert any("Pass num_classes=2" in msg for msg in warning_msgs), (
            f"Expected 'Pass num_classes=2' warning, got: {warning_msgs}"
        )

    def test_class_count_match_no_warning(self, caplog):
        """Matching class count — no warning emitted."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {
            "args": ckpt_args,
            "model": {"class_embed.bias": torch.randn(91)},
        }
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=90)

        rf_detr_logger = logging.getLogger("rf-detr")
        prev_propagate = rf_detr_logger.propagate
        rf_detr_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="rf-detr"):
                validate_checkpoint_compatibility(checkpoint, model_args)
        finally:
            rf_detr_logger.propagate = prev_propagate

        warning_msgs = [r.getMessage() for r in caplog.records if r.name == "rf-detr" and r.levelno >= logging.WARNING]
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"

    def test_class_count_missing_model_key_no_warning(self, caplog):
        """Checkpoint without 'model' key — no warning (backward compat)."""
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14)
        checkpoint = {"args": ckpt_args}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=90)

        rf_detr_logger = logging.getLogger("rf-detr")
        prev_propagate = rf_detr_logger.propagate
        rf_detr_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="rf-detr"):
                validate_checkpoint_compatibility(checkpoint, model_args)
        finally:
            rf_detr_logger.propagate = prev_propagate

        warning_msgs = [r.getMessage() for r in caplog.records if r.name == "rf-detr" and r.levelno >= logging.WARNING]
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"
