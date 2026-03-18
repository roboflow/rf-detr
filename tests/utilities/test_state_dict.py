# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for validate_checkpoint_compatibility in rfdetr.utilities.state_dict."""

from types import SimpleNamespace

import pytest

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
