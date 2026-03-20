# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for rfdetr.utilities.state_dict."""

import logging
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from rfdetr.utilities.state_dict import _make_fit_loop_state, validate_checkpoint_compatibility

# ---------------------------------------------------------------------------
# _make_fit_loop_state
# ---------------------------------------------------------------------------


class TestMakeFitLoopState:
    """Tests for _make_fit_loop_state epoch counter encoding."""

    @pytest.mark.parametrize(
        "epoch,expected_n",
        [
            pytest.param(0, 1, id="epoch_0"),
            pytest.param(4, 5, id="epoch_4"),
            pytest.param(9, 10, id="epoch_9"),
        ],
    )
    def test_epoch_progress_completed_is_epoch_plus_one(self, epoch: int, expected_n: int) -> None:
        """epoch_progress.current.completed == epoch + 1 so PTL sets current_epoch correctly."""
        state = _make_fit_loop_state(epoch)
        assert state["epoch_progress"]["current"]["completed"] == expected_n
        assert state["epoch_progress"]["total"]["completed"] == expected_n

    def test_epoch_progress_all_counters_equal(self) -> None:
        """All four counters in epoch_progress should be equal (epoch fully completed)."""
        state = _make_fit_loop_state(7)
        for scope in ("total", "current"):
            ep = state["epoch_progress"][scope]
            vals = [ep["ready"], ep["started"], ep["processed"], ep["completed"]]
            assert len(set(vals)) == 1, f"epoch_progress.{scope} counters differ: {ep}"

    def test_batches_that_stepped_is_zero(self) -> None:
        """Optimizer/scheduler state should start fresh; _batches_that_stepped must be 0."""
        state = _make_fit_loop_state(3)
        assert state["epoch_loop.state_dict"]["_batches_that_stepped"] == 0

    def test_batch_progress_is_zero(self) -> None:
        """Batch progress counters should be zeroed out (not mid-batch resume)."""
        state = _make_fit_loop_state(5)
        for key in ("epoch_loop.batch_progress", "epoch_loop.val_loop.batch_progress"):
            bp = state[key]
            assert bp["is_last_batch"] is False
            for scope in ("total", "current"):
                assert all(v == 0 for v in bp[scope].values()), f"{key}.{scope} not zero: {bp[scope]}"

    def test_ptl_accepts_fit_loop_state(self) -> None:
        """PTL's _FitLoop.load_state_dict must not raise with our synthesised state dict."""

        class _DummyModule(LightningModule):
            def training_step(self, batch, idx):
                return torch.tensor(0.0, requires_grad=True)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=1e-3)

        trainer = Trainer(max_epochs=10, accelerator="cpu", enable_progress_bar=False, logger=False)
        trainer.strategy.connect(_DummyModule())

        epoch = 4
        state = _make_fit_loop_state(epoch)
        trainer.fit_loop.load_state_dict(state)
        assert trainer.current_epoch == epoch + 1

    def test_required_top_level_keys_present(self) -> None:
        """State dict must contain all keys the FitLoop accesses during load."""
        required = {
            "state_dict",
            "epoch_loop.state_dict",
            "epoch_loop.batch_progress",
            "epoch_loop.scheduler_progress",
            "epoch_loop.automatic_optimization.state_dict",
            "epoch_loop.automatic_optimization.optim_progress",
            "epoch_loop.manual_optimization.state_dict",
            "epoch_loop.manual_optimization.optim_step_progress",
            "epoch_loop.val_loop.state_dict",
            "epoch_loop.val_loop.batch_progress",
            "epoch_progress",
        }
        state = _make_fit_loop_state(0)
        missing = required - set(state.keys())
        assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
# validate_checkpoint_compatibility
# ---------------------------------------------------------------------------


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
