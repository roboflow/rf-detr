# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr.utilities.state_dict."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from rfdetr.utilities.state_dict import (
    _make_fit_loop_state,
    remap_projector_to_cross_attn,
    strip_checkpoint,
    validate_checkpoint_compatibility,
)

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
    # patch_size inferred from projection weight (no "args" key)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "ckpt_patch_size,model_patch_size,should_raise",
        [
            pytest.param(16, 12, True, id="ckpt_16_model_12_raises"),
            pytest.param(14, 16, True, id="ckpt_14_model_16_raises"),
            pytest.param(16, 16, False, id="matching_16_no_raise"),
        ],
    )
    def test_patch_size_inferred_from_projection_weight(
        self, ckpt_patch_size: int, model_patch_size: int, should_raise: bool
    ) -> None:
        """Projection weight shape used to infer ckpt patch_size when 'args' key absent.

        Regression test for #965 — pretrained COCO weights lack 'args', so the shape-based fallback must fire before
        load_state_dict raises a cryptic RuntimeError.
        """
        proj_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        proj_weight = torch.zeros(384, 3, ckpt_patch_size, ckpt_patch_size)
        checkpoint = {"model": {proj_key: proj_weight}}  # no "args" key
        model_args = SimpleNamespace(patch_size=model_patch_size)

        if should_raise:
            with pytest.raises(
                ValueError,
                match=rf"patch_size={ckpt_patch_size}.*patch_size={model_patch_size}"
                rf"|patch_size={model_patch_size}.*patch_size={ckpt_patch_size}",
            ):
                validate_checkpoint_compatibility(checkpoint, model_args)
        else:
            validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    @pytest.mark.parametrize(
        "checkpoint,model_args_kwargs",
        [
            pytest.param(
                {},
                {"patch_size": 16},
                id="no_model_key_skips",
            ),
            pytest.param(
                {"model": {}},
                {"patch_size": 16},
                id="no_projection_key_skips",
            ),
            pytest.param(
                {
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3, 16, 16
                        )
                    }
                },
                {},
                id="model_no_patch_size_attr_skips",
            ),
            pytest.param(
                {
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3
                        )  # 2D — not a Conv2d weight; rank guard must skip cleanly
                    }
                },
                {"patch_size": 16},
                id="proj_weight_2d_skips",
            ),
            pytest.param(
                {
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3, 16
                        )  # 3D — not a Conv2d weight; rank guard must skip cleanly
                    }
                },
                {"patch_size": 16},
                id="proj_weight_3d_skips",
            ),
            pytest.param(
                {
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3, 16, 16, 16
                        )  # 5D — Conv3d-like; rank guard (== 4) must skip cleanly
                    }
                },
                {"patch_size": 8},
                id="proj_weight_5d_skips",
            ),
            pytest.param(
                {
                    "args": SimpleNamespace(patch_size=14),
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3, 16, 16
                        )  # projection suggests 16, but args.patch_size=14 takes precedence
                    },
                },
                {"patch_size": 14},
                id="args_patch_size_suppresses_projection_inference",
            ),
            pytest.param(
                {
                    "args": {"patch_size": 14},  # PTL-style dict args (not SimpleNamespace)
                    "model": {
                        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": torch.zeros(
                            384, 3, 16, 16
                        )  # projection suggests 16, but dict args["patch_size"]=14 takes precedence
                    },
                },
                {"patch_size": 14},
                id="dict_args_patch_size_suppresses_projection_inference",
            ),
        ],
    )
    def test_projection_inference_silently_skips_when_incomplete(
        self, checkpoint: dict, model_args_kwargs: dict
    ) -> None:
        """Shape-based patch_size check is skipped when key or attribute is absent.

        Verifies backward compatibility: missing projection key, missing model key, model_args without patch_size
        attribute, non-4D projection weights, or an explicit args.patch_size (SimpleNamespace or dict) must all be
        handled without error.
        """
        model_args = SimpleNamespace(**model_args_kwargs)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    def test_non_square_projection_kernel_skips_check(self) -> None:
        """Non-square patch projection kernel is skipped — patch_size cannot be inferred reliably.

        Guards against hypothetical future backbones with non-square Conv2d kernels where shape[-1] would not equal
        patch_size.
        """
        proj_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        proj_weight = torch.zeros(384, 3, 16, 14)  # non-square: h=16, w=14
        checkpoint = {"model": {proj_key: proj_weight}}
        model_args = SimpleNamespace(patch_size=16)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

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


class TestRemapProjectorToCrossAttn:
    """Tests for dual-projector checkpoint key remapping."""

    def test_clones_projector_weights_when_dual_projector_enabled(self) -> None:
        """Dual-projector models clone projector keys into cross_attn_projector when missing."""
        state_dict = {
            "backbone.0.projector.0.weight": torch.randn(4, 4, 1, 1),
            "backbone.0.projector.0.bias": torch.randn(4),
        }
        model = SimpleNamespace(backbone=[SimpleNamespace(dual_projector=True)])

        remapped = remap_projector_to_cross_attn(state_dict, model)

        assert remapped is state_dict
        assert "backbone.0.cross_attn_projector.0.weight" in remapped
        assert "backbone.0.cross_attn_projector.0.bias" in remapped
        assert torch.equal(
            remapped["backbone.0.cross_attn_projector.0.weight"],
            state_dict["backbone.0.projector.0.weight"],
        )
        assert torch.equal(
            remapped["backbone.0.cross_attn_projector.0.bias"],
            state_dict["backbone.0.projector.0.bias"],
        )

    def test_skips_when_cross_attn_keys_already_present(self) -> None:
        """No remap is applied when cross_attn_projector keys already exist."""
        state_dict = {
            "backbone.0.projector.0.weight": torch.randn(4, 4, 1, 1),
            "backbone.0.cross_attn_projector.0.weight": torch.randn(4, 4, 1, 1),
        }
        model = SimpleNamespace(backbone=[SimpleNamespace(dual_projector=True)])

        remapped = remap_projector_to_cross_attn(state_dict, model)

        assert remapped is state_dict
        assert len([key for key in remapped if key.startswith("backbone.0.cross_attn_projector.")]) == 1

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


class TestStripCheckpoint:
    """Tests for strip_checkpoint loop-stub backfill."""

    def _make_minimal_ckpt(self, tmp_path, extra: dict | None = None) -> Path:
        """Write a minimal checkpoint to a temp file."""
        payload = {"model": {"w": torch.tensor(1.0)}, "args": {"lr": 1e-4}}
        if extra:
            payload.update(extra)
        ckpt_path = Path(tmp_path) / "ckpt.pth"
        torch.save(payload, ckpt_path)
        return ckpt_path

    def test_strip_adds_validate_loop_stub_when_loops_present_but_missing_key(self, tmp_path) -> None:
        """Old checkpoints with loops but no validate_loop/test_loop get stubs backfilled."""
        ckpt_path = self._make_minimal_ckpt(
            tmp_path,
            extra={"loops": {"fit_loop": {"state_dict": {}}}},
        )
        strip_checkpoint(ckpt_path)
        result = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert result["loops"]["validate_loop"] == {"state_dict": {}}
        assert result["loops"]["test_loop"] == {"state_dict": {}}

    def test_strip_preserves_existing_validate_loop_stub(self, tmp_path) -> None:
        """Checkpoints with validate_loop already present are not overwritten."""
        original_stub = {"state_dict": {"some_key": 1}}
        ckpt_path = self._make_minimal_ckpt(
            tmp_path,
            extra={"loops": {"fit_loop": {"state_dict": {}}, "validate_loop": original_stub}},
        )
        strip_checkpoint(ckpt_path)
        result = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert result["loops"]["validate_loop"] == original_stub

    def test_strip_no_loops_key_leaves_loops_absent(self, tmp_path) -> None:
        """Checkpoints without a loops key must not gain one after stripping."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        strip_checkpoint(ckpt_path)
        result = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "loops" not in result
