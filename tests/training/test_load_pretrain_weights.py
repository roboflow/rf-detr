# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for fine-tuned checkpoint weight destruction.

When a user loads a fine-tuned N-class checkpoint but has ``num_classes``
configured to a LARGER value (e.g. default 90), the second reinit in
``load_pretrain_weights`` (models/weights.py) must NOT erroneously resize the
detection head to ``num_classes + 1``, destroying the loaded weights.

The fix changes the second reinit condition from:
    ``checkpoint_num_classes != args.num_classes + 1``
to the user-override-aware logic that auto-aligns to the checkpoint when the
user did not explicitly set ``num_classes``.

These tests exercise ``rfdetr.models.weights.load_pretrain_weights`` directly,
which is the unified function that replaced the two prior separate implementations
(``detr.py:_load_pretrain_weights_into`` and
``module_model.py:RFDETRModelModule._load_pretrain_weights``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from rfdetr.models.weights import load_pretrain_weights

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(num_classes=91, num_queries=300, group_detr=13):
    """Build a minimal checkpoint dict with the given class count.

    Args:
        num_classes: Total classes including background (bias shape).
        num_queries: Number of object queries per group.
        group_detr: Number of groups.
    """
    total_queries = num_queries * group_detr
    state = {
        "class_embed.weight": torch.randn(num_classes, 256),
        "class_embed.bias": torch.randn(num_classes),
        "refpoint_embed.weight": torch.randn(total_queries, 4),
        "query_feat.weight": torch.randn(total_queries, 256),
        "other_layer.weight": torch.randn(10, 10),
    }
    ckpt_args = SimpleNamespace(
        segmentation_head=False,
        patch_size=14,
        class_names=[],
    )
    return {"model": state, "args": ckpt_args}


def _make_train_config():
    """Return a minimal TrainConfig for use in load_pretrain_weights.

    Returns:
        Minimal TrainConfig with placeholder dataset and output dirs.
    """
    return TrainConfig(
        dataset_dir="/nonexistent/dataset",
        output_dir="/nonexistent/output",
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        lr_drop=8,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        tensorboard=False,
    )


# ---------------------------------------------------------------------------
# Regression tests: load_pretrain_weights (models/weights.py)
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsSecondReinit:
    """Regression tests for ``load_pretrain_weights`` in ``rfdetr.models.weights``.

    Validates that the second reinitialize_detection_head call only fires when
    the checkpoint has MORE classes than configured (backbone pretrain scenario),
    not when it has fewer (fine-tuned checkpoint scenario).
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_finetune_checkpoint_preserves_weights(self, monkeypatch):
        """Fine-tuned checkpoint (fewer classes) must NOT trigger second reinit.

        Scenario: 2-class fine-tuned checkpoint (bias shape [3]) loaded with
        default num_classes=90. The first reinit correctly resizes the head to 3
        so load_state_dict works. The second reinit must NOT resize to 91 —
        that would destroy the loaded fine-tuned weights.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        checkpoint = _make_checkpoint(num_classes=3)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls[0] == call(3), f"First reinit should resize to checkpoint size 3, got {calls[0]}"
        assert len(calls) == 1, (
            f"Expected exactly 1 reinit call (to checkpoint size), but got {len(calls)}: "
            f"{calls}. The second reinit to 91 destroys loaded weights."
        )
        assert mc.num_classes == 2, (
            f"mc.num_classes must be auto-aligned to 2 (checkpoint_logits - 1), got {mc.num_classes}"
        )

    def test_no_mismatch_no_reinit(self, monkeypatch):
        """Checkpoint class count matches config — no reinit at all.

        Scenario: COCO checkpoint (91 classes) with num_classes=90.
        91 == 90 + 1, so no reinit should fire.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        fake_model.reinitialize_detection_head.assert_not_called()

    def test_backbone_pretrain_still_reinits(self, monkeypatch):
        """Backbone pretrain (more classes in checkpoint) must still reinit.

        Scenario: COCO 91-class checkpoint loaded for 2-class fine-tuning
        (num_classes=2). Both reinits are correct here: first to 91 for
        load_state_dict, second to 3 for the configured class count.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=2)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(3)], f"Expected reinit to [91, 3] (expand then trim), got {calls}"

    def test_user_override_larger_than_checkpoint_reexpands_head(self, monkeypatch):
        """Explicit larger num_classes must be restored after checkpoint load.

        Scenario: 91-class checkpoint loaded with explicit num_classes=93.
        Loader must temporarily match checkpoint size for load_state_dict, then
        expand to 94 logits and keep args.num_classes unchanged.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=93)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(94)], f"Expected reinit to [91, 94] (load then expand), got {calls}"
        assert mc.num_classes == 93, "Explicitly configured num_classes must not be overwritten."

    # All non-deprecated model configs (RFDETRLargeDeprecatedConfig and
    # RFDETRBaseConfig are excluded; the former is deprecated, the latter
    # serves as the base class for the concrete variants below).
    @pytest.mark.parametrize(
        "config_cls",
        [
            pytest.param(RFDETRNanoConfig, id="nano"),
            pytest.param(RFDETRSmallConfig, id="small"),
            pytest.param(RFDETRMediumConfig, id="medium"),
            pytest.param(RFDETRLargeConfig, id="large"),
            pytest.param(RFDETRSegNanoConfig, id="seg_nano"),
            pytest.param(RFDETRSegSmallConfig, id="seg_small"),
            pytest.param(RFDETRSegMediumConfig, id="seg_medium"),
            pytest.param(RFDETRSegLargeConfig, id="seg_large"),
            pytest.param(RFDETRSegXLargeConfig, id="seg_xlarge"),
            pytest.param(RFDETRSeg2XLargeConfig, id="seg_2xlarge"),
        ],
    )
    def test_eight_class_finetune_checkpoint_auto_aligns_num_classes_and_reinits_once(self, monkeypatch, config_cls):
        """Auto-align ``mc.num_classes`` and avoid a second reinit for 8-class checkpoints.

        Scenario (from user bug report): user trains on 8 categories (IDs 0–7).
        The checkpoint stores ``class_embed.bias`` with shape [9] (8 user classes
        + 1 background). Loading without specifying ``num_classes`` must NOT
        trigger a second reinit to 91 after temporarily matching the checkpoint
        size for ``load_state_dict``.

        This test asserts the loader auto-aligns ``mc.num_classes`` to 8 (9 - 1)
        and fires exactly one reinit call — to 9 (the checkpoint size).
        """
        # 8 dataset categories → training builds a model with 8+1=9 logits.
        checkpoint = _make_checkpoint(num_classes=9)
        mc = config_cls(pretrain_weights="/fake/weights.pth", device="cpu")
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert len(calls) == 1, (
            f"Expected exactly 1 reinit call (to checkpoint size 9), but got {len(calls)}: "
            f"{calls}. A second reinit to 91 would produce OOB class IDs like 73."
        )
        assert calls[0] == call(9), f"Reinit must resize to checkpoint's 9 logits, got {calls[0]}"
        assert mc.num_classes == 8, (
            f"mc.num_classes must be auto-aligned to 8 (checkpoint_logits - 1), got {mc.num_classes}"
        )


# ---------------------------------------------------------------------------
# Regression #960: PE interpolation for custom resolution
# ---------------------------------------------------------------------------

PE_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"


class TestLoadPretrainWeightsPEInterpolation:
    """Regression tests for #960 — PE must be interpolated when resolution changes.

    ``load_pretrain_weights`` must bicubic-interpolate the checkpoint's DINOv2
    positional embeddings to match the model's ``positional_encoding_size`` before
    calling ``load_state_dict``.  Without this, any custom ``resolution`` that
    changes the PE grid size causes a ``RuntimeError: size mismatch``.
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    @pytest.mark.parametrize(
        "src_pe_size, tgt_resolution, patch_size, expected_tgt_pe_size",
        [
            pytest.param(24, 640, 16, 40, id="nano_24x24_upscale_to_40x40"),
            pytest.param(40, 384, 16, 24, id="nano_40x40_downscale_to_24x24"),
            pytest.param(32, 640, 16, 40, id="small_32x32_upscale_to_40x40"),
        ],
    )
    def test_pe_in_checkpoint_is_interpolated_to_model_resolution(
        self, monkeypatch, src_pe_size, tgt_resolution, patch_size, expected_tgt_pe_size
    ):
        """Checkpoint PE is bicubic-interpolated to match model_config.positional_encoding_size.

        Regression for #960: ``load_pretrain_weights`` must not raise ``RuntimeError``
        when model resolution differs from checkpoint resolution.  The PE tensor in the
        checkpoint must be resized in-place before ``load_state_dict`` is called.
        """
        mc = RFDETRNanoConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            resolution=tgt_resolution,
            patch_size=patch_size,
        )
        assert mc.positional_encoding_size == tgt_resolution // patch_size

        dim = 384
        src_n = src_pe_size * src_pe_size + 1  # patches + class token
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"][PE_KEY] = torch.randn(1, src_n, dim).half()  # float16 to verify dtype round-trip

        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        pe = checkpoint["model"][PE_KEY]
        expected_n = expected_tgt_pe_size * expected_tgt_pe_size + 1
        assert pe.shape == torch.Size([1, expected_n, dim]), (
            f"Expected PE shape [1, {expected_n}, {dim}], got {tuple(pe.shape)}. "
            f"PE was not interpolated from {src_pe_size}x{src_pe_size} "
            f"to {expected_tgt_pe_size}x{expected_tgt_pe_size}."
        )
        assert pe.dtype == torch.float16, f"Dtype must be preserved after interpolation, got {pe.dtype}"

    def test_matching_pe_shape_is_not_modified(self, monkeypatch):
        """When checkpoint PE matches model expectations, the tensor is not changed.

        Ensures PE interpolation is a no-op for same-resolution checkpoints so that
        normal weight loading is unaffected.
        """
        mc = RFDETRNanoConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        # Default: positional_encoding_size=24 → PE = [1, 24*24+1, 384] = [1, 577, 384]

        dim = 384
        original_pe = torch.randn(1, 577, dim)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"][PE_KEY] = original_pe.clone()

        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        pe = checkpoint["model"][PE_KEY]
        assert pe.shape == torch.Size([1, 577, dim]), "Matching PE shape must not be modified."
        assert torch.equal(pe, original_pe), "Matching PE tensor values must not be modified."

    def test_base_config_non_formula_pe_is_interpolated_from_smaller_checkpoint(self, monkeypatch):
        """RFDETRBaseConfig PE=37 (not formula-derived) is interpolated when checkpoint differs.

        RFDETRBaseConfig.positional_encoding_size=37 is not updated by
        ``_sync_pe_with_resolution`` because 37 ≠ 560//16=35 (not formula-derived).
        Loading a checkpoint with a smaller PE grid (e.g., 24×24) must still
        trigger interpolation to the model's fixed PE=37×37 target.
        """
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        assert mc.positional_encoding_size == 37, "RFDETRBaseConfig PE must remain 37 (not formula-derived)"

        dim = 384
        src_pe_size = 24
        src_n = src_pe_size * src_pe_size + 1
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"][PE_KEY] = torch.randn(1, src_n, dim)

        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        pe = checkpoint["model"][PE_KEY]
        expected_n = 37 * 37 + 1
        assert pe.shape == torch.Size([1, expected_n, dim]), (
            f"Expected PE shape [1, {expected_n}, {dim}] (37×37 grid), got {tuple(pe.shape)}. "
            "BaseConfig's non-formula-derived PE must be the interpolation target."
        )

    def test_non_square_source_pe_logs_warning_and_is_not_modified(self, monkeypatch):
        """Non-square source PE grids are skipped with a warning and left unchanged.

        When ``n_source`` is not a perfect square the interpolation is skipped to
        avoid producing malformed embeddings.  The tensor must remain untouched and
        a warning must be emitted via the weights module logger.
        """
        mc = RFDETRNanoConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        # positional_encoding_size=24 → n_target=576 (perfect square, so the
        # target-side guard does not trigger; only the source-side guard fires)

        dim = 384
        # 17 is not a perfect square: isqrt(17)=4, 4*4=16 ≠ 17
        non_square_n_source = 17
        original_pe = torch.randn(1, non_square_n_source + 1, dim)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"][PE_KEY] = original_pe.clone()

        warning_calls: list[tuple] = []
        monkeypatch.setattr("rfdetr.models.weights.logger.warning", lambda *a, **kw: warning_calls.append(a))
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        pe = checkpoint["model"][PE_KEY]
        assert torch.equal(pe, original_pe), "Non-square source PE must not be modified."
        assert any("not a perfect square" in str(args) for args in warning_calls), (
            f"Expected a 'not a perfect square' warning; got calls: {warning_calls}"
        )


# ---------------------------------------------------------------------------
# Deprecation: train_config argument
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsDeprecation:
    """Passing train_config must emit a DeprecationWarning."""

    def test_emits_deprecation_warning_when_train_config_passed(self, monkeypatch):
        """Any non-None train_config triggers a DeprecationWarning."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        tc = _make_train_config()

        with pytest.warns(DeprecationWarning, match="train_config.*deprecated"):
            load_pretrain_weights(MagicMock(), mc, tc)
