# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for fine-tuned checkpoint weight destruction.

When a user loads a fine-tuned N-class checkpoint but has ``num_classes`` configured to a LARGER value (e.g. default
90), the second reinit in ``load_pretrain_weights`` (models/weights.py) must NOT erroneously resize the detection head
to ``num_classes + 1``, destroying the loaded weights.

The fix changes the second reinit condition from:
    ``checkpoint_num_classes != args.num_classes + 1``
to the user-override-aware logic that auto-aligns to the checkpoint when the user did not explicitly set
``num_classes``.

These tests exercise ``rfdetr.models.weights.load_pretrain_weights`` directly, which is the unified function that
replaced the two prior separate implementations (``detr.py:_load_pretrain_weights_into`` and
``module_model.py:RFDETRModelModule._load_pretrain_weights``).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from rfdetr import RFDETRNano, RFDETRSegNano
from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.utilities.state_dict import validate_checkpoint_compatibility

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(num_classes=91, num_queries=300, group_detr=13):
    """Build a minimal checkpoint dict with the given class count.

    Args:
        num_classes: Total classes including background (bias shape).
        num_queries: Number of object queries per group.
        group_detr: Number of groups.

    Examples:
        >>> checkpoint = _make_checkpoint(num_classes=3, num_queries=2, group_detr=4)
        >>> checkpoint["model"]["class_embed.bias"].shape
        torch.Size([3])
        >>> checkpoint["model"]["query_feat.weight"].shape
        torch.Size([8, 256])
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


@pytest.fixture
def suppress_pretrain_io(monkeypatch) -> None:
    """Suppress download/validate/file-existence side effects on the canonical load path."""
    monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
    monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
    monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
    monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)


# ---------------------------------------------------------------------------
# Regression tests: load_pretrain_weights (models/weights.py)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("suppress_pretrain_io")
class TestLoadPretrainWeightsSecondReinit:
    """Regression tests for ``load_pretrain_weights`` in ``rfdetr.models.weights``.

    Validates that the second reinitialize_detection_head call only fires when the checkpoint has MORE classes than
    configured (backbone pretrain scenario), not when it has fewer (fine-tuned checkpoint scenario).
    """

    @pytest.mark.parametrize(
        ("config_kwargs", "checkpoint_classes", "expected_reinit_sizes", "expected_num_classes"),
        [
            pytest.param({}, 3, [3], 2, id="finetune-fewer-classes-single-reinit"),
            pytest.param({"num_classes": 90}, 91, [], 90, id="matching-class-count-no-reinit"),
            pytest.param({"num_classes": 2}, 91, [91, 3], 2, id="backbone-pretrain-expands-then-trims"),
            pytest.param({"num_classes": 93}, 91, [91, 94], 93, id="explicit-larger-num-classes-reexpands"),
        ],
    )
    def test_head_reinit_sequence_and_num_classes(
        self, monkeypatch, config_kwargs, checkpoint_classes, expected_reinit_sizes, expected_num_classes
    ):
        """The loader resizes the head only as needed and auto-aligns ``num_classes`` only when the user left it unset.

        Fine-tuned checkpoint (3 logits, default ``num_classes``): one reinit to the checkpoint size and ``num_classes``
        becomes 2; a second reinit to 91 would destroy the loaded weights. COCO checkpoint (91 logits) with a matching
        ``num_classes=90``: no reinit. COCO checkpoint with ``num_classes=2``: expand to 91 for ``load_state_dict``,
        then trim to 3. COCO checkpoint with an explicit ``num_classes=93``: expand to 91, then to 94, keeping
        ``num_classes``.
        """
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", **config_kwargs)
        checkpoint = _make_checkpoint(num_classes=checkpoint_classes)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()

        load_pretrain_weights(fake_model, mc)

        assert fake_model.reinitialize_detection_head.call_args_list == [call(size) for size in expected_reinit_sizes]
        assert mc.num_classes == expected_num_classes

    def test_keep_and_resize_warning_agrees_with_loader_behaviour(self, monkeypatch, caplog):
        """The real validator's "Keeping num_classes" warning matches the head the loader leaves behind.

        Every other loader test stubs the validator, so the message and the loader's keep-and-resize decision are
        otherwise pinned only separately. Here a 91-row checkpoint is loaded with an explicit ``num_classes=93`` through
        the real validator: if the loader's decision ever changes without the message, one of the two assertions fails.
        """
        mc = RFDETRNanoConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=93)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["args"].patch_size = mc.patch_size
        monkeypatch.setattr(
            "rfdetr.models.weights.validate_checkpoint_compatibility", validate_checkpoint_compatibility
        )
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        fake_model = MagicMock()

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            load_pretrain_weights(fake_model, mc)

        warning_msgs = [r.getMessage() for r in caplog.records if r.name == "rf-detr"]
        assert any("Keeping num_classes=93" in msg for msg in warning_msgs), warning_msgs
        assert fake_model.reinitialize_detection_head.call_args_list == [call(91), call(94)]
        assert mc.num_classes == 93

    # Every non-deprecated model config except RFDETRBaseConfig, which is the base class of the variants below.
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

        Scenario (from user bug report): user trains on 8 categories (IDs 0-7), so the checkpoint stores
        ``class_embed.bias`` with shape [9] (8 classes + 1 background). Loading without ``num_classes`` must NOT trigger
        a second reinit to 91 after temporarily matching the checkpoint size for ``load_state_dict``, which would
        produce out-of-bounds class IDs like 73.
        The loader must auto-align ``mc.num_classes`` to 8 (9 - 1) with exactly one reinit call, to 9.
        """
        checkpoint = _make_checkpoint(num_classes=9)
        mc = config_cls(pretrain_weights="/fake/weights.pth", device="cpu")
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        load_pretrain_weights(fake_model, mc)

        assert fake_model.reinitialize_detection_head.call_args_list == [call(9)]
        assert mc.num_classes == 8


# ---------------------------------------------------------------------------
# Regression #960: PE interpolation for custom resolution
# ---------------------------------------------------------------------------

PE_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"


@pytest.mark.usefixtures("suppress_pretrain_io")
class TestLoadPretrainWeightsPEInterpolation:
    """Regression tests for #960 — PE must be interpolated when resolution changes.

    ``load_pretrain_weights`` must bicubic-interpolate the checkpoint's DINOv2 positional embeddings to match the
    model's ``positional_encoding_size`` before calling ``load_state_dict``.  Without this, any custom ``resolution``
    that changes the PE grid size causes a ``RuntimeError: size mismatch``.
    """

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

        Regression for #960: ``load_pretrain_weights`` must not raise ``RuntimeError`` when model resolution differs
        from checkpoint resolution.  The PE tensor in the checkpoint must be resized in-place before ``load_state_dict``
        is called.
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

        Ensures PE interpolation is a no-op for same-resolution checkpoints so that normal weight loading is unaffected.
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

        RFDETRBaseConfig.positional_encoding_size=37 is not updated by ``_sync_pe_with_resolution`` because 37 ≠
        560//16=35 (not formula-derived). Loading a checkpoint with a smaller PE grid (e.g., 24×24) must still trigger
        interpolation to the model's fixed PE=37×37 target.
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

        When ``n_source`` is not a perfect square the interpolation is skipped to avoid producing malformed embeddings.
        The tensor must remain untouched and a warning must be emitted via the weights module logger.
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


class TestL1FacadePEInterpolationEndToEnd:
    """An L1 facade variant built at a custom ``resolution`` loads a checkpoint trained at its default resolution.

    In v1.6.5 the facades (``RFDETRNano``, ``RFDETRSegNano``, ...) used a private ``_load_pretrain_weights_into`` helper
    in ``detr.py`` that bypassed the PE bicubic interpolation of ``models.weights.load_pretrain_weights``. This pins the
    wiring through the unified loader (``inference._build_model_context``) so a divergent loader path cannot silently
    skip PE interpolation. Only upward interpolation is covered; add a downward row (high-res checkpoint into a lower-
    res model) when extending.
    """

    @pytest.mark.parametrize(
        ("model_cls", "source_num_classes", "ckpt_args", "default_pe_grid", "patch_size", "new_resolution"),
        [
            pytest.param(RFDETRNano, 2, {"class_names": ["a", "b"]}, 24, 16, 640, id="nano-detection"),
            pytest.param(
                RFDETRSegNano,
                90,
                {"class_names": [], "segmentation_head": True},
                26,
                12,
                624,
                id="seg-nano-segmentation",
            ),
        ],
    )
    def test_default_pe_checkpoint_loads_at_custom_resolution(
        self, tmp_path, model_cls, source_num_classes, ckpt_args, default_pe_grid, patch_size, new_resolution
    ):
        """A state_dict saved at the default resolution loads at a higher resolution via PE interpolation in the L1
        facade.

        Detection mirrors issue 990 (a PE size mismatch raised from ``LWDETR.load_state_dict``), reduced to RFDETRNano
        for speed. Segmentation is the regression for issue 1023, where the same ``RuntimeError`` hit ``RFDETRSegNano``;
        its source model has 90 classes like an official COCO checkpoint, so loading with 2 classes exercises head
        reinit and PE interpolation together. The 84x84 PE grid reported in 1023 is deferred to a follow-up row.
        """
        default_model = model_cls(pretrain_weights=None, num_classes=source_num_classes, device="cpu")
        assert default_model.model_config.positional_encoding_size == default_pe_grid
        assert default_model.model_config.patch_size == patch_size
        default_state = default_model.model.model.state_dict()
        pe_dim = default_state[PE_KEY].shape[-1]
        assert default_state[PE_KEY].shape == torch.Size([1, default_pe_grid * default_pe_grid + 1, pe_dim])
        # Saved as .pth (not .pt) so the tmp_path does not trigger the ModelWeights registry / MD5 lookup.
        ckpt_path = tmp_path / "finetuned.pth"
        torch.save(
            {"model": dict(default_state), "args": SimpleNamespace(patch_size=patch_size, **ckpt_args)}, ckpt_path
        )

        loaded = model_cls(pretrain_weights=str(ckpt_path), resolution=new_resolution, num_classes=2, device="cpu")

        expected_pe_grid = new_resolution // patch_size
        assert loaded.model_config.positional_encoding_size == expected_pe_grid
        loaded_pe = loaded.model.model.state_dict()[PE_KEY]
        assert loaded_pe.shape == torch.Size([1, expected_pe_grid * expected_pe_grid + 1, pe_dim]), (
            f"Backbone PE was not interpolated to the requested resolution; got shape {tuple(loaded_pe.shape)}, "
            f"expected [1, {expected_pe_grid**2 + 1}, {pe_dim}]."
        )


# ---------------------------------------------------------------------------
# Regression #1038: PE interpolation for custom resolution — training path
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("suppress_pretrain_io")
class TestModuleLoadPretrainWeightsPEInterpolationCustomResolution:
    """Regression for #1038 — PE interpolation must fire through ``RFDETRModelModule.__init__``.

    The L2 training entry path (``RFDETRSegLarge(resolution=1008).train(...)``) constructs an
    :class:`~rfdetr.training.module_model.RFDETRModelModule` whose ``__init__`` delegates to
    :func:`~rfdetr.models.weights.load_pretrain_weights`. That helper must bicubic-interpolate the checkpoint's DINOv2
    positional embeddings to match ``model_config.positional_encoding_size`` before calling ``load_state_dict``.
    Without this, any ``model.train()`` call with a custom ``resolution`` that changes the PE grid raises::

        RuntimeError: Error(s) in loading state_dict for LWDETR:
            size mismatch for backbone.0.encoder.encoder.embeddings.position_embeddings

    These tests exercise the construction path end-to-end (mocking only the heavy ``build_model_from_config`` /
    ``build_criterion_from_config`` calls and disk I/O), so the regression cannot reappear if the in-init delegation to
    ``load_pretrain_weights`` is removed.
    """

    def _construct_module(self, mc, checkpoint, monkeypatch, tmp_path):
        """Construct an RFDETRModelModule with all heavy work mocked.

        Returns the constructed module and the fake nn_model whose ``load_state_dict`` receives the (now-interpolated)
        state dict.

        Examples:
            Needs pytest's ``monkeypatch`` and ``tmp_path`` fixtures, so tests cover the live call.

            >>> cls = TestModuleLoadPretrainWeightsPEInterpolationCustomResolution
            >>> callable(cls._construct_module)  # doctest: +SKIP
            True
        """
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = MagicMock()
        fake_model.num_classes = mc.num_classes  # head already aligned, so the loader need not introspect the mock
        monkeypatch.setattr(
            "rfdetr.training.module_model.build_model_from_config",
            lambda *a, **kw: fake_model,
        )
        monkeypatch.setattr(
            "rfdetr.training.module_model.build_criterion_from_config",
            lambda *a, **kw: (MagicMock(), MagicMock()),
        )

        tc = TrainConfig(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=1,
            lr=1e-4,
            lr_encoder=1.5e-4,
            batch_size=2,
            weight_decay=1e-4,
            lr_scheduler_kwargs={"lr_drop": 1},
            warmup_epochs=0.0,
            drop_path=0.0,
            multi_scale=False,
            expanded_scales=False,
            grad_accum_steps=1,
            tensorboard=False,
        )
        from rfdetr.training.module_model import RFDETRModelModule

        module = RFDETRModelModule(mc, tc)
        return module, fake_model

    @pytest.mark.parametrize(
        "config_cls, src_pe_size, tgt_pe_size",
        [
            # All 7 segmentation variants listed in #1038, plus detection up/downscale.
            pytest.param(RFDETRSegNanoConfig, 26, 84, id="seg_nano_upscale_26_to_84"),
            pytest.param(RFDETRSegSmallConfig, 32, 84, id="seg_small_upscale_32_to_84"),
            pytest.param(RFDETRSegMediumConfig, 36, 84, id="seg_medium_upscale_36_to_84"),
            pytest.param(RFDETRSegLargeConfig, 42, 84, id="seg_large_upscale_42_to_84"),
            pytest.param(RFDETRSegPreviewConfig, 24, 84, id="seg_preview_upscale_24_to_84"),
            pytest.param(RFDETRSegXLargeConfig, 52, 84, id="seg_xlarge_upscale_52_to_84"),
            pytest.param(RFDETRSeg2XLargeConfig, 64, 84, id="seg_2xlarge_upscale_64_to_84"),
            pytest.param(RFDETRNanoConfig, 24, 40, id="nano_upscale_24_to_40"),
            pytest.param(RFDETRNanoConfig, 40, 24, id="nano_downscale_40_to_24"),
        ],
    )
    def test_pe_interpolated_in_training_path(self, monkeypatch, config_cls, src_pe_size, tgt_pe_size, tmp_path):
        """Module construction interpolates PE to match ``positional_encoding_size``.

        Regression for #1038 — ``RFDETRModelModule.__init__`` must trigger PE interpolation through the canonical loader
        so ``load_state_dict`` does not raise ``RuntimeError: size mismatch`` at custom training resolutions.
        """
        mc = config_cls(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            positional_encoding_size=tgt_pe_size,
        )

        dim = 384
        src_n = src_pe_size * src_pe_size + 1
        checkpoint = _make_checkpoint(num_classes=mc.num_classes + 1)
        checkpoint["model"][PE_KEY] = torch.randn(1, src_n, dim)

        _, fake_model = self._construct_module(mc, checkpoint, monkeypatch, tmp_path)

        pe = checkpoint["model"][PE_KEY]
        expected_n = tgt_pe_size * tgt_pe_size + 1
        assert pe.shape == torch.Size([1, expected_n, dim]), (
            f"Expected PE shape [1, {expected_n}, {dim}] after interpolation from "
            f"{src_pe_size}x{src_pe_size} to {tgt_pe_size}x{tgt_pe_size}, got {tuple(pe.shape)}. "
            "PE interpolation must fire during RFDETRModelModule.__init__ via canonical load_pretrain_weights."
        )
        # load_state_dict was called on the model with the interpolated state dict.
        fake_model.load_state_dict.assert_called_once()

    def test_matching_pe_not_modified_in_training_path(self, monkeypatch, tmp_path):
        """Same-resolution checkpoint PE is untouched in the training path."""
        pe_size = 24
        mc = RFDETRNanoConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            positional_encoding_size=pe_size,
        )

        dim = 384
        original_pe = torch.randn(1, pe_size * pe_size + 1, dim)
        checkpoint = _make_checkpoint(num_classes=mc.num_classes + 1)
        checkpoint["model"][PE_KEY] = original_pe.clone()

        self._construct_module(mc, checkpoint, monkeypatch, tmp_path)

        pe = checkpoint["model"][PE_KEY]
        assert pe.shape == torch.Size([1, pe_size * pe_size + 1, dim]), "Matching PE must not be modified."
        assert torch.equal(pe, original_pe), "Matching PE values must not be modified."


# ---------------------------------------------------------------------------
# Regression: keypoint schema auto-align from checkpoint _kp_active_mask
# ---------------------------------------------------------------------------


def _make_kp_checkpoint(kp_schema: list[int], num_queries: int = 300, group_detr: int = 13) -> dict:
    """Build a minimal keypoint checkpoint encoding *kp_schema* via ``_kp_active_mask``.

    Args:
        kp_schema: Keypoints-per-class list, e.g. ``[0, 17]`` for bg-first or ``[17]`` for active-first.
        num_queries: Number of queries per group.
        group_detr: Number of decoder groups.

    Returns:
        Checkpoint dict with ``model``, ``args``, and schema-derived ``_kp_active_mask``.

    Examples:
        >>> mask = _make_kp_checkpoint([0, 17], num_queries=2, group_detr=2)["model"]["_kp_active_mask"]
        >>> mask.shape, mask.sum(dim=1).tolist()
        (torch.Size([2, 17]), [0, 17])
    """
    num_classes = len(kp_schema)
    total_queries = num_queries * group_detr
    max_kp = max(kp_schema) if kp_schema else 0
    mask = torch.zeros(num_classes, max_kp, dtype=torch.bool)
    for idx, n_kp in enumerate(kp_schema):
        mask[idx, :n_kp] = True
    state = {
        "class_embed.weight": torch.randn(num_classes + 1, 256),
        "class_embed.bias": torch.randn(num_classes + 1),
        "refpoint_embed.weight": torch.randn(total_queries, 4),
        "query_feat.weight": torch.randn(total_queries, 256),
        "_kp_active_mask": mask,
    }
    ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14, class_names=[])
    return {"model": state, "args": ckpt_args}


def _make_kp_fake_model(initial_schema: list[int]) -> MagicMock:
    """Build a fake nn_model mock that mimics the GroupPose keypoint model interface.

    Args:
        initial_schema: Keypoints-per-class schema the model was built with.

    Returns:
        Configured MagicMock with keypoint schema methods and ``_kp_active_mask`` state.

    Examples:
        >>> model = _make_kp_fake_model([17])
        >>> model.reinitialize_keypoint_head([0, 17])
        >>> model.get_num_keypoints_per_class(), model.state_dict()["_kp_active_mask"].shape
        ([0, 17], torch.Size([2, 17]))
    """
    max_kp = max(initial_schema) if initial_schema else 0
    current_schema = list(initial_schema)
    mask = torch.zeros(len(initial_schema), max_kp, dtype=torch.bool)
    for idx, n in enumerate(initial_schema):
        mask[idx, :n] = True

    fake_model = MagicMock()
    fake_model.state_dict.return_value = {"_kp_active_mask": mask.clone()}

    def _reinit_kp(schema):
        current_schema[:] = schema
        new_max = max(schema) if schema else 0
        new_mask = torch.zeros(len(schema), new_max, dtype=torch.bool)
        for i, n in enumerate(schema):
            new_mask[i, :n] = True
        fake_model.state_dict.return_value = {"_kp_active_mask": new_mask}
        fake_model.get_num_keypoints_per_class.return_value = list(schema)

    fake_model.reinitialize_keypoint_head.side_effect = _reinit_kp
    fake_model.get_num_keypoints_per_class.return_value = list(initial_schema)
    fake_model.get_num_keypoints_per_class_from_checkpoint = lambda sd: (
        [int(n) for n in sd["_kp_active_mask"].sum(dim=1).tolist()] if "_kp_active_mask" in sd else None
    )
    return fake_model


@pytest.mark.usefixtures("suppress_pretrain_io")
class TestLoadPretrainWeightsKeypointSchemaAutoAlign:
    """Regression for AP=0 when loading bg-first checkpoint into active-first default config.

    Before the fix, ``load_pretrain_weights`` would restore ``mc.num_keypoints_per_class`` to the config default
    ``[17]`` after loading a ``[0, 17]`` pretrained checkpoint.  The detection-head trimming kept rows 0 (background)
    and 1 (person) from the checkpoint but the active-first schema treats row 0 as person, producing AP≈0.

    The fix auto-aligns ``mc.num_keypoints_per_class`` from the checkpoint ``_kp_active_mask`` when the user did not
    explicitly set the field, mirroring the existing ``num_classes`` auto-align pattern.
    """

    def test_bg_first_checkpoint_auto_aligns_active_first_config(self, monkeypatch):
        """Loading bg-first [0,17] checkpoint into active-first [17] config auto-aligns schema."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights="/fake/kp.pth", device="cpu")
        assert mc.num_keypoints_per_class == [17], "Precondition: config default is active-first [17]."

        checkpoint = _make_kp_checkpoint([0, 17])
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = _make_kp_fake_model([17])

        load_pretrain_weights(fake_model, mc)

        assert mc.num_keypoints_per_class == [0, 17], (
            f"expected auto-align to [0, 17], got {mc.num_keypoints_per_class}"
        )
        assert mc.num_classes == 2, (
            f"mc.num_classes must be auto-aligned to 2 (checkpoint has 3 logit slots), got {mc.num_classes}"
        )

    def test_matching_schema_no_change(self, monkeypatch):
        """Config and checkpoint with same schema leaves mc.num_keypoints_per_class unchanged."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights="/fake/kp.pth", device="cpu")
        mc.num_keypoints_per_class = [17]

        checkpoint = _make_kp_checkpoint([17])
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = _make_kp_fake_model([17])

        load_pretrain_weights(fake_model, mc)

        assert mc.num_keypoints_per_class == [17]

    def test_user_explicit_schema_not_overridden(self, monkeypatch):
        """Explicit num_keypoints_per_class from user survives even when checkpoint differs."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights="/fake/kp.pth", device="cpu")
        mc.num_keypoints_per_class = [0, 33]
        mc.model_fields_set.add("num_keypoints_per_class")

        checkpoint = _make_kp_checkpoint([0, 17])
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = _make_kp_fake_model([0, 33])

        load_pretrain_weights(fake_model, mc)

        assert mc.num_keypoints_per_class == [0, 33], (
            "Explicit user schema must not be overridden by checkpoint auto-align."
        )

    def test_1d_kp_active_mask_skips_auto_align_with_warning(self, monkeypatch):
        """Malformed 1-D _kp_active_mask skips auto-align and emits a logger.warning."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights="/fake/kp.pth", device="cpu")
        assert mc.num_keypoints_per_class == [17], "Precondition: config default is active-first [17]."

        checkpoint = _make_kp_checkpoint([0, 17])
        # Replace the 2-D mask with a malformed 1-D tensor.
        checkpoint["model"]["_kp_active_mask"] = torch.ones(17, dtype=torch.bool)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = _make_kp_fake_model([17])
        # The fake's get_num_keypoints_per_class_from_checkpoint sums over dim=1 (IndexError on 1-D): return None.
        fake_model.get_num_keypoints_per_class_from_checkpoint = lambda sd: None

        warned: list[str] = []
        monkeypatch.setattr("rfdetr.models.weights.logger.warning", lambda msg, *args: warned.append(msg % args))

        load_pretrain_weights(fake_model, mc)

        assert mc.num_keypoints_per_class == [17], (
            "Auto-align must not fire for a 1-D _kp_active_mask; config default should be unchanged."
        )
        assert any("unexpected shape" in msg for msg in warned), (
            f"Expected a warning mentioning 'unexpected shape'; got: {warned}"
        )

    def test_all_zero_kp_active_mask_skips_auto_align_with_warning(self, monkeypatch):
        """All-zero 2-D _kp_active_mask (degenerate) skips auto-align and emits a logger.warning."""
        mc = RFDETRKeypointPreviewConfig(pretrain_weights="/fake/kp.pth", device="cpu")
        assert mc.num_keypoints_per_class == [17], "Precondition: config default is active-first [17]."

        checkpoint = _make_kp_checkpoint([0, 17])
        # Replace mask with a valid 2-D shape but all zeros (no active slots).
        checkpoint["model"]["_kp_active_mask"] = torch.zeros(2, 17, dtype=torch.bool)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)
        fake_model = _make_kp_fake_model([17])

        warned: list[str] = []
        monkeypatch.setattr("rfdetr.models.weights.logger.warning", lambda msg, *args: warned.append(msg % args))

        load_pretrain_weights(fake_model, mc)

        assert mc.num_keypoints_per_class == [17], (
            "Auto-align must not fire for an all-zero _kp_active_mask; config default should be unchanged."
        )
        assert any("no active slots" in msg for msg in warned), (
            f"Expected a warning mentioning 'no active slots'; got: {warned}"
        )
