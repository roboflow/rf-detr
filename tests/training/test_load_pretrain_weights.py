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

from rfdetr.config import RFDETRBaseConfig, TrainConfig

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
