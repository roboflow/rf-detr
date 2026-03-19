# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for fine-tuned checkpoint weight destruction.

When a user loads a fine-tuned N-class checkpoint but has ``num_classes``
configured to a LARGER value (e.g. default 90), the second reinit in both
``_load_pretrain_weights_into`` (detr.py) and ``_load_pretrain_weights``
(training/module.py) erroneously resizes the detection head to
``args.num_classes + 1``, destroying the loaded weights.

The fix changes the second reinit condition from:
    ``checkpoint_num_classes != args.num_classes + 1``
to:
    ``args.num_classes + 1 < checkpoint_num_classes``

This preserves the "backbone pretrain" scenario (checkpoint has MORE classes
than configured, e.g. COCO 91-class pretrain to fine-tune on 2 classes) while
no longer clobbering loaded weights when the checkpoint has fewer classes.

Each scenario is tested for both code paths:
  - Path 1: ``_load_pretrain_weights_into`` in ``src/rfdetr/detr.py``
  - Path 2: ``_load_pretrain_weights`` in ``src/rfdetr/training/module.py``
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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


# ---------------------------------------------------------------------------
# Path 1: _load_pretrain_weights_into (detr.py)
# ---------------------------------------------------------------------------


def _make_detr_args(num_classes=90, num_queries=300, group_detr=13):
    """Return a SimpleNamespace shaped like the args for _load_pretrain_weights_into."""
    return SimpleNamespace(
        pretrain_weights="/fake/weights.pth",
        num_classes=num_classes,
        num_queries=num_queries,
        group_detr=group_detr,
        segmentation_head=False,
        patch_size=14,
    )


class TestLoadPretrainWeightsIntoSecondReinit:
    """Regression tests for _load_pretrain_weights_into (detr.py path).

    Validates that the second reinitialize_detection_head call only fires when
    the checkpoint has MORE classes than configured (backbone pretrain scenario),
    not when it has fewer (fine-tuned checkpoint scenario).
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.detr.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.detr.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.detr.os.path.isfile", lambda _: True)

    def test_finetune_checkpoint_preserves_weights(self, monkeypatch):
        """Fine-tuned checkpoint (fewer classes) must NOT trigger second reinit.

        Scenario: 2-class fine-tuned checkpoint (bias shape [3]) loaded with
        num_classes=90 (the default). The first reinit correctly resizes the
        head to 3 so load_state_dict works. The second reinit must NOT resize
        to 91 — that would destroy the loaded fine-tuned weights.
        """
        from rfdetr.detr import _load_pretrain_weights_into

        checkpoint = _make_checkpoint(num_classes=3)  # 2-class fine-tuned
        monkeypatch.setattr("rfdetr.detr.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        args = _make_detr_args(num_classes=90)

        _load_pretrain_weights_into(fake_model, args)

        # First reinit to checkpoint size (3) must happen.
        # Second reinit to 91 must NOT happen — that is the bug.
        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls[0] == call(3), f"First reinit should resize to checkpoint size 3, got {calls[0]}"
        assert len(calls) == 1, (
            f"Expected exactly 1 reinit call (to checkpoint size), but got {len(calls)}: "
            f"{calls}. The second reinit to {args.num_classes + 1} destroys loaded weights."
        )

    def test_no_mismatch_no_reinit(self, monkeypatch):
        """Checkpoint class count matches config — no reinit at all.

        Scenario: COCO checkpoint (91 classes) with num_classes=90.
        91 == 90 + 1, so no reinit should fire.
        """
        from rfdetr.detr import _load_pretrain_weights_into

        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.detr.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        args = _make_detr_args(num_classes=90)

        _load_pretrain_weights_into(fake_model, args)

        fake_model.reinitialize_detection_head.assert_not_called()

    def test_backbone_pretrain_still_reinits(self, monkeypatch):
        """Backbone pretrain (more classes in checkpoint) must still reinit.

        Scenario: COCO 91-class checkpoint loaded for 2-class fine-tuning
        (num_classes=2). Both reinits are correct here: first to 91 for
        load_state_dict, second to 3 for the configured class count.
        """
        from rfdetr.detr import _load_pretrain_weights_into

        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.detr.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        args = _make_detr_args(num_classes=2)

        _load_pretrain_weights_into(fake_model, args)

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(3)], f"Expected reinit to [91, 3] (expand then trim), got {calls}"


# ---------------------------------------------------------------------------
# Path 2: RFDETRModule._load_pretrain_weights (training/module.py)
# ---------------------------------------------------------------------------


def _fake_model():
    """Return a MagicMock that behaves enough like an LWDETR model."""
    from torch import nn

    model = MagicMock(spec=nn.Module)
    real_param = nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = iter([real_param])
    model.named_parameters.return_value = iter([("weight", real_param)])
    model.update_drop_path = MagicMock()
    model.update_dropout = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    return model


def _build_module(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRModule with build_model and build_criterion_and_postprocessors mocked."""
    mc = model_config or RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=5)
    tc = train_config or TrainConfig(
        dataset_dir=str(tmp_path / "dataset") if tmp_path else "/nonexistent/dataset",
        output_dir=str(tmp_path / "output") if tmp_path else "/nonexistent/output",
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
    fake = _fake_model()
    fake_criterion = MagicMock()
    fake_criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    fake_postprocess = MagicMock()
    with (
        patch("rfdetr.training.module.build_model", return_value=fake),
        patch(
            "rfdetr.training.module.build_criterion_and_postprocessors",
            return_value=(fake_criterion, fake_postprocess),
        ),
    ):
        from rfdetr.training.module import RFDETRModule

        module = RFDETRModule(mc, tc)
    return module, fake


class TestModuleLoadPretrainWeightsSecondReinit:
    """Regression tests for RFDETRModule._load_pretrain_weights (module.py path).

    Validates that the second reinitialize_detection_head call only fires when
    the checkpoint has MORE classes than configured (backbone pretrain scenario),
    not when it has fewer (fine-tuned checkpoint scenario).
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.training.module.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.training.module.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.training.module.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.training.module.os.path.isfile", lambda _: True)

    @patch("rfdetr.training.module.torch.load")
    def test_finetune_checkpoint_preserves_weights(self, mock_torch_load, tmp_path):
        """Fine-tuned checkpoint (fewer classes) must NOT trigger second reinit.

        Scenario: 2-class fine-tuned checkpoint (bias shape [3]) loaded with
        default num_classes=90. The first reinit correctly resizes the head to 3 so
        load_state_dict works. The second reinit must NOT resize to 91.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        module, fake_model = _build_module(model_config=mc, tmp_path=tmp_path)

        checkpoint = _make_checkpoint(num_classes=3)  # 2-class fine-tuned
        mock_torch_load.return_value = checkpoint
        module._args.pretrain_weights = "/fake/weights.pth"

        module._load_pretrain_weights()

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls[0] == call(3), f"First reinit should resize to checkpoint size 3, got {calls[0]}"
        assert len(calls) == 1, (
            f"Expected exactly 1 reinit call (to checkpoint size), but got {len(calls)}: "
            f"{calls}. The second reinit to {module._args.num_classes + 1} destroys loaded weights."
        )
        assert module._args.num_classes == 2, "Default num_classes should auto-align to checkpoint classes."

    @patch("rfdetr.training.module.torch.load")
    def test_user_override_larger_than_checkpoint_reexpands_head(self, mock_torch_load, tmp_path):
        """Explicit larger num_classes must be restored after checkpoint load.

        Scenario: 91-class checkpoint loaded with explicit num_classes=93.
        Loader must temporarily match checkpoint size for load_state_dict, then
        expand to 94 logits and keep args.num_classes unchanged.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=93)
        module, fake_model = _build_module(model_config=mc, tmp_path=tmp_path)

        checkpoint = _make_checkpoint(num_classes=91)
        mock_torch_load.return_value = checkpoint
        module._args.pretrain_weights = "/fake/weights.pth"

        module._load_pretrain_weights()

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(94)], f"Expected reinit to [91, 94] (load then expand), got {calls}"
        assert module._args.num_classes == 93, "Explicitly configured num_classes must not be overwritten."

    @patch("rfdetr.training.module.torch.load")
    def test_no_mismatch_no_reinit(self, mock_torch_load, tmp_path):
        """Checkpoint class count matches config — no reinit at all.

        Scenario: COCO checkpoint (91 classes) with num_classes=90.
        91 == 90 + 1, so no reinit should fire.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=90)
        module, fake_model = _build_module(model_config=mc, tmp_path=tmp_path)

        checkpoint = _make_checkpoint(num_classes=91)
        mock_torch_load.return_value = checkpoint
        module._args.pretrain_weights = "/fake/weights.pth"

        module._load_pretrain_weights()

        fake_model.reinitialize_detection_head.assert_not_called()

    @patch("rfdetr.training.module.torch.load")
    def test_backbone_pretrain_still_reinits(self, mock_torch_load, tmp_path):
        """Backbone pretrain (more classes in checkpoint) must still reinit.

        Scenario: COCO 91-class checkpoint loaded for 2-class fine-tuning
        (num_classes=2). Both reinits are correct here: first to 91 for
        load_state_dict, second to 3 for the configured class count.
        """
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=2)
        module, fake_model = _build_module(model_config=mc, tmp_path=tmp_path)

        checkpoint = _make_checkpoint(num_classes=91)
        mock_torch_load.return_value = checkpoint
        module._args.pretrain_weights = "/fake/weights.pth"

        module._load_pretrain_weights()

        calls = fake_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(3)], f"Expected reinit to [91, 3] (expand then trim), got {calls}"
