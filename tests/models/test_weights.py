# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ``rfdetr.models.weights`` — the unified weight-loading and LoRA module.

These tests cover ``load_pretrain_weights`` and ``apply_lora`` directly,
exercising the unified logic extracted from ``detr.py`` and ``module_model.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(num_classes: int = 91, num_queries: int = 300, group_detr: int = 13) -> dict:
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
        class_names=["cat", "dog"],
    )
    return {"model": state, "args": ckpt_args}


def _make_train_config(tmp_path=None) -> TrainConfig:
    """Return a minimal TrainConfig for use in load_pretrain_weights.

    Args:
        tmp_path: Optional pytest tmp_path fixture value.
    """
    return TrainConfig(
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


def _fake_nn_model() -> MagicMock:
    """Return a MagicMock that behaves enough like an LWDETR nn.Module.

    Returns:
        MagicMock with reinitialize_detection_head and load_state_dict stubs.
    """
    model = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    model.load_state_dict = MagicMock()
    return model


# ---------------------------------------------------------------------------
# load_pretrain_weights — reinit scenarios
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsReinitScenarios:
    """Verify reinitialize_detection_head call patterns for all class-count scenarios."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch):
        """Suppress all download, file-existence, and validation side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_characterization_fine_tuned_checkpoint_auto_aligns_default_num_classes(self, monkeypatch, tmp_path):
        """Fine-tuned checkpoint (fewer classes) + default num_classes → 1 reinit to ckpt size.

        When the user did NOT explicitly set num_classes (default=90), the loader
        auto-aligns to the checkpoint's class count (3 classes = bias shape [3]).
        Only one reinit fires; no second reinit back to 91.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        checkpoint = _make_checkpoint(num_classes=3)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls[0] == call(3), f"First reinit must resize to checkpoint size 3, got {calls[0]}"
        assert len(calls) == 1, (
            f"Expected exactly 1 reinit call; got {len(calls)}: {calls}. "
            "A second reinit to 91 would destroy loaded fine-tuned weights."
        )
        assert mc.num_classes == 2, "Auto-aligned checkpoint class count must be persisted back onto ModelConfig."

    def test_characterization_backbone_pretrain_two_reinits(self, monkeypatch, tmp_path):
        """Backbone pretrain (more classes in checkpoint) + explicit small num_classes → 2 reinits.

        Scenario: 91-class COCO checkpoint, user explicitly requested num_classes=2.
        First reinit to 91 so load_state_dict works; second reinit to 3 to match config.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=2)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(3)], f"Expected reinit to [91, 3] (expand then trim), got {calls}"

    def test_characterization_user_override_larger_than_checkpoint_reexpands(self, monkeypatch, tmp_path):
        """Explicit num_classes larger than checkpoint → 2 reinits (load then expand back).

        Scenario: 91-class checkpoint, user explicitly set num_classes=93.
        The head must temporarily align to 91 for loading, then expand back to 94.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=93)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == [call(91), call(94)], f"Expected reinit to [91, 94] (load then expand), got {calls}"

    def test_characterization_no_mismatch_no_reinit(self, monkeypatch, tmp_path):
        """Checkpoint class count matches config → no reinit.

        Scenario: 91-class checkpoint with num_classes=90. 91 == 90 + 1 → no reinit.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.reinitialize_detection_head.assert_not_called()


# ---------------------------------------------------------------------------
# load_pretrain_weights — class_names extraction
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsClassNames:
    """Verify that class_names are extracted from checkpoint and returned."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch):
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_characterization_class_names_extracted_from_checkpoint(self, monkeypatch, tmp_path):
        """class_names stored in checkpoint args are returned as a list of strings."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["args"] = SimpleNamespace(
            segmentation_head=False,
            patch_size=14,
            class_names=["cat", "dog", "bird"],
        )
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        assert result == ["cat", "dog", "bird"], f"Expected class names from checkpoint, got {result!r}"

    def test_characterization_empty_class_names_when_absent_from_checkpoint(self, monkeypatch, tmp_path):
        """Empty list returned when checkpoint has no args or no class_names key."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint.pop("args", None)  # no args key at all
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        assert result == [], f"Expected empty list when checkpoint has no class_names, got {result!r}"

    def test_none_pretrain_weights_returns_empty_list_immediately(self, tmp_path):
        """load_pretrain_weights returns [] without any I/O when pretrain_weights is None."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        nn_model = _fake_nn_model()

        result = load_pretrain_weights(nn_model, mc)

        assert result == [], f"Expected [] for None pretrain_weights, got {result!r}"
        nn_model.load_state_dict.assert_not_called()
        nn_model.reinitialize_detection_head.assert_not_called()


# ---------------------------------------------------------------------------
# load_pretrain_weights — PTL .ckpt format
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsPTLCkptFormat:
    """Verify that PTL-native .ckpt checkpoints (state_dict, no model key) are handled."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch):
        """Suppress all download, file-existence, and validation side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def _make_ptl_checkpoint(
        self,
        num_classes: int = 91,
        num_queries: int = 300,
        group_detr: int = 13,
    ) -> dict:
        """Build a fake PyTorch Lightning (PTL) native checkpoint with state_dict keys prefixed by 'model.'.

        Args:
            num_classes: Total classes including background (bias shape).
            num_queries: Number of object queries per group.
            group_detr: Number of groups.
        """
        total_queries = num_queries * group_detr
        raw_state = {
            "class_embed.weight": torch.randn(num_classes, 256),
            "class_embed.bias": torch.randn(num_classes),
            "refpoint_embed.weight": torch.randn(total_queries, 4),
            "query_feat.weight": torch.randn(total_queries, 256),
            "other_layer.weight": torch.randn(10, 10),
        }
        return {
            "state_dict": {f"model.{k}": v for k, v in raw_state.items()},
            "epoch": 10,
            "global_step": 1000,
        }

    def test_ptl_ckpt_loads_successfully(self, monkeypatch):
        """PTL .ckpt checkpoints (state_dict without model key) must load without KeyError."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        assert result == [], f"Expected [] (no args/class_names in checkpoint), got {result!r}"

    def test_ptl_ckpt_model_prefix_stripped_before_load_state_dict(self, monkeypatch):
        """Model weights passed to load_state_dict must not carry the 'model.' prefix."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        loaded_state = nn_model.load_state_dict.call_args[0][0]
        assert all(not k.startswith("model.") for k in loaded_state), (
            f"Keys passed to load_state_dict must not have 'model.' prefix, got: {list(loaded_state.keys())[:5]}"
        )

    def test_ptl_ckpt_no_model_prefix_in_state_dict_raises_value_error(self, monkeypatch):
        """A checkpoint with state_dict but no 'model.'-prefixed keys raises ValueError."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = {"state_dict": {"some_other.key": torch.zeros(1)}, "epoch": 10}
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        with pytest.raises(ValueError, match="model\\."):
            load_pretrain_weights(nn_model, mc)

    def test_ptl_ckpt_class_names_from_hyper_parameters(self, monkeypatch):
        """Class names stored in hyper_parameters are returned when args key is absent."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        checkpoint["hyper_parameters"] = {"class_names": ["cat", "dog"]}
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        assert result == ["cat", "dog"], f"Expected class names from hyper_parameters, got {result!r}"

    def test_ptl_ckpt_args_takes_precedence_over_hyper_parameters(self, monkeypatch):
        """When both args and hyper_parameters are present, args takes precedence."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        checkpoint["args"] = {"class_names": ["from_args"]}
        checkpoint["hyper_parameters"] = {"class_names": ["from_hyper_params"]}
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        assert result == ["from_args"], f"args must take precedence over hyper_parameters, got {result!r}"

    def test_ptl_ckpt_non_model_keys_in_state_dict_are_excluded(self, monkeypatch):
        """Non-model. keys in state_dict (optimizer, lr_scheduler) must not appear in checkpoint['model'].

        Real PTL checkpoints contain keys like 'optimizer.param_groups' and
        'lr_scheduler.last_epoch' alongside the 'model.*' weights.  The loader must
        exclude these non-model keys so they do not pollute the state dict passed to
        load_state_dict and do not cause KeyError or unexpected parameter names.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        # Inject non-model keys that a real PTL checkpoint would contain
        checkpoint["state_dict"]["optimizer.param_groups"] = torch.zeros(1)
        checkpoint["state_dict"]["lr_scheduler.last_epoch"] = torch.tensor(10)
        checkpoint["state_dict"]["callback_states.ema.shadow_params"] = torch.zeros(4)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        loaded_state = nn_model.load_state_dict.call_args[0][0]
        non_model_keys = [k for k in loaded_state if k.startswith(("optimizer.", "lr_scheduler.", "callback_states."))]
        assert not non_model_keys, f"Non-model keys must be excluded from loaded state; found: {non_model_keys}"

    def test_ptl_ckpt_torch_compile_orig_mod_prefix_stripped(self, monkeypatch):
        """PTL .ckpt from a torch.compile-wrapped model must load without KeyError.

        When a model is wrapped with torch.compile before training, PTL records weights
        under keys like "model._orig_mod.class_embed.bias".  The loader must strip both
        the "model." and the subsequent "_orig_mod." segment so the resulting keys match
        the bare parameter names expected by load_state_dict.
        """
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/last.ckpt", device="cpu", num_classes=90)
        raw_state = {
            "class_embed.weight": torch.randn(91, 256),
            "class_embed.bias": torch.randn(91),
            "refpoint_embed.weight": torch.randn(300 * 13, 4),
            "query_feat.weight": torch.randn(300 * 13, 256),
        }
        # Simulate torch.compile: keys are prefixed with "model._orig_mod."
        checkpoint = {
            "state_dict": {f"model._orig_mod.{k}": v for k, v in raw_state.items()},
            "epoch": 5,
        }
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        loaded_state = nn_model.load_state_dict.call_args[0][0]
        assert all(not k.startswith(("model.", "_orig_mod.")) for k in loaded_state), (
            f"Keys must have both 'model.' and '_orig_mod.' stripped; got: {list(loaded_state.keys())[:5]}"
        )

    def test_best_model_callback_format_with_both_model_and_state_dict_still_works(self, monkeypatch):
        """Checkpoints with both 'model' and 'state_dict' (BestModelCallback format) must still load."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(pretrain_weights="/fake/checkpoint_best_total.pth", device="cpu", num_classes=90)
        # BestModelCallback writes both "model" (raw keys) and "state_dict" (prefixed keys).
        raw_state = {
            "class_embed.weight": torch.randn(91, 256),
            "class_embed.bias": torch.randn(91),
            "refpoint_embed.weight": torch.randn(300 * 13, 4),
            "query_feat.weight": torch.randn(300 * 13, 256),
        }
        checkpoint = {
            "model": raw_state,
            "state_dict": {f"model.{k}": v for k, v in raw_state.items()},
            "epoch": 5,
        }
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


class TestApplyLora:
    """Verify that apply_lora applies LoRA adapters to the backbone encoder.

    ``apply_lora`` lazily imports ``peft`` inside the function body, so we use
    ``patch.dict("sys.modules", ...)`` to intercept the import rather than
    patching a module-level name.
    """

    def test_characterization_apply_lora_wraps_backbone_encoder(self):
        """apply_lora must call get_peft_model on nn_model.backbone[0].encoder."""
        from rfdetr.models.weights import apply_lora

        nn_model = MagicMock()
        fake_peft_model = MagicMock()

        mock_peft = MagicMock()
        mock_peft.get_peft_model.return_value = fake_peft_model

        with patch.dict("sys.modules", {"peft": mock_peft}):
            apply_lora(nn_model)

        mock_peft.LoraConfig.assert_called_once()
        lora_kwargs = mock_peft.LoraConfig.call_args.kwargs
        assert lora_kwargs.get("r") == 16, "LoRA rank must be 16"
        assert lora_kwargs.get("lora_alpha") == 16, "LoRA alpha must be 16"
        assert lora_kwargs.get("use_dora") is True, "DoRA must be enabled"

        assert mock_peft.get_peft_model.call_count == 1, "get_peft_model must be called exactly once"
        assert nn_model.backbone[0].encoder is fake_peft_model, "backbone encoder must be replaced with the peft model"

    def test_characterization_apply_lora_target_modules(self):
        """apply_lora must target exactly the 9 expected module names."""
        from rfdetr.models.weights import apply_lora

        nn_model = MagicMock()
        mock_peft = MagicMock()

        with patch.dict("sys.modules", {"peft": mock_peft}):
            apply_lora(nn_model)

        expected_targets = {
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        }
        actual_targets = set(mock_peft.LoraConfig.call_args.kwargs.get("target_modules", []))
        assert actual_targets == expected_targets, (
            f"LoRA target_modules mismatch.\nExpected: {expected_targets}\nGot: {actual_targets}"
        )
