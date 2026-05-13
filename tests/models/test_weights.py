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
from rfdetr.models.weights import _warn_on_partial_load

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


# ---------------------------------------------------------------------------
# Per-group query embedding slicing
# ---------------------------------------------------------------------------


def _labelled_query_tensor(num_queries: int, group_detr: int, dim: int = 2) -> torch.Tensor:
    """Build a query embedding tensor where row ``g * num_queries + q`` encodes
    ``[g * 100 + q, 0, ...]``.

    This lets tests check the per-group ordering of the result without floating-point
    fuzz: the first column carries the (group, query) identity directly.
    """
    rows = []
    for g in range(group_detr):
        for q in range(num_queries):
            rows.append([float(g * 100 + q)] + [0.0] * (dim - 1))
    return torch.tensor(rows, dtype=torch.float32)


class TestSliceQueryParamPerGroup:
    """Direct unit tests for ``_slice_query_param_per_group``.

    The helper is the fix for a latent bug where a flat ``tensor[:N]`` slice
    scrambled per-group structure when ``num_queries`` decreased with
    ``group_detr > 1``.  See the docstring in ``rfdetr.models.weights`` for the
    ``LWDETR`` packing layout that motivates these tests.
    """

    def test_returns_input_unchanged_when_dimensions_match(self):
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=4, target_group_detr=3)
        assert out is tensor

    def test_num_queries_decrease_preserves_per_group_structure(self):
        """The bug being fixed: 4→2 queries with 3 groups must keep first 2 of each group."""
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=2, target_group_detr=3)
        # Expect rows: g0q0, g0q1, g1q0, g1q1, g2q0, g2q1 → labels 0, 1, 100, 101, 200, 201.
        labels = out[:, 0].int().tolist()
        assert labels == [0, 1, 100, 101, 200, 201], (
            f"Per-group structure scrambled. A flat slice would give {tensor[:6, 0].int().tolist()}."
        )

    def test_group_detr_decrease_drops_tail_groups(self):
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=4, target_group_detr=2)
        labels = out[:, 0].int().tolist()
        # First 2 groups, all 4 queries each.
        assert labels == [0, 1, 2, 3, 100, 101, 102, 103]

    def test_both_decrease(self):
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=2, target_group_detr=2)
        labels = out[:, 0].int().tolist()
        assert labels == [0, 1, 100, 101]

    def test_falls_back_to_flat_slice_on_inconsistent_shape(self):
        """If args don't match the tensor's flat length, defer to legacy behavior."""
        from rfdetr.models.weights import _slice_query_param_per_group

        weird = torch.arange(7, dtype=torch.float32).unsqueeze(1)  # shape [7, 1], not 4*3=12
        out = _slice_query_param_per_group(weird, 4, 3, target_num_queries=2, target_group_detr=2)
        # Legacy: tensor[:4]
        assert out.shape == (4, 1)
        assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        "ckpt_nq,ckpt_g,tgt_nq,tgt_g,expected_labels",
        [
            pytest.param(
                4,
                3,
                8,
                3,
                [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203],
                id="nq_expands_g_equal",
            ),
            pytest.param(
                4,
                2,
                4,
                4,
                [0, 1, 2, 3, 100, 101, 102, 103],
                id="g_expands_nq_equal",
            ),
            pytest.param(
                4,
                3,
                8,
                2,
                [0, 1, 2, 3, 100, 101, 102, 103],
                id="nq_expands_g_shrinks",
            ),
            pytest.param(
                4,
                3,
                2,
                4,
                [0, 1, 100, 101, 200, 201],
                id="nq_shrinks_g_expands",
            ),
            pytest.param(
                4,
                3,
                8,
                4,
                [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203],
                id="both_expand",
            ),
        ],
    )
    def test_expansion_combos(
        self,
        ckpt_nq: int,
        ckpt_g: int,
        tgt_nq: int,
        tgt_g: int,
        expected_labels: list[int],
    ) -> None:
        """min(target, ckpt) along each axis produces the correct per-group prefix."""
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=ckpt_nq, group_detr=ckpt_g)
        out = _slice_query_param_per_group(tensor, ckpt_nq, ckpt_g, tgt_nq, tgt_g)
        assert out[:, 0].int().tolist() == expected_labels

    def test_num_queries_expansion_returns_smaller_tensor(self):
        """When target > ckpt, return min-per-group; load_state_dict will reject."""
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=8, target_group_detr=3)
        # min(4, 8) = 4 per group, all 3 groups → 12 rows == input length.
        assert out.shape == (12, 2)

    @pytest.mark.parametrize(
        "ckpt_nq,ckpt_g,tgt_nq,tgt_g",
        [
            pytest.param(0, 3, 2, 3, id="ckpt_nq=0"),
            pytest.param(-1, 3, 2, 3, id="ckpt_nq=-1"),
            pytest.param(4, 0, 2, 3, id="ckpt_g=0"),
            pytest.param(4, -1, 2, 3, id="ckpt_g=-1"),
            pytest.param(4, 3, 0, 3, id="tgt_nq=0"),
            pytest.param(4, 3, -1, 3, id="tgt_nq=-1"),
            pytest.param(4, 3, 2, 0, id="tgt_g=0"),
            pytest.param(4, 3, 2, -1, id="tgt_g=-1"),
        ],
    )
    def test_raises_on_non_positive_dimension(self, ckpt_nq: int, ckpt_g: int, tgt_nq: int, tgt_g: int) -> None:
        """ValueError raised when any dimension arg is zero or negative."""
        from rfdetr.models.weights import _slice_query_param_per_group

        tensor = torch.zeros(12, 2)
        with pytest.raises(ValueError, match="must be positive"):
            _slice_query_param_per_group(tensor, ckpt_nq, ckpt_g, tgt_nq, tgt_g)


class TestLoadPretrainWeightsPerGroupQuerySlice:
    """End-to-end check that ``load_pretrain_weights`` invokes per-group slicing."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch):
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def _make_args_dict_checkpoint(self, num_queries: int, group_detr: int) -> dict:
        """Build a checkpoint with labelled query weights and dict-style args."""
        labelled_refpoint = _labelled_query_tensor(num_queries, group_detr, dim=4)
        labelled_query_feat = _labelled_query_tensor(num_queries, group_detr, dim=256)
        state = {
            "class_embed.weight": torch.randn(91, 256),
            "class_embed.bias": torch.randn(91),
            "refpoint_embed.weight": labelled_refpoint,
            "query_feat.weight": labelled_query_feat,
        }
        # Dict-style args payload used to exercise the checkpoint-loading path.
        return {"model": state, "args": {"num_queries": num_queries, "group_detr": group_detr}}

    def test_decreasing_num_queries_preserves_per_group_structure(self, monkeypatch, tmp_path):
        """Real flow: checkpoint(nq=4, g=3) → model(nq=2, g=3). Group structure must be preserved."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=2,
            num_select=2,
            group_detr=3,
        )
        checkpoint = self._make_args_dict_checkpoint(num_queries=4, group_detr=3)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        refpoint = passed_state["refpoint_embed.weight"]
        query_feat = passed_state["query_feat.weight"]
        expected = [0, 1, 100, 101, 200, 201]
        # First column carries (group, query) identity (see _labelled_query_tensor).
        assert refpoint[:, 0].int().tolist() == expected, (
            "Per-group structure was not preserved in refpoint_embed.weight."
        )
        assert query_feat[:, 0].int().tolist() == expected, (
            "Per-group structure was not preserved in query_feat.weight."
        )

    def test_legacy_checkpoint_without_args_falls_back_to_flat_slice(self, monkeypatch, tmp_path):
        """No ``args`` in checkpoint → preserve the legacy flat slice (backward compat)."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=2,
            num_select=2,
            group_detr=3,
        )
        checkpoint = self._make_args_dict_checkpoint(num_queries=4, group_detr=3)
        del checkpoint["args"]  # legacy
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        refpoint = passed_state["refpoint_embed.weight"]
        query_feat = passed_state["query_feat.weight"]
        # Legacy flat slice: first 2*3=6 rows of the original 12.  Original rows
        # are labelled 0,1,2,3,100,101,102,103,200,201,202,203 → first 6 are
        # 0,1,2,3,100,101.
        expected = [0, 1, 2, 3, 100, 101]
        assert refpoint[:, 0].int().tolist() == expected
        assert query_feat[:, 0].int().tolist() == expected

    def test_decreasing_group_detr_drops_tail_groups(self, monkeypatch, tmp_path):
        """checkpoint(nq=4, g=3) → model(nq=4, g=2): tail group dropped, retained groups intact."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=4,
            num_select=4,
            group_detr=2,
        )
        checkpoint = self._make_args_dict_checkpoint(num_queries=4, group_detr=3)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        refpoint = passed_state["refpoint_embed.weight"]
        query_feat = passed_state["query_feat.weight"]
        expected = [0, 1, 2, 3, 100, 101, 102, 103]
        assert refpoint[:, 0].int().tolist() == expected
        assert query_feat[:, 0].int().tolist() == expected

    def test_decreasing_num_queries_namespace_args(self, monkeypatch, tmp_path):
        """Namespace-style args in checkpoint trigger per-group slice identical to dict-style."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=2,
            num_select=2,
            group_detr=3,
        )
        labelled_refpoint = _labelled_query_tensor(num_queries=4, group_detr=3, dim=4)
        labelled_query_feat = _labelled_query_tensor(num_queries=4, group_detr=3, dim=256)
        checkpoint = {
            "model": {
                "class_embed.weight": torch.randn(91, 256),
                "class_embed.bias": torch.randn(91),
                "refpoint_embed.weight": labelled_refpoint,
                "query_feat.weight": labelled_query_feat,
            },
            "args": SimpleNamespace(num_queries=4, group_detr=3),
        }
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        refpoint = passed_state["refpoint_embed.weight"]
        query_feat = passed_state["query_feat.weight"]
        expected = [0, 1, 100, 101, 200, 201]
        assert refpoint[:, 0].int().tolist() == expected
        assert query_feat[:, 0].int().tolist() == expected

    def test_legacy_fallback_multigroup_emits_warning(self, monkeypatch) -> None:
        """group_detr > 1 legacy checkpoint (no num_queries/group_detr in args) emits scramble-risk warning."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=2,
            num_select=2,
            group_detr=3,
        )
        labelled_refpoint = _labelled_query_tensor(num_queries=4, group_detr=3, dim=4)
        labelled_query_feat = _labelled_query_tensor(num_queries=4, group_detr=3, dim=256)
        checkpoint = {
            "model": {
                "class_embed.weight": torch.randn(91, 256),
                "class_embed.bias": torch.randn(91),
                "refpoint_embed.weight": labelled_refpoint,
                "query_feat.weight": labelled_query_feat,
            },
            "args": {},  # no num_queries / group_detr keys
        }
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        captured: list[str] = []

        def _capture(msg: str, *args: object, **kwargs: object) -> None:
            try:
                captured.append(msg % args if args else msg)
            except TypeError:
                captured.append(msg)

        monkeypatch.setattr("rfdetr.models.weights.logger.warning", _capture)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        assert any("group_detr" in msg and ("scramble" in msg or "flat slice" in msg) for msg in captured), (
            f"Expected scramble-risk warning for group_detr > 1; got: {captured}"
        )

    def test_legacy_fallback_when_args_missing_num_queries_key(self, monkeypatch, tmp_path):
        """When checkpoint args dict lacks num_queries/group_detr keys, falls back to flat legacy slice."""
        from rfdetr.models.weights import load_pretrain_weights

        mc = RFDETRBaseConfig(
            pretrain_weights="/fake/weights.pth",
            device="cpu",
            num_queries=2,
            num_select=2,
            group_detr=1,
        )
        labelled_refpoint = _labelled_query_tensor(num_queries=4, group_detr=1, dim=4)
        labelled_query_feat = _labelled_query_tensor(num_queries=4, group_detr=1, dim=256)
        checkpoint = {
            "model": {
                "class_embed.weight": torch.randn(91, 256),
                "class_embed.bias": torch.randn(91),
                "refpoint_embed.weight": labelled_refpoint,
                "query_feat.weight": labelled_query_feat,
            },
            "args": {},
        }
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        refpoint = passed_state["refpoint_embed.weight"]
        query_feat = passed_state["query_feat.weight"]
        expected = [0, 1]
        assert refpoint[:, 0].int().tolist() == expected
        assert query_feat[:, 0].int().tolist() == expected


# Partial-load detector
# ---------------------------------------------------------------------------


class TestPartialLoadDetector:
    """Tests for ``_warn_on_partial_load`` — surfaces silent partial loads.

    The rf-detr logger has ``propagate=False`` so pytest's ``caplog`` does not
    see its records.  These tests monkeypatch ``logger.warning`` directly to
    capture the message text.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        """Capture every call to ``rfdetr.models.weights.logger.warning`` as a formatted string."""
        captured: list[str] = []

        def _capture(msg, *args, **kwargs):
            try:
                captured.append(msg % args if args else msg)
            except TypeError:
                captured.append(msg)

        monkeypatch.setattr("rfdetr.models.weights.logger.warning", _capture)
        return captured

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(
                SimpleNamespace(missing_keys=[], unexpected_keys=[]),
                id="clean_load",
            ),
            pytest.param(
                SimpleNamespace(
                    missing_keys=[
                        "class_embed.weight",
                        "bbox_embed.layers.0.weight",
                        "refpoint_embed.weight",
                        "query_feat.weight",
                        "transformer.enc_out_class_embed.0.weight",
                        "transformer.enc_out_bbox_embed.0.layers.0.weight",
                    ],
                    unexpected_keys=[],
                ),
                id="intentional_head_keys",
            ),
            pytest.param(
                SimpleNamespace(missing_keys=42, unexpected_keys=[]),
                id="non_iterable_missing_keys",
            ),
        ],
    )
    def test_no_warning_cases(self, captured, result: SimpleNamespace) -> None:
        """Cases that must not emit any partial-load warning.

        Covers: clean load, intentional head keys, and non-iterable missing_keys.
        """
        _warn_on_partial_load(result, "/fake/weights.pth")
        assert captured == []

    def test_unexpected_backbone_missing_keys_warn(self, captured):
        """Missing backbone keys (e.g. register_tokens) must trigger the warning."""

        result = SimpleNamespace(
            missing_keys=[
                "backbone.0.encoder.encoder.embeddings.register_tokens",
                "backbone.0.encoder.encoder.layers.0.register_block.weight",
            ],
            unexpected_keys=[],
        )
        _warn_on_partial_load(result, "/fake/weights.pth")
        assert len(captured) == 1
        assert "/fake/weights.pth" in captured[0]
        assert "register_tokens" in captured[0]

    def test_unexpected_keys_warn(self, captured):
        """Unexpected checkpoint keys (model has no slot for them) must trigger the warning."""

        result = SimpleNamespace(
            missing_keys=[],
            unexpected_keys=["backbone.0.encoder.legacy_module.weight"],
        )
        _warn_on_partial_load(result, "/fake/weights.pth")
        assert len(captured) == 1
        assert "not consumed by model" in captured[0]

    def test_handles_non_iterable_input_gracefully(self, captured):
        """A MagicMock-style result (used in many existing tests) must not raise."""

        _warn_on_partial_load(MagicMock(), "/fake/weights.pth")
        # The crucial assertion is "did not raise"; whether captured is empty
        # depends on MagicMock truthiness — both outcomes are acceptable.

    @pytest.mark.parametrize(
        "missing_keys, unexpected_keys, count_str",
        [
            pytest.param(
                [f"backbone.0.encoder.layer.{i}.weight" for i in range(10)],
                [],
                "10 model parameter",
                id="long_missing_keys",
            ),
            pytest.param(
                [],
                [f"backbone.0.legacy.{i}.weight" for i in range(8)],
                "8 checkpoint key(s)",
                id="long_unexpected_keys",
            ),
        ],
    )
    def test_truncates_long_key_lists_in_message(
        self,
        captured,
        missing_keys: list[str],
        unexpected_keys: list[str],
        count_str: str,
    ) -> None:
        """Sample key lists in the warning are bounded to 5 entries with a trailing ellipsis."""
        result = SimpleNamespace(missing_keys=missing_keys, unexpected_keys=unexpected_keys)
        _warn_on_partial_load(result, "/fake/weights.pth")
        assert len(captured) == 1
        assert count_str in captured[0]
        assert "..." in captured[0]

    def test_mixed_intentional_and_unintentional_keys_warn_only_for_unexpected(self, captured) -> None:
        """Only unintentional missing keys appear in the warning; intentional reinit keys are filtered.

        When a checkpoint load returns both head-reinit keys (class_embed.weight, etc.) and
        a genuine backbone mismatch (backbone.0.encoder.register_tokens), the warning must
        fire exactly once and must reference the unintentional key, not the filtered ones.
        """
        result = SimpleNamespace(
            missing_keys=[
                "class_embed.weight",
                "bbox_embed.layers.0.weight",
                "refpoint_embed.weight",
                "backbone.0.encoder.encoder.embeddings.register_tokens",
            ],
            unexpected_keys=[],
        )
        _warn_on_partial_load(result, "/fake/mixed.pth")
        assert len(captured) == 1, f"Expected exactly one warning, got {len(captured)}: {captured}"
        assert "register_tokens" in captured[0], "Warning must reference the unintentional backbone key"
        assert "class_embed" not in captured[0], "Intentional head key must be filtered from warning text"

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.os.path.isfile", return_value=True)
    @patch("rfdetr.models.weights.validate_checkpoint_compatibility")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    @patch("rfdetr.models.weights.download_pretrain_weights")
    def test_partial_load_is_invoked_during_load_pretrain_weights(
        self,
        mock_download: MagicMock,
        mock_validate_weights: MagicMock,
        mock_validate_compat: MagicMock,
        mock_isfile: MagicMock,
        mock_torch_load: MagicMock,
        monkeypatch,
    ) -> None:
        """Integration check: load_pretrain_weights wires up the partial-load detector."""
        from rfdetr.models.weights import load_pretrain_weights

        mock_torch_load.return_value = _make_checkpoint(num_classes=91)

        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu")
        nn_model = _fake_nn_model()
        nn_model.load_state_dict.return_value = SimpleNamespace(
            missing_keys=["backbone.0.encoder.something_required.weight"],
            unexpected_keys=[],
        )

        captured: list[str] = []

        def _capture_warning(msg, *args, **kwargs):
            try:
                captured.append(msg % args if args else msg)
            except TypeError:
                captured.append(msg)

        monkeypatch.setattr("rfdetr.models.weights.logger.warning", _capture_warning)
        load_pretrain_weights(nn_model, mc)

        assert any("partially" in m for m in captured), (
            f"Expected partial-load warning to fire; got messages: {captured}"
        )

        # Conversely, a clean load must not fire the warning.
        captured.clear()
        nn_model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        load_pretrain_weights(nn_model, mc)
        assert not any("partially" in m for m in captured), f"Clean load must not warn; got messages: {captured}"
