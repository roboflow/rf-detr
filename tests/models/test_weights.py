# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for ``rfdetr.models.weights`` — the unified weight-loading and LoRA module.

These tests cover ``load_pretrain_weights`` and ``apply_lora`` directly, exercising the unified logic extracted from
``detr.py`` and ``module_model.py``. Checkpoints saved by a ``backbone_lora=True`` run (#1540) are also loaded through
``RFDETR.from_checkpoint`` and ``RFDETRModelModule``, which reach them through the inference and training builders.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, call

import pytest
import torch

from rfdetr import RFDETR, RFDETRNano
from rfdetr.config import RFDETRBaseConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.models.weights import (
    _pad_query_param_per_group,
    _slice_query_param_per_group,
    _warn_on_partial_load,
    apply_lora,
    load_pretrain_weights,
)
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.reproducibility import seed_all
from tests._markers import _requires_peft

#: ``num_classes`` default of ``RFDETRBaseConfig``; an explicit value equal to it must still count as explicit.
_DEFAULT_NUM_CLASSES = RFDETRBaseConfig.model_fields["num_classes"].default

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(num_classes: int = 91, num_queries: int = 300, group_detr: int = 13) -> dict:
    """Build a minimal checkpoint dict with the given class count.

    Args:
        num_classes: Total classes including background (bias shape).
        num_queries: Number of object queries per group.
        group_detr: Number of groups.

    Examples:
        >>> ckpt = _make_checkpoint(num_classes=3, num_queries=2, group_detr=2)
        >>> ckpt["model"]["query_feat.weight"].shape
        torch.Size([4, 256])
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


def _make_train_config() -> TrainConfig:
    """Return a minimal TrainConfig for use in load_pretrain_weights.

    Examples:
        >>> _make_train_config().dataset_dir.endswith("dataset")
        True
    """
    return TrainConfig(
        dataset_dir="/nonexistent/dataset",
        output_dir="/nonexistent/output",
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        lr_scheduler_kwargs={"lr_drop": 8},
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        grad_accum_steps=1,
        tensorboard=False,
    )


def _base_config(pretrain_weights: str | None = "/fake/weights.pth", **config: Any) -> RFDETRBaseConfig:
    """Return a CPU ``RFDETRBaseConfig`` pointing at a fake weights path.

    Args:
        pretrain_weights: Value for ``pretrain_weights``; ``None`` disables loading.
        **config: Extra ``RFDETRBaseConfig`` fields.

    Examples:
        >>> _base_config(num_classes=2).num_classes
        2
    """
    return RFDETRBaseConfig(pretrain_weights=pretrain_weights, device="cpu", **config)


def _fake_nn_model() -> MagicMock:
    """Return a MagicMock that behaves enough like an LWDETR nn.Module.

    Returns:
        MagicMock with reinitialize_detection_head and load_state_dict stubs.

    Examples:
        >>> model = _fake_nn_model()
        >>> hasattr(model, "load_state_dict")
        True
    """
    model = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    model.load_state_dict = MagicMock()
    return model


def _serve_checkpoint(monkeypatch: pytest.MonkeyPatch, checkpoint: dict) -> None:
    """Make ``torch.load`` as called by ``rfdetr.models.weights`` return ``checkpoint``.

    Args:
        monkeypatch: Patches ``torch.load`` for the duration of the test.
        checkpoint: Checkpoint object handed back for any path.

    Examples:
        >>> with pytest.MonkeyPatch.context() as patch:
        ...     _serve_checkpoint(patch, {"model": {}})
        ...     torch.load("ignored")
        {'model': {}}
    """
    monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture every call to ``rfdetr.models.weights.logger.warning`` as a formatted string.

    The rf-detr logger has ``propagate=False``, so pytest's ``caplog`` does not see its records.

    Examples:
        Pytest fixtures cannot be called directly outside fixture injection.

        >>> captured(None)  # doctest: +SKIP
    """
    messages: list[str] = []

    def _capture(msg: str, *args: object, **kwargs: object) -> None:
        try:
            messages.append(msg % args if args else msg)
        except TypeError:
            messages.append(msg)

    monkeypatch.setattr("rfdetr.models.weights.logger.warning", _capture)
    return messages


class _PatchWeightsIO:
    """Mixin suppressing all download, file-existence, and validation side effects on ``rfdetr.models.weights``."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Suppress external weights-I/O side effects for inheriting tests.

        Examples:
            Pytest applies this autouse fixture through fixture injection, so direct invocation is skipped.

            >>> _PatchWeightsIO()._patch_io(pytest.MonkeyPatch())  # doctest: +SKIP
        """
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_checkpoint_compatibility", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)


# ---------------------------------------------------------------------------
# load_pretrain_weights — reinit scenarios
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsReinitScenarios(_PatchWeightsIO):
    """Verify reinitialize_detection_head call patterns for all class-count scenarios."""

    def test_fine_tuned_checkpoint_auto_aligns_default_num_classes(self, monkeypatch):
        """Fine-tuned checkpoint (fewer classes) + default num_classes → 1 reinit to ckpt size.

        When the user did NOT explicitly set num_classes (default=90), the loader auto-aligns to the checkpoint's class
        count (3 classes = bias shape [3]). Only one reinit fires; no second reinit back to 91.
        """
        mc = _base_config()
        _serve_checkpoint(monkeypatch, _make_checkpoint(num_classes=3))

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == [call(3)], (
            f"Expected exactly one reinit to checkpoint size 3, got {calls}. "
            "A second reinit to 91 would destroy loaded fine-tuned weights."
        )
        assert mc.num_classes == 2, "Auto-aligned checkpoint class count must be persisted back onto ModelConfig."

    def test_auto_aligned_num_classes_is_not_marked_explicit(self, monkeypatch):
        """A checkpoint-derived num_classes must not count as user-set, so a larger dataset can still raise it.

        Scenario: ``RFDETRSegXLarge(pretrain_weights="own_18_class.pth")`` then ``train()`` on a dataset with 20
        classes. The constructor's load auto-aligns ``num_classes`` to the checkpoint; that assignment lands in
        ``model_fields_set`` and ``RFDETR._align_num_classes_from_dataset`` then treats it as explicit, keeps the
        checkpoint's count with a warning, and the training module loads a head and a criterion that are too small
        for the dataset's labels (index out of bounds at the first step). The flag must be cleared after the
        auto-alignment, as ``from_checkpoint`` clears every checkpoint-derived field, so the later dataset-driven
        assignment is honoured by the second load (load then expand).
        """
        mc = _base_config()
        _serve_checkpoint(monkeypatch, _make_checkpoint(num_classes=3))
        load_pretrain_weights(_fake_nn_model(), mc)

        assert mc.num_classes == 2
        assert "num_classes" not in mc.model_fields_set, (
            "A checkpoint-derived num_classes must not be recorded as user-set; "
            "_align_num_classes_from_dataset would refuse to raise it for a dataset with more classes."
        )

        mc.num_classes = 5  # what _align_num_classes_from_dataset does for a 5-class dataset
        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == [call(3), call(6)], f"Expected reinit to [3, 6] (load then expand to the dataset), got {calls}"
        assert mc.num_classes == 5, "The dataset-aligned num_classes must survive the training module's load."

    @pytest.mark.parametrize(
        ("num_classes", "checkpoint_classes", "expected_calls"),
        [
            # 91-class COCO checkpoint, explicit small num_classes: reinit to 91 so loading works, then trim to 3.
            pytest.param(2, 91, [call(91), call(3)], id="smaller_than_checkpoint_loads_then_trims"),
            # The head aligns to 91 for loading, then expands back to 94.
            pytest.param(93, 91, [call(91), call(94)], id="larger_than_checkpoint_loads_then_expands"),
            # Guards against re-introducing a `num_classes != default` clause, which made an explicit default behave
            # like "unset" and silently adopted the checkpoint's class count.
            pytest.param(
                _DEFAULT_NUM_CLASSES,
                3,
                [call(3), call(_DEFAULT_NUM_CLASSES + 1)],
                id="explicit_default_is_honored_like_any_explicit_value",
            ),
            # 91 == 90 + 1: the checkpoint already fits the head.
            pytest.param(90, 91, [], id="matching_checkpoint_skips_reinit"),
        ],
    )
    def test_explicit_num_classes_is_kept_across_reinits(
        self, monkeypatch, num_classes: int, checkpoint_classes: int, expected_calls: list
    ):
        """An explicit num_classes is never replaced by the checkpoint's: the head aligns for loading, then resizes."""
        mc = _base_config(num_classes=num_classes)
        assert "num_classes" in mc.model_fields_set
        _serve_checkpoint(monkeypatch, _make_checkpoint(num_classes=checkpoint_classes))

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == expected_calls, f"Expected reinit calls {expected_calls}, got {calls}"
        assert mc.num_classes == num_classes, "Explicit num_classes must be preserved, not auto-aligned."

    def test_num_classes_assigned_after_construction_treated_as_explicit(self, monkeypatch):
        """num_classes assigned post-construction wins over a smaller checkpoint (load then expand).

        Scenario: config constructed without an explicit num_classes, then ``num_classes`` assigned to 5 — mirroring
        what ``RFDETR._align_num_classes_from_dataset`` does during ``train()`` for a model loaded via
        ``from_checkpoint`` (issue #1092).  Loading a 3-class checkpoint must align to the checkpoint for loading and
        expand back to 6, NOT auto-align the config back down to the checkpoint's class count.
        """
        mc = _base_config()
        mc.num_classes = 5  # Pydantic records assigned fields in model_fields_set.
        _serve_checkpoint(monkeypatch, _make_checkpoint(num_classes=3))

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        calls = nn_model.reinitialize_detection_head.call_args_list
        assert calls == [call(3), call(6)], f"Expected reinit to [3, 6] (load then expand), got {calls}"
        assert mc.num_classes == 5, "Dataset-aligned num_classes must not be clobbered by the checkpoint."

    def test_keypoint_active_mask_mismatch_is_dropped(self, monkeypatch):
        """Checkpoint `_kp_active_mask` with mismatched shape is dropped before load_state_dict."""
        mc = _base_config()
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"]["_kp_active_mask"] = torch.ones(2, 17, dtype=torch.bool)
        _serve_checkpoint(monkeypatch, checkpoint)

        nn_model = _fake_nn_model()
        nn_model.state_dict = MagicMock(return_value={"_kp_active_mask": torch.ones(1, 17, dtype=torch.bool)})
        nn_model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])

        load_pretrain_weights(nn_model, mc)

        loaded_state = nn_model.load_state_dict.call_args[0][0]
        assert "_kp_active_mask" not in loaded_state

    def test_keypoint_checkpoint_schema_reinitializes_before_and_after_load(self, monkeypatch):
        """Schema-dependent keypoint tensors should match checkpoint shape during load, then return to config schema."""
        mc = _base_config(use_grouppose_keypoints=True, num_keypoints_per_class=[0, 17])
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["model"]["_kp_active_mask"] = torch.ones(1, 17, dtype=torch.bool)
        _serve_checkpoint(monkeypatch, checkpoint)

        nn_model = _fake_nn_model()
        nn_model.reinitialize_keypoint_head = MagicMock()
        nn_model.get_num_keypoints_per_class_from_checkpoint = MagicMock(return_value=[17])
        nn_model.state_dict = MagicMock(return_value={"_kp_active_mask": torch.ones(2, 17, dtype=torch.bool)})
        nn_model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])

        load_pretrain_weights(nn_model, mc)

        assert nn_model.reinitialize_keypoint_head.call_args_list == [call([17]), call([0, 17])]


# ---------------------------------------------------------------------------
# load_pretrain_weights — class_names extraction
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsClassNames(_PatchWeightsIO):
    """Verify that class_names are extracted from checkpoint and returned."""

    def test_class_names_extracted_from_checkpoint(self, monkeypatch):
        """class_names stored in checkpoint args are returned as a list of strings."""
        mc = _base_config(num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint["args"] = SimpleNamespace(segmentation_head=False, patch_size=14, class_names=["cat", "dog", "bird"])
        _serve_checkpoint(monkeypatch, checkpoint)

        result = load_pretrain_weights(_fake_nn_model(), mc)

        assert result == ["cat", "dog", "bird"], f"Expected class names from checkpoint, got {result!r}"

    def test_class_names_empty_when_absent_from_checkpoint(self, monkeypatch):
        """Empty list returned when checkpoint has no args or no class_names key."""
        mc = _base_config(num_classes=90)
        checkpoint = _make_checkpoint(num_classes=91)
        checkpoint.pop("args")  # no args key at all
        _serve_checkpoint(monkeypatch, checkpoint)

        result = load_pretrain_weights(_fake_nn_model(), mc)

        assert result == [], f"Expected empty list when checkpoint has no class_names, got {result!r}"

    def test_none_pretrain_weights_returns_empty_list_immediately(self):
        """load_pretrain_weights returns [] without any I/O when pretrain_weights is None."""
        mc = _base_config(pretrain_weights=None)
        nn_model = _fake_nn_model()

        result = load_pretrain_weights(nn_model, mc)

        assert result == [], f"Expected [] for None pretrain_weights, got {result!r}"
        nn_model.load_state_dict.assert_not_called()
        nn_model.reinitialize_detection_head.assert_not_called()


# ---------------------------------------------------------------------------
# load_pretrain_weights — trust propagation
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsTrustPropagation(_PatchWeightsIO):
    """Verify the ``trust`` kwarg reaches ``_safe_torch_load`` unchanged.

    Regression coverage for a gap where ``RFDETR.from_checkpoint(path, trust_checkpoint=True)`` bypassed the safe-load
    check only for its own metadata read, then silently reverted to ``trust=False`` when the constructed model reloaded
    the same file here — making ``trust_checkpoint=True`` inert for any checkpoint that actually needed it.
    """

    @pytest.mark.parametrize(
        ("trust_kwargs", "expected_trust"),
        [
            pytest.param({"trust": True}, True, id="trust_true_forwarded"),
            pytest.param({}, False, id="trust_defaults_to_false"),
        ],
    )
    def test_trust_forwarded_to_safe_torch_load(self, mocker, trust_kwargs: dict, expected_trust: bool):
        """load_pretrain_weights passes its trust kwarg (default False) on to _safe_torch_load."""
        mc = _base_config()
        mock_load = mocker.patch("rfdetr.utilities.io._safe_torch_load", return_value=_make_checkpoint(num_classes=91))

        load_pretrain_weights(_fake_nn_model(), mc, **trust_kwargs)

        mock_load.assert_called_once_with(mc.pretrain_weights, trust=expected_trust)


# ---------------------------------------------------------------------------
# load_pretrain_weights — PTL .ckpt format
# ---------------------------------------------------------------------------


class TestLoadPretrainWeightsPTLCkptFormat(_PatchWeightsIO):
    """Verify that PTL-native .ckpt checkpoints (state_dict, no model key) are handled."""

    @staticmethod
    def _make_ptl_checkpoint(num_classes: int = 91, num_queries: int = 300, group_detr: int = 13) -> dict:
        """Build a fake PyTorch Lightning (PTL) native checkpoint with state_dict keys prefixed by 'model.'.

        Args:
            num_classes: Total classes including background (bias shape).
            num_queries: Number of object queries per group.
            group_detr: Number of groups.

        Examples:
            >>> checkpoint = TestLoadPretrainWeightsPTLCkptFormat._make_ptl_checkpoint(3, 2, 2)
            >>> sorted(checkpoint["state_dict"])[0], checkpoint["epoch"]
            ('model.class_embed.bias', 10)
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

    def test_loads_without_model_key(self, monkeypatch):
        """PTL .ckpt checkpoints (state_dict without model key) must load without KeyError."""
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        _serve_checkpoint(monkeypatch, self._make_ptl_checkpoint(num_classes=91))

        nn_model = _fake_nn_model()
        result = load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        assert result == [], f"Expected [] (no args/class_names in checkpoint), got {result!r}"

    def test_model_prefix_stripped_before_load_state_dict(self, monkeypatch):
        """Model weights passed to load_state_dict must not carry the 'model.' prefix."""
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        _serve_checkpoint(monkeypatch, self._make_ptl_checkpoint(num_classes=91))

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        loaded_state = nn_model.load_state_dict.call_args[0][0]
        assert all(not k.startswith("model.") for k in loaded_state), (
            f"Keys passed to load_state_dict must not have 'model.' prefix, got: {list(loaded_state.keys())[:5]}"
        )

    def test_missing_model_prefix_in_state_dict_raises_value_error(self, monkeypatch):
        """A checkpoint with state_dict but no 'model.'-prefixed keys raises ValueError."""
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        _serve_checkpoint(monkeypatch, {"state_dict": {"some_other.key": torch.zeros(1)}, "epoch": 10})

        with pytest.raises(ValueError) as exc_info:
            load_pretrain_weights(_fake_nn_model(), mc)

        assert "model." in str(exc_info.value)

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            pytest.param(
                {"hyper_parameters": {"class_names": ["cat", "dog"]}}, ["cat", "dog"], id="hyper_parameters_only"
            ),
            pytest.param(
                {"args": {"class_names": ["from_args"]}, "hyper_parameters": {"class_names": ["from_hyper_params"]}},
                ["from_args"],
                id="args_take_precedence_over_hyper_parameters",
            ),
        ],
    )
    def test_class_names_source(self, monkeypatch, extra: dict, expected: list[str]):
        """Class names come from hyper_parameters when args is absent, and from args when both are present."""
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        _serve_checkpoint(monkeypatch, {**self._make_ptl_checkpoint(num_classes=91), **extra})

        result = load_pretrain_weights(_fake_nn_model(), mc)

        assert result == expected, f"Expected class names {expected!r}, got {result!r}"

    def test_non_model_keys_in_state_dict_are_excluded(self, monkeypatch):
        """Non-model keys in state_dict (optimizer, lr_scheduler) must not appear in checkpoint['model'].

        Real PTL checkpoints contain keys like 'optimizer.param_groups' and 'lr_scheduler.last_epoch' alongside
        'model.*' weights. The loader must exclude these non-model keys so they do not pollute the state dict passed to
        load_state_dict and do not cause KeyError or unexpected parameter names.
        """
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        checkpoint = self._make_ptl_checkpoint(num_classes=91)
        checkpoint["state_dict"]["optimizer.param_groups"] = torch.zeros(1)
        checkpoint["state_dict"]["lr_scheduler.last_epoch"] = torch.tensor(10)
        checkpoint["state_dict"]["callback_states.ema.shadow_params"] = torch.zeros(4)
        _serve_checkpoint(monkeypatch, checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        loaded_state = nn_model.load_state_dict.call_args[0][0]
        non_model_keys = [k for k in loaded_state if k.startswith(("optimizer.", "lr_scheduler.", "callback_states."))]
        assert not non_model_keys, f"Non-model keys must be excluded from loaded state; found: {non_model_keys}"

    def test_torch_compile_orig_mod_prefix_stripped(self, monkeypatch):
        """PTL .ckpt from a torch.compile-wrapped model must load without KeyError.

        When a model is wrapped with torch.compile before training, PTL records weights under keys like
        "model._orig_mod.class_embed.bias".  The loader must strip both the "model." and the subsequent "_orig_mod."
        segment so the resulting keys match the bare parameter names expected by load_state_dict.
        """
        mc = _base_config("/fake/last.ckpt", num_classes=90)
        ptl_state = self._make_ptl_checkpoint(num_classes=91)["state_dict"]
        # Simulate torch.compile: keys are prefixed with "model._orig_mod."
        checkpoint = {"state_dict": {k.replace("model.", "model._orig_mod.", 1): v for k, v in ptl_state.items()}}
        _serve_checkpoint(monkeypatch, checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()
        loaded_state = nn_model.load_state_dict.call_args[0][0]
        assert all(not k.startswith(("model.", "_orig_mod.")) for k in loaded_state), (
            f"Keys must have both 'model.' and '_orig_mod.' stripped; got: {list(loaded_state.keys())[:5]}"
        )

    def test_best_model_callback_format_with_model_and_state_dict_loads(self, monkeypatch):
        """Checkpoints with both 'model' and 'state_dict' (BestModelCallback format) must still load."""
        mc = _base_config("/fake/checkpoint_best_total.pth", num_classes=90)
        # BestModelCallback writes both "model" (raw keys) and "state_dict" (prefixed keys).
        ptl_state = self._make_ptl_checkpoint(num_classes=91)["state_dict"]
        checkpoint = {"model": {k.removeprefix("model."): v for k, v in ptl_state.items()}, "state_dict": ptl_state}
        _serve_checkpoint(monkeypatch, checkpoint)

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        nn_model.load_state_dict.assert_called_once()


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


class TestApplyLora:
    """Verify that apply_lora applies LoRA adapters to the backbone encoder.

    ``apply_lora`` lazily imports ``peft`` inside the function body, so the tests put a stand-in ``peft`` module into
    ``sys.modules`` to intercept the import rather than patching a module-level name.
    """

    @pytest.fixture
    def mock_peft(self, mocker) -> MagicMock:
        """Install a stand-in ``peft`` module whose ``PeftModel`` is a real class, so ``isinstance`` checks work.

        Examples:
            Pytest fixtures cannot be called directly outside fixture injection.

            >>> TestApplyLora.mock_peft(None, None)  # doctest: +SKIP
        """
        module = MagicMock(PeftModel=type("PeftModel", (), {}))
        mocker.patch.dict("sys.modules", {"peft": module})
        return module

    def test_wraps_backbone_encoder(self, mock_peft):
        """apply_lora must call get_peft_model on nn_model.backbone[0].encoder."""
        nn_model = MagicMock()
        fake_peft_model = MagicMock()
        mock_peft.get_peft_model.return_value = fake_peft_model

        apply_lora(nn_model)

        mock_peft.LoraConfig.assert_called_once()
        lora_kwargs = mock_peft.LoraConfig.call_args.kwargs
        assert lora_kwargs.get("r") == 16, "LoRA rank must be 16"
        assert lora_kwargs.get("lora_alpha") == 16, "LoRA alpha must be 16"
        assert lora_kwargs.get("use_dora") is True, "DoRA must be enabled"

        assert mock_peft.get_peft_model.call_count == 1, "get_peft_model must be called exactly once"
        assert nn_model.backbone[0].encoder is fake_peft_model, "backbone encoder must be replaced with the peft model"

    def test_targets_expected_modules(self, mock_peft):
        """apply_lora must target exactly the 9 expected module names."""
        apply_lora(MagicMock())

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

    def test_already_wrapped_encoder_is_left_alone(self, mock_peft) -> None:
        """An encoder that ``load_pretrain_weights`` already wrapped for a LoRA checkpoint gets no second adapter set.

        Both model builders call ``apply_lora`` after loading, so a second wrap would nest one PEFT model inside another
        and rename every encoder key again.
        """
        nn_model = MagicMock()
        nn_model.backbone[0].encoder = mock_peft.PeftModel()

        apply_lora(nn_model)

        mock_peft.get_peft_model.assert_not_called()


# ---------------------------------------------------------------------------
# load_pretrain_weights — checkpoints saved by a backbone_lora=True run (#1540)
# ---------------------------------------------------------------------------

#: Seed for the synthetic LoRA and plain checkpoints below; module-scoped fixtures run before the autouse reseed.
_LORA_CHECKPOINT_SEED = 1540


def _perturb_parameters(model: torch.nn.Module) -> None:
    """Shift every parameter so saved weights differ from any fresh initialization, even LoRA's all-zero ``lora_B``.

    Examples:
        >>> layer = torch.nn.Linear(2, 2)
        >>> _ = torch.nn.init.zeros_(layer.weight)
        >>> _perturb_parameters(layer)
        >>> bool(layer.weight.abs().sum() > 0)
        True
    """
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)


def _build_nano(
    builder: Literal["inference", "training"], pretrain_weights: Path | None, **config: Any
) -> torch.nn.Module:
    """Build a Nano ``LWDETR`` through one of the two entry points that load pretrained weights.

    Args:
        builder: ``"inference"`` for ``RFDETR.from_checkpoint`` (the issue's call, which goes through the same
            builder as the ``RFDETRNano`` constructor), ``"training"`` for ``RFDETRModelModule`` (the path behind
            ``train()``).
        pretrain_weights: Checkpoint to load, or ``None`` for a randomly initialized model.
        **config: Extra ``RFDETRNanoConfig`` fields, e.g. ``backbone_lora=True``.

    Examples:
        >>> type(_build_nano("training", None)).__name__
        'LWDETR'
    """
    if builder == "inference":
        if pretrain_weights is None:
            return RFDETRNano(pretrain_weights=None, **config).model.model
        return RFDETR.from_checkpoint(pretrain_weights, **config).model.model
    weights = None if pretrain_weights is None else str(pretrain_weights)
    model_config = RFDETRNanoConfig(pretrain_weights=weights, **config)
    return RFDETRModelModule(model_config, _make_train_config()).model


def _mismatched_keys(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> list[str]:
    """List the keys of *expected* that *actual* lacks or holds with a different value.

    Examples:
        >>> _mismatched_keys({"a": torch.zeros(1), "b": torch.ones(1)}, {"a": torch.zeros(1), "b": torch.zeros(1)})
        ['b']
        >>> _mismatched_keys({"a": torch.zeros(1)}, {})
        ['a']
    """
    return [key for key, value in expected.items() if key not in actual or not torch.equal(actual[key], value)]


def _encoder_features(model: torch.nn.Module) -> torch.Tensor:
    """Run the backbone encoder of *model* in eval mode on a fixed image and flatten every feature map into one tensor.

    Examples:
        >>> _encoder_features(_build_nano("inference", None)).ndim
        1
    """
    images = torch.rand(1, 3, 384, 384, generator=torch.Generator().manual_seed(0))
    encoder = model.backbone[0].encoder.eval()
    with torch.no_grad():
        return torch.cat([feature.flatten() for feature in encoder(images)])


@pytest.fixture(scope="module")
def lora_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, torch.nn.Module]:
    """Save a checkpoint the way a ``backbone_lora=True`` run does: PEFT-named encoder keys, trained adapters.

    Returns:
        The checkpoint path and the source model it was saved from, in eval mode.

    Examples:
        Pytest fixtures cannot be called directly outside fixture injection.

        >>> lora_checkpoint(None)  # doctest: +SKIP
    """
    seed_all(_LORA_CHECKPOINT_SEED)
    source = _build_nano("inference", None, backbone_lora=True).eval()
    _perturb_parameters(source)
    path = tmp_path_factory.mktemp("lora_checkpoint") / "checkpoint_best_total.pth"
    torch.save({"model": source.state_dict(), "args": {}, "model_name": "RFDETRNano"}, path)
    return path, source


@pytest.fixture(scope="module")
def plain_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a checkpoint without LoRA, like the published COCO weights a LoRA fine-tune starts from.

    Examples:
        Pytest fixtures cannot be called directly outside fixture injection.

        >>> plain_checkpoint(None)  # doctest: +SKIP
    """
    seed_all(_LORA_CHECKPOINT_SEED)
    source = _build_nano("inference", None)
    _perturb_parameters(source)
    path = tmp_path_factory.mktemp("plain_checkpoint") / "plain_checkpoint.pth"
    torch.save({"model": source.state_dict(), "args": {}, "model_name": "RFDETRNano"}, path)
    return path


@_requires_peft
class TestLoadPretrainWeightsLoraCheckpoint:
    """A checkpoint saved by a ``backbone_lora=True`` run reloads with its trained encoder (#1540).

    Such a checkpoint stores the encoder under PEFT names (``backbone.0.encoder.base_model.model.…``). Both builders
    used to load it before wrapping the encoder with LoRA, so none of those keys matched and the encoder stayed at
    random init: the issue's ``evaluate()`` reported mAP 0 on a model that scored 0.99 in training.
    """

    @pytest.mark.parametrize("builder", ["inference", "training"])
    def test_lora_run_reloads_every_checkpoint_tensor(
        self, lora_checkpoint: tuple[Path, torch.nn.Module], builder: Literal["inference", "training"]
    ) -> None:
        """With ``backbone_lora=True`` the base weights and the trained adapters all come back unchanged."""
        checkpoint_path, source = lora_checkpoint
        loaded = _build_nano(builder, checkpoint_path, backbone_lora=True)

        assert _mismatched_keys(source.state_dict(), loaded.state_dict()) == []

    def test_plain_run_encoder_computes_the_trained_lora_features(
        self, lora_checkpoint: tuple[Path, torch.nn.Module]
    ) -> None:
        """With ``backbone_lora=False`` the encoder computes what the trained LoRA encoder computed.

        ``test_plain_run_matches_a_model_built_without_lora`` checks that it is a plain encoder, not a wrapped one.
        """
        checkpoint_path, source = lora_checkpoint

        loaded = _build_nano("inference", checkpoint_path)

        torch.testing.assert_close(_encoder_features(loaded), _encoder_features(source), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("freeze_encoder", [False, True])
    def test_plain_run_matches_a_model_built_without_lora(
        self, lora_checkpoint: tuple[Path, torch.nn.Module], freeze_encoder: bool
    ) -> None:
        """With ``backbone_lora=False`` the parameters and their trainable flags match a Nano that never had LoRA.

        peft freezes the base encoder when it wraps it, and ``merge_and_unload`` keeps that flag. Without restoring it,
        a ``train()`` started from a LoRA checkpoint with ``backbone_lora=False`` would silently keep the encoder
        frozen.
        """
        checkpoint_path, _ = lora_checkpoint
        loaded = _build_nano("training", checkpoint_path, freeze_encoder=freeze_encoder)
        expected = _build_nano("training", None, freeze_encoder=freeze_encoder)

        assert {name: param.requires_grad for name, param in loaded.named_parameters()} == {
            name: param.requires_grad for name, param in expected.named_parameters()
        }

    def test_plain_checkpoint_loads_before_fresh_adapters(self, plain_checkpoint: Path) -> None:
        """A plain checkpoint with ``backbone_lora=True`` still fills the encoder that LoRA then wraps.

        This is how a LoRA fine-tune starts from the COCO weights. Fresh adapters are an identity, so the wrapped
        encoder must compute what the same checkpoint computes without LoRA.
        """
        with_lora = _build_nano("inference", plain_checkpoint, backbone_lora=True)
        without_lora = _build_nano("inference", plain_checkpoint)

        torch.testing.assert_close(_encoder_features(with_lora), _encoder_features(without_lora), atol=1e-4, rtol=1e-4)

    def test_lora_checkpoint_without_peft_raises(
        self, lora_checkpoint: tuple[Path, torch.nn.Module], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``peft`` a LoRA checkpoint raises instead of loading into a randomly initialized encoder."""
        checkpoint_path, _ = lora_checkpoint
        monkeypatch.setitem(sys.modules, "peft", None)

        with pytest.raises(ImportError, match=r"backbone_lora=True run.*rfdetr\[lora\]"):
            _build_nano("inference", checkpoint_path)

    def test_already_wrapped_model_loads_without_merging(self, lora_checkpoint: tuple[Path, torch.nn.Module]) -> None:
        """``load_pretrain_weights`` into a model the caller already wrapped loads every tensor and keeps the wrap.

        Neither builder gets here, since both load before they wrap, but ``load_pretrain_weights`` is public: the merge
        must not try to undo a wrap it did not make, even with the default ``backbone_lora=False``.
        """
        checkpoint_path, source = lora_checkpoint
        model = _build_nano("inference", None)
        apply_lora(model)

        load_pretrain_weights(model, RFDETRNanoConfig(pretrain_weights=str(checkpoint_path)))

        assert _mismatched_keys(source.state_dict(), model.state_dict()) == []


# ---------------------------------------------------------------------------
# Per-group query embedding slicing
# ---------------------------------------------------------------------------


def _labelled_query_tensor(num_queries: int, group_detr: int, dim: int = 2, label_stride: int = 100) -> torch.Tensor:
    """Build a query embedding tensor where row ``g * num_queries + q`` encodes ``[g * label_stride + q, 0, ...]``.

    This lets tests check the per-group ordering of the result without floating-point
    fuzz: the first column carries the (group, query) identity directly. ``label_stride``
    must exceed the largest ``q`` used by a caller, or labels from different groups collide
    (e.g. the default stride of 100 collides once ``num_queries >= 100``).

    Examples:
        >>> _labelled_query_tensor(2, 2, dim=1).squeeze(-1).tolist()
        [0.0, 1.0, 100.0, 101.0]
        >>> _labelled_query_tensor(150, 2, dim=1, label_stride=1000).squeeze(-1).tolist()[100]
        100.0
    """
    rows = []
    for g in range(group_detr):
        for q in range(num_queries):
            rows.append([float(g * label_stride + q)] + [0.0] * (dim - 1))
    return torch.tensor(rows, dtype=torch.float32)


def _query_config(num_queries: int, group_detr: int) -> RFDETRBaseConfig:
    """Return a CPU ``RFDETRBaseConfig`` with the given query layout.

    Examples:
        >>> _query_config(2, 3).group_detr
        3
    """
    return _base_config(num_queries=num_queries, num_select=num_queries, group_detr=group_detr)


def _query_checkpoint(num_queries: int, group_detr: int, **extra: Any) -> dict:
    """Build a checkpoint whose query embeddings carry ``(group, query)`` labels in their first column.

    Args:
        num_queries: Queries per group in the checkpoint.
        group_detr: Groups in the checkpoint.
        **extra: Further top-level checkpoint entries, e.g. ``args``.

    Examples:
        >>> _query_checkpoint(2, 3, args={})["model"]["refpoint_embed.weight"].shape
        torch.Size([6, 4])
    """
    return {
        "model": {
            "class_embed.weight": torch.randn(91, 256),
            "class_embed.bias": torch.randn(91),
            "refpoint_embed.weight": _labelled_query_tensor(num_queries, group_detr, dim=4),
            "query_feat.weight": _labelled_query_tensor(num_queries, group_detr, dim=256),
        },
        **extra,
    }


class TestSliceQueryParamPerGroup:
    """Direct unit tests for ``_slice_query_param_per_group``.

    The helper is the fix for a latent bug where a flat ``tensor[:N]`` slice scrambled per-group structure when
    ``num_queries`` decreased with ``group_detr > 1``.  See the docstring in ``rfdetr.models.weights`` for the
    ``LWDETR`` packing layout that motivates these tests.
    """

    def test_returns_input_unchanged_when_dimensions_match(self):
        """Matching checkpoint and target dimensions return the very same tensor object."""
        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=4, target_group_detr=3)
        assert out is tensor

    @pytest.mark.parametrize(
        ("ckpt_nq", "ckpt_g", "tgt_nq", "tgt_g", "expected_labels"),
        [
            # The bug being fixed: 4→2 queries with 3 groups must keep the first 2 of each group, not a flat prefix.
            pytest.param(4, 3, 2, 3, [0, 1, 100, 101, 200, 201], id="nq_shrinks"),
            pytest.param(4, 3, 4, 2, [0, 1, 2, 3, 100, 101, 102, 103], id="g_shrinks_drops_tail_groups"),
            pytest.param(4, 3, 2, 2, [0, 1, 100, 101], id="both_shrink"),
            pytest.param(4, 3, 8, 3, [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203], id="nq_expands_g_equal"),
            pytest.param(4, 2, 4, 4, [0, 1, 2, 3, 100, 101, 102, 103], id="g_expands_nq_equal"),
            pytest.param(4, 3, 8, 2, [0, 1, 2, 3, 100, 101, 102, 103], id="nq_expands_g_shrinks"),
            pytest.param(4, 3, 2, 4, [0, 1, 100, 101, 200, 201], id="nq_shrinks_g_expands"),
            pytest.param(4, 3, 8, 4, [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203], id="both_expand"),
        ],
    )
    def test_keeps_min_prefix_per_group(
        self, ckpt_nq: int, ckpt_g: int, tgt_nq: int, tgt_g: int, expected_labels: list[int]
    ) -> None:
        """Min(target, ckpt) along each axis produces the correct per-group prefix."""
        tensor = _labelled_query_tensor(num_queries=ckpt_nq, group_detr=ckpt_g)
        out = _slice_query_param_per_group(tensor, ckpt_nq, ckpt_g, tgt_nq, tgt_g)
        flat_labels = tensor[: len(expected_labels), 0].int().tolist()
        assert out[:, 0].int().tolist() == expected_labels, (
            f"Per-group structure scrambled; a flat slice gives {flat_labels}."
        )

    def test_falls_back_to_flat_slice_on_inconsistent_shape(self):
        """If args don't match the tensor's flat length, defer to legacy behavior."""
        weird = torch.arange(7, dtype=torch.float32).unsqueeze(1)  # shape [7, 1], not 4*3=12
        out = _slice_query_param_per_group(weird, 4, 3, target_num_queries=2, target_group_detr=2)
        # Legacy: tensor[:4]
        assert out.shape == (4, 1)
        assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_num_queries_expansion_returns_smaller_tensor(self):
        """When target > ckpt, return min-per-group; padding to the target size is left to the caller."""
        tensor = _labelled_query_tensor(num_queries=4, group_detr=3)
        out = _slice_query_param_per_group(tensor, 4, 3, target_num_queries=8, target_group_detr=3)
        # min(4, 8) = 4 per group, all 3 groups → 12 rows == input length.
        assert out.shape == (12, 2)

    @pytest.mark.parametrize(
        ("ckpt_nq", "ckpt_g", "tgt_nq", "tgt_g"),
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
        with pytest.raises(ValueError) as exc_info:
            _slice_query_param_per_group(torch.zeros(12, 2), ckpt_nq, ckpt_g, tgt_nq, tgt_g)
        assert "must be positive" in str(exc_info.value)


def _fresh_query_init(num_queries: int, group_detr: int) -> dict[str, torch.Tensor]:
    """Return the freshly initialised query tensors of a model, every row labelled ``-1 - row_index``.

    Negative labels never collide with the non-negative ``_labelled_query_tensor`` checkpoint labels, so a row in the
    padded result is attributable to the checkpoint or to the model's own initialisation by its sign.

    Examples:
        >>> _fresh_query_init(2, 2)["refpoint_embed.weight"][:, 0].tolist()
        [-1.0, -2.0, -3.0, -4.0]
    """
    total = num_queries * group_detr
    labels = -1.0 - torch.arange(total, dtype=torch.float32)
    return {
        "refpoint_embed.weight": labels.unsqueeze(1).repeat(1, 4),
        "query_feat.weight": labels.unsqueeze(1).repeat(1, 256),
    }


class TestPadQueryParamPerGroup:
    """Direct unit tests for ``_pad_query_param_per_group``."""

    @pytest.mark.parametrize(
        ("ckpt_nq", "target_nq", "target_g", "expected_labels"),
        [
            pytest.param(2, 4, 2, [0, 1, -3, -4, 100, 101, -7, -8], id="two_groups"),
            pytest.param(2, 4, 1, [0, 1, -3, -4], id="single_group"),
            pytest.param(3, 5, 2, [0, 1, 2, -4, -5, 100, 101, 102, -9, -10], id="uneven_split"),
        ],
    )
    def test_keeps_pretrained_rows_at_the_head_of_each_group(
        self, ckpt_nq: int, target_nq: int, target_g: int, expected_labels: list[int]
    ) -> None:
        """Pretrained rows land at the start of every group, and the model's fresh rows fill the rest of it.

        A flat append would shift checkpoint group 1 into the new slots of group 0, which inference reads.
        """
        sliced = _labelled_query_tensor(num_queries=ckpt_nq, group_detr=target_g, dim=4)
        fresh = _fresh_query_init(target_nq, target_g)["refpoint_embed.weight"]

        padded = _pad_query_param_per_group(sliced, ckpt_nq, target_nq, target_g, fresh)

        assert padded[:, 0].int().tolist() == expected_labels

    def test_leaves_the_fresh_tensor_untouched(self) -> None:
        """The padded result is a copy, so the model's own initial tensor is not overwritten by pretrained rows."""
        sliced = _labelled_query_tensor(num_queries=2, group_detr=1, dim=4)
        fresh = _fresh_query_init(4, 1)["refpoint_embed.weight"]
        before = fresh.clone()

        _pad_query_param_per_group(sliced, 2, 4, 1, fresh)

        assert torch.equal(fresh, before)


class TestLoadPretrainWeightsPadsQueryGrowth(_PatchWeightsIO):
    """``load_pretrain_weights`` pads a checkpoint to a larger ``num_queries`` instead of failing."""

    @staticmethod
    def _model_with_query_init(num_queries: int, group_detr: int) -> MagicMock:
        """Fake model whose ``state_dict`` exposes its freshly initialised query tensors.

        Examples:
            >>> model = TestLoadPretrainWeightsPadsQueryGrowth._model_with_query_init(2, 1)
            >>> model.state_dict()["query_feat.weight"].shape
            torch.Size([2, 256])
        """
        model = _fake_nn_model()
        model.state_dict.return_value = _fresh_query_init(num_queries, group_detr)
        return model

    def test_pretrained_rows_are_kept_and_new_rows_come_from_the_model(self, monkeypatch, captured) -> None:
        """A 4 -> 6 queries-per-group config loads every pretrained row and initialises only the 2 extra per group.

        The first column carries each row's identity, so the check reads which rows are pretrained (non-negative) and
        which are the model's own (negative), group by group.
        """
        mc = _query_config(6, 3)
        _serve_checkpoint(monkeypatch, _query_checkpoint(4, 3, args={"num_queries": 4, "group_detr": 3}))
        nn_model = self._model_with_query_init(6, 3)

        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        expected = [0, 1, 2, 3, -5, -6, 100, 101, 102, 103, -11, -12, 200, 201, 202, 203, -17, -18]
        assert passed_state["refpoint_embed.weight"][:, 0].int().tolist() == expected
        assert passed_state["query_feat.weight"][:, 0].int().tolist() == expected

    def test_saved_model_config_preserves_custom_group_layout(self, monkeypatch, captured) -> None:
        """Saved architecture metadata takes precedence over the variant default group count."""
        checkpoint = _query_checkpoint(4, 3, args={}, model_config={"num_queries": 4, "group_detr": 3})
        _serve_checkpoint(monkeypatch, checkpoint)
        nn_model = self._model_with_query_init(6, 3)

        load_pretrain_weights(nn_model, _query_config(6, 3))

        expected = [0, 1, 2, 3, -5, -6, 100, 101, 102, 103, -11, -12, 200, 201, 202, 203, -17, -18]
        assert nn_model.load_state_dict.call_args[0][0]["query_feat.weight"][:, 0].int().tolist() == expected

    def test_infers_groups_for_query_growth_with_fewer_groups(self, monkeypatch, captured) -> None:
        """Encoder heads identify checkpoint groups even when the target has fewer total rows."""
        checkpoint = _query_checkpoint(4, 3, args={})
        for group in range(3):
            checkpoint["model"][f"transformer.enc_out_class_embed.{group}.weight"] = torch.zeros(1)
            checkpoint["model"][f"transformer.enc_out_bbox_embed.{group}.layers.0.weight"] = torch.zeros(1)
        _serve_checkpoint(monkeypatch, checkpoint)
        nn_model = self._model_with_query_init(5, 2)

        mc = _query_config(5, 2)
        mc.pretrain_weights = "rf-detr-nano.pth"
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: True)
        load_pretrain_weights(nn_model, mc)

        assert nn_model.load_state_dict.call_args[0][0]["query_feat.weight"][:, 0].int().tolist() == [
            0,
            1,
            2,
            3,
            -5,
            100,
            101,
            102,
            103,
            -10,
        ]

    def test_rejects_ambiguous_query_growth(self, monkeypatch, captured) -> None:
        """A divisible row count alone cannot establish the checkpoint group layout."""
        _serve_checkpoint(monkeypatch, _query_checkpoint(4, 13, args={}))

        with pytest.raises(ValueError, match="layout"):
            load_pretrain_weights(self._model_with_query_init(6, 13), _query_config(6, 13))

    def test_rejects_unknown_layout_when_total_rows_shrink(self, monkeypatch, captured) -> None:
        """A smaller target total does not prove queries per group did not grow."""
        _serve_checkpoint(monkeypatch, _query_checkpoint(100, 13, args={}))

        with pytest.raises(ValueError, match="layout"):
            load_pretrain_weights(self._model_with_query_init(200, 6), _query_config(200, 6))

    def test_rejects_partial_encoder_heads_as_layout_proof(self, monkeypatch, captured) -> None:
        """Matching incomplete encoder heads cannot establish the original group count."""
        checkpoint = _query_checkpoint(3, 4, args={})
        for group in range(3):
            checkpoint["model"][f"transformer.enc_out_class_embed.{group}.weight"] = torch.zeros(1)
            checkpoint["model"][f"transformer.enc_out_bbox_embed.{group}.layers.0.weight"] = torch.zeros(1)
        _serve_checkpoint(monkeypatch, checkpoint)

        with pytest.raises(ValueError, match="layout"):
            load_pretrain_weights(self._model_with_query_init(6, 3), _query_config(6, 3))

    def test_warns_that_new_queries_are_untrained(self, monkeypatch, captured) -> None:
        """The padding is announced, so a user knows the extra slots need fine-tuning before they are useful."""
        mc = _query_config(6, 3)
        _serve_checkpoint(monkeypatch, _query_checkpoint(4, 3, args={"num_queries": 4, "group_detr": 3}))

        load_pretrain_weights(self._model_with_query_init(6, 3), mc)

        assert any("untrained" in msg and "num_queries" in msg for msg in captured), captured

    def test_recovers_layout_of_checkpoint_without_args(self, monkeypatch, captured) -> None:
        """Published checkpoints that record no layout are padded too, splitting their rows by the default groups."""
        mc = _query_config(6, 13)
        checkpoint = _query_checkpoint(4, 13, args={})
        for group in range(13):
            checkpoint["model"][f"transformer.enc_out_class_embed.{group}.weight"] = torch.zeros(1)
            checkpoint["model"][f"transformer.enc_out_bbox_embed.{group}.layers.0.weight"] = torch.zeros(1)
        _serve_checkpoint(monkeypatch, checkpoint)
        nn_model = self._model_with_query_init(6, 13)

        mc.pretrain_weights = "rf-detr-nano.pth"
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: True)
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        assert passed_state["query_feat.weight"][:, 0].int().tolist()[:6] == [0, 1, 2, 3, -5, -6]


class TestLoadPretrainWeightsPerGroupQuerySlice(_PatchWeightsIO):
    """End-to-end check that ``load_pretrain_weights`` invokes per-group slicing."""

    @pytest.mark.parametrize(
        ("config_shape", "checkpoint_shape", "checkpoint_extra", "expected"),
        [
            # (num_queries, group_detr) of the config and of the checkpoint; the checkpoint args describe its layout.
            pytest.param(
                (2, 3),
                (4, 3),
                {"args": {"num_queries": 4, "group_detr": 3}},
                [0, 1, 100, 101, 200, 201],
                id="dict_args",
            ),
            pytest.param(
                (2, 3),
                (4, 3),
                {"args": SimpleNamespace(num_queries=4, group_detr=3)},
                [0, 1, 100, 101, 200, 201],
                id="namespace_args",
            ),
            pytest.param(
                (4, 2),
                (4, 3),
                {"args": {"num_queries": 4, "group_detr": 3}},
                [0, 1, 2, 3, 100, 101, 102, 103],
                id="group_detr_decrease_drops_tail_groups",
            ),
        ],
    )
    def test_query_embeddings_follow_target_layout(
        self, monkeypatch, config_shape: tuple, checkpoint_shape: tuple, checkpoint_extra: dict, expected: list[int]
    ):
        """Checkpoint query embeddings are re-partitioned per group when the checkpoint layout is known.

        The first column carries the (group, query) identity (see ``_labelled_query_tensor``).
        """
        mc = _query_config(*config_shape)
        _serve_checkpoint(monkeypatch, _query_checkpoint(*checkpoint_shape, **checkpoint_extra))

        nn_model = _fake_nn_model()
        load_pretrain_weights(nn_model, mc)

        passed_state = nn_model.load_state_dict.call_args[0][0]
        assert passed_state["refpoint_embed.weight"][:, 0].int().tolist() == expected
        assert passed_state["query_feat.weight"][:, 0].int().tolist() == expected

    @pytest.mark.parametrize(
        ("config_shape", "checkpoint_shape", "checkpoint_args"),
        [
            pytest.param((4, 5), (4, 3), {"num_queries": 4, "group_detr": 3}, id="group_detr_grows"),
            # No recorded layout and 7 rows do not split into 3 groups, so the layout cannot be recovered to pad.
            pytest.param((5, 3), (7, 1), {}, id="unknown_layout_not_divisible"),
            # 12 rows read as 6 groups of 2 would pad group growth as query growth and drop half of group 0's rows.
            pytest.param((4, 6), (4, 3), {}, id="unknown_layout_with_group_growth"),
        ],
    )
    def test_rejects_query_layout_larger_than_checkpoint(
        self, monkeypatch, config_shape: tuple, checkpoint_shape: tuple, checkpoint_args: dict
    ) -> None:
        """A layout the loader cannot pad fails with an actionable error.

        Growing ``group_detr`` needs new per-group encoder heads, and a checkpoint with no recorded layout may not split
        into the configured groups, so the loader names the offending settings and the ``pretrain_weights=None`` escape
        hatch instead of surfacing a bare ``load_state_dict`` size mismatch.
        """
        mc = _query_config(*config_shape)
        _serve_checkpoint(monkeypatch, _query_checkpoint(*checkpoint_shape, args=checkpoint_args))

        with pytest.raises(ValueError, match=r"num_queries.*pretrain_weights=None"):
            load_pretrain_weights(_fake_nn_model(), mc)

    @pytest.mark.parametrize(
        ("config_shape", "checkpoint_shape"),
        [
            pytest.param((2, 3), (4, 3), id="shorter_query_count"),
            pytest.param((4, 1), (8, 1), id="single_target_group"),
            pytest.param((100, 3), (150, 2), id="same_total_different_layout"),
        ],
    )
    def test_unknown_layout_rejects_changed_configuration(
        self, monkeypatch, config_shape: tuple, checkpoint_shape: tuple
    ) -> None:
        """Changed layouts cannot be inferred from the total number of checkpoint rows."""
        _serve_checkpoint(monkeypatch, _query_checkpoint(*checkpoint_shape, args={}))

        with pytest.raises(ValueError, match="layout"):
            load_pretrain_weights(_fake_nn_model(), _query_config(*config_shape))

    def test_unknown_layout_with_default_configuration_keeps_rows(self, monkeypatch) -> None:
        """An unchanged legacy default continues to load its query tensors without repartitioning."""
        checkpoint = _query_checkpoint(300, 13, args={})
        expected = checkpoint["model"]["query_feat.weight"].clone()
        _serve_checkpoint(monkeypatch, checkpoint)
        nn_model = _fake_nn_model()

        load_pretrain_weights(nn_model, _query_config(300, 13))

        assert torch.equal(nn_model.load_state_dict.call_args[0][0]["query_feat.weight"], expected)


_REMOVED_KEYPOINT_PROJECTION_KEYS = [
    "keypoint_head.keypoint_proj.0.weight",
    "keypoint_head.keypoint_proj.0.bias",
    "keypoint_head.keypoint_proj.2.weight",
    "keypoint_head.keypoint_proj.2.bias",
]


class TestPartialLoadDetector:
    """Tests for ``_warn_on_partial_load`` — surfaces silent partial loads."""

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(SimpleNamespace(missing_keys=[], unexpected_keys=[]), id="clean_load"),
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
                SimpleNamespace(missing_keys=["_kp_active_mask"], unexpected_keys=[]),
                id="legacy_checkpoint_missing_deterministic_keypoint_mask",
            ),
            pytest.param(
                SimpleNamespace(missing_keys=["transformer.decoder._kp_active_mask"], unexpected_keys=[]),
                id="submodule_prefixed_missing_mask",
            ),
            pytest.param(
                SimpleNamespace(missing_keys=[], unexpected_keys=_REMOVED_KEYPOINT_PROJECTION_KEYS),
                id="removed_keypoint_projection_keys",
            ),
            pytest.param(SimpleNamespace(missing_keys=42, unexpected_keys=[]), id="non_iterable_missing_keys"),
        ],
    )
    def test_no_warning_cases(self, captured, result: SimpleNamespace) -> None:
        """Cases that must not emit any partial-load warning.

        Covers: clean load, intentional head keys, deterministic buffers (bare or under a submodule path), removed
        legacy keypoint projection tensors, and non-iterable missing_keys.
        """
        _warn_on_partial_load(result, "/fake/weights.pth")
        assert captured == []

    @pytest.mark.parametrize(
        ("missing_keys", "unexpected_keys", "present", "absent"),
        [
            pytest.param(
                [
                    "backbone.0.encoder.encoder.embeddings.register_tokens",
                    "backbone.0.encoder.encoder.layers.0.register_block.weight",
                ],
                [],
                ["/fake/weights.pth", "register_tokens"],
                [],
                id="unexpected_backbone_missing_keys",
            ),
            pytest.param(
                ["_kp_active_mask", "backbone.0.encoder.encoder.embeddings.register_tokens"],
                [],
                ["register_tokens"],
                ["_kp_active_mask"],
                id="missing_keypoint_mask_does_not_hide_real_missing_key",
            ),
            pytest.param(
                [
                    "class_embed.weight",
                    "bbox_embed.layers.0.weight",
                    "refpoint_embed.weight",
                    "backbone.0.encoder.encoder.embeddings.register_tokens",
                ],
                [],
                ["register_tokens"],
                ["class_embed"],
                id="intentional_head_keys_filtered_from_mixed_missing_keys",
            ),
            pytest.param(
                ["_kp_active_mask_projection.weight"],
                [],
                ["_kp_active_mask_projection.weight"],
                [],
                id="similarly_named_missing_key",
            ),
            pytest.param(
                ["backbone.custom_kp_active_mask"],
                [],
                ["backbone.custom_kp_active_mask"],
                [],
                id="suffix_collision_without_separator",
            ),
            pytest.param([], ["_kp_active_mask"], ["_kp_active_mask"], [], id="unexpected_keypoint_mask"),
            pytest.param(
                [],
                ["backbone.0.encoder.legacy_module.weight"],
                ["not consumed by model"],
                [],
                id="unexpected_checkpoint_keys",
            ),
            pytest.param(
                [],
                [*_REMOVED_KEYPOINT_PROJECTION_KEYS, "backbone.0.encoder.legacy_module.weight"],
                ["legacy_module"],
                ["keypoint_proj"],
                id="removed_projection_keys_do_not_mask_other_unexpected_keys",
            ),
            pytest.param(
                [f"backbone.0.encoder.layer.{i}.weight" for i in range(10)],
                [],
                ["10 model parameter", "..."],
                [],
                id="long_missing_keys_truncated",
            ),
            pytest.param(
                [],
                [f"backbone.0.legacy.{i}.weight" for i in range(8)],
                ["8 checkpoint key(s)", "..."],
                [],
                id="long_unexpected_keys_truncated",
            ),
        ],
    )
    def test_warns_once_naming_only_unintentional_keys(
        self, captured, missing_keys: list[str], unexpected_keys: list[str], present: list[str], absent: list[str]
    ) -> None:
        """Genuine gaps trigger exactly one warning that names them, without the intentionally ignored keys.

        Key lists in the message are bounded to 5 entries with a trailing ellipsis.
        """
        result = SimpleNamespace(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

        _warn_on_partial_load(result, "/fake/weights.pth")

        assert len(captured) == 1, f"Expected exactly one warning, got {len(captured)}: {captured}"
        assert all(text in captured[0] for text in present), f"Warning must contain {present}; got: {captured[0]}"
        assert not any(text in captured[0] for text in absent), f"Warning must not contain {absent}; got: {captured[0]}"

    def test_handles_non_iterable_input_gracefully(self, captured):
        """A MagicMock-style result (used in many existing tests) must not raise.

        The crucial assertion is "did not raise"; whether anything is captured depends on MagicMock truthiness, and both
        outcomes are acceptable.
        """
        _warn_on_partial_load(MagicMock(), "/fake/weights.pth")


class TestPartialLoadWiring(_PatchWeightsIO):
    """``load_pretrain_weights`` wires up the partial-load detector."""

    @pytest.mark.parametrize(
        ("missing_keys", "expect_warning"),
        [
            pytest.param(["backbone.0.encoder.something_required.weight"], True, id="partial_load_warns"),
            pytest.param([], False, id="clean_load_is_silent"),
        ],
    )
    def test_partial_load_warning_follows_load_result(
        self, monkeypatch, captured, missing_keys: list[str], expect_warning: bool
    ) -> None:
        """A load that leaves model parameters missing warns about a partial load; a clean load does not."""
        _serve_checkpoint(monkeypatch, _make_checkpoint(num_classes=91))
        nn_model = _fake_nn_model()
        nn_model.load_state_dict.return_value = SimpleNamespace(missing_keys=missing_keys, unexpected_keys=[])

        load_pretrain_weights(nn_model, _base_config())

        assert any("partially" in m for m in captured) is expect_warning, f"Unexpected warnings: {captured}"
