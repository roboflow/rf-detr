# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the RFDETR training shim: legacy checkpoints, pretrained-weight loading, public exports and deploy."""

import argparse
import importlib
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.training.module_model import RFDETRModelModule

# ---------------------------------------------------------------------------
# 3. convert_legacy_checkpoint
# ---------------------------------------------------------------------------


class _CustomArgs:
    """Module-level class so torch.save can pickle instances of it."""

    lr: float
    epochs: int


def _make_legacy_pth(tmp_path, epoch=5, include_ema=False, args_value="namespace") -> str:
    """Write a minimal legacy .pth checkpoint and return its path.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     sorted(torch.load(_make_legacy_pth(Path(directory), epoch=7), weights_only=False))
        ['args', 'epoch', 'model']
    """
    path = str(tmp_path / "legacy.pth")
    state = {
        "layer.weight": torch.ones(2, 3),
        "layer.bias": torch.zeros(3),
    }
    ckpt: dict[str, Any] = {"model": state, "epoch": epoch}

    if args_value == "namespace":
        ns = argparse.Namespace(lr=1e-4, epochs=100)
        ckpt["args"] = ns
    elif args_value == "dict":
        ckpt["args"] = {"lr": 1e-4, "epochs": 100}
    elif args_value is None:
        ckpt["args"] = None
    elif args_value == "missing":
        pass  # no "args" key at all
    else:
        ckpt["args"] = args_value

    if include_ema:
        ckpt["ema_model"] = {k: v.clone() * 0.99 for k, v in state.items()}

    torch.save(ckpt, path)
    return path


class TestConvertLegacyCheckpoint:
    """convert_legacy_checkpoint() produces a valid PTL .ckpt file."""

    def test_state_dict_keys_prefixed_with_model(self, tmp_path):
        """All state_dict keys must be prefixed with 'model.'."""
        src = _make_legacy_pth(tmp_path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert all(k.startswith("model.") for k in ckpt["state_dict"])

    def test_state_dict_keys_with_dots_prefixed_once(self, tmp_path):
        """Keys already containing dots are prefixed exactly once."""
        path = str(tmp_path / "dot_keys.pth")
        torch.save({"model": {"backbone.layer.weight": torch.zeros(1)}, "epoch": 0}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "model.backbone.layer.weight" in ckpt["state_dict"]
        assert "model.model.backbone.layer.weight" not in ckpt["state_dict"]

    @pytest.mark.parametrize(
        ("key", "expected"),
        [("epoch", 42), ("global_step", 0), ("legacy_checkpoint_format", True)],
    )
    def test_checkpoint_scalar_fields(self, tmp_path, key, expected):
        """Epoch is copied from the source, global_step is always 0 and legacy_checkpoint_format is always True."""
        src = _make_legacy_pth(tmp_path, epoch=42)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt[key] == expected
        assert type(ckpt[key]) is type(expected)

    def test_epoch_defaults_to_zero_when_missing(self, tmp_path):
        """Missing epoch key in source defaults to 0."""
        path = str(tmp_path / "no_epoch.pth")
        torch.save({"model": {"w": torch.zeros(1)}}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0

    @pytest.mark.parametrize(
        "args_value, expected_hyper_parameters",
        [
            pytest.param("namespace", {"lr": pytest.approx(1e-4), "epochs": 100}, id="namespace-converted-via-vars"),
            pytest.param("dict", {"lr": pytest.approx(1e-4), "epochs": 100}, id="dict-kept-as-dict"),
            pytest.param(None, {}, id="none-gives-empty"),
            pytest.param("missing", {}, id="missing-key-gives-empty"),
        ],
    )
    def test_args_converted_to_hyper_parameters(
        self, tmp_path, args_value: str | None, expected_hyper_parameters: dict[str, object]
    ):
        """convert_legacy_checkpoint() normalizes source 'args' into hyper_parameters.

        An argparse.Namespace is converted via vars(), a dict is kept as-is, and a None or altogether-missing 'args' key
        produces an empty dict.
        """
        src = _make_legacy_pth(tmp_path, args_value=args_value)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == expected_hyper_parameters

    def test_args_custom_object_with_dict_converted_via_vars(self, tmp_path):
        """A custom object with __dict__ is converted via vars()."""
        opts = _CustomArgs()
        opts.lr = 2e-4
        opts.epochs = 50

        path = str(tmp_path / "custom_args.pth")
        torch.save({"model": {"w": torch.zeros(1)}, "epoch": 0, "args": opts}, path)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"]["lr"] == pytest.approx(2e-4)

    def test_ema_model_preserved_as_legacy_ema_state_dict(self, tmp_path):
        """ema_model present in source is written as legacy_ema_state_dict."""
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" in ckpt
        assert "layer.weight" in ckpt["legacy_ema_state_dict"]

    def test_no_ema_model_no_legacy_ema_state_dict(self, tmp_path):
        """No ema_model in source means legacy_ema_state_dict is absent."""
        src = _make_legacy_pth(tmp_path, include_ema=False)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "legacy_ema_state_dict" not in ckpt

    def test_round_trip_with_on_load_checkpoint(self, tmp_path):
        """convert_legacy_checkpoint output is handled correctly by on_load_checkpoint.

        After conversion, loading the .ckpt via on_load_checkpoint must NOT re-apply the 'model.' prefix because
        'state_dict' already exists.
        """
        src = _make_legacy_pth(tmp_path, include_ema=True)
        dst = str(tmp_path / "out.ckpt")
        convert_legacy_checkpoint(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)

        class _FakeModule:
            model_config = SimpleNamespace(positional_encoding_size=36)
            train_config = SimpleNamespace(optimizer="torch.optim.AdamW")

        fake = _FakeModule()
        original_state_dict = dict(ckpt["state_dict"])  # copy before mutation

        RFDETRModelModule.on_load_checkpoint(fake, ckpt)

        # state_dict must NOT have been re-prefixed (already had "state_dict")
        assert ckpt["state_dict"] == original_state_dict
        # EMA stashed
        assert hasattr(fake, "_pending_legacy_ema_state")

    def test_missing_model_key_raises_value_error(self, tmp_path):
        """Source file with no 'model' key raises ValueError with a clear message."""
        path = str(tmp_path / "no_model.pth")
        torch.save({"epoch": 5}, path)
        dst = str(tmp_path / "out.ckpt")

        with pytest.raises(ValueError) as exc_info:
            convert_legacy_checkpoint(path, dst)
        assert "'model' key" in str(exc_info.value)

    def test_args_primitive_type_falls_back_to_empty_dict(self, tmp_path):
        """Args of a non-dict, non-Namespace type (e.g. string) falls back to {} with a warning."""
        path = str(tmp_path / "prim_args.pth")
        torch.save({"model": {"w": torch.zeros(1)}, "args": "legacy_string_value"}, path)
        dst = str(tmp_path / "out.ckpt")

        convert_legacy_checkpoint(path, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["hyper_parameters"] == {}


# ---------------------------------------------------------------------------
# 4. RFDETRModule.on_load_checkpoint
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal object supporting attribute assignment for on_load_checkpoint tests.

    A checkpoint carrying optimizer state also has the hook resolve the resumed learning-rate settings (#1613), which
    reads ``train_config.model_fields_set`` and ``global_rank``.
    """

    model_config = SimpleNamespace(positional_encoding_size=36)
    train_config = TrainConfig(dataset_dir="data", optimizer="torch.optim.AdamW")
    global_rank = 0
    _adopt_resumed_param_group_settings = RFDETRModelModule._adopt_resumed_param_group_settings


class TestOnLoadCheckpoint:
    """RFDETRModule.on_load_checkpoint auto-detects legacy formats."""

    def test_raw_pth_writes_state_dict_with_prefix(self):
        """'model' key without 'state_dict' → state_dict written with 'model.' prefix."""
        fake = _FakeModule()
        ckpt = {"model": {"backbone.weight": torch.zeros(2)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "state_dict" in ckpt
        assert "model.backbone.weight" in ckpt["state_dict"]

    def test_raw_pth_original_model_key_preserved(self):
        """Original 'model' key is not deleted after state_dict is written."""
        fake = _FakeModule()
        ckpt = {"model": {"w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert "model" in ckpt  # PTL may inspect it; must not be deleted

    def test_empty_model_dict_produces_empty_state_dict(self):
        """Empty 'model' dict without 'state_dict' → empty state_dict written."""
        fake = _FakeModule()
        ckpt = {"model": {}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] == {}

    def test_native_ptl_format_no_op(self):
        """Native PTL checkpoint (has 'state_dict', no 'model') → no mutation."""
        fake = _FakeModule()
        sentinel = {"model.layer.weight": torch.zeros(1)}
        ckpt = {"state_dict": sentinel}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] is sentinel  # not replaced
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_both_model_and_state_dict_present_state_dict_not_overwritten(self):
        """'state_dict' is NOT overwritten when both 'model' and 'state_dict' exist."""
        fake = _FakeModule()
        existing_sd = {"model.existing": torch.zeros(1)}
        ckpt = {
            "state_dict": existing_sd,
            "model": {"new_key": torch.ones(1)},
        }
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert ckpt["state_dict"] is existing_sd
        assert "model.new_key" not in ckpt["state_dict"]

    def test_legacy_ema_state_dict_stashed(self):
        """'legacy_ema_state_dict' in checkpoint → stashed on _pending_legacy_ema_state."""
        fake = _FakeModule()
        ema_weights = {"layer.weight": torch.ones(2)}
        ckpt = {
            "state_dict": {"model.layer.weight": torch.zeros(2)},
            "legacy_ema_state_dict": ema_weights,
        }
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert fake._pending_legacy_ema_state is ema_weights

    def test_no_legacy_ema_attribute_not_set(self):
        """No 'legacy_ema_state_dict' → _pending_legacy_ema_state not set on module."""
        fake = _FakeModule()
        ckpt = {"state_dict": {"model.w": torch.zeros(1)}}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_empty_checkpoint_is_noop(self):
        """Completely empty checkpoint {} triggers no mutation and no error."""
        fake = _FakeModule()
        ckpt: dict[str, Any] = {}
        RFDETRModelModule.on_load_checkpoint(fake, ckpt)  # must not raise
        assert ckpt == {}
        assert not hasattr(fake, "_pending_legacy_ema_state")

    def test_second_call_overwrites_pending_ema(self):
        """Calling on_load_checkpoint twice with EMA overwrites the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        second_ema = {"w": torch.ones(1)}
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": second_ema})
        assert fake._pending_legacy_ema_state is second_ema

    def test_second_call_without_ema_leaves_first_stash(self):
        """Second call without 'legacy_ema_state_dict' does not clear the stash."""
        fake = _FakeModule()
        first_ema = {"w": torch.zeros(1)}
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}, "legacy_ema_state_dict": first_ema})
        RFDETRModelModule.on_load_checkpoint(fake, {"state_dict": {}})
        assert fake._pending_legacy_ema_state is first_ema

    def test_pre_merge_optimizer_state_is_regrouped(self) -> None:
        """on_load_checkpoint itself regroups one-group-per-parameter optimizer state.

        tests/training/optim/test_param_groups.py exercises regroup_unmerged_optimizer_state() directly; this proves the
        hook actually calls it, since a wiring mistake there (wrong argument, wrong order relative to the other
        normalisation steps, a silently swallowed exception) would not be caught by calling the helper on its own.
        """
        fake = _FakeModule()
        ckpt = {
            "state_dict": {"model.w": torch.zeros(1)},
            "optimizer_states": [
                {
                    "state": {0: {"exp_avg": torch.ones(1)}, 1: {"exp_avg": torch.full((1,), 2.0)}},
                    "param_groups": [
                        {"lr": 0.1, "weight_decay": 0.0, "params": [0]},
                        {"lr": 0.1, "weight_decay": 0.0, "params": [1]},
                    ],
                }
            ],
        }

        RFDETRModelModule.on_load_checkpoint(fake, ckpt)

        groups = ckpt["optimizer_states"][0]["param_groups"]
        assert len(groups) == 1
        assert groups[0]["params"] == [0, 1]


# ---------------------------------------------------------------------------
# 5. Public API exports
# ---------------------------------------------------------------------------


_PTL_EXPORTS = ["RFDETRModelModule", "RFDETRDataModule", "build_trainer"]


class TestPublicAPIExports:
    """rfdetr.__init__ exposes PTL names via __getattr__ (rfdetr[train] extra)."""

    @pytest.mark.parametrize("name", _PTL_EXPORTS)
    def test_symbol_importable_from_rfdetr(self, name):
        """Each PTL export is accessible as rfdetr.<name> via lazy __getattr__."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} is missing"

    @pytest.mark.parametrize("name", _PTL_EXPORTS)
    def test_symbol_is_same_object_as_rfdetr_training(self, name):
        """rfdetr.<name> is the identical object to rfdetr.training.<name>."""
        import rfdetr
        import rfdetr.training

        assert getattr(rfdetr, name) is getattr(rfdetr.training, name)

    @pytest.mark.parametrize("name", _PTL_EXPORTS)
    def test_ptl_name_not_in_all(self, name):
        """PTL exports are optional (rfdetr[train]) and must not be in rfdetr.__all__."""
        import rfdetr

        assert name not in rfdetr.__all__, f"{name} must not be in __all__ (optional extra)"

    def test_rfdetr_all_no_duplicates(self):
        """rfdetr.__all__ contains no duplicate names."""
        import rfdetr

        assert len(rfdetr.__all__) == len(set(rfdetr.__all__))

    def test_plus_symbol_resolution_does_not_mutate_all(self, monkeypatch):
        """Top-level __all__ remains static when plus-only symbols resolve lazily."""
        import rfdetr
        import rfdetr.platform.models

        sentinel = object()
        monkeypatch.setitem(rfdetr.platform.models.__dict__, "RFDETRXLarge", sentinel)
        # Track the prior state even when the lazy export has never been cached.
        monkeypatch.setitem(rfdetr.__dict__, "RFDETRXLarge", sentinel)
        monkeypatch.delitem(rfdetr.__dict__, "RFDETRXLarge")

        original_all = list(rfdetr.__all__)
        assert rfdetr.RFDETRXLarge is sentinel
        assert rfdetr.__all__ == original_all

    @pytest.mark.parametrize("name", ["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"])
    def test_existing_export_still_present(self, name):
        """Original RFDETR* class exports are unchanged."""
        import rfdetr

        assert hasattr(rfdetr, name), f"rfdetr.{name} unexpectedly missing"

    def test_convert_legacy_checkpoint_not_in_rfdetr_namespace(self):
        """convert_legacy_checkpoint is in rfdetr.training but not the top-level rfdetr namespace."""
        import rfdetr
        from rfdetr.training import convert_legacy_checkpoint  # noqa: F401

        # It is NOT directly on rfdetr (Phase 7.7 spec lists only the three PTL exports)
        assert not hasattr(rfdetr, "convert_legacy_checkpoint")


class TestRemovedLegacyModuleAliases:
    """Removed legacy modules raise migration-hint ImportErrors on any access."""

    @staticmethod
    def _simulate_missing_removed_module_specs(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        """Force the removed-module finder to behave as if shim files no longer exist."""
        import rfdetr

        path_finder = rfdetr._RemovedModuleFinder._PATH_FINDER
        original_find_spec = path_finder.find_spec

        def _fake_find_spec(fullname: str, path: list[str] | None = None, target: object | None = None) -> object:
            if fullname in names:
                return None
            return original_find_spec(fullname, path, target)

        monkeypatch.setattr(path_finder, "find_spec", _fake_find_spec)
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)

        root_names = {name.removeprefix("rfdetr.").split(".", maxsplit=1)[0] for name in names}
        for root_name in root_names:
            monkeypatch.delitem(rfdetr.__dict__, root_name, raising=False)

    def test_removed_shim_missing_raises_importerror_with_getattr(self, mocker: MockerFixture) -> None:
        """Missing removed shim should raise ImportError with migration hint."""
        import rfdetr

        missing_name = "rfdetr.missing_removed_shim"
        missing_exc = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
        mocker.patch.dict(rfdetr._REMOVE_IN_VERSION_1_9, {"missing_removed_shim": "migration hint"})
        mocker.patch("rfdetr.importlib.import_module", side_effect=missing_exc)

        with pytest.raises(ImportError) as exc_info:
            rfdetr.missing_removed_shim
        assert "migration hint" in str(exc_info.value)

    def test_nested_module_not_found_is_not_masked_for_package_attribute(self, mocker: MockerFixture) -> None:
        """Nested ModuleNotFoundError from inside a shim import should propagate."""
        import rfdetr

        mocker.patch.dict(rfdetr._REMOVE_IN_VERSION_1_9, {"missing_dep_shim": "migration hint"})
        mocker.patch(
            "rfdetr.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'torchvision_ops'", name="torchvision_ops"),
        )

        with pytest.raises(ModuleNotFoundError) as exc_info:
            rfdetr.missing_dep_shim
        assert "torchvision_ops" in str(exc_info.value)

    def test_removed_util_import_raises_migration_hint_when_shim_is_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dotted legacy imports get a migration hint once the util shim package is removed."""
        self._simulate_missing_removed_module_specs(monkeypatch, "rfdetr.util")

        with pytest.raises(ImportError) as exc_info:
            importlib.import_module("rfdetr.util")
        assert "rfdetr.util was removed in v1.9" in str(exc_info.value)

    def test_removed_deploy_submodule_import_raises_migration_hint_when_shim_is_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dotted legacy submodule imports get a migration hint once the deploy shim is removed."""
        self._simulate_missing_removed_module_specs(monkeypatch, "rfdetr.deploy", "rfdetr.deploy.benchmark")

        with pytest.raises(ImportError) as exc_info:
            importlib.import_module("rfdetr.deploy.benchmark")
        assert "rfdetr.deploy was removed in v1.9" in str(exc_info.value)

    def test_find_spec_ignores_non_rfdetr_top_level_imports(self) -> None:
        """find_spec must not intercept bare top-level imports like 'util' or 'deploy'."""
        import rfdetr

        finder = rfdetr._RemovedModuleFinder()
        assert finder.find_spec("util", None) is None
        assert finder.find_spec("deploy", None) is None
        assert finder.find_spec("os", None) is None

    def test_meta_path_insertion_is_idempotent_across_reload(self) -> None:
        """importlib.reload(rfdetr) must not insert a second finder into sys.meta_path."""
        import rfdetr

        count_before = sum(type(f).__name__ == "_RemovedModuleFinder" for f in sys.meta_path)
        importlib.reload(rfdetr)
        count_after = sum(type(f).__name__ == "_RemovedModuleFinder" for f in sys.meta_path)
        assert count_after == count_before, (
            f"reload added {count_after - count_before} extra finder(s) to sys.meta_path"
        )


# ---------------------------------------------------------------------------
# 6. load_pretrain_weights — detr.py path (the non-PTL scenario from #806)
# ---------------------------------------------------------------------------


def _make_detr_checkpoint(
    num_classes=91,
    num_queries=300,
    group_detr=13,
    segmentation_head=False,
    patch_size=14,
):
    """Return a minimal checkpoint dict for _load_pretrain_weights_into tests.

    Examples:
        >>> checkpoint = _make_detr_checkpoint(num_classes=5)
        >>> checkpoint["model"]["class_embed.bias"].shape[0], checkpoint["args"].patch_size
        (5, 14)
    """
    total_queries = num_queries * group_detr
    state = {
        "class_embed.bias": torch.zeros(num_classes),
        "refpoint_embed.weight": torch.zeros(total_queries, 4),
        "query_feat.weight": torch.zeros(total_queries, 256),
    }
    ckpt_args = SimpleNamespace(
        segmentation_head=segmentation_head,
        patch_size=patch_size,
        class_names=[],
    )
    return {"model": state, "args": ckpt_args}


class TestLoadPretrainWeightsInto:
    """Checkpoint compatibility validation in load_pretrain_weights (models/weights.py).

    Exercised when RFDETRNano(pretrain_weights=...) is called (issue #806).
    """

    @pytest.fixture(autouse=True)
    def _patch_download(self, monkeypatch):
        """Suppress all download and file-existence side effects."""
        monkeypatch.setattr("rfdetr.models.weights.download_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.validate_pretrain_weights", lambda *a, **kw: None)
        monkeypatch.setattr("rfdetr.models.weights.os.path.isfile", lambda _: True)

    def test_seg_ckpt_into_detection_model_raises_via_detr_path(self, monkeypatch, tmp_path):
        """Segmentation checkpoint must raise ValueError when loaded into a detection model."""
        from rfdetr.models.weights import load_pretrain_weights

        checkpoint = _make_detr_checkpoint(segmentation_head=True, patch_size=14)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", segmentation_head=False)

        with pytest.raises(ValueError) as exc_info:
            load_pretrain_weights(fake_model, mc)
        assert "segmentation head" in str(exc_info.value)

    def test_patch_size_mismatch_raises_via_detr_path(self, monkeypatch, tmp_path):
        """patch_size mismatch must raise ValueError via the load_pretrain_weights path."""
        from rfdetr.models.weights import load_pretrain_weights

        checkpoint = _make_detr_checkpoint(segmentation_head=False, patch_size=12)
        monkeypatch.setattr("rfdetr.models.weights.torch.load", lambda *a, **kw: checkpoint)

        fake_model = MagicMock()
        mc = RFDETRBaseConfig(pretrain_weights="/fake/weights.pth", device="cpu", patch_size=16)

        with pytest.raises(ValueError) as exc_info:
            load_pretrain_weights(fake_model, mc)
        assert "patch_size" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. RFDETR.class_names property — empty-list identity check
# ---------------------------------------------------------------------------


class TestClassNamesProperty:
    """RFDETR.class_names property returns List[str] (0-indexed)."""

    def test_empty_class_names_returns_empty_list_not_coco(self):
        """class_names property returns [] when model.class_names is [], NOT COCO fallback.

        Regression test for #509: the truthiness check `and self.model.class_names:` treated [] as falsy and fell
        through to return COCO_CLASSES, defeating the detr.py sync-back even after training on a dataset that reports
        empty names. The fix uses `is not None` so that [] is preserved.
        """
        mock_self = MagicMock()
        mock_self.model.class_names = []

        result = RFDETR.class_names.fget(mock_self)

        assert result == [], "class_names=[] must return [] (empty list), not COCO fallback"

    def test_none_class_names_returns_coco(self):
        """class_names property falls back to COCO_CLASS_NAMES when model.class_names is None."""
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        mock_self = MagicMock()
        mock_self.model.class_names = None

        result = RFDETR.class_names.fget(mock_self)

        assert result == COCO_CLASS_NAMES
        assert result is not COCO_CLASS_NAMES, "COCO fallback must return a copy, not the mutable global"

    def test_custom_class_names_returned_as_list(self):
        """Non-empty class_names are returned as a 0-indexed list."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)

        assert result == ["cat", "dog"]

    def test_custom_class_names_returns_shallow_copy(self):
        """Mutating the returned class_names list must not mutate model state."""
        mock_self = MagicMock()
        mock_self.model.class_names = ["cat", "dog"]

        result = RFDETR.class_names.fget(mock_self)
        result.append("bird")

        assert result == ["cat", "dog", "bird"]
        assert mock_self.model.class_names == ["cat", "dog"]


# ---------------------------------------------------------------------------
# 8. RFDETR.deploy_to_roboflow — class_names.txt and args.class_names
# ---------------------------------------------------------------------------


class TestDeployToRoboflow:
    """deploy_to_roboflow writes class_names.txt and embeds class_names in args.

    Regression tests for the bug where RFDETRSeg models (and any model whose args namespace lacks a ``class_names``
    attribute) failed to upload to Roboflow with a FileNotFoundError from the Roboflow client library.
    """

    @pytest.fixture
    def mock_self(self):
        """Return a minimal RFDETR-like mock for deploy_to_roboflow tests."""
        class_names = ["cat", "dog"]
        mock_self = MagicMock(spec=RFDETR)
        mock_self.size = "rfdetr-small"
        mock_self.class_names = class_names  # the property, resolved to a plain list
        # `model` is an instance attribute (set in __init__), not a class attribute, so
        # MagicMock(spec=RFDETR).__getattr__ would raise AttributeError for it.  Assign
        # it directly via __setattr__ so sub-attribute chaining works correctly.
        mock_self.model = MagicMock()
        mock_self.model.model.state_dict.return_value = {}
        mock_self.model.args = SimpleNamespace(num_classes=len(class_names))
        # deploy_to_roboflow now delegates bundle-writing to export_for_roboflow; bind the
        # real method so these end-to-end tests exercise it (a bare MagicMock attribute
        # would no-op the class_names.txt / torch.save side effects).
        mock_self.export_for_roboflow = lambda output_dir: RFDETR.export_for_roboflow(mock_self, output_dir)
        return mock_self

    @staticmethod
    def _set_class_names(mock_self: MagicMock, class_names: list[str]) -> None:
        """Update class names and keep args.num_classes in sync."""
        mock_self.class_names = class_names
        mock_self.model.args.num_classes = len(class_names)

    @pytest.fixture
    def roboflow_project(self, mocker: MockerFixture) -> MagicMock:
        """Patch the Roboflow client and return the project mock that deploy_to_roboflow talks to."""
        return mocker.patch("roboflow.Roboflow").return_value.workspace.return_value.project.return_value

    @pytest.fixture
    def roboflow_deploy(self, roboflow_project: MagicMock) -> MagicMock:
        """Return the ``deploy`` mock of the Roboflow version that deploy_to_roboflow uploads to."""
        return roboflow_project.version.return_value.deploy

    @staticmethod
    def _deploy(mock_self: MagicMock, **deploy_kwargs: Any) -> None:
        """Call deploy_to_roboflow with a fixed workspace, project and API key.

        Examples:
            >>> TestDeployToRoboflow._deploy(model, version=1)  # doctest: +SKIP
            (needs the mocked RFDETR instance built by the ``mock_self`` fixture)
        """
        RFDETR.deploy_to_roboflow(
            mock_self, workspace="test-workspace", project_id="test-project", api_key="dummy-key", **deploy_kwargs
        )

    def test_class_names_txt_written_with_correct_content(self, tmp_path, monkeypatch, mock_self, roboflow_deploy):
        """deploy_to_roboflow must write class_names.txt with one name per line.

        Regression: RFDETRSeg models were failing with FileNotFoundError from the Roboflow client library because
        class_names.txt was absent.
        """
        monkeypatch.chdir(tmp_path)
        self._set_class_names(mock_self, ["cat", "dog", "bird"])
        captured: dict = {}

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            # Inspect class_names.txt while the temp dir still exists (before cleanup).
            f = (Path(model_path) / "class_names.txt").resolve()
            if f.exists():
                captured["content"] = f.read_text()

        roboflow_deploy.side_effect = deploy_side_effect

        self._deploy(mock_self, version=1)

        assert "content" in captured, "class_names.txt was not present in the upload directory during deploy"
        assert captured["content"] == "cat\ndog\nbird"

    @pytest.mark.parametrize(
        ("args_overrides", "expected_class_names"),
        [
            pytest.param({}, ["cat", "dog"], id="attribute-absent"),
            pytest.param({"class_names": None}, ["cat", "dog"], id="attribute-none"),
            pytest.param(
                {"class_names": ["existing_cat", "existing_dog"]}, ["existing_cat", "existing_dog"], id="existing-kept"
            ),
        ],
    )
    @pytest.mark.usefixtures("roboflow_project")
    def test_checkpoint_args_class_names(
        self,
        tmp_path,
        monkeypatch,
        mocker,
        mock_self,
        args_overrides: dict[str, Any],
        expected_class_names: list[str],
    ):
        """The saved checkpoint's args carry the model's class names unless args already holds a non-None list.

        Regression: args.class_names was absent after switching to PTL training, causing the Roboflow client library
        to raise FileNotFoundError.
        """
        monkeypatch.chdir(tmp_path)
        vars(mock_self.model.args).update(args_overrides)
        save = mocker.patch("torch.save")  # skip the actual disk I/O

        self._deploy(mock_self, version=1)

        assert save.called, "torch.save must have been called"
        assert save.call_args_list[0].args[0]["args"].class_names == expected_class_names

    def test_temp_dir_cleaned_up_after_deploy(self, tmp_path, monkeypatch, mock_self, roboflow_deploy):
        """The temporary upload directory must be removed after a successful deploy."""
        monkeypatch.chdir(tmp_path)
        self._set_class_names(mock_self, ["cat"])
        deployed_paths: list[Path] = []

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            deployed_paths.append(Path(model_path))

        roboflow_deploy.side_effect = deploy_side_effect

        self._deploy(mock_self, version=1)

        assert deployed_paths, "deploy must receive a temporary model_path"
        assert not deployed_paths[0].exists(), "Temporary upload dir must be removed after deploy"
        assert not (tmp_path / ".roboflow_temp_upload").exists(), "Fixed-name temp dir must not be created"

    def test_temp_dir_cleaned_up_after_deploy_failure(self, tmp_path, monkeypatch, mock_self, roboflow_deploy):
        """Temp upload dir must be removed even when deploy() raises an exception."""
        monkeypatch.chdir(tmp_path)
        self._set_class_names(mock_self, ["cat"])
        deployed_paths: list[Path] = []

        def deploy_side_effect(model_type, model_path, filename, **kwargs):
            deployed_paths.append(Path(model_path))
            raise RuntimeError("upload failed")

        roboflow_deploy.side_effect = deploy_side_effect

        with pytest.raises(RuntimeError) as exc_info:
            self._deploy(mock_self, version=1)

        assert "upload failed" in str(exc_info.value)
        assert deployed_paths, "deploy must receive a temporary model_path"
        assert not deployed_paths[0].exists(), "Temporary upload dir must be removed even after a failed deploy"
        assert not (tmp_path / ".roboflow_temp_upload").exists(), "Fixed-name temp dir must not be created"

    def test_explicit_size_overrides_model_size(self, tmp_path, monkeypatch, mock_self, roboflow_deploy):
        """An explicitly passed size must win over self.size (documented precedence).

        Regression: ``size = self.size or size`` inverted the precedence, silently ignoring the user's argument.
        """
        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._deploy(mock_self, version=1, size="rfdetr-medium")

        assert roboflow_deploy.call_args.kwargs["model_type"] == "rfdetr-medium"

    def test_size_defaults_to_model_size_when_not_provided(self, tmp_path, monkeypatch, mock_self, roboflow_deploy):
        """Without an explicit size the model's own size is deployed."""
        monkeypatch.chdir(tmp_path)

        self._deploy(mock_self, version=1)

        assert roboflow_deploy.call_args.kwargs["model_type"] == "rfdetr-small"

    @pytest.mark.usefixtures("roboflow_project")
    def test_warns_when_explicit_size_differs_from_model_size(self, tmp_path, monkeypatch, mock_self):
        """A UserWarning is emitted when the explicit size conflicts with the model's own size."""
        monkeypatch.chdir(tmp_path)
        with pytest.warns(UserWarning, match="rfdetr-medium.*rfdetr-small"):
            self._deploy(mock_self, version=1, size="rfdetr-medium")

    @pytest.mark.usefixtures("roboflow_project")
    def test_no_warning_when_explicit_size_matches_model_size(self, tmp_path, monkeypatch, mock_self):
        """No size-conflict warning is emitted when the explicit size equals the model's own size."""
        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._deploy(mock_self, version=1, size="rfdetr-small")

        conflict_warnings = [w for w in caught if "deploy_to_roboflow" in str(w.message)]
        assert not conflict_warnings

    def test_omitted_version_resolves_to_latest(self, tmp_path, monkeypatch, mock_self, roboflow_project):
        """Omitting version deploys to the highest existing dataset version.

        Project has versions 9 and 10 (server list order not guaranteed); auto-resolution must pick 10 so the model
        lands on the newest dataset snapshot without the caller tracking version numbers.
        """
        monkeypatch.chdir(tmp_path)
        roboflow_project.get_version_information.return_value = [{"id": "ws/proj/9"}, {"id": "ws/proj/10"}]

        self._deploy(mock_self)

        roboflow_project.version.assert_called_once_with(10)

    def test_omitted_version_falls_back_to_one_for_empty_project(
        self, tmp_path, monkeypatch, mock_self, roboflow_project
    ):
        """Omitting version on a project with no versions falls back to version 1.

        The SDK's Version lookup then raises its own "Version number 1 is not found." for a genuinely empty project, so
        the fallback never silently deploys anywhere unexpected.
        """
        monkeypatch.chdir(tmp_path)
        roboflow_project.get_version_information.return_value = []

        self._deploy(mock_self)

        roboflow_project.version.assert_called_once_with(1)

    def test_explicit_version_skips_lookup(self, tmp_path, monkeypatch, mock_self, roboflow_project):
        """An explicitly passed version is used verbatim without any version-list lookup.

        Guards the no-surprise contract: existing callers pinning a version must not trigger the extra
        get_version_information network round-trip nor have their choice overridden by a newer version.
        """
        monkeypatch.chdir(tmp_path)
        roboflow_project.get_version_information.return_value = [{"id": "ws/proj/9"}]

        self._deploy(mock_self, version=2)

        roboflow_project.get_version_information.assert_not_called()
        roboflow_project.version.assert_called_once_with(2)
