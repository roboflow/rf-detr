# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""A resumed run trains with the optimizer settings it records, explicit ones winning over the checkpoint (#1613).

``trainer.fit(ckpt_path=...)`` restores every parameter group's learning rate and weight decay, and the scheduler's base
learning rates, from the checkpoint. A changed ``lr`` was therefore silently ignored while ``training_config.json`` and
the checkpoints' ``args`` recorded it. Precedence is now defaults, then the checkpoint, then what the caller set.
"""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ReduceLROnPlateau, SequentialLR

from rfdetr import RFDETRNano
from rfdetr.config import TrainConfig
from rfdetr.detr import _resolve_resumed_optimizer_settings
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import (
    _PARAM_GROUP_SETTINGS,
    _apply_configured_param_group_settings,
    _resolve_resumed_param_group_settings,
)
from rfdetr.utilities.reproducibility import seed_all
from tests.conftest import build_synthetic_dataset

#: A value no setting defaults to, so a restored value cannot be mistaken for the default.
_CHECKPOINT_VALUE = 0.123
#: A second distinct value, set explicitly on the resumed run.
_EXPLICIT_VALUE = 0.456
#: Seed for the untrained model; module-scoped fixtures run before the autouse per-test reseed.
_TRAIN_SEED = 1613
#: Learning rates and weight decay of the first run; none is the ``TrainConfig`` default.
_FIRST_RUN = {"lr": 5e-5, "lr_encoder": 3e-4, "weight_decay": 2e-4}
#: Settings the resumed run passes explicitly; ``lr_encoder`` is left out so it comes from the checkpoint.
_RESUMED_RUN = {"lr": 1e-5, "weight_decay": 5e-4}
#: ``train()`` arguments shared by both runs: an untrained Nano on the CPU at 224 px.
_TRAIN_KWARGS = {
    "batch_size": 4,
    "num_workers": 0,
    "multi_scale": False,
    "augmentation_backend": "torchvision",
    "tensorboard": False,
    "run_test": False,
    "device": "cpu",
}


def _write_checkpoint(path: Path, args: object, *, optimizer_state: bool = True) -> Path:
    """Save a stand-in Lightning checkpoint holding only what the resume resolution reads.

    Args:
        path: File to write.
        args: Recorded training settings; ``None`` leaves them out, as a ``last.ckpt`` from before 1.11.1 does.
        optimizer_state: Whether the checkpoint carries optimizer state, as ``last.ckpt`` does.

    Returns:
        ``path``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     written = _write_checkpoint(Path(directory) / "last.ckpt", {"lr": 5e-5})
        ...     torch.load(written, weights_only=True)["args"]
        {'lr': 5e-05}
    """
    checkpoint: dict[str, Any] = {"optimizer_states": [{"state": {}, "param_groups": []}] if optimizer_state else []}
    if args is not None:
        checkpoint["args"] = args
    torch.save(checkpoint, path)
    return path


def _adopt(train_config: TrainConfig, args: dict[str, float] | None, *, global_rank: int = 0) -> SimpleNamespace:
    """Run ``RFDETRModelModule._adopt_resumed_param_group_settings`` on a stand-in holding only what it reads.

    Args:
        train_config: The module's config.
        args: The checkpoint's recorded settings, or ``None`` for a checkpoint without them.
        global_rank: The stand-in's process rank.

    Returns:
        The stand-in, with the attributes the hook set.

    Examples:
        >>> config = TrainConfig(dataset_dir="data", lr=1e-5)
        >>> module = _adopt(config, {"lr": 5e-5, "lr_encoder": 3e-4}, global_rank=1)
        >>> module.train_config.lr_encoder, module._reapplies_param_group_settings
        (0.0003, True)
    """
    module = SimpleNamespace(train_config=train_config, global_rank=global_rank)
    checkpoint: dict[str, Any] = {"optimizer_states": [{"state": {}, "param_groups": []}]}
    if args is not None:
        checkpoint["args"] = args
    RFDETRModelModule._adopt_resumed_param_group_settings(module, checkpoint)
    return module


@pytest.fixture
def rf_detr_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """``caplog`` capturing the ``rf-detr`` logger at INFO; that logger does not propagate to the root logger.

    Examples:
        >>> rf_detr_caplog  # doctest: +SKIP
        A pytest fixture; it needs pytest's caplog and monkeypatch.
    """
    monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
    caplog.set_level(logging.INFO, logger="rf-detr")
    return caplog


class TestResolveResumedParamGroupSettings:
    """Defaults, then the checkpoint's recorded ``args``, then what the caller set."""

    @pytest.mark.parametrize("setting", _PARAM_GROUP_SETTINGS)
    def test_unset_setting_takes_the_checkpoint_value(self, setting: str) -> None:
        config = TrainConfig(dataset_dir="data")
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {"args": {setting: _CHECKPOINT_VALUE}})
        assert getattr(resolved, setting) == _CHECKPOINT_VALUE

    @pytest.mark.parametrize("setting", _PARAM_GROUP_SETTINGS)
    def test_explicit_setting_wins_over_the_checkpoint(self, setting: str) -> None:
        config = TrainConfig(dataset_dir="data", **{setting: _EXPLICIT_VALUE})
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {"args": {setting: _CHECKPOINT_VALUE}})
        assert getattr(resolved, setting) == _EXPLICIT_VALUE

    def test_restored_setting_is_not_marked_explicit(self) -> None:
        config = TrainConfig(dataset_dir="data")
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {"args": {"lr": _CHECKPOINT_VALUE}})
        assert "lr" not in resolved.model_fields_set

    def test_setting_missing_from_the_checkpoint_keeps_its_default(self) -> None:
        config = TrainConfig(dataset_dir="data")
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {"args": {"lr": _CHECKPOINT_VALUE}})
        assert resolved.weight_decay == TrainConfig.model_fields["weight_decay"].default

    def test_explicit_setting_equal_to_the_checkpoint_is_no_override(self) -> None:
        config = TrainConfig(dataset_dir="data", lr=5e-5)
        _, _, overridden = _resolve_resumed_param_group_settings(config, {"args": {"lr": 5e-5}})
        assert overridden == {}

    def test_checkpoint_without_recorded_settings_leaves_the_config_alone(self) -> None:
        config = TrainConfig(dataset_dir="data", lr=1e-5)
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {})
        assert resolved is config

    def test_config_built_from_a_complete_mapping_takes_unchanged_defaults_from_the_checkpoint(self) -> None:
        """LightningCLI's parser (``rfdetr fit --ckpt_path``) marks every field as set, defaults included."""
        config = TrainConfig(**TrainConfig(dataset_dir="data", lr=1e-5).model_dump())
        resolved, _, _ = _resolve_resumed_param_group_settings(config, {"args": {"lr_encoder": _CHECKPOINT_VALUE}})
        assert resolved.lr_encoder == _CHECKPOINT_VALUE


class TestResolveResumedOptimizerSettings:
    """``RFDETR.train()`` resolves the settings from the ``resume`` file before anything records the config."""

    def test_fills_unset_settings_from_the_checkpoint(self, tmp_path: Path) -> None:
        resume = _write_checkpoint(tmp_path / "last.ckpt", {"lr_encoder": _CHECKPOINT_VALUE})
        config = TrainConfig(dataset_dir="data", resume=str(resume))
        assert _resolve_resumed_optimizer_settings(config).lr_encoder == _CHECKPOINT_VALUE

    def test_checkpoint_without_optimizer_state_leaves_the_config_alone(self, tmp_path: Path) -> None:
        """A lightweight ``.pth`` restarts the optimizer from the config, so nothing is taken from it."""
        resume = _write_checkpoint(tmp_path / "last_ema.pth", {"lr": _CHECKPOINT_VALUE}, optimizer_state=False)
        config = TrainConfig(dataset_dir="data", resume=str(resume))
        assert _resolve_resumed_optimizer_settings(config) is config

    @pytest.mark.parametrize("resume", ["missing.ckpt", "last"])
    def test_resume_naming_no_file_leaves_the_config_alone(self, tmp_path: Path, resume: str) -> None:
        """Lightning resolves a sentinel such as ``"last"`` itself, and the module resolves the settings then."""
        config = TrainConfig(dataset_dir="data", resume=str(tmp_path / resume))
        assert _resolve_resumed_optimizer_settings(config) is config

    @pytest.mark.parametrize("content", ["pickle-only", "truncated"])
    def test_checkpoint_unreadable_weights_only_leaves_the_config_alone(self, tmp_path: Path, content: str) -> None:
        """Nothing is unpickled beyond what a weights-only ``torch.load`` allows, and a broken file is left to
        Lightning; a pre-Lightning ``.pth``, for one, holds argparse objects."""
        resume = tmp_path / "checkpoint.pth"
        if content == "pickle-only":
            torch.save(
                {
                    "optimizer_states": [{"state": {}, "param_groups": []}],
                    "args": {"lr": _CHECKPOINT_VALUE},
                    "legacy_args": argparse.Namespace(lr=_CHECKPOINT_VALUE),
                },
                resume,
            )
        else:
            resume.write_bytes(_write_checkpoint(tmp_path / "whole.ckpt", {"lr": _CHECKPOINT_VALUE}).read_bytes()[:64])
        config = TrainConfig(dataset_dir="data", resume=str(resume))
        assert _resolve_resumed_optimizer_settings(config) is config

    def test_training_config_written_before_fit_records_checkpoint_settings(self, tmp_path: Path) -> None:
        """A resumed run interrupted inside ``trainer.fit`` leaves only the start-of-run ``training_config.json``."""
        resume = _write_checkpoint(tmp_path / "last.ckpt", {"lr_encoder": _CHECKPOINT_VALUE})
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer") as build_trainer,
        ):
            build_trainer.return_value.fit.side_effect = KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu").train(
                    dataset_dir=str(tmp_path), output_dir=str(tmp_path / "output"), resume=str(resume), device="cpu"
                )
        recorded = json.loads((tmp_path / "output" / "training_config.json").read_text())["train_config"]
        assert recorded["lr_encoder"] == _CHECKPOINT_VALUE


class TestAdoptResumedParamGroupSettings:
    """``RFDETRModelModule.on_load_checkpoint`` resolves the same order from the checkpoint Lightning loads."""

    def test_unset_setting_takes_the_checkpoint_value(self) -> None:
        module = _adopt(TrainConfig(dataset_dir="data"), {"lr_encoder": _CHECKPOINT_VALUE})
        assert module.train_config.lr_encoder == _CHECKPOINT_VALUE

    @pytest.mark.parametrize(
        ("explicit", "args", "reapplies"),
        [
            pytest.param({}, {"lr": 5e-5}, False, id="unchanged-resume"),
            pytest.param({"lr": 5e-5}, {"lr": 5e-5}, True, id="explicit-equal-to-checkpoint"),
            pytest.param({"weight_decay": 2e-4}, {"weight_decay": 2e-4}, True, id="explicit-equal-weight-decay"),
            pytest.param({"lr": 1e-5}, {"lr": 5e-5}, True, id="explicit-change"),
            pytest.param({}, None, False, id="no-recorded-settings"),
            pytest.param({"lr": 1e-5}, None, True, id="explicit-without-recorded-settings"),
        ],
    )
    def test_reapplies_when_a_setting_is_passed_explicitly(
        self, explicit: dict[str, float], args: dict[str, float] | None, reapplies: bool
    ) -> None:
        """A resume that passes none continues exactly as saved.

        One that passes any is compared group by group, not with ``args``: a resume hit by #1613 recorded ``args`` its
        optimizer never trained with.
        """
        module = _adopt(TrainConfig(dataset_dir="data", **explicit), args)
        assert module._reapplies_param_group_settings is reapplies

    @pytest.mark.parametrize(
        ("explicit", "args", "reapplies"),
        [
            pytest.param({}, None, False, id="no-recorded-settings"),
            pytest.param({}, {"lr": 5e-5}, False, id="unchanged-resume"),
            pytest.param({"lr": 1e-5}, {"lr": 5e-5}, True, id="explicit-change"),
        ],
    )
    def test_config_built_from_a_complete_mapping_reapplies_only_what_differs_from_its_default(
        self, explicit: dict[str, float], args: dict[str, float] | None, reapplies: bool
    ) -> None:
        """``rfdetr fit --ckpt_path`` builds its config through LightningCLI, which marks every field as set."""
        config = TrainConfig(**TrainConfig(dataset_dir="data", **explicit).model_dump())
        assert _adopt(config, args)._reapplies_param_group_settings is reapplies

    @pytest.mark.parametrize(
        ("explicit", "args", "restarts"),
        [
            pytest.param({"lr": 1e-5}, {"lr": 5e-5}, True, id="lr-changed"),
            pytest.param({"lr_encoder": 1e-5}, {"lr_encoder": 5e-5}, True, id="lr-encoder-changed"),
            pytest.param({"lr": 5e-5}, {"lr": 5e-5}, False, id="lr-unchanged"),
            pytest.param({"weight_decay": 5e-4}, {"weight_decay": 1e-4}, False, id="weight-decay-changed"),
            pytest.param({"lr": 1e-5}, None, True, id="lr-without-recorded-settings"),
            pytest.param({"weight_decay": 5e-4}, None, False, id="weight-decay-without-recorded-settings"),
        ],
    )
    def test_restarts_groups_without_a_base_only_for_a_learning_rate_change(
        self, explicit: dict[str, float], args: dict[str, float] | None, restarts: bool
    ) -> None:
        """Restarting is the only way to apply a new lr to a group saved without ``initial_lr``, and it loses the
        reductions of an older ``ReduceLROnPlateau`` checkpoint."""
        module = _adopt(TrainConfig(dataset_dir="data", **explicit), args)
        assert module._restarts_groups_without_base is restarts

    def test_other_ranks_resolve_the_same_settings(self) -> None:
        """Every DDP rank must build the same optimizer; only the logging is limited to rank zero."""
        module = _adopt(TrainConfig(dataset_dir="data", lr=1e-5), {"lr_encoder": _CHECKPOINT_VALUE}, global_rank=1)
        assert (module.train_config.lr_encoder, module._reapplies_param_group_settings) == (_CHECKPOINT_VALUE, True)

    def test_logs_restored_and_overridden_settings(self, rf_detr_caplog: pytest.LogCaptureFixture) -> None:
        _adopt(TrainConfig(dataset_dir="data", lr=1e-5), {"lr": 5e-5, "lr_encoder": 3e-4})
        assert "lr_encoder=0.0003 from the checkpoint; lr=1e-05 (was 5e-05) set explicitly" in rf_detr_caplog.text

    def test_explicit_setting_equal_to_the_checkpoint_is_not_logged_as_set(
        self, rf_detr_caplog: pytest.LogCaptureFixture
    ) -> None:
        _adopt(TrainConfig(dataset_dir="data", lr=5e-5), {"lr": 5e-5, "lr_encoder": 3e-4})
        assert "set explicitly" not in rf_detr_caplog.text

    def test_logs_nothing_off_rank_zero(self, rf_detr_caplog: pytest.LogCaptureFixture) -> None:
        _adopt(TrainConfig(dataset_dir="data", lr=1e-5), {"lr": 5e-5, "lr_encoder": 3e-4}, global_rank=1)
        assert rf_detr_caplog.text == ""

    @pytest.mark.parametrize(
        ("explicit", "warns"),
        [
            pytest.param({"lr": 1e-5}, True, id="some-set"),
            pytest.param({}, False, id="none-set"),
            pytest.param(dict.fromkeys(_PARAM_GROUP_SETTINGS, 1e-5), False, id="all-set"),
        ],
    )
    def test_checkpoint_without_recorded_settings_warns_only_when_defaults_replace_it(
        self, rf_detr_caplog: pytest.LogCaptureFixture, explicit: dict[str, float], warns: bool
    ) -> None:
        """A ``last.ckpt`` written before 1.11.1 has optimizer state but no ``args``.

        Resumed with none set it keeps its saved values; with all set nothing falls back to a default.
        """
        _adopt(TrainConfig(dataset_dir="data", **explicit), None)
        assert any(record.levelno == logging.WARNING for record in rf_detr_caplog.records) is warns

    def test_warning_names_the_settings_that_take_their_defaults(
        self, rf_detr_caplog: pytest.LogCaptureFixture
    ) -> None:
        _adopt(TrainConfig(dataset_dir="data", lr=1e-5), None)
        assert (
            "lr_encoder, lr_vit_layer_decay, lr_component_decay, weight_decay use their defaults" in rf_detr_caplog.text
        )


def _restored_optimizer(
    restored_lrs: list[float], *, scheduler_kind: str, steps: int, restored_weight_decay: float = 1e-4
) -> tuple[torch.optim.Optimizer, Any]:
    """Build an AdamW and scheduler as Lightning leaves them after restoring a checkpoint trained at ``restored_lrs``.

    A first pair is configured at ``restored_lrs`` and stepped ``steps`` times. A second pair, configured at other
    learning rates as a resumed run would, then loads its state, which is what ``trainer.fit(ckpt_path=...)`` does.

    Args:
        restored_lrs: Base learning rate of each parameter group in the checkpoint.
        scheduler_kind: ``"lambda"``, ``"warmup"`` (``LinearLR`` then ``LambdaLR``) or ``"plateau"``.
        steps: Scheduler steps taken before the checkpoint was saved.
        restored_weight_decay: Weight decay recorded in the checkpoint.

    Returns:
        ``(optimizer, scheduler)`` holding the restored state.

    Examples:
        >>> optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        >>> [round(group["lr"], 4) for group in optimizer.param_groups], scheduler.base_lrs
        ([0.025, 0.05], [0.1, 0.2])
    """

    def build(lrs: list[float], weight_decay: float) -> tuple[torch.optim.Optimizer, Any]:
        """Configure one single-parameter group per learning rate, with the ``scheduler_kind`` scheduler."""
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in lrs]
        groups = [{"params": [param], "lr": lr} for param, lr in zip(params, lrs)]
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        if scheduler_kind == "plateau":
            scheduler: Any = ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        elif scheduler_kind == "warmup":
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=4)
            scheduler = SequentialLR(optimizer, [warmup, LambdaLR(optimizer, lambda _: 1.0)], milestones=[4])
        else:
            scheduler = LambdaLR(optimizer, lambda step: 0.5**step)
        return optimizer, scheduler

    saved_optimizer, saved_scheduler = build(restored_lrs, restored_weight_decay)
    for _ in range(steps):
        saved_optimizer.step()
        if scheduler_kind == "plateau":
            saved_scheduler.step(1.0)
        else:
            saved_scheduler.step()
    optimizer, scheduler = build([0.9 for _ in restored_lrs], 0.9)
    optimizer.load_state_dict(saved_optimizer.state_dict())
    scheduler.load_state_dict(saved_scheduler.state_dict())
    return optimizer, scheduler


class TestApplyConfiguredParamGroupSettings:
    """After Lightning restores a checkpoint, each group takes the configured settings and keeps its progress."""

    def test_configured_lr_becomes_the_schedule_base(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        assert scheduler.base_lrs == [0.01, 0.02]

    def test_configured_lr_becomes_the_group_initial_lr(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        assert [group["initial_lr"] for group in optimizer.param_groups] == [0.01, 0.02]

    def test_schedule_progress_carries_over(self) -> None:
        """Two halvings were taken before the checkpoint, so the new base starts at a quarter."""
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.0025, 0.005])

    def test_next_scheduler_step_follows_the_configured_base(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        optimizer.step()
        scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.00125, 0.0025])

    def test_nested_warmup_scheduler_bases_follow(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="warmup", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        assert [nested.base_lrs for nested in scheduler._schedulers] == [[0.01, 0.02], [0.01, 0.02]]

    def test_warmup_reaches_the_configured_lr(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="warmup", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        for _ in range(4):
            optimizer.step()
            scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.01, 0.02])

    def test_plateau_reductions_carry_over(self) -> None:
        """Two reductions by half were taken before the checkpoint."""
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="plateau", steps=3)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)])
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.0025, 0.005])

    @pytest.mark.parametrize(
        ("restart_without_base", "expected_lrs"),
        [
            pytest.param(True, [0.01, 0.02], id="lr-changed"),
            pytest.param(False, [0.025, 0.05], id="lr-unchanged"),
        ],
    )
    def test_group_without_a_recorded_base_restarts_only_when_asked(
        self, restart_without_base: bool, expected_lrs: list[float]
    ) -> None:
        """``ReduceLROnPlateau`` checkpoints written before this fix record no ``initial_lr`` to scale from; two
        reductions by half were taken."""
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="plateau", steps=3)
        for group in optimizer.param_groups:
            del group["initial_lr"]
        _apply_configured_param_group_settings(
            optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)], restart_without_base=restart_without_base
        )
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected_lrs)

    def test_group_without_a_recorded_base_is_reported(self, rf_detr_caplog: pytest.LogCaptureFixture) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="plateau", steps=3)
        for group in optimizer.param_groups:
            del group["initial_lr"]
        _apply_configured_param_group_settings(
            optimizer, scheduler, [(0.01, 1e-4), (0.02, 1e-4)], restart_without_base=True
        )
        assert (
            "2 resumed parameter groups were saved without the learning rate they started from" in rf_detr_caplog.text
        )

    def test_zero_restored_base_takes_the_configured_lr(self) -> None:
        """No schedule progress can be read off a group that trained at zero, so it starts at its configured lr."""
        optimizer, scheduler = _restored_optimizer([0.1, 0.0], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.1, 1e-4), (0.02, 1e-4)])
        assert optimizer.param_groups[1]["lr"] == 0.02

    def test_configured_weight_decay_replaces_the_restored_one(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        _apply_configured_param_group_settings(optimizer, scheduler, [(0.1, 5e-4), (0.2, 0.0)])
        assert [group["weight_decay"] for group in optimizer.param_groups] == [5e-4, 0.0]

    def test_unchanged_settings_leave_every_group_alone(self) -> None:
        optimizer, scheduler = _restored_optimizer([0.1, 0.2], scheduler_kind="lambda", steps=2)
        assert _apply_configured_param_group_settings(optimizer, scheduler, [(0.1, 1e-4), (0.2, 1e-4)]) == 0


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic COCO detection dataset.

    Examples:
        >>> dataset_dir  # doctest: +SKIP
        A module-scoped pytest fixture; it needs tmp_path_factory.
    """
    path = tmp_path_factory.mktemp("resume_optimizer_settings_dataset")
    build_synthetic_dataset(path, task="detection", num_images=16)
    return path


@pytest.fixture(scope="module")
def training_runs(dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train one epoch at :data:`_FIRST_RUN`, then resume it for a second with :data:`_RESUMED_RUN` set explicitly.

    Returns:
        The output directories of the first and the resumed run.

    Examples:
        >>> training_runs  # doctest: +SKIP
        A module-scoped pytest fixture that trains a model twice.
    """
    seed_all(_TRAIN_SEED)
    first_output_dir = tmp_path_factory.mktemp("resume_optimizer_settings_first")
    RFDETRNano(pretrain_weights=None, resolution=224, device="cpu").train(
        dataset_dir=str(dataset_dir), output_dir=str(first_output_dir), epochs=1, **_FIRST_RUN, **_TRAIN_KWARGS
    )
    resumed_output_dir = tmp_path_factory.mktemp("resume_optimizer_settings_resumed")
    RFDETRNano(pretrain_weights=None, resolution=224, device="cpu").train(
        dataset_dir=str(dataset_dir),
        output_dir=str(resumed_output_dir),
        epochs=2,
        resume=str(first_output_dir / "last.ckpt"),
        **_RESUMED_RUN,
        **_TRAIN_KWARGS,
    )
    return first_output_dir, resumed_output_dir


@pytest.fixture(scope="module")
def stale_args_resumed_ckpt(
    dataset_dir: Path, training_runs: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Resume a first-run ``last.ckpt`` whose ``args.lr`` says :data:`_RESUMED_RUN`, passing that ``lr`` again.

    A resume on rfdetr 1.11.1 or 1.11.2 with a changed ``lr`` wrote such a file: ``args`` recorded the new ``lr`` while
    its optimizer kept training at the old one (#1613). Its author then resumes with the ``lr`` they asked for.

    Returns:
        The ``last.ckpt`` of that resumed run.

    Examples:
        >>> stale_args_resumed_ckpt  # doctest: +SKIP
        A module-scoped pytest fixture that trains a model.
    """
    seed_all(_TRAIN_SEED)
    stale_dir = tmp_path_factory.mktemp("resume_optimizer_settings_stale_args")
    checkpoint = torch.load(training_runs[0] / "last.ckpt", map_location="cpu", weights_only=True)
    checkpoint["args"]["lr"] = _RESUMED_RUN["lr"]
    torch.save(checkpoint, stale_dir / "stale.ckpt")
    RFDETRNano(pretrain_weights=None, resolution=224, device="cpu").train(
        dataset_dir=str(dataset_dir),
        output_dir=str(stale_dir),
        epochs=2,
        resume=str(stale_dir / "stale.ckpt"),
        lr=_RESUMED_RUN["lr"],
        **_TRAIN_KWARGS,
    )
    return torch.load(stale_dir / "last.ckpt", map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def last_ckpts(training_runs: tuple[Path, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``last.ckpt`` of the first run and of the resumed run.

    Examples:
        >>> last_ckpts  # doctest: +SKIP
        A module-scoped pytest fixture over the training runs.
    """
    return tuple(  # type: ignore[return-value]
        torch.load(output_dir / "last.ckpt", map_location="cpu", weights_only=True) for output_dir in training_runs
    )


@pytest.fixture(scope="module")
def resumed_training_config(training_runs: tuple[Path, Path]) -> dict[str, Any]:
    """The ``train_config`` section of the resumed run's ``training_config.json``.

    Examples:
        >>> resumed_training_config  # doctest: +SKIP
        A module-scoped pytest fixture over the training runs.
    """
    return json.loads((training_runs[1] / "training_config.json").read_text())["train_config"]


class TestResumeWithChangedSettings:
    """The issue's scenario: stop a run, change ``lr``, resume from ``last.ckpt``."""

    def test_schedule_follows_the_explicit_lr_and_the_restored_lr_encoder(
        self, last_ckpts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Groups trained at ``lr`` (and the decoder's ``lr * lr_component_decay``) move to the new ``lr``; the
        backbone's, built from the unchanged ``lr_encoder``, stay where they were."""
        first, resumed = last_ckpts
        component_decay = TrainConfig.model_fields["lr_component_decay"].default
        new_base = {
            _FIRST_RUN["lr"]: _RESUMED_RUN["lr"],
            _FIRST_RUN["lr"] * component_decay: _RESUMED_RUN["lr"] * component_decay,
        }
        expected = [new_base.get(base, base) for base in first["lr_schedulers"][0]["base_lrs"]]
        assert resumed["lr_schedulers"][0]["base_lrs"] == expected

    def test_explicit_weight_decay_applies_where_the_group_decays(
        self, last_ckpts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Parameters exempt from weight decay (biases, norms) keep zero."""
        first, resumed = last_ckpts
        expected = [
            0.0 if group["weight_decay"] == 0.0 else _RESUMED_RUN["weight_decay"]
            for group in first["optimizer_states"][0]["param_groups"]
        ]
        assert [group["weight_decay"] for group in resumed["optimizer_states"][0]["param_groups"]] == expected

    def test_explicit_lr_applies_over_stale_recorded_args(self, stale_args_resumed_ckpt: dict[str, Any]) -> None:
        """The issue's own checkpoint, after the upgrade: ``args`` already say the new ``lr``."""
        assert stale_args_resumed_ckpt["lr_schedulers"][0]["base_lrs"][0] == _RESUMED_RUN["lr"]

    @pytest.mark.parametrize("setting", ["lr", "lr_encoder", "weight_decay"])
    def test_training_config_records_what_ran(self, resumed_training_config: dict[str, Any], setting: str) -> None:
        assert resumed_training_config[setting] == {**_FIRST_RUN, **_RESUMED_RUN}[setting]

    @pytest.mark.parametrize("setting", ["lr", "lr_encoder", "weight_decay"])
    def test_checkpoint_args_record_what_ran(
        self, last_ckpts: tuple[dict[str, Any], dict[str, Any]], setting: str
    ) -> None:
        assert last_ckpts[1]["args"][setting] == {**_FIRST_RUN, **_RESUMED_RUN}[setting]
