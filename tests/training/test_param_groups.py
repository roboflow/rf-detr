# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for optimizer parameter grouping and its pre-merge checkpoint migration."""

import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models import build_model
from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import _build_param_dicts, get_param_dict, regroup_unmerged_optimizer_state


@pytest.fixture
def nano_model_and_args(monkeypatch: pytest.MonkeyPatch) -> tuple[LWDETR, Any]:
    """Build an RF-DETR Nano LWDETR and the argument namespace the grouping helpers read.

    Pretrained weights are skipped so the fixture needs no network.

    Examples:
        >>> nano_model_and_args(pytest.MonkeyPatch())  # doctest: +SKIP
    """
    monkeypatch.setattr(
        WindowedDinov2WithRegistersBackbone,
        "from_pretrained",
        classmethod(lambda cls, name, config: cls(config)),
    )
    args = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=3, pretrain_weights=None),
        TrainConfig(dataset_dir=".", output_dir="."),
    )
    return build_model(args), args


def build_adamw(param_groups: list[dict[str, Any]], args: Any) -> torch.optim.AdamW:
    """Build AdamW over `param_groups` with the run's base learning rate and weight decay.

    Examples:
        >>> from types import SimpleNamespace
        >>> parameter = torch.nn.Parameter(torch.zeros(2))
        >>> optimizer = build_adamw([{"params": parameter}], SimpleNamespace(lr=0.1, weight_decay=0.0))
        >>> len(optimizer.param_groups)
        1
    """
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def step_once(optimizer: torch.optim.Optimizer, model: nn.Module, seed: int) -> None:
    """Take one optimizer step on reproducible pseudo-gradients, so state tensors exist to regroup.

    Examples:
        >>> model = nn.Linear(2, 2)
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>> step_once(optimizer, model, seed=0)
        >>> "exp_avg" in optimizer.state[model.weight]
        True
    """
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn(parameter.shape, generator=generator)
    optimizer.step()


def hyperparameters(param_group: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return a parameter group's overrides, without its parameters, as a comparable key.

    Examples:
        >>> hyperparameters({"params": [], "lr": 0.1, "weight_decay": 0.0})
        (('lr', 0.1), ('weight_decay', 0.0))
    """
    return tuple(sorted((key, value) for key, value in param_group.items() if key != "params"))


@pytest.fixture
def nano_module(nano_model_and_args: tuple[LWDETR, Any], tmp_path: Path) -> RFDETRModelModule:
    """Build the training module around the already-built Nano model, ready for `configure_optimizers`.

    Examples:
        >>> nano_module((None, None), Path("."))  # doctest: +SKIP
    """
    model, _ = nano_model_and_args
    model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None)
    train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=model),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ),
    ):
        module = RFDETRModelModule(model_config, train_config)
    trainer = MagicMock()
    trainer.estimated_stepping_batches = 1000
    module._trainer = trainer
    type(module).trainer = property(lambda self: self._trainer)
    return module


class TestGetParamDict:
    """`get_param_dict` groups parameters by hyperparameters without changing any parameter's own."""

    def test_every_trainable_parameter_appears_exactly_once(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """No parameter is dropped or duplicated by the merge."""
        model, args = nano_model_and_args

        grouped = [id(p) for group in get_param_dict(args, model) for p in group["params"]]

        assert sorted(grouped) == sorted(id(p) for p in model.parameters() if p.requires_grad)

    def test_one_group_per_distinct_hyperparameter_combination(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Groups correspond to distinct hyperparameter combinations, not to parameters."""
        model, args = nano_model_and_args

        groups = get_param_dict(args, model)

        assert len(groups) == len({hyperparameters(g) for g in _build_param_dicts(args, model)})

    def test_merge_collapses_groups(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Fewer groups is the point: it is what keeps `torch.optim`'s multi-tensor kernels batched."""
        model, args = nano_model_and_args

        assert len(get_param_dict(args, model)) < len(_build_param_dicts(args, model))

    def test_each_parameter_keeps_its_own_hyperparameters(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Layer-wise LR decay and backbone weight decay survive the merge, parameter by parameter."""
        model, args = nano_model_and_args

        merged = {id(p): hyperparameters(group) for group in get_param_dict(args, model) for p in group["params"]}

        assert merged == {id(g["params"]): hyperparameters(g) for g in _build_param_dicts(args, model)}

    def test_frozen_parameters_are_excluded(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """`configure_optimizers` no longer filters the groups, so the grouping itself must drop frozen tensors."""
        model, args = nano_model_and_args
        frozen = model.transformer.decoder
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

        grouped = {id(p) for group in get_param_dict(args, model) for p in group["params"]}

        assert grouped.isdisjoint({id(p) for p in frozen.parameters()})

    def test_no_group_is_empty(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """An empty group would reach the optimizer as a no-op group; buckets are built from real parameters."""
        model, args = nano_model_and_args

        assert all(group["params"] for group in get_param_dict(args, model))


class TestConfigureOptimizersUsesMergedGroups:
    """The production optimizer is built from the merged groups, not from a mocked stand-in."""

    def test_optimizer_group_count_matches_the_merged_grouping(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """`configure_optimizers` passes the merged groups straight through to AdamW."""
        model, args = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        assert len(optimizer.param_groups) == len(get_param_dict(args, model))

    def test_optimizer_covers_every_trainable_parameter_once(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """Dropping `configure_optimizers`' own requires_grad filter must not change what reaches the optimizer."""
        model, _ = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert sorted(owned) == sorted(id(p) for p in model.parameters() if p.requires_grad)

    def test_each_parameter_keeps_its_layerwise_learning_rate(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """Backbone layer-wise LR decay must survive the merge through the real optimizer."""
        model, args = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        assigned = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
        assert assigned == {id(group["params"]): group["lr"] for group in _build_param_dicts(args, model)}


class TestUnmergedOptimizerStateMigration:
    """A checkpoint written with one parameter group per parameter still resumes."""

    def test_saved_state_loads_into_the_merged_optimizer(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """`Optimizer.load_state_dict` rejects a differing group count, so the migration must run first."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        step_once(unmerged, model, seed=0)
        checkpoint = {"optimizer_states": [copy.deepcopy(unmerged.state_dict())]}

        regroup_unmerged_optimizer_state(checkpoint)

        merged = build_adamw(get_param_dict(args, model), args)
        merged.load_state_dict(checkpoint["optimizer_states"][0])

        assert len(merged.param_groups) == len(get_param_dict(args, model))

    def test_migrated_state_stays_with_its_own_parameter(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Reindexing must not hand one parameter's momentum to another."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        step_once(unmerged, model, seed=0)
        saved = {id(p): unmerged.state[p]["exp_avg"].clone() for g in unmerged.param_groups for p in g["params"]}
        checkpoint = {"optimizer_states": [copy.deepcopy(unmerged.state_dict())]}

        regroup_unmerged_optimizer_state(checkpoint)

        merged = build_adamw(get_param_dict(args, model), args)
        merged.load_state_dict(checkpoint["optimizer_states"][0])

        assert all(
            torch.equal(merged.state[p]["exp_avg"], saved[id(p)]) for g in merged.param_groups for p in g["params"]
        )

    def test_scheduler_lists_collapse_with_the_groups(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Per-group scheduler lists are positional too, so they must shrink to the merged group count."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(unmerged, lr_lambda=lambda step: 0.5)
        checkpoint = {
            "optimizer_states": [copy.deepcopy(unmerged.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }

        regroup_unmerged_optimizer_state(checkpoint)

        assert len(checkpoint["lr_schedulers"][0]["base_lrs"]) == len(get_param_dict(args, model))

    def test_scheduler_keeps_each_merged_group_base_lr(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """The collapsed `base_lrs` must be the learning rates the merged groups were built with."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(unmerged, lr_lambda=lambda step: 0.5)
        checkpoint = {
            "optimizer_states": [copy.deepcopy(unmerged.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint["lr_schedulers"][0]["base_lrs"] == [g["lr"] for g in get_param_dict(args, model)]

    def test_wrapped_scheduler_lists_collapse_too(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """`_wrap_with_warmup`'s `SequentialLR` nests its wrapped scheduler's own per-group lists one level deeper
        (`_schedulers`), so the collapse must recurse into that nesting instead of only handling the `SequentialLR`
        object's own top-level lists."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        warmup = torch.optim.lr_scheduler.LinearLR(unmerged, start_factor=0.1, end_factor=1.0, total_iters=2)
        main = torch.optim.lr_scheduler.LambdaLR(unmerged, lr_lambda=lambda step: 0.5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(unmerged, schedulers=[warmup, main], milestones=[2])
        checkpoint = {
            "optimizer_states": [copy.deepcopy(unmerged.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }

        regroup_unmerged_optimizer_state(checkpoint)

        merged_group_count = len(get_param_dict(args, model))
        for wrapped_state in checkpoint["lr_schedulers"][0]["_schedulers"]:
            assert len(wrapped_state["base_lrs"]) == merged_group_count

    def test_wrapped_scheduler_activates_after_migration(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """The migrated state must let the wrapped scheduler take over at its milestone, not just load."""
        model, args = nano_model_and_args
        unmerged = build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)
        warmup = torch.optim.lr_scheduler.LinearLR(unmerged, start_factor=0.1, end_factor=1.0, total_iters=2)
        main = torch.optim.lr_scheduler.LambdaLR(unmerged, lr_lambda=lambda step: 0.5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(unmerged, schedulers=[warmup, main], milestones=[2])
        checkpoint = {
            "optimizer_states": [copy.deepcopy(unmerged.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }

        regroup_unmerged_optimizer_state(checkpoint)

        merged = build_adamw(get_param_dict(args, model), args)
        merged.load_state_dict(checkpoint["optimizer_states"][0])
        merged_warmup = torch.optim.lr_scheduler.LinearLR(merged, start_factor=0.1, end_factor=1.0, total_iters=2)
        merged_main = torch.optim.lr_scheduler.LambdaLR(merged, lr_lambda=lambda step: 0.5)
        merged_scheduler = torch.optim.lr_scheduler.SequentialLR(
            merged, schedulers=[merged_warmup, merged_main], milestones=[2]
        )
        merged_scheduler.load_state_dict(checkpoint["lr_schedulers"][0])

        merged_scheduler.step()
        merged_scheduler.step()  # crosses the milestone into `main`; raised before the recursive collapse.

    def test_already_merged_state_is_left_alone(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """A checkpoint written after the merge must pass through untouched."""
        model, args = nano_model_and_args
        merged = build_adamw(get_param_dict(args, model), args)
        step_once(merged, model, seed=0)
        checkpoint = {"optimizer_states": [copy.deepcopy(merged.state_dict())]}
        before = copy.deepcopy(checkpoint)

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint["optimizer_states"][0]["param_groups"] == before["optimizer_states"][0]["param_groups"]

    def test_unrecognised_optimizer_state_is_left_alone(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """State that is neither layout (e.g. a custom optimizer's) must pass through untouched."""
        model, args = nano_model_and_args
        checkpoint: dict[str, Any] = {"optimizer_states": [{"state": {}, "param_groups": []}]}
        before = copy.deepcopy(checkpoint)

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint == before
