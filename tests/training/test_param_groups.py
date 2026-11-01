# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for optimizer parameter grouping and its pre-merge checkpoint migration."""

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models import build_model
from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import (
    _build_param_dicts,
    get_param_dict,
    regroup_unmerged_optimizer_state,
    regroup_unmerged_scheduler_kwargs,
)


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


def constant_lr_lambda(step: int) -> float:
    """Return a constant schedule factor for explicit scheduler migration tests.

    Examples:
        >>> constant_lr_lambda(3)
        1.0
    """
    return 1.0


@pytest.fixture
def nano_module(
    nano_model_and_args: tuple[LWDETR, Any], tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> RFDETRModelModule:
    """Build the training module around the already-built Nano model, ready for `configure_optimizers`.

    Examples:
        >>> nano_module((None, None), Path("."), None, pytest.MonkeyPatch())  # doctest: +SKIP
    """
    model, _ = nano_model_and_args
    model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None)
    train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
    mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=model)
    mocker.patch(
        "rfdetr.training.module_model.build_criterion_from_config", return_value=(mocker.Mock(), mocker.Mock())
    )
    module = RFDETRModelModule(model_config, train_config)
    trainer = mocker.Mock()
    trainer.estimated_stepping_batches = 1000
    module._trainer = trainer
    monkeypatch.setattr(type(module), "trainer", property(lambda self: self._trainer))
    return module


@pytest.fixture
def unmerged_adamw(nano_model_and_args: tuple[LWDETR, Any]) -> torch.optim.AdamW:
    """Build AdamW with one parameter group per parameter, the layout of a pre-merge checkpoint.

    Examples:
        >>> unmerged_adamw((None, None))  # doctest: +SKIP
    """
    model, args = nano_model_and_args
    return build_adamw([dict(group) for group in _build_param_dicts(args, model)], args)


def checkpoint_of(optimizer: torch.optim.Optimizer, scheduler: Any = None) -> dict[str, Any]:
    """Return a Lightning-style checkpoint dict holding deep copies of the optimizer (and scheduler) state.

    Examples:
        >>> optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))])
        >>> list(checkpoint_of(optimizer))
        ['optimizer_states']
    """
    checkpoint = {"optimizer_states": [copy.deepcopy(optimizer.state_dict())]}
    if scheduler is not None:
        checkpoint["lr_schedulers"] = [copy.deepcopy(scheduler.state_dict())]
    return checkpoint


class TestGetParamDict:
    """`get_param_dict` groups parameters by hyperparameters without changing any parameter's own."""

    def test_every_trainable_parameter_appears_once(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """No parameter is dropped or duplicated by the merge."""
        model, args = nano_model_and_args

        grouped = [id(p) for group in get_param_dict(args, model) for p in group["params"]]

        assert sorted(grouped) == sorted(id(p) for p in model.parameters() if p.requires_grad)

    def test_one_group_per_hyperparameter_combination(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Groups correspond to distinct hyperparameter combinations, not to parameters."""
        model, args = nano_model_and_args

        groups = get_param_dict(args, model)

        assert len(groups) == len({hyperparameters(g) for g in _build_param_dicts(args, model)})

    def test_merge_collapses_groups(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Fewer groups is the point: it is what keeps `torch.optim`'s multi-tensor kernels batched."""
        model, args = nano_model_and_args

        assert len(get_param_dict(args, model)) < len(_build_param_dicts(args, model))

    def test_parameter_keeps_own_hyperparameters(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """Layer-wise LR decay and backbone weight decay survive the merge, parameter by parameter."""
        model, args = nano_model_and_args

        merged = {id(p): hyperparameters(group) for group in get_param_dict(args, model) for p in group["params"]}

        assert merged == {id(g["params"]): hyperparameters(g) for g in _build_param_dicts(args, model)}

    def test_frozen_parameters_excluded(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """`configure_optimizers` no longer filters the groups, so the grouping itself must drop frozen tensors."""
        model, args = nano_model_and_args
        frozen = model.transformer.decoder
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

        grouped = {id(p) for group in get_param_dict(args, model) for p in group["params"]}

        assert grouped.isdisjoint({id(p) for p in frozen.parameters()})

    def test_no_empty_group(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """An empty group would reach the optimizer as a no-op group; buckets are built from real parameters."""
        model, args = nano_model_and_args

        assert all(group["params"] for group in get_param_dict(args, model))


class TestLegacySchedulerKwargMigration:
    """Migrate scheduler constructor kwargs that followed the pre-merge group topology."""

    def test_lambda_callbacks_collapse_from_two_legacy_groups(self) -> None:
        """An explicit LambdaLR builds after two equivalent legacy callbacks collapse to one group."""
        callback = constant_lr_lambda
        parameters = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.ones(1))]
        legacy_groups = [{"params": parameter, "lr": 0.1} for parameter in parameters]
        scheduler_kwargs = {"lr_lambda": [callback, callback]}

        migrated_kwargs = regroup_unmerged_scheduler_kwargs(scheduler_kwargs, legacy_groups)
        optimizer = torch.optim.AdamW([{"params": parameters, "lr": 0.1}])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **migrated_kwargs)

        assert scheduler.lr_lambdas == [callback]
        assert scheduler_kwargs == {"lr_lambda": [callback, callback]}


class TestConfigureOptimizersUsesMergedGroups:
    """The production optimizer is built from the merged groups, not from a mocked stand-in."""

    def test_optimizer_group_count_matches_merged_grouping(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """`configure_optimizers` passes the merged groups straight through to AdamW."""
        model, args = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        assert len(optimizer.param_groups) == len(get_param_dict(args, model))

    def test_optimizer_covers_each_trainable_parameter_once(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """Dropping `configure_optimizers`' own requires_grad filter must not change what reaches the optimizer."""
        model, _ = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert sorted(owned) == sorted(id(p) for p in model.parameters() if p.requires_grad)

    def test_parameter_keeps_layerwise_learning_rate(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """Backbone layer-wise LR decay must survive the merge through the real optimizer."""
        model, args = nano_model_and_args

        optimizer = nano_module.configure_optimizers()["optimizer"]

        assigned = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
        assert assigned == {id(group["params"]): group["lr"] for group in _build_param_dicts(args, model)}

    def test_lambda_scheduler_migrates_legacy_callback_list(
        self, nano_module: RFDETRModelModule, nano_model_and_args: tuple[LWDETR, Any]
    ) -> None:
        """The scheduler constructor receives callbacks regrouped for the merged optimizer topology."""
        model, args = nano_model_and_args
        callback = constant_lr_lambda
        nano_module.train_config = nano_module.train_config.model_copy(
            update={
                "lr_scheduler": "torch.optim.lr_scheduler.LambdaLR",
                "lr_scheduler_kwargs": {"lr_lambda": [callback] * len(_build_param_dicts(args, model))},
            }
        )

        scheduler = nano_module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert scheduler.lr_lambdas == [callback] * len(get_param_dict(args, model))


class TestUnmergedOptimizerStateMigration:
    """A checkpoint written with one parameter group per parameter still resumes."""

    def test_saved_state_loads_into_merged_optimizer(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """`Optimizer.load_state_dict` rejects a differing group count, so the migration must run first."""
        model, args = nano_model_and_args
        step_once(unmerged_adamw, model, seed=0)
        checkpoint = checkpoint_of(unmerged_adamw)

        regroup_unmerged_optimizer_state(checkpoint)

        merged = build_adamw(get_param_dict(args, model), args)
        merged.load_state_dict(checkpoint["optimizer_states"][0])

        assert len(merged.param_groups) == len(get_param_dict(args, model))

    def test_migrated_state_stays_with_own_parameter(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """Reindexing must not hand one parameter's momentum to another."""
        model, args = nano_model_and_args
        step_once(unmerged_adamw, model, seed=0)
        saved = {
            id(p): unmerged_adamw.state[p]["exp_avg"].clone() for g in unmerged_adamw.param_groups for p in g["params"]
        }
        checkpoint = checkpoint_of(unmerged_adamw)

        regroup_unmerged_optimizer_state(checkpoint)

        merged = build_adamw(get_param_dict(args, model), args)
        merged.load_state_dict(checkpoint["optimizer_states"][0])

        assert all(
            torch.equal(merged.state[p]["exp_avg"], saved[id(p)]) for g in merged.param_groups for p in g["params"]
        )

    def test_scheduler_lists_collapse_with_groups(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """Per-group scheduler lists are positional too, so they must shrink to the merged group count."""
        model, args = nano_model_and_args
        scheduler = torch.optim.lr_scheduler.LambdaLR(unmerged_adamw, lr_lambda=lambda step: 0.5)
        checkpoint = checkpoint_of(unmerged_adamw, scheduler)

        regroup_unmerged_optimizer_state(checkpoint)

        assert len(checkpoint["lr_schedulers"][0]["base_lrs"]) == len(get_param_dict(args, model))

    def test_scheduler_keeps_merged_group_base_lr(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """The collapsed `base_lrs` must be the learning rates the merged groups were built with."""
        model, args = nano_model_and_args
        scheduler = torch.optim.lr_scheduler.LambdaLR(unmerged_adamw, lr_lambda=lambda step: 0.5)
        checkpoint = checkpoint_of(unmerged_adamw, scheduler)

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint["lr_schedulers"][0]["base_lrs"] == [g["lr"] for g in get_param_dict(args, model)]

    def test_wrapped_scheduler_lists_collapse_too(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """The collapse recurses into `SequentialLR._schedulers`, not only its own top-level per-group lists.

        `_wrap_with_warmup`'s `SequentialLR` nests the wrapped schedulers' per-group lists one level deeper.
        """
        model, args = nano_model_and_args
        warmup = torch.optim.lr_scheduler.LinearLR(unmerged_adamw, start_factor=0.1, end_factor=1.0, total_iters=2)
        main = torch.optim.lr_scheduler.LambdaLR(unmerged_adamw, lr_lambda=lambda step: 0.5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(unmerged_adamw, schedulers=[warmup, main], milestones=[2])
        checkpoint = checkpoint_of(unmerged_adamw, scheduler)

        regroup_unmerged_optimizer_state(checkpoint)

        merged_group_count = len(get_param_dict(args, model))
        for wrapped_state in checkpoint["lr_schedulers"][0]["_schedulers"]:
            assert len(wrapped_state["base_lrs"]) == merged_group_count

    def test_wrapped_scheduler_keeps_children_when_count_matches_groups(self) -> None:
        """SequentialLR recurses into its children instead of treating them as per-group values."""
        parameters = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.ones(1))]
        unmerged = torch.optim.AdamW([{"params": parameter, "lr": 0.1} for parameter in parameters])
        warmup = torch.optim.lr_scheduler.LinearLR(unmerged, start_factor=0.1, end_factor=1.0, total_iters=2)
        main = torch.optim.lr_scheduler.LambdaLR(unmerged, lr_lambda=constant_lr_lambda)
        scheduler = torch.optim.lr_scheduler.SequentialLR(unmerged, schedulers=[warmup, main], milestones=[2])
        checkpoint = checkpoint_of(unmerged, scheduler)

        regroup_unmerged_optimizer_state(checkpoint)

        wrapped_states = checkpoint["lr_schedulers"][0]["_schedulers"]
        assert len(wrapped_states) == 2
        assert all(len(wrapped_state["base_lrs"]) == 1 for wrapped_state in wrapped_states)

    def test_wrapped_scheduler_activates_after_migration(
        self, nano_model_and_args: tuple[LWDETR, Any], unmerged_adamw: torch.optim.AdamW
    ) -> None:
        """The migrated state must let the wrapped scheduler take over at its milestone, not just load."""
        model, args = nano_model_and_args
        warmup = torch.optim.lr_scheduler.LinearLR(unmerged_adamw, start_factor=0.1, end_factor=1.0, total_iters=2)
        main = torch.optim.lr_scheduler.LambdaLR(unmerged_adamw, lr_lambda=lambda step: 0.5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(unmerged_adamw, schedulers=[warmup, main], milestones=[2])
        checkpoint = checkpoint_of(unmerged_adamw, scheduler)

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

    def test_merged_state_left_alone(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """A checkpoint written after the merge must pass through untouched."""
        model, args = nano_model_and_args
        merged = build_adamw(get_param_dict(args, model), args)
        step_once(merged, model, seed=0)
        checkpoint = checkpoint_of(merged)
        before = copy.deepcopy(checkpoint)

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint["optimizer_states"][0]["param_groups"] == before["optimizer_states"][0]["param_groups"]

    def test_unrecognised_state_left_alone(self, nano_model_and_args: tuple[LWDETR, Any]) -> None:
        """State that is neither layout (e.g. a custom optimizer's) must pass through untouched."""
        model, args = nano_model_and_args
        checkpoint: dict[str, Any] = {"optimizer_states": [{"state": {}, "param_groups": []}]}
        before = copy.deepcopy(checkpoint)

        regroup_unmerged_optimizer_state(checkpoint)

        assert checkpoint == before
