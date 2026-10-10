# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Functions to get params dict."""

from collections.abc import Mapping
from typing import Any, cast

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from rfdetr.config import TrainConfig
from rfdetr.models.backbone import Joiner
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: ``TrainConfig`` settings that set a parameter group's learning rate or weight decay, read by
#: :func:`_build_param_dicts` and the backbone's ``get_named_param_lr_pairs``.
_PARAM_GROUP_SETTINGS = ("lr", "lr_encoder", "lr_vit_layer_decay", "lr_component_decay", "weight_decay")


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: parameter name.
        lr_decay_rate: base lr decay rate.
        num_layers: number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    # NOTE: near-duplicate of get_dinov2_lr_decay_rate in models/backbone/backbone.py (same formula,
    # different layer-key pattern: this matches ".blocks.", that matches ".layer.").
    # If updating this formula, update the sibling too.
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    logger.debug(f"name: {name}, lr_decay: {lr_decay_rate ** (num_layers + 1 - layer_id)}")
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Calculate weight decay rate for different ViT parameters.

    Args:
        name: parameter name.
        weight_decay_rate: base weight decay rate.

    Returns:
        weight decay rate for the given parameter.
    """
    if ("gamma" in name) or ("pos_embed" in name) or ("rel_pos" in name) or ("bias" in name) or ("norm" in name):
        weight_decay_rate = 0.0
    logger.debug(f"name: {name}, weight_decay rate: {weight_decay_rate}")
    return weight_decay_rate


def _hyperparameter_key(param_group: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the hyperparameter overrides of ``param_group``, without its parameters.

    Values are compared by ``repr`` so a group carrying an unhashable setting (a third-party
    optimizer's list-valued option, say) still yields a key. Distinct floats keep distinct ``repr``,
    and anything without a value-based ``repr`` lands in its own bucket, which under-merges rather
    than merging two differently configured parameters.

    Args:
        param_group: A single optimizer parameter group.

    Returns:
        The group's non-``params`` items, sorted by key so groups configured identically compare equal.
    """
    return tuple(sorted((key, repr(value)) for key, value in param_group.items() if key != "params"))


def _merge_buckets(param_dicts: list[dict[str, Any]]) -> list[list[int]]:
    """Bucket ``param_dicts`` indices by hyperparameter overrides, in first-appearance order.

    Args:
        param_dicts: Single-parameter groups as built by :func:`_build_param_dicts`.

    Returns:
        One list of ``param_dicts`` indices per distinct hyperparameter combination.
    """
    buckets: dict[tuple[tuple[str, str], ...], list[int]] = {}
    ordered: list[list[int]] = []
    for index, param_group in enumerate(param_dicts):
        key = _hyperparameter_key(param_group)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = []
            ordered.append(bucket)
        bucket.append(index)
    return ordered


def _build_param_dicts(args: Any, model_without_ddp: nn.Module) -> list[dict[str, Any]]:
    """Build one single-parameter group per trainable parameter, with its LR/weight-decay overrides.

    Args:
        args: Namespace supplying the learning-rate and weight-decay knobs.
        model_without_ddp: The model whose parameters the optimizer will own.

    Returns:
        One group per trainable parameter, ordered head/neck parameters first, then backbone, then
        decoder. :func:`get_param_dict` merges these; the order is also the parameter order of
        checkpoints written before that merge.
    """
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = cast("Any", model_without_ddp.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [p for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad]

    decoder_param_lr_pairs = [{"params": param, "lr": args.lr * args.lr_component_decay} for param in decoder_params]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters()
        if (n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [{"params": param, "lr": args.lr} for param in other_params]

    final_param_dicts = other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs

    return final_param_dicts


def get_param_dict(args: Any, model_without_ddp: nn.Module) -> list[dict[str, Any]]:
    """Build optimizer parameter groups with layer-wise LR (and backbone weight-decay) overrides.

    Parameters that end up configured identically share one group: ``torch.optim``'s foreach and
    fused kernels batch a group's parameters into a single multi-tensor launch, so one group per
    parameter would run ~500 single-tensor launches (and ~500 Python iterations of the optimizer's
    per-group loop) per step instead of one launch per distinct configuration.

    Args:
        args: Namespace supplying ``lr``, ``lr_encoder``, ``lr_component_decay``,
            ``lr_vit_layer_decay``, ``weight_decay``, and ``out_feature_indexes``.
        model_without_ddp: The model whose parameters the optimizer will own.

    Returns:
        Optimizer parameter groups, each holding every trainable parameter that shares its
        hyperparameter overrides.
    """
    param_dicts = _build_param_dicts(args, model_without_ddp)
    return [
        {
            **{key: value for key, value in param_dicts[bucket[0]].items() if key != "params"},
            "params": [param_dicts[index]["params"] for index in bucket],
        }
        for bucket in _merge_buckets(param_dicts)
    ]


def regroup_unmerged_scheduler_kwargs(
    scheduler_kwargs: dict[str, Any], unmerged_param_groups: list[dict[str, Any]]
) -> dict[str, Any]:
    """Collapse legacy ``LambdaLR`` callbacks onto the merged optimizer groups.

    Legacy RF-DETR versions created one optimizer group per parameter. A dotted
    ``LambdaLR`` configuration could therefore provide one ``lr_lambda`` callback
    per parameter, while the current optimizer merges parameters with identical
    hyperparameters before the scheduler constructor validates that list. This
    helper returns a copy with one callback per merged group when every callback
    in that group is the same object.

    Args:
        scheduler_kwargs: Explicit scheduler constructor keyword arguments.
        unmerged_param_groups: Legacy single-parameter groups in their original order.

    Returns:
        Scheduler keyword arguments compatible with the merged group layout.

    Raises:
        ValueError: If one merged group would need distinct ``lr_lambda`` callbacks.
    """
    lr_lambdas = scheduler_kwargs.get("lr_lambda")
    if not isinstance(lr_lambdas, list) or len(lr_lambdas) != len(unmerged_param_groups):
        return scheduler_kwargs

    buckets = _merge_buckets(unmerged_param_groups)
    regrouped_lambdas: list[Any] = []
    for bucket in buckets:
        callbacks = [lr_lambdas[index] for index in bucket]
        if any(callback is not callbacks[0] for callback in callbacks[1:]):
            raise ValueError(
                "Cannot merge optimizer groups with distinct LambdaLR callbacks. "
                "Use one shared callback for parameters that share optimizer hyperparameters, "
                "or keep those parameters in separate optimizer groups."
            )
        regrouped_lambdas.append(callbacks[0])

    regrouped_kwargs = dict(scheduler_kwargs)
    regrouped_kwargs["lr_lambda"] = regrouped_lambdas
    return regrouped_kwargs


def regroup_unmerged_optimizer_state(checkpoint: dict[str, Any]) -> None:
    """Rewrite one-group-per-parameter optimizer/scheduler state onto the merged parameter groups.

    :func:`get_param_dict` used to emit one parameter group per parameter, so ``torch.optim`` numbered
    the saved optimizer state by each parameter's position in that layout, and every per-group
    scheduler list (``base_lrs``, ``_last_lr``, ``lr_lambdas``, ``min_lrs``) had one entry per
    parameter. Groups now hold every parameter sharing their hyperparameters — a group count
    ``Optimizer.load_state_dict`` rejects outright — so reindex the saved state rather than fail the
    resume.

    Each optimizer's scheduler state is collapsed alongside it, matched by position the way
    PyTorch Lightning stores the two lists.

    The merged layout is derived from the saved groups themselves: bucketing them by the
    hyperparameters they recorded, in the order they were saved, reproduces the buckets
    :func:`get_param_dict` builds for the same run, since those are the same parameters in the same
    order. State already saved in the merged layout has groups holding several parameters and is left
    untouched.

    Args:
        checkpoint: Checkpoint dict carrying ``optimizer_states`` (and optionally ``lr_schedulers``),
            mutated in-place.
    """
    scheduler_states = checkpoint.get("lr_schedulers") or []
    for index, optimizer_state in enumerate(checkpoint.get("optimizer_states") or []):
        saved_groups = optimizer_state.get("param_groups") or []
        if not saved_groups or any(len(saved_group["params"]) != 1 for saved_group in saved_groups):
            continue
        buckets = _merge_buckets(saved_groups)
        saved_state = optimizer_state.get("state", {})
        merged_state: dict[int, Any] = {}
        merged_groups: list[dict[str, Any]] = []
        slot = 0
        for bucket in buckets:
            merged_group = {key: value for key, value in saved_groups[bucket[0]].items() if key != "params"}
            slots = []
            for unmerged_index in bucket:
                saved_id = saved_groups[unmerged_index]["params"][0]
                if saved_id in saved_state:
                    merged_state[slot] = saved_state[saved_id]
                slots.append(slot)
                slot += 1
            merged_group["params"] = slots
            merged_groups.append(merged_group)
        optimizer_state["state"] = merged_state
        optimizer_state["param_groups"] = merged_groups
        if index < len(scheduler_states):
            _regroup_scheduler_lists(scheduler_states[index], len(saved_groups), buckets)
        if len(merged_groups) < len(saved_groups):
            logger.info(
                "Regrouped resumed optimizer state from %d single-parameter groups onto %d merged groups.",
                len(saved_groups),
                len(merged_groups),
            )


def _regroup_scheduler_lists(scheduler_state: dict[str, Any], unmerged_count: int, buckets: list[list[int]]) -> None:
    """Collapse a scheduler's per-parameter-group lists the way its optimizer's groups were collapsed.

    A composite scheduler (``SequentialLR``, ``ChainedScheduler``) nests its wrapped schedulers' own
    state dicts under a ``_schedulers`` list rather than holding per-group lists at the top level, so
    those nested dicts need the same collapse applied recursively.

    Args:
        scheduler_state: Saved scheduler state, mutated in-place.
        unmerged_count: Number of parameter groups the scheduler state was saved with.
        buckets: Saved-group indices per merged group, as produced by :func:`_merge_buckets`.
    """
    # A bucket's parameters all shared one group's hyperparameters before the merge, so its first
    # entry is the value the merged group inherits.
    for key, value in scheduler_state.items():
        if key == "_schedulers" and isinstance(value, list):
            for nested_state in value:
                if isinstance(nested_state, dict):
                    _regroup_scheduler_lists(nested_state, unmerged_count, buckets)
        elif isinstance(value, list) and len(value) == unmerged_count:
            scheduler_state[key] = [value[bucket[0]] for bucket in buckets]


def _explicit_param_group_settings(train_config: TrainConfig) -> list[str]:
    """Return the settings in :data:`_PARAM_GROUP_SETTINGS` that the caller of a resumed run chose.

    That is ``model_fields_set``, except for a config built from a complete mapping: LightningCLI's parser (``rfdetr
    fit``) and ``TrainConfig(**json.load(f)["train_config"])`` on a current ``training_config.json`` mark every field
    as set, so membership says nothing about what the caller chose. For such a config a setting counts when it differs
    from its default.

    Args:
        train_config: Config of the resumed run.

    Returns:
        The chosen settings, in :data:`_PARAM_GROUP_SETTINGS` order.

    Examples:
        >>> _explicit_param_group_settings(TrainConfig(dataset_dir="data", lr=1e-4))
        ['lr']
        >>> complete = TrainConfig(**TrainConfig(dataset_dir="data", weight_decay=5e-4).model_dump())
        >>> _explicit_param_group_settings(complete)
        ['weight_decay']
    """
    fields = type(train_config).model_fields
    if train_config.model_fields_set >= fields.keys():
        return [name for name in _PARAM_GROUP_SETTINGS if getattr(train_config, name) != fields[name].default]
    return [name for name in _PARAM_GROUP_SETTINGS if name in train_config.model_fields_set]


def _resolve_resumed_param_group_settings(
    train_config: TrainConfig, checkpoint: Mapping[str, Any]
) -> tuple[TrainConfig, dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Order a resumed run's parameter-group settings as defaults, then the checkpoint, then the caller (#1613).

    A setting in :data:`_PARAM_GROUP_SETTINGS` that the caller left out takes the value recorded in the checkpoint's
    ``args``, and one they set keeps their value; :func:`_explicit_param_group_settings` tells the two apart. A
    checkpoint without ``args`` (a ``last.ckpt`` written before rfdetr 1.11.1) records nothing to take, so the config
    is returned as it is.

    Args:
        train_config: Config of the resumed run.
        checkpoint: The checkpoint being resumed, or at least its ``args``.

    Returns:
        ``(resolved, restored, overridden)``: the config to train with, whose ``model_fields_set`` leaves out what it
        took from the checkpoint; the settings taken from the checkpoint; and ``{name: (recorded, set)}`` for each
        setting the caller set to a value other than the recorded one.

    Examples:
        >>> config = TrainConfig(dataset_dir="data", lr=1e-5)
        >>> resolved, restored, overridden = _resolve_resumed_param_group_settings(
        ...     config, {"args": {"lr": 5e-5, "weight_decay": 2e-4}}
        ... )
        >>> resolved.lr, resolved.weight_decay, restored, overridden
        (1e-05, 0.0002, {'weight_decay': 0.0002}, {'lr': (5e-05, 1e-05)})
    """
    recorded = checkpoint.get("args")
    if not isinstance(recorded, dict):
        return train_config, {}, {}
    explicit = _explicit_param_group_settings(train_config)
    restored = {name: recorded[name] for name in _PARAM_GROUP_SETTINGS if name in recorded and name not in explicit}
    overridden = {
        name: (recorded[name], getattr(train_config, name))
        for name in _PARAM_GROUP_SETTINGS
        if name in recorded and name in explicit and recorded[name] != getattr(train_config, name)
    }
    resolved = train_config.model_copy(update=restored)
    # The caller did not set these; same convention as RFDETR.from_checkpoint's checkpoint-derived num_classes.
    resolved.model_fields_set.difference_update(restored)
    return resolved, restored, overridden


def _apply_configured_param_group_settings(
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    configured: list[tuple[float, float | None]],
    *,
    restart_without_base: bool = False,
) -> int:
    """Give each restored parameter group the base learning rate and weight decay this run configured.

    Resuming from a checkpoint restores every group's ``lr``, ``initial_lr`` and ``weight_decay``, and the scheduler's
    ``base_lrs``, over what ``configure_optimizers`` built (#1613). A group whose configured base differs takes it as
    its new ``initial_lr`` and scheduler base, and its current ``lr`` is scaled by the same factor, so the schedule
    carries on from the step it reached instead of restarting: a cosine half-way down stays half-way down, and
    ``ReduceLROnPlateau`` keeps its reductions.

    A group restored without ``initial_lr`` (a ``ReduceLROnPlateau`` checkpoint written by an earlier rfdetr release)
    has no base to compare or scale from: it keeps its restored ``lr`` unless ``restart_without_base`` is set, and then
    restarts at its configured base, which is logged. A group restored at a zero base shows no schedule progress, so it
    restarts at its configured base.

    Args:
        optimizer: The optimizer after Lightning restored its state.
        scheduler: Its scheduler after Lightning restored its state, if any.
        configured: ``(initial_lr, weight_decay)`` of each group as ``configure_optimizers`` built it, in group order;
            ``weight_decay`` is ``None`` for an optimizer without one.
        restart_without_base: Restart groups restored without ``initial_lr`` at their configured base; set when a
            learning-rate setting changed.

    Returns:
        How many groups changed.

    Examples:
        >>> import torch
        >>> optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
        >>> optimizer.step(); scheduler.step()
        >>> _apply_configured_param_group_settings(optimizer, scheduler, [(0.01, 0.01)])
        1
        >>> scheduler.base_lrs, optimizer.param_groups[0]["lr"]
        ([0.01], 0.005)
    """
    changed = 0
    without_base = 0
    for index, (group, (configured_lr, configured_weight_decay)) in enumerate(
        zip(optimizer.param_groups, configured, strict=True)
    ):
        restored_lr = group.get("initial_lr")
        lr_changed = restored_lr != configured_lr and (restored_lr is not None or restart_without_base)
        if lr_changed:
            without_base += restored_lr is None
            group["lr"] = group["lr"] * configured_lr / restored_lr if restored_lr else configured_lr
            group["initial_lr"] = configured_lr
            _set_scheduler_group_lr(scheduler, index, base_lr=configured_lr, lr=group["lr"])
        weight_decay_changed = (
            configured_weight_decay is not None and group.get("weight_decay") != configured_weight_decay
        )
        if weight_decay_changed:
            group["weight_decay"] = configured_weight_decay
        changed += lr_changed or weight_decay_changed
    if without_base:
        logger.warning(
            "%d resumed parameter groups were saved without the learning rate they started from (a ReduceLROnPlateau "
            "checkpoint written by an earlier rfdetr release), so they restart at the configured learning rate "
            "instead of keeping the reductions already taken.",
            without_base,
        )
    return changed


def _set_scheduler_group_lr(scheduler: object, index: int, *, base_lr: float, lr: float) -> None:
    """Set one parameter group's base and last learning rate in ``scheduler`` and every scheduler it wraps.

    A composite scheduler (``SequentialLR``, ``ChainedScheduler``) keeps the schedulers it wraps under ``_schedulers``,
    and each closed-form one (``LambdaLR``, the managed presets) computes the group's rate from its own ``base_lrs``.

    Args:
        scheduler: A scheduler, or ``None``.
        index: The parameter group's position in the optimizer.
        base_lr: The group's new base learning rate.
        lr: The group's new current learning rate.

    Examples:
        >>> import torch
        >>> optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        >>> _set_scheduler_group_lr(scheduler, 0, base_lr=0.2, lr=0.2)
        >>> scheduler.base_lrs, scheduler.get_last_lr()
        ([0.2], [0.2])
    """
    base_lrs = getattr(scheduler, "base_lrs", None)
    if isinstance(base_lrs, list):
        base_lrs[index] = base_lr
    last_lrs = getattr(scheduler, "_last_lr", None)
    if isinstance(last_lrs, list):
        last_lrs[index] = lr
    for nested in getattr(scheduler, "_schedulers", ()):
        _set_scheduler_group_lr(nested, index, base_lr=base_lr, lr=lr)
