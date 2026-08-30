# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Functions to get params dict."""

from typing import Any, cast

from torch import nn

from rfdetr.models.backbone import Joiner
from rfdetr.utilities.logger import get_logger

logger = get_logger()


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
        if isinstance(value, list) and len(value) == unmerged_count:
            scheduler_state[key] = [value[bucket[0]] for bucket in buckets]
        elif key == "_schedulers" and isinstance(value, list):
            for nested_state in value:
                if isinstance(nested_state, dict):
                    _regroup_scheduler_lists(nested_state, unmerged_count, buckets)
