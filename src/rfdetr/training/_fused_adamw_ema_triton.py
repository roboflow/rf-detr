# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Triton kernels for the combined gradient-clipping, AdamW, and EMA update."""

# Triton's JIT DSL is intentionally untyped and does not ship a ``py.typed`` marker.
# mypy: disable-error-code="import-not-found, import-untyped, untyped-decorator, no-untyped-def"

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def global_norm_kernel(grad_ptrs, tensor_ids, block_starts, lengths, norm_sq, BLOCK: tl.constexpr):  # noqa: N803
    """Accumulate the squared L2 norm of all gradients into one scalar."""
    block_id = tl.program_id(0)
    tensor_id = tl.load(tensor_ids + block_id)
    start = tl.load(block_starts + block_id)
    length = tl.load(lengths + tensor_id)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < length
    grad = tl.load(grad_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    values = tl.load(grad + offsets, mask=mask, other=0.0)
    tl.atomic_add(norm_sq, tl.sum(values * values))


@triton.jit
def fused_adamw_ema_kernel(
    param_ptrs,
    grad_ptrs,
    exp_avg_ptrs,
    exp_avg_sq_ptrs,
    step_ptrs,
    ema_ptrs,
    tensor_group_ids,
    tensor_ids,
    block_starts,
    lengths,
    group_lrs,
    group_weight_decays,
    group_beta1s,
    group_beta2s,
    group_epsilons,
    total_norm_sq,
    max_norm,
    ema_decay,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Apply clipped AdamW and update the corresponding EMA parameter in one pass."""
    block_id = tl.program_id(0)
    tensor_id = tl.load(tensor_ids + block_id)
    group_id = tl.load(tensor_group_ids + tensor_id)
    start = tl.load(block_starts + block_id)
    length = tl.load(lengths + tensor_id)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < length

    param = tl.load(param_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    grad = tl.load(grad_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    exp_avg = tl.load(exp_avg_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    exp_avg_sq = tl.load(exp_avg_sq_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    step_ptr = tl.load(step_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    ema = tl.load(ema_ptrs + tensor_id).to(tl.pointer_type(tl.float32))

    lr = tl.load(group_lrs + group_id)
    weight_decay = tl.load(group_weight_decays + group_id)
    beta1 = tl.load(group_beta1s + group_id)
    beta2 = tl.load(group_beta2s + group_id)
    epsilon = tl.load(group_epsilons + group_id)
    step = tl.load(step_ptr)
    norm = tl.sqrt(tl.load(total_norm_sq))
    clip_coefficient = tl.minimum(max_norm / (norm + 1.0e-6), 1.0)

    parameter = tl.load(param + offsets, mask=mask)
    gradient = tl.load(grad + offsets, mask=mask) * clip_coefficient
    first_moment = tl.load(exp_avg + offsets, mask=mask)
    second_moment = tl.load(exp_avg_sq + offsets, mask=mask)

    parameter = libdevice.fma(-lr * weight_decay, parameter, parameter)
    first_moment = libdevice.fma(beta1, first_moment, libdevice.fma(-beta1, gradient, gradient))
    squared_gradient = gradient * gradient
    second_moment = libdevice.fma(
        beta2,
        second_moment,
        libdevice.fma(-beta2, squared_gradient, squared_gradient),
    )
    bias_correction1 = 1.0 - libdevice.pow(beta1, step)
    bias_correction2 = 1.0 - libdevice.pow(beta2, step)
    denominator = libdevice.sqrt(second_moment) / libdevice.sqrt(bias_correction2) + epsilon
    parameter = parameter - (lr / bias_correction1) * first_moment / denominator
    averaged = tl.load(ema + offsets, mask=mask)
    averaged = libdevice.fma(parameter, 1.0 - ema_decay, averaged * ema_decay)

    tl.store(param + offsets, parameter, mask=mask)
    tl.store(exp_avg + offsets, first_moment, mask=mask)
    tl.store(exp_avg_sq + offsets, second_moment, mask=mask)
    tl.store(ema + offsets, averaged, mask=mask)


@triton.jit
def ema_only_kernel(
    source_ptrs,
    ema_ptrs,
    tensor_ids,
    block_starts,
    lengths,
    ema_decay,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Update floating-point EMA buffers that are not optimizer parameters."""
    block_id = tl.program_id(0)
    tensor_id = tl.load(tensor_ids + block_id)
    start = tl.load(block_starts + block_id)
    length = tl.load(lengths + tensor_id)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < length
    source = tl.load(source_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    ema = tl.load(ema_ptrs + tensor_id).to(tl.pointer_type(tl.float32))
    value = tl.load(source + offsets, mask=mask)
    averaged = tl.load(ema + offsets, mask=mask)
    tl.store(ema + offsets, libdevice.fma(value, 1.0 - ema_decay, averaged * ema_decay), mask=mask)
