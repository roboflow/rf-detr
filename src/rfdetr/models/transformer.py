# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Transformer class."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.models._types import BuilderArgs
from rfdetr.models.heads.keypoints import ConditionalQueryInitializer
from rfdetr.models.math import MLP
from rfdetr.models.ops.modules import MSDeformAttn
from rfdetr.utilities.compiler import cuda_autocast_dtype as _cuda_autocast_dtype
from rfdetr.utilities.compiler import is_compiling
from rfdetr.utilities.compiler import is_tracing as _is_tracing


def _tracer_absent() -> bool:
    """Fallback for ``torch.compiler.is_exporting`` on torch versions that lack it.

    ``is_exporting`` was added in torch 2.7. Hoisting this fallback avoids allocating a fresh
    ``lambda`` in the hot decoder ``forward`` path.

    Returns:
        Always ``False``.
    """
    return False


#: ROCm builds report AMD devices as ``cuda``. The decoder rewrites were measured (speed and bitwise parity) on NVIDIA
#: CUDA only, so a HIP build keeps the previous ops.
_IS_HIP = torch.version.hip is not None


def _eager_cuda(tensor: Tensor) -> bool:
    """Return whether ``tensor`` is on CUDA in eager execution, the only case the decoder rewrites are taken.

    Every eager CUDA rewrite in this module gates on this one predicate, so a device or execution-mode policy is a
    single edit; each call site adds its own module and dtype checks after it. ROCm (HIP) builds keep the plain ops
    because the rewrites are unmeasured there. ``torch.compile`` and tracing keep them too: Inductor fuses them itself
    and the exporters expect no custom autograd function. So does any ``torch.func`` transform (``vmap``, ``grad``,
    ``jvp``, ...): the rewrites' autograd functions define no ``setup_context`` and would raise under one.

    Args:
        tensor: The input whose device decides eligibility.

    Returns:
        ``True`` for a CUDA tensor on a non-HIP build outside ``torch.compile``, tracing and ``torch.func`` transforms,
        ``False`` otherwise.

    Examples:
        >>> _eager_cuda(torch.zeros(1))
        False
    """
    return (
        tensor.is_cuda
        and not _IS_HIP
        and not is_compiling()
        and not _is_tracing()
        and not torch._C._are_functorch_transforms_active()
    )


def _safe_multinormalize(dim: int) -> int:
    """Clamp a MultiheadAttention head count to at least one."""
    return max(1, dim)


#: Hook registries ``Module.__call__`` consults around ``forward``; each also has a ``_global`` twin in
#: ``torch.nn.modules.module`` that applies to every module. These, the ``_global`` twins and the
#: ``_compiled_call_impl`` that ``Module.compile()`` sets are the private torch state :func:`_module_call_is_plain`
#: reads; a torch release that renames or drops any of them makes it return ``False``, never a silent ``True``.
_MODULE_HOOK_ATTRS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_MODULE_HOOK_ATTRS = tuple(f"_global{name}" for name in _MODULE_HOOK_ATTRS)
_MODULE_CALL_ATTRS = (*_MODULE_HOOK_ATTRS, "_compiled_call_impl")


def _module_call_is_plain(*modules: nn.Module) -> bool:
    """Return whether calling each of ``modules`` runs its ``forward`` and nothing else.

    Code that reads a module's parameters instead of calling the module stands in for the call only when
    ``Module.__call__`` adds nothing: no instance-level ``forward``, no ``compile()`` wrapper
    (``_compiled_call_impl``), and no forward, forward-pre, backward or backward-pre hook, whether registered on the
    module or for every module through ``torch.nn.modules.module``. The check fails closed: if any of those torch
    attributes is missing, the call is not plain, so the caller keeps the module call.

    Args:
        *modules: The modules the caller would otherwise call.

    Returns:
        ``True`` when none of them is overridden or observed, ``False`` otherwise or when an attribute the check reads
        is missing.

    Examples:
        >>> linear = nn.Linear(2, 2)
        >>> _module_call_is_plain(linear)
        True
        >>> handle = linear.register_full_backward_hook(lambda *_: None)
        >>> _module_call_is_plain(linear)
        False
        >>> handle.remove()
    """
    module_registry = nn.modules.module
    if not all(hasattr(module_registry, name) for name in _GLOBAL_MODULE_HOOK_ATTRS):
        return False
    if any(getattr(module_registry, name) for name in _GLOBAL_MODULE_HOOK_ATTRS):
        return False
    return all(
        all(hasattr(module, name) for name in _MODULE_CALL_ATTRS)
        and "forward" not in module.__dict__
        and module._compiled_call_impl is None
        and not any(getattr(module, name) for name in _MODULE_HOOK_ATTRS)
        for module in modules
    )


class _AddInDtype(torch.autograd.Function):
    """``(a + b).to(dtype)`` as one kernel that never writes the full-precision sum.

    ``torch.add`` with a lower-precision ``out=`` computes in the promoted operand dtype and rounds once on the store,
    so the result is bitwise the two-step one. The backward mirrors the two-step graph as well: the cast's backward
    upcasts the incoming gradient, and the add hands it to both operands, casting it back for a lower-precision one —
    that round trip returns the incoming bits, so ``grad.to(b.dtype)`` is it. That holds for a single consumer of the
    result. When it fans out (the self-attention query and key projections, or the two deformable-attention heads),
    autograd sums the consumers' gradients in ``dtype`` before this backward upcasts them, where the two-step graph
    upcasts each one and sums in full precision: the operand gradients carry one extra rounding to ``dtype`` (at most
    half an ulp), while forward values and losses stay bitwise.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, a: Tensor, b: Tensor, dtype: torch.dtype) -> Tensor:
        ctx.operand_dtypes = (a.dtype, b.dtype)  # type: ignore[attr-defined]
        out = torch.empty(a.shape, dtype=dtype, device=a.device)
        return torch.add(a, b, out=out)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor, Tensor, None]:
        a_dtype, b_dtype = ctx.operand_dtypes  # type: ignore[attr-defined]
        return grad.to(a_dtype), grad.to(b_dtype), None


class _LinearReLU(torch.autograd.Function):
    """``relu(x @ weight.T + bias)`` with the bias and the ReLU applied in the GEMM epilogue.

    ``torch._addmm_activation`` rounds ``acc + bias`` once and clamps, and ``round(max(0, y)) == max(0, round(y))`` for
    finite values under a rounding that keeps sign and zero, so for finite inputs the forward is bitwise
    ``F.relu(F.linear(x, weight, bias))`` while the separate ReLU pass over the ``[tokens, dim_feedforward]`` activation
    disappears. Parity for NaN and for the sign of a zero result (``-0.0``) is unverified on CUDA: the cuBLASLt epilogue
    may differ from ``torch.relu`` there. torch defines no derivative for the fused op, so the backward runs the ops
    autograd runs for the two-op graph: ``threshold_backward`` on the saved output, the two GEMMs ``AddmmBackward0``
    issues for a row-major input and an ``nn.Linear`` weight (``grad.mm(weight)``, and ``grad.t().mm(rows)``, which
    leaves the weight gradient contiguous like the parameter) and the bias reduce, each skipped when its input needs no
    gradient. Only the input and the returned output are saved (the flattened views are rebuilt in the backward), so the
    backward is itself differentiable and a double backward (``create_graph=True``) gets every term the two-op graph
    gives. ``torch.utils.flop_counter.FlopCounterMode`` has no formula for ``_addmm_activation``, so it undercounts
    ``linear1`` on a CUDA model in eager execution.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        result = torch._addmm_activation(bias, rows, weight.t()).view(*x.shape[:-1], weight.shape[0])
        # A saved tensor that is neither an input nor an output (such as ``rows``) is cut off from the graph, which
        # silently drops its terms from a double backward; save ``x`` and ``result`` and rebuild the views instead.
        ctx.save_for_backward(x, weight, result)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        x, weight, result = ctx.saved_tensors  # type: ignore[attr-defined]
        needs_x, needs_weight, needs_bias = ctx.needs_input_grad  # type: ignore[attr-defined]
        out = result.reshape(-1, result.shape[-1])
        grad_rows = grad.reshape(-1, grad.shape[-1])
        grad_pre = torch.ops.aten.threshold_backward(grad_rows, out, 0)
        grad_x = grad_pre.mm(weight).view(x.shape) if needs_x else None
        grad_weight = grad_pre.t().mm(x.reshape(-1, x.shape[-1])) if needs_weight else None
        grad_bias = grad_pre.sum(0) if needs_bias else None
        return grad_x, grad_weight, grad_bias


def _additive_attn_mask(mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    """Return the float form of a boolean attention mask: ``-inf`` where blocked, ``0`` elsewhere.

    ``nn.MultiheadAttention`` builds this itself from a boolean mask with ``zeros_like(mask, dtype=...)``.
    coremltools (9.0, 9.1) drops that ``dtype`` keyword when it converts a ``torch.export`` program, so the mask
    stays boolean and ``scaled_dot_product_attention`` reads it with the opposite meaning. Passing the float mask
    keeps eager results identical and gives the converters nothing to reinterpret.

    Args:
        mask: Boolean mask, ``True`` where attention is blocked, or ``None``.
        dtype: Floating dtype of the attention inputs.

    Returns:
        The additive mask in *dtype*, or *mask* unchanged when it is ``None`` or already floating.

    Examples:
        >>> _additive_attn_mask(torch.tensor([[False, True]]), torch.float32)
        tensor([[0., -inf]])
        >>> _additive_attn_mask(None, torch.float32) is None
        True
    """
    if mask is None or mask.dtype != torch.bool:
        return mask
    return mask.to(dtype).masked_fill(mask, float("-inf"))


class _CastThenExpand(torch.autograd.Function):
    """``x.to(dtype).expand(groups, ...)`` whose backward sums the group gradients straight into ``x``'s dtype.

    The plain graph casts the expanded tensor (a materialised ``groups``-fold copy in ``dtype``) and, in the backward,
    upcasts the ``groups``-fold gradient before ``expand``'s sum. Casting once before the expand gives the same forward
    values; the backward reduces the lower-precision gradient with accumulation and output in ``x``'s dtype, the same
    fp32 sum of the same values (summation order may differ), without the intermediate full-width upcast.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor, groups: int, dtype: torch.dtype) -> Tensor:
        ctx.x_dtype = x.dtype  # type: ignore[attr-defined]
        return x.to(dtype).unsqueeze(0).expand(groups, *x.shape)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor, None, None]:
        return grad.sum(0, dtype=ctx.x_dtype), None, None  # type: ignore[attr-defined]


class _InterleavedSinCos(torch.autograd.Function):
    """``stack((sin(angle), cos(angle)), -1)`` written straight into an interleaved output in ``dtype``.

    Each of ``sin`` and ``cos`` computes in the angle's precision and stores into its own strided half of the output,
    rounding once — bitwise ``sin(angle).to(dtype)`` — so neither the full-precision values nor the stacked copy is ever
    written. The backward is the two-op graph's: ``sin`` contributes ``grad * cos(angle)``, ``cos`` contributes ``-(grad
    * sin(angle))``, each from the incoming gradient upcast to the angle's precision.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, angle: Tensor, dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(angle)
        out = torch.empty(*angle.shape, 2, dtype=dtype, device=angle.device)
        torch.sin(angle, out=out[..., 0])
        torch.cos(angle, out=out[..., 1])
        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        (angle,) = ctx.saved_tensors  # type: ignore[attr-defined]
        grad_sin = grad[..., 0].to(angle.dtype)
        grad_cos = grad[..., 1].to(angle.dtype)
        return grad_sin * angle.cos() - grad_cos * angle.sin(), None


def _sineembed_interleaved(pos_tensor: Tensor, dim: int, out_dtype: torch.dtype | None) -> Tensor:
    """:func:`gen_sineembed_for_position` for CUDA eager execution, without its strided slices.

    ``dim_t`` repeats every frequency twice (``dim_t // 2`` maps 2i and 2i+1 to the same exponent), and the even
    entries feed ``sin`` while the odd entries feed ``cos``. Dividing by the ``dim // 2`` distinct frequencies once
    gives, bitwise, the angles the interleaved ``[0::2]`` / ``[1::2]`` slices read, so ``sin`` and ``cos`` run on one
    contiguous tensor instead of two strided views (whose backward zero-fills and scatters a full-width buffer each),
    the division does half the work, and :class:`_InterleavedSinCos` writes both straight into the output.

    Args:
        pos_tensor: Coordinates of shape ``(bs, n_query, 2)`` or ``(bs, n_query, 4)``, on CUDA.
        dim: Embedding width per coordinate; must be even (the caller routes odd widths to the plain ops).
        out_dtype: Dtype of the returned embedding, or ``None`` for ``pos_tensor.dtype``.

    Returns:
        The embedding, ``(bs, n_query, dim * pos_tensor.shape[-1])``, in y, x[, w, h] order.

    Raises:
        ValueError: If the last dimension of ``pos_tensor`` is neither 2 nor 4.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    frequencies = dim_t[0::2]
    # Coordinates in output order (y, x[, w, h]) through slices: an index list would build a host index
    # tensor, and its device copy is not allowed while a CUDA graph is being captured.
    if pos_tensor.size(-1) == 2:
        coords = torch.cat((pos_tensor[:, :, 1:2], pos_tensor[:, :, 0:1]), dim=-1)
    elif pos_tensor.size(-1) == 4:
        coords = torch.cat((pos_tensor[:, :, 1:2], pos_tensor[:, :, 0:1], pos_tensor[:, :, 2:]), dim=-1)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    # (bs, n_query, coords, dim // 2): one angle tensor for every coordinate, in output order.
    angle = (coords * scale)[..., None] / frequencies
    pos = cast(Tensor, _InterleavedSinCos.apply(angle, pos_tensor.dtype if out_dtype is None else out_dtype))
    return pos.flatten(2)


def gen_sineembed_for_position(pos_tensor: Tensor, dim: int = 128, out_dtype: torch.dtype | None = None) -> Tensor:
    """Sine/cosine positional embedding of box coordinates, ``dim`` values per coordinate.

    CUDA eager execution takes :func:`_sineembed_interleaved`. Every other case (CPU, MPS, XLA, ``torch.compile`` and
    tracing) runs the plain ops: compiled and traced graphs keep them (Inductor fuses them itself and the exporters
    expect no custom autograd function), and the interleaved write is a CUDA rewrite, not one to run unmeasured
    elsewhere. The plain ops build the frequency table in at least float32 and round it once to the positions' dtype.
    float16, float32 and float64 keep the values and gradients they always had. In bfloat16, 10 of the 128 frequencies
    at the default ``dim`` are float32 values rounded once rather than values computed in bfloat16, which moves the
    outputs and coordinate gradients that use them slightly.

    Args:
        pos_tensor: Coordinates of shape ``(bs, n_query, 2)`` or ``(bs, n_query, 4)`` in ``[0, 1]``.
        dim: Embedding width per coordinate; consecutive (sin, cos) pairs share one of ``dim // 2``
            frequencies.
        out_dtype: Dtype of the returned embedding. ``None`` keeps ``pos_tensor.dtype``. A lower-precision
            dtype rounds each sin/cos once, exactly what a later cast of the full-precision result would
            give; on CUDA in eager execution that rounding happens on store, so the full-precision result
            is never written.

    Returns:
        The embedding, ``(bs, n_query, dim * pos_tensor.shape[-1])``, in y, x[, w, h] order.
    """
    # An odd ``dim`` has no (sin, cos) pairing: the plain ops raise for it, the interleaved write would not.
    if dim % 2 == 0 and _eager_cuda(pos_tensor):
        return _sineembed_interleaved(pos_tensor, dim, out_dtype)
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    # Computed in at least float32, as PositionEmbeddingSine does, then cast once to the positions' dtype. torch
    # 2.14 lowers a half-precision ``dim_t // 2`` to ``libdevice.isinf`` on a float16 or bfloat16 operand, which
    # Triton cannot compile (pytorch/pytorch#197002), so ``inference(dtype="float16",
    # compile_backend="inductor")`` failed. float32 and float64 positions compute in their own dtype as before.
    dim_t = torch.arange(dim, dtype=torch.promote_types(pos_tensor.dtype, torch.float32), device=pos_tensor.device)
    dim_t = (10000 ** (2 * (dim_t // 2) / dim)).to(pos_tensor.dtype)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos if out_dtype is None else pos.to(out_dtype)


def select_top_rows(scores: Tensor, k: int) -> Callable[[Tensor], Tensor]:
    """Rank ``(batch, tokens)`` scores and return a function that picks the ``k`` best rows of a token tensor.

    The per-group two-stage query selection loop calls this through ``Transformer.select_top_rows``, so an exporter can
    replace the ranking with one its target runs natively (see :mod:`rfdetr.export._neural_engine`). That loop runs in
    eval and export, and in training whenever the batched ``Transformer._two_stage_group_selection`` path is not taken;
    the batched path calls ``torch.topk`` itself and never reaches this hook.

    Args:
        scores: One score per token.
        k: Number of rows to keep, best first.

    Returns:
        A function from a ``(batch, tokens, C)`` tensor to its ``(batch, k, C)`` selected rows.

    Examples:
        >>> rows = torch.arange(4.0).reshape(1, 4, 1)
        >>> select_top_rows(torch.tensor([[0.1, 0.9, 0.5, 0.3]]), 2)(rows).flatten().tolist()
        [1.0, 2.0]
    """
    indices = torch.topk(scores, k, dim=1)[1]
    # Tensor.expand keeps the index a broadcast view (stride 0) instead of materialising it per channel.
    return lambda rows: torch.gather(rows, 1, indices.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))


def gen_encoder_output_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor | None = None,
    spatial_shapes: Sequence[tuple[int, int]] | Tensor | None = None,
    unsigmoid: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    proposals = []
    _cur = 0
    batch_size, _, _ = memory.shape
    assert spatial_shapes is not None
    for lvl, (height, width) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            # reshape(-1, ...) infers batch dynamically in ONNX instead of baking it in as constants.
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + height * width)].reshape(batch_size, height, width, 1)

            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            # Avoid baking constants in ONNX.
            valid_height = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + height
            valid_width = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + width

        # arange(n) equals linspace(0, n - 1, n) bit for bit, but linspace's integer step count
        # specialises symbolic sizes under torch.compile(dynamic=True) and recompiles per resolution.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=memory.device),
            torch.arange(width, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2

        # Keep symbolic batch in ONNX.
        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).reshape(-1, 1, 1, 2)
        proposals_grid = (grid.unsqueeze(0) + 0.5) / scale.float()

        wh = torch.ones_like(proposals_grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((proposals_grid, wh), -1).reshape(batch_size, -1, 4)
        proposals.append(proposal)
        _cur += height * width

    output_proposals = proposals[0] if len(proposals) == 1 else torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


def _stack_linear_params(modules: Sequence[nn.Linear]) -> tuple[Tensor, Tensor]:
    """Stack same-shaped ``nn.Linear`` modules' live parameters for a batched matmul.

    Reads ``.weight``/``.bias`` fresh on every call (never cached) so the stacked tensors always
    reflect the current, possibly just-updated, parameter values.

    Args:
        modules: Per-group ``nn.Linear`` layers sharing identical ``in_features``/``out_features``.

    Returns:
        Stacked ``(weight, bias)``, shaped ``(group, out_features, in_features)`` and
        ``(group, out_features)``.
    """
    return torch.stack([m.weight for m in modules]), torch.stack([m.bias for m in modules])


def _batched_group_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply ``group`` independent affine transforms via one batched matmul.

    Mathematically equivalent to calling a separate ``nn.Linear`` per group. The batched GEMM may
    use a different floating-point accumulation order, so callers compare within dtype-appropriate
    tolerance rather than requiring bit equality.

    Args:
        x: Input of shape ``(group, ..., in_features)``.
        weight: Stacked per-group weights of shape ``(group, out_features, in_features)``.
        bias: Stacked per-group biases of shape ``(group, out_features)``.

    Returns:
        Output of shape ``(group, ..., out_features)``.
    """
    leading_shape = x.shape[1:-1]
    x_flat = x.reshape(x.shape[0], -1, x.shape[-1])
    out = torch.baddbmm(bias.unsqueeze(1), x_flat, weight.transpose(1, 2))
    return out.reshape(*x.shape[:1], *leading_shape, weight.shape[-2])


def _batched_group_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """Apply ``group`` independent ``nn.LayerNorm`` affines after one shared normalization pass.

    ``LayerNorm``'s mean/variance reduction is computed per position regardless of which group's
    affine follows it, so normalizing once (without affine) and then applying each group's own
    ``weight``/``bias`` is exactly what ``group`` separate ``nn.LayerNorm`` calls compute, just
    without ``group`` separate reduction kernels.

    Args:
        x: Input of shape ``(group, ..., channels)``.
        weight: Stacked per-group scale of shape ``(group, channels)``.
        bias: Stacked per-group shift of shape ``(group, channels)``.
        eps: Shared normalization epsilon; the eligibility guard requires equality across groups.

    Returns:
        Output of shape ``(group, ..., channels)``.
    """
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    view_shape = (weight.shape[0], *([1] * (x.dim() - 2)), weight.shape[-1])
    return normalized * weight.view(view_shape) + bias.view(view_shape)


class Transformer(nn.Module):
    """Transformer with optional GroupPose keypoint decoder stream support."""

    def __init__(
        self,
        d_model: int = 512,
        sa_nhead: int = 8,
        ca_nhead: int = 8,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        lite_refpoint_refine: bool = False,
        decoder_norm_type: str = "LN",
        bbox_reparam: bool = False,
        use_grouppose_keypoints: bool = False,
        num_keypoints_per_class: list[int] | None = None,
        grouppose_keypoint_dim_downscale: int = 1,
        keypoint_cross_attn: bool = True,
        inter_instance_kp_attn: bool = False,
        num_registers: int = 0,
        dual_projector_kp_only: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = None
        self.enc_out_class_embed: nn.ModuleList | None = None
        self.enc_out_bbox_embed: nn.ModuleList | None = None
        self.enc_out_keypoint_embed: nn.ModuleList | None = None

        self.use_grouppose_keypoints = use_grouppose_keypoints
        self.dual_projector_kp_only = dual_projector_kp_only
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.num_registers = num_registers

        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
            enable_keypoint_processing=use_grouppose_keypoints,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
            keypoint_cross_attn=keypoint_cross_attn,
            inter_instance_kp_attn=inter_instance_kp_attn,
        )
        assert decoder_norm_type in ["LN", "Identity"]
        norm_ctors: dict[str, Callable[[int], nn.Module]] = {
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm_ctors[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
            enable_keypoint_processing=use_grouppose_keypoints,
            num_keypoints_per_class=self.num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
        )

        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

            if use_grouppose_keypoints and self.num_keypoints_per_class:
                total_keypoints = sum(self.num_keypoints_per_class)
                if total_keypoints > 0:
                    keypoint_dim = d_model // grouppose_keypoint_dim_downscale
                    self.keypoint_query_initializer = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    self.keypoint_query_initializer_enc = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    self.enc_out_keypoint_embed = nn.ModuleList(
                        [MLP(keypoint_dim, d_model, keypoint_dim, 2) for _ in range(group_detr)]
                    )

        self._reset_parameters()

        # Register tokens used by GroupPose path.
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.empty(num_registers, d_model).normal_())
            self.register_ref_points = nn.Parameter(torch.zeros(num_registers, 4))

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        # Query selection of the per-group two-stage loop: eval, export, and training when the batched
        # _two_stage_group_selection path (which calls torch.topk directly) is not taken. An exporter can swap it on
        # its copy of the model.
        self.select_top_rows: Callable[[Tensor, int], Callable[[Tensor], Tensor]] = select_top_rows
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False
        self._cuda_graph_spatial_shapes: dict[tuple[torch.device, tuple[tuple[int, int], ...]], Tensor] | None = None

    def export(self) -> None:
        self._export = True

    def enable_cuda_graph_capture(self) -> None:
        """Cache immutable spatial-shape tensors before their captured reuse.

        The cache is opt-in so eager, compile, and export tensor construction remain unchanged. Device is part of the
        key to keep a later model move safe. ``RFDETRModelModule._configure_cuda_graph_runner`` only builds the eager
        graph runner that calls this when the module was not compiled, so the cached branch in ``forward`` and its
        ``is_compiling()`` branch are mutually exclusive.
        """
        self._cuda_graph_spatial_shapes = {}

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height.float() / height
        valid_ratio_w = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _two_stage_batching_eligible(self) -> bool:
        """Return whether the per-group modules can safely share stacked operations.

        :meth:`_two_stage_group_selection` stacks each group's own ``.weight``/``.bias`` directly instead
        of calling the module generically, so it only gives correct results for the exact
        ``nn.Linear``/``nn.LayerNorm``/:class:`~rfdetr.models.math.MLP` types ``LWDETR.__init__`` always
        deepcopies per group, each with its affine parameters present. Exact ``type(...) is ...`` checks
        (not ``isinstance``) are required: a subclass of one of these types would otherwise pass an
        ``isinstance`` check while its own overridden ``forward`` is silently bypassed, since the batched
        path never calls the module -- it only reads ``.weight``/``.bias``. The ``is not None`` checks
        reject a ``bias=False`` ``nn.Linear`` or an ``elementwise_affine=False`` ``nn.LayerNorm``, which
        would otherwise reach ``torch.stack`` over a ``None`` and crash instead of falling back. Modules
        must also agree on parameter shapes, dtypes, devices, LayerNorm epsilon, and MLP depth because one
        stacked operation cannot preserve heterogeneous group contracts. Hooks, instance-level ``forward``
        overrides, and individually compiled children also require the generic call path. A test double, a
        future custom head, or any of these edge configurations all fall back to the per-group loop in
        :meth:`forward`, preserving each module's normal call semantics.
        """
        assert self.enc_out_class_embed is not None
        assert self.enc_out_bbox_embed is not None
        enc_output = cast(Sequence[nn.Linear], self.enc_output)
        enc_output_norm = cast(Sequence[nn.LayerNorm], self.enc_output_norm)
        class_embeds = cast(Sequence[nn.Linear], self.enc_out_class_embed)
        bbox_mlps = cast(Sequence[MLP], self.enc_out_bbox_embed)
        module_groups = (enc_output, enc_output_norm, class_embeds, bbox_mlps)
        if any(len(modules) != self.group_detr for modules in module_groups):
            return False
        if not (
            all(type(m) is nn.Linear and m.bias is not None for m in enc_output)
            and all(type(m) is nn.LayerNorm and m.weight is not None and m.bias is not None for m in enc_output_norm)
            and all(type(m) is nn.Linear and m.bias is not None for m in class_embeds)
            and all(
                type(m) is MLP and all(type(layer) is nn.Linear and layer.bias is not None for layer in m.layers)
                for m in bbox_mlps
            )
        ):
            return False

        bbox_layers = [layer for mlp in bbox_mlps for layer in mlp.layers]
        call_modules = [*enc_output, *enc_output_norm, *class_embeds, *bbox_mlps, *bbox_layers]
        if not _module_call_is_plain(*call_modules):
            return False

        for modules in (enc_output, class_embeds):
            first_weight = modules[0].weight
            first_bias = cast(Tensor, modules[0].bias)
            if any(
                (m.weight.shape, m.weight.dtype, m.weight.device)
                != (first_weight.shape, first_weight.dtype, first_weight.device)
                or (cast(Tensor, m.bias).shape, cast(Tensor, m.bias).dtype, cast(Tensor, m.bias).device)
                != (first_bias.shape, first_bias.dtype, first_bias.device)
                for m in modules[1:]
            ):
                return False

        first_norm = enc_output_norm[0]
        if first_norm.normalized_shape != (self.d_model,) or any(
            m.normalized_shape != first_norm.normalized_shape
            or m.eps != first_norm.eps
            or cast(Tensor, m.weight).dtype != cast(Tensor, first_norm.weight).dtype
            or cast(Tensor, m.weight).device != cast(Tensor, first_norm.weight).device
            or cast(Tensor, m.bias).dtype != cast(Tensor, first_norm.bias).dtype
            or cast(Tensor, m.bias).device != cast(Tensor, first_norm.bias).device
            for m in enc_output_norm[1:]
        ):
            return False

        num_layers = bbox_mlps[0].num_layers
        if any(m.num_layers != num_layers or len(m.layers) != num_layers for m in bbox_mlps):
            return False
        for layer_idx in range(num_layers):
            first_layer = cast(nn.Linear, bbox_mlps[0].layers[layer_idx])
            first_weight = first_layer.weight
            first_bias = cast(Tensor, first_layer.bias)
            if any(
                (
                    cast(nn.Linear, m.layers[layer_idx]).weight.shape,
                    cast(nn.Linear, m.layers[layer_idx]).weight.dtype,
                    cast(nn.Linear, m.layers[layer_idx]).weight.device,
                )
                != (first_weight.shape, first_weight.dtype, first_weight.device)
                or (
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).shape,
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).dtype,
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).device,
                )
                != (first_bias.shape, first_bias.dtype, first_bias.device)
                for m in bbox_mlps[1:]
            ):
                return False
        return True

    def _two_stage_group_selection(
        self, output_memory: Tensor, output_proposals: Tensor, group_detr: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run every ``group_detr`` group's two-stage top-k proposal selection in one batched pass.

        Replaces a Python loop that calls each group's own ``enc_output``/``enc_output_norm``/
        ``enc_out_class_embed``/``enc_out_bbox_embed`` on the SAME ``output_memory`` -- the loop issues
        one kernel per op per group, and on a host-dispatch-bound GPU that launch count dominates the
        block's wall time far more than its (small) GEMMs do. Stacking the ``group_detr`` copies of
        each op's weights and running one batched matmul keeps every group's own weights and drops the
        per-group launch count to a constant.

        Only called for ``group_detr > 1``, which the caller (:meth:`forward`) only reaches while
        training -- the ``assert self.training`` below enforces that mechanically. Eval/export always
        pass ``group_detr=1`` and use the original single-group code path. If a training-mode call is
        nevertheless traced, the cast-once rewrite below also keeps the plain expand-then-cast graph.

        The batched GEMMs can use a different accumulation order than separate calls. The paths match
        within float32 tolerance at real model scale, but under bf16/fp16 a near-tied class score can
        cross the discrete ``topk`` boundary. Tests therefore cover float32 output/gradient parity and
        finite mixed-precision gradients, while the accompanying benchmark checks short-run training
        quality through the public API.

        Args:
            output_memory: Encoder memory of shape ``(bs, S, d_model)``, shared by every group.
            output_proposals: Encoder anchor proposals of shape ``(bs, S, 4)``, shared by every group.
            group_detr: Number of independent groups (``self.group_detr`` while training, and always
                greater than 1 for every call site).

        Returns:
            ``(refpoint_embed_ts, memory_ts, boxes_ts, cls_ts)``, matching the per-group loop's own
            ``torch.cat(parts, dim=1)`` outputs and query order (group 0's queries first, then group
            1's, ...). ``cls_ts`` is ``enc_out_class_embed``'s output at the same selected positions,
            gathered from the ranking pass rather than recomputed by the caller.
        """
        # Training-only by contract, and the contract is load-bearing: the eval/export loop in forward
        # gathers the pre-norm rows so an fp16 CoreML program keeps its Neural Engine placement, while
        # this path norms the full length and gathers post-norm. An eval or exported graph routed here
        # would still produce correct numbers, so nothing would fail -- the model would just silently
        # lose the ANE and fall back to CPU. Assert instead of trusting the caller's guard.
        assert self.training
        assert self.enc_out_class_embed is not None
        assert self.enc_out_bbox_embed is not None
        bs = output_memory.shape[0]
        class_embeds = cast(Sequence[nn.Linear], self.enc_out_class_embed)
        bbox_mlps = cast(Sequence[MLP], self.enc_out_bbox_embed)
        topk = min(self.num_queries, output_memory.shape[-2])

        enc_output_weight, enc_output_bias = _stack_linear_params(cast(Sequence[nn.Linear], self.enc_output))
        norm_weight = torch.stack([cast(nn.LayerNorm, m).weight for m in self.enc_output_norm])
        norm_bias = torch.stack([cast(nn.LayerNorm, m).bias for m in self.enc_output_norm])
        norm_eps = cast(nn.LayerNorm, self.enc_output_norm[0]).eps

        compute_dtype = _cuda_autocast_dtype() if _eager_cuda(output_memory) else None
        if compute_dtype is None or output_memory.dtype != torch.float32 or compute_dtype == output_memory.dtype:
            memory_expanded = output_memory.unsqueeze(0).expand(group_detr, -1, -1, -1)
        else:
            # Under autocast the batched GEMM casts its fp32 input: cast the shared memory once instead of its
            # ``group_detr``-fold expansion, and let the backward sum the group gradients without upcasting.
            memory_expanded = cast(Tensor, _CastThenExpand.apply(output_memory, group_detr, compute_dtype))
        output_memory_all = _batched_group_linear(memory_expanded, enc_output_weight, enc_output_bias)
        output_memory_all = _batched_group_layer_norm(output_memory_all, norm_weight, norm_bias, norm_eps)

        class_weight, class_bias = _stack_linear_params(class_embeds)
        class_logits_all = _batched_group_linear(output_memory_all, class_weight, class_bias)
        # (group, bs, S) -> (group, bs, nq); torch.topk batches over every leading dim natively.
        topk_proposals_all = torch.topk(class_logits_all.max(-1)[0], topk, dim=-1)[1]

        # enc_out_class_embed is a plain per-position Linear, so gathering its already-computed
        # output at the same indices used below is exactly what re-running it on the gathered
        # hidden state would produce (Linear(x)[idx] == Linear(x[idx])) -- reuse instead of the
        # caller re-running enc_out_class_embed a second time on the gathered subset.
        cls_logits_selected = torch.gather(
            class_logits_all, 2, topk_proposals_all.unsqueeze(-1).expand(-1, -1, -1, class_logits_all.shape[-1])
        )

        tgt_undetach_all = torch.gather(
            output_memory_all, 2, topk_proposals_all.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        )
        proposals_expanded = output_proposals.unsqueeze(0).expand(group_detr, -1, -1, -1)
        # See the loop's own comment: gather before the pointwise box MLP, not after.
        output_proposals_all = torch.gather(
            proposals_expanded, 2, topk_proposals_all.unsqueeze(-1).expand(-1, -1, -1, 4)
        )

        hidden = tgt_undetach_all
        num_layers = bbox_mlps[0].num_layers
        for layer_idx in range(num_layers):
            layer_weight, layer_bias = _stack_linear_params(
                cast(Sequence[nn.Linear], [mlp.layers[layer_idx] for mlp in bbox_mlps])
            )
            hidden = _batched_group_linear(hidden, layer_weight, layer_bias)
            if layer_idx < num_layers - 1:
                hidden = F.relu(hidden)
        enc_outputs_coord_delta_all = hidden

        if self.bbox_reparam:
            coord_cxcy_all = (
                enc_outputs_coord_delta_all[..., :2] * output_proposals_all[..., 2:] + output_proposals_all[..., :2]
            )
            coord_wh_all = enc_outputs_coord_delta_all[..., 2:].exp() * output_proposals_all[..., 2:]
            refpoint_embed_all_undetach = torch.concat([coord_cxcy_all, coord_wh_all], dim=-1)
        else:
            refpoint_embed_all_undetach = enc_outputs_coord_delta_all + output_proposals_all
        refpoint_embed_all = refpoint_embed_all_undetach.detach()

        def _merge_groups(t: Tensor) -> Tensor:
            """Flatten the leading group dimension into the query dimension, group 0 first.

            Args:
                t: Tensor of shape ``(group, bs, nq, C)``.

            Returns:
                Tensor of shape ``(bs, group * nq, C)``, matching the per-group loop's own
                ``torch.cat(parts, dim=1)`` order (group 0's queries first, then group 1's, ...).
            """
            return t.permute(1, 0, 2, 3).reshape(bs, group_detr * topk, t.shape[-1])

        return (
            _merge_groups(refpoint_embed_all),
            _merge_groups(tgt_undetach_all),
            _merge_groups(refpoint_embed_all_undetach),
            _merge_groups(cls_logits_selected),
        )

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor] | None,
        pos_embeds: list[Tensor],
        refpoint_embed: Tensor,
        query_feat: Tensor,
        cross_attn_srcs: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor | None, ...]:
        """Flatten the multi-scale features, run the two-stage query selection, then the decoder.

        Args:
            srcs: Per-level feature maps, each ``(bs, d_model, h, w)``.
            masks: Per-level padding masks, each ``(bs, h, w)``, or ``None`` for unpadded input.
            pos_embeds: Per-level positional embeddings matching ``srcs``.
            refpoint_embed: Learned reference points, ``(num_queries * group_detr, 4)``.
            query_feat: Learned query features, ``(num_queries * group_detr, d_model)``.
            cross_attn_srcs: Optional separate feature maps for decoder cross-attention.

        Returns:
            ``(hs, references, memory_ts, boxes_ts)``, followed by ``(keypoint_hs, enc_kp_predictions,
            keypoint_memory_ts)`` when grouped-pose keypoints are enabled, and always ending with ``cls_ts``.
            The two-stage entries are ``None`` when ``two_stage`` is off; ``hs``/``references`` are ``None``
            without a decoder.
        """
        src_flatten = []
        mask_flatten_parts: list[Tensor] | None = [] if masks is not None else None
        lvl_pos_embed_flatten_parts = []
        spatial_shapes_hw: list[tuple[int, int]] = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            _, c, h, w = src.shape
            spatial_shapes_hw.append((h, w))

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed_flatten_parts.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)  # bs, hw
                assert mask_flatten_parts is not None
                mask_flatten_parts.append(mask)

        # MultiScaleProjector ends each stage with its channel LayerNorm, which returns channels-last storage viewed
        # as NCHW. Its real flattened/transposed output is already contiguous; contiguous() preserves the old layout
        # contract for custom strided inputs.
        memory = src_flatten[0].contiguous() if len(src_flatten) == 1 else torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten: Tensor | None = None
        valid_ratios: Tensor | None = None
        if masks is not None:
            assert mask_flatten_parts is not None
            # Real padding masks are contiguous after flatten(1), so the single-level contiguous() is a no-op.
            # It preserves cat's contiguous-layout contract for custom strided masks.
            mask_flatten = (
                mask_flatten_parts[0].contiguous() if len(mask_flatten_parts) == 1 else torch.cat(mask_flatten_parts, 1)
            )  # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # PositionEmbeddingSine produces channels-last storage viewed as NCHW, so flatten+transpose above is already
        # contiguous. Current nondeprecated models use one projector level; contiguous() is then a no-op, while
        # preserving cat's contiguous-layout contract for custom strided position tensors.
        lvl_pos_embed_flatten = (
            lvl_pos_embed_flatten_parts[0].contiguous()
            if len(lvl_pos_embed_flatten_parts) == 1
            else torch.cat(lvl_pos_embed_flatten_parts, 1)
        )  # bs, \sum{hxw}, c
        # spatial_shapes: one form per execution mode, each forced by a constraint the others break. Never
        # torch.empty(...) + in-place index assignment — the ScatterND it emits feeds a shape tensor
        # (level_start_index), and TensorRT rejects "IScatterLayer cannot be used to compute a shape tensor".
        #   cuda-graph capture -> as_tensor cached per (device, resolution): replay needs one immutable tensor
        #                         per signature (see enable_cuda_graph_capture).
        #   torch.export       -> as_tensor, ScatterND-free and constant-baking: _shape_as_tensor is untraceable
        #                         there ("the tensor has a non-zero number of elements, but its data is not
        #                         allocated yet"), and static export shapes make the baked Constant exact.
        #   torch.compile      -> one 0-d tensor per size, the only symbolic form: Dynamo polyfills
        #                         _shape_as_tensor to a torch.Size ("expected Tensor as element 0 in argument
        #                         0, but got torch.Size" aborts the compile), while as_tensor (torch.tensor on
        #                         older torch) specialises every size under dynamic=True, recompiling the whole
        #                         transformer per resolution until multi-scale training exhausts Dynamo's
        #                         recompile limit.
        #   eager / jit.trace  -> _shape_as_tensor(src)[2:4], a private ATen op returning src's 1-D int64 dim
        #                         sizes ([2:4] = (H, W) of NCHW): the Constant it bakes into a TorchScript ONNX
        #                         graph is one TensorRT accepts as a shape-tensor source, unlike ScatterND (#1155).
        # Predicates: is_compiling() is public from torch 2.3 (the compat helper uses the legacy Dynamo predicate
        # on 2.2); is_exporting() is absent below 2.7, so its probe is always False there and a strict
        # torch.export reports is_compiling() instead, taking the stacked-0-d branch for the same values —
        # is_exporting() implies is_compiling(), not the reverse.
        if self._cuda_graph_spatial_shapes is not None:
            spatial_key = (srcs[0].device, tuple(spatial_shapes_hw))
            spatial_shapes = self._cuda_graph_spatial_shapes.get(spatial_key)
            if spatial_shapes is None:
                spatial_shapes = torch.as_tensor(spatial_shapes_hw, device=srcs[0].device, dtype=torch.long)
                self._cuda_graph_spatial_shapes[spatial_key] = spatial_shapes
        # Export must precede compile: non-strict export sets both flags; compile alone does not set is_exporting().
        elif getattr(torch.compiler, "is_exporting", _tracer_absent)():
            spatial_shapes = torch.as_tensor(spatial_shapes_hw, device=srcs[0].device, dtype=torch.long)
        elif is_compiling():
            spatial_shapes = torch.stack(
                [
                    torch.stack([torch.scalar_tensor(size, dtype=torch.long, device=srcs[0].device) for size in hw])
                    for hw in spatial_shapes_hw
                ]
            )
        else:
            spatial_shapes = torch.stack([torch._shape_as_tensor(src)[2:4] for src in srcs]).to(
                device=srcs[0].device, dtype=torch.long
            )
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Flatten optional dual-projector features for keypoint-specific cross-attention.
        # NOTE: cross-attention reuses ``spatial_shapes_hw`` derived from ``srcs`` above — this assumes
        # ``cross_attn_srcs`` share the per-level (H, W) geometry of ``srcs`` (they differ only in channel
        # features from the dual projector). If a future variant produces cross-attn features with a different
        # level count or spatial geometry, the deformable sampling would mis-index; build a separate
        # ``cross_attn_spatial_shapes_hw`` at that point.
        cross_attn_memory = None
        if cross_attn_srcs is not None:
            ca_flatten = []
            for cross_src in cross_attn_srcs:
                tensor = getattr(cross_src, "tensors", cross_src)
                ca_flatten.append(tensor.flatten(2).transpose(1, 2))
            # The dual-projector path uses the same MultiScaleProjector layout as memory above.
            cross_attn_memory = ca_flatten[0].contiguous() if len(ca_flatten) == 1 else torch.cat(ca_flatten, 1)

        cls_ts = None
        if self.two_stage:
            assert self.enc_out_class_embed is not None
            assert self.enc_out_bbox_embed is not None
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes_hw, unsigmoid=not self.bbox_reparam
            )
            # group detr for first stage
            group_detr = self.group_detr if self.training else 1
            if group_detr > 1 and self._two_stage_batching_eligible():
                # Stack each group's own weights to replace the per-group launches with one batched
                # call per operation. The eligibility guard keeps custom or heterogeneous modules on
                # the loop below. Eval/export use group_detr=1, so tracing never sees the batched path.
                refpoint_embed_ts, memory_ts, boxes_ts, cls_ts = self._two_stage_group_selection(
                    output_memory, output_proposals, group_detr
                )
            else:
                refpoint_embed_ts_parts, memory_ts_parts, boxes_ts_parts = [], [], []
                for g_idx in range(group_detr):
                    output_memory_prenorm_gidx = self.enc_output[g_idx](output_memory)
                    output_memory_gidx = self.enc_output_norm[g_idx](output_memory_prenorm_gidx)

                    enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                    topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
                    select_rows = self.select_top_rows(enc_outputs_class_unselected_gidx.max(-1)[0], topk)

                    # get memory tgt. LayerNorm acts per token, so gathering the pre-norm rows and normalizing
                    # only the selected ones is mathematically the same as gathering the normalized rows. It
                    # also keeps the full-length norm output from crossing into the CPU-resident topk/gather.
                    # That crossing makes Apple's Neural Engine compiler reject a whole fp16 CoreML program
                    # ("Invalid layer") when enc_output_norm still has its identity affine (weight all ones,
                    # bias all zeros) and the token count is a multiple of 32. This reordering relies on
                    # enc_output_norm being token-pointwise, like the box-MLP reordering below; a cross-token
                    # norm must gather after.
                    # _two_stage_group_selection still gathers post-norm: it runs only while training, and export
                    # always takes this loop, so no exported graph reaches the ANE through that path.
                    tgt_undetach_gidx = self.enc_output_norm[g_idx](select_rows(output_memory_prenorm_gidx))
                    # Ranking needs every position's class score, but the box MLP is a pointwise (no
                    # cross-token mixing) transform of a single token's features -- gather the selected
                    # tokens first and run the MLP only on those, instead of on every encoder position and
                    # discarding all but ``topk`` of the results. This is equivalent only while
                    # ``enc_out_bbox_embed`` remains token-pointwise; a future stateful or cross-token head
                    # must move the MLP back before this gather.
                    output_proposals_gidx = select_rows(output_proposals)
                    if self.bbox_reparam:
                        enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](tgt_undetach_gidx)
                        enc_outputs_coord_cxcy_gidx = (
                            enc_outputs_coord_delta_gidx[..., :2] * output_proposals_gidx[..., 2:]
                            + output_proposals_gidx[..., :2]
                        )
                        enc_outputs_coord_wh_gidx = (
                            enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals_gidx[..., 2:]
                        )
                        refpoint_embed_gidx_undetach = torch.concat(
                            [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                        )
                    else:
                        refpoint_embed_gidx_undetach = (
                            self.enc_out_bbox_embed[g_idx](tgt_undetach_gidx) + output_proposals_gidx
                        )  # unsigmoid
                    # for decoder layer, detached as initial ones, (bs, nq, 4)
                    refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

                    refpoint_embed_ts_parts.append(refpoint_embed_gidx)
                    memory_ts_parts.append(tgt_undetach_gidx)
                    boxes_ts_parts.append(refpoint_embed_gidx_undetach)
                # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d). Eval/export run one group;
                # a single-input Concat would reach the ONNX graph, where CoreML rejects it and splits the graph.
                if group_detr == 1:
                    # refpoint_embed_ts is a .detach() view sharing boxes_ts's storage; the cat below used to copy.
                    refpoint_embed_ts, memory_ts, boxes_ts = (
                        refpoint_embed_ts_parts[0],
                        memory_ts_parts[0],
                        boxes_ts_parts[0],
                    )
                else:
                    refpoint_embed_ts = torch.cat(refpoint_embed_ts_parts, dim=1)
                    # (bs, nq, d)
                    memory_ts = torch.cat(memory_ts_parts, dim=1)
                    boxes_ts = torch.cat(boxes_ts_parts, dim=1)
                # This loop discards its own per-group class ranking scores after topk (same as the
                # batched path above) instead of gathering them -- unlike the batched path, this rare
                # fallback (custom/heterogeneous group modules, or group_detr==1) is left as is; the
                # caller re-runs enc_out_class_embed for this case, same as before this change.
                cls_ts = None

        enc_kp_predictions = None
        init_kp_ref_xy = None
        keypoint_memory_ts = None
        if self.two_stage and self.use_grouppose_keypoints and hasattr(self, "keypoint_query_initializer"):
            assert self.enc_out_keypoint_embed is not None
            batch_size, _, _ = memory_ts.shape
            keypoint_memory_ts = self.keypoint_query_initializer_enc(memory_ts)
            boxes_ref = boxes_ts if self.bbox_reparam else boxes_ts.sigmoid()
            group_detr = len(self.enc_out_keypoint_embed) if self.training else 1

            kp_mem_chunks = keypoint_memory_ts.chunk(group_detr, dim=1)
            boxes_chunks = boxes_ref.chunk(group_detr, dim=1)
            kp_pred_chunks = []
            for g_idx in range(group_detr):
                kp_delta = self.enc_out_keypoint_embed[g_idx](kp_mem_chunks[g_idx])
                # Sanitize the full encoder prediction at this model boundary: its channels feed both the
                # shared box-reference multiply and outer classification/matching consumers.
                kp_delta = torch.nan_to_num(kp_delta, nan=0.0, posinf=0.0, neginf=0.0)
                ref_wh = boxes_chunks[g_idx][..., 2:].unsqueeze(-2)
                ref_xy = boxes_chunks[g_idx][..., :2].unsqueeze(-2)
                kp_xy = kp_delta[..., :2] * ref_wh + ref_xy
                kp_pred_chunks.append(torch.cat([kp_xy, kp_delta[..., 2:]], dim=-1))

            enc_kp_predictions = torch.cat(kp_pred_chunks, dim=1)
            init_kp_ref_xy = enc_kp_predictions[..., :2].detach()

        if self.dec_layers > 0:
            # Use memory.shape[0] (symbolic) instead of a Python-int `bs` constant.
            bs = memory.shape[0]
            tgt = query_feat.unsqueeze(0).expand(bs, -1, -1).contiguous()
            refpoint_embed = refpoint_embed.unsqueeze(0).expand(bs, -1, -1).contiguous()
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

                # When every query comes from the two-stage selection, the remainder is empty; concatenating it
                # puts a zero-sized tensor in the exported graph, which CoreML rejects.
                if refpoint_embed_subset.shape[-2] == 0:
                    refpoint_embed = refpoint_embed_ts_subset
                else:
                    refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            # Insert register tokens per group
            original_num_queries_per_group = None
            if self.num_registers > 0:
                group_count = self.group_detr if self.training else 1
                original_num_queries_per_group = tgt.shape[1] // group_count
                reg_tgt = self.register_tokens.unsqueeze(0).expand(bs, -1, -1)
                reg_ref = self.register_ref_points.unsqueeze(0).expand(bs, -1, -1)
                tgt_chunks = list(tgt.split(original_num_queries_per_group, dim=1))  # type: ignore[no-untyped-call]
                ref_chunks = list(
                    refpoint_embed.split(original_num_queries_per_group, dim=1)  # type: ignore[no-untyped-call]
                )
                tgt = torch.cat([torch.cat([chunk, reg_tgt], dim=1) for chunk in tgt_chunks], dim=1)
                refpoint_embed = torch.cat([torch.cat([chunk, reg_ref], dim=1) for chunk in ref_chunks], dim=1)
                if init_kp_ref_xy is not None:
                    num_keypoints = init_kp_ref_xy.shape[2]
                    reg_kp_xy = self.register_ref_points[:, :2].sigmoid()
                    reg_kp_xy = reg_kp_xy.unsqueeze(0).unsqueeze(2).expand(bs, -1, num_keypoints, -1)
                    kp_ref_chunks = list(
                        init_kp_ref_xy.split(original_num_queries_per_group, dim=1)  # type: ignore[no-untyped-call]
                    )
                    init_kp_ref_xy = torch.cat([torch.cat([chunk, reg_kp_xy], dim=1) for chunk in kp_ref_chunks], dim=1)

            tgt_keypoints = None
            if self.use_grouppose_keypoints:
                if not hasattr(self, "keypoint_query_initializer"):
                    raise ValueError("use_grouppose_keypoints=True requires keypoint initializers")
                tgt_keypoints = self.keypoint_query_initializer(tgt)

            # Route memories: kp_only mode keeps main features for detection and
            # second projector memory for keypoint cross-attention.
            if self.dual_projector_kp_only and cross_attn_memory is not None:
                decoder_memory = memory
                kp_cross_attn_memory = cross_attn_memory
            else:
                decoder_memory = cross_attn_memory if cross_attn_memory is not None else memory
                kp_cross_attn_memory = None

            decoder_outputs = self.decoder(
                tgt,
                decoder_memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                spatial_shapes_hw=spatial_shapes_hw,
                valid_ratios=valid_ratios.to(decoder_memory.dtype) if valid_ratios is not None else valid_ratios,
                tgt_keypoints=tgt_keypoints,
                init_kp_ref_xy=init_kp_ref_xy,
                kp_cross_attn_memory=kp_cross_attn_memory,
            )

            if self.use_grouppose_keypoints and len(decoder_outputs) > 2:
                hs, references, keypoint_hs = decoder_outputs[:3]
            else:
                hs, references = decoder_outputs[:2]
                keypoint_hs = None

            # Remove register tokens from decoder outputs.
            if self.num_registers > 0 and original_num_queries_per_group is not None:
                group_count = self.group_detr if self.training else 1
                n_with_reg = hs.shape[2] // group_count
                hs = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in hs.split(n_with_reg, dim=2)],
                    dim=2,
                )
                references = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in references.split(n_with_reg, dim=2)],
                    dim=2,
                )
                if keypoint_hs is not None:
                    keypoint_hs = torch.cat(
                        [c[:, :, :original_num_queries_per_group] for c in keypoint_hs.split(n_with_reg, dim=2)],
                        dim=2,
                    )
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None
            keypoint_hs = None

        return_values = [hs, references]
        if self.two_stage:
            return_values.append(memory_ts)
            if self.bbox_reparam:
                return_values.append(boxes_ts)
            else:
                return_values.append(boxes_ts.sigmoid())
        else:
            return_values.extend([None, None])

        if self.use_grouppose_keypoints:
            return_values.append(keypoint_hs)
            return_values.append(enc_kp_predictions)
            return_values.append(keypoint_memory_ts if self.two_stage else None)

        # Always last: callers that need it grab it by position ([-1] or an explicit slice), so this
        # append must never move without updating every unpacking site (LWDETR.forward, forward_export).
        return_values.append(cls_ts)

        return tuple(return_values)


class TransformerDecoder(nn.Module):
    """Decoder stack used by DETR transformer."""

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
        d_model: int = 256,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
        enable_keypoint_processing: bool = False,
        num_keypoints_per_class: list[int] | None = None,
        grouppose_keypoint_dim_downscale: int = 1,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        self.enable_keypoint_processing = enable_keypoint_processing
        self.num_keypoints_per_class = num_keypoints_per_class
        self.grouppose_keypoint_dim_downscale = grouppose_keypoint_dim_downscale
        # Populated externally (e.g. by LWDETR) when iterative bbox refinement is active.
        # Declared here so that ``hasattr(self, "bbox_embed")`` short-circuits even without an
        # external injection, and so that mypy sees a stable attribute type.
        self.bbox_embed: nn.Module | None = None

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.keypoint_pos_embed = None
        if enable_keypoint_processing and num_keypoints_per_class:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            self.keypoint_pos_embed = nn.Parameter(torch.randn(sum(num_keypoints_per_class), kp_dim))
            self._create_keypoint_class_mask()
        self._export = False

    def export(self) -> None:
        self._export = True

    def _create_keypoint_class_mask(self) -> Tensor:
        """Create attention mask that blocks cross-class keypoint interactions."""
        # NOTE: near-duplicate of LWDETR._create_keypoint_class_mask in models/lwdetr.py (same
        # mask logic; that one is a pure static taking num_keypoints_per_class, this reads
        # self.num_keypoints_per_class and registers the keypoint_class_mask buffer). Keep in sync.
        if not self.num_keypoints_per_class:
            mask = torch.zeros(1, 1, dtype=torch.bool)
        else:
            total_kp = sum(self.num_keypoints_per_class)
            mask = torch.zeros(1 + total_kp, 1 + total_kp, dtype=torch.bool)
            offset = 1
            for class_idx_i, num_kp_i in enumerate(self.num_keypoints_per_class):
                if num_kp_i == 0:
                    continue
                start_i = offset + sum(self.num_keypoints_per_class[:class_idx_i])
                end_i = start_i + num_kp_i
                for class_idx_j, num_kp_j in enumerate(self.num_keypoints_per_class):
                    if num_kp_j == 0 or class_idx_i == class_idx_j:
                        continue
                    start_j = offset + sum(self.num_keypoints_per_class[:class_idx_j])
                    end_j = start_j + num_kp_j
                    mask[start_i:end_i, start_j:end_j] = True

        if "keypoint_class_mask" in self._buffers:
            self._buffers["keypoint_class_mask"] = mask
        else:
            self.register_buffer("keypoint_class_mask", mask, persistent=True)
        return cast(Tensor, self.keypoint_class_mask)

    def refpoints_refine(self, refpoints_unsigmoid: Tensor, new_refpoints_delta: Tensor) -> Tensor:
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            )
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat([new_refpoints_cxcy, new_refpoints_wh], dim=-1)
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        refpoints_unsigmoid: Tensor | None = None,
        # for memory
        level_start_index: Tensor | None = None,  # num_levels
        spatial_shapes: Tensor | None = None,  # num_levels, 2
        spatial_shapes_hw: list[tuple[int, int]] | None = None,  # num_levels (H, W) Python ints
        valid_ratios: Tensor | None = None,
        # keypoints
        tgt_keypoints: Tensor | None = None,
        init_kp_ref_xy: Tensor | None = None,
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        assert refpoints_unsigmoid is not None
        output = tgt

        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        keypoint_tgt = None
        kp_query_pos = None
        intermediate_keypoints = []

        if self.enable_keypoint_processing:
            assert self.lite_refpoint_refine, "Keypoint processing requires lite_refpoint_refine"
            if tgt_keypoints is None:
                raise ValueError("Keypoint processing is enabled but tgt_keypoints was not provided")
            if init_kp_ref_xy is None:
                raise ValueError("Keypoint processing is enabled but init_kp_ref_xy was not provided")
            keypoint_tgt = tgt_keypoints
            assert self.keypoint_pos_embed is not None, "keypoint_pos_embed must be initialized for keypoint processing"
            kp_query_pos = (
                self.keypoint_pos_embed.unsqueeze(0)
                .unsqueeze(0)
                .expand(keypoint_tgt.shape[0], keypoint_tgt.shape[1], -1, -1)
            )

        def get_reference(refpoints: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4]

            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model // 2)  # bs, nq, 256*2
                # "Materialize one shared box to per-level references" happens at different layers per mode:
                # eager (below) multiplies by valid_ratios to produce per-level refs, while export keeps the
                # singleton level dim here and defers expansion to MSDeformAttn's internal .expand(). Export's
                # expand omits eager's valid_ratios scaling — sound only under the no-padding export assumption.
                refpoints_input = obj_center[:, :, None]  # bs, nq, 1, 4
            else:
                assert valid_ratios is not None
                refpoints_input = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                # ``ref_point_head`` and its first Linear are the embedding's only consumers. When their calls are
                # plain, autocast casts an fp32 input inside Linear, so producing it in the compute dtype is bitwise
                # the same. Hooks, overrides, subclasses, compile wrappers and non-fp32 references (autocast leaves
                # fp64 alone) keep the original module input.
                # ``.layers`` is read only after the type check: a replaced head need not have it.
                ref_point_head = self.ref_point_head
                ref_point_head_is_plain = (
                    _eager_cuda(refpoints_input)
                    and type(ref_point_head) is MLP
                    and type(ref_point_head.layers[0]) is nn.Linear
                    and _module_call_is_plain(ref_point_head, ref_point_head.layers[0])
                )
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :],
                    self.d_model // 2,
                    out_dtype=(
                        _cuda_autocast_dtype()
                        if ref_point_head_is_plain and refpoints_input.dtype == torch.float32
                        else None
                    ),
                )

            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(
                        refpoints_unsigmoid.sigmoid()
                    )

            if self.enable_keypoint_processing and keypoint_tgt is not None:
                layer_outputs = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    query_pos=query_pos,
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_hw=spatial_shapes_hw,
                    level_start_index=level_start_index,
                    keypoint_tgt=keypoint_tgt,
                    keypoint_pos=kp_query_pos,
                    keypoint_class_mask=self.keypoint_class_mask,
                    kp_cross_attn_memory=kp_cross_attn_memory,
                )
                output, keypoint_tgt = layer_outputs
                intermediate_keypoints.append(keypoint_tgt)
            else:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    query_pos=query_pos,
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_hw=spatial_shapes_hw,
                    level_start_index=level_start_index,
                )

            if not self.lite_refpoint_refine:
                assert self.bbox_embed is not None
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                assert self.norm is not None
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid

                if self.enable_keypoint_processing:
                    return hs, ref, intermediate_keypoints[-1]
                return hs, ref

            results = []
            if self.bbox_embed is not None:
                results.append(torch.stack(intermediate))
                results.append(torch.stack(hs_refpoints_unsigmoid))
            else:
                results.append(torch.stack(intermediate))
                results.append(refpoints_unsigmoid.unsqueeze(0))

            if self.enable_keypoint_processing:
                results.append(torch.stack(intermediate_keypoints))

            return tuple(results)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    """A single decoder layer with optional keypoint subnetwork."""

    def __init__(
        self,
        d_model: int,
        sa_nhead: int,
        ca_nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        group_detr: int = 1,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        skip_self_attn: bool = False,
        enable_keypoint_processing: bool = False,
        grouppose_keypoint_dim_downscale: int = 1,
        keypoint_cross_attn: bool = True,
        inter_instance_kp_attn: bool = False,
    ) -> None:
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

        self.enable_keypoint_processing = enable_keypoint_processing
        self.inter_instance_kp_attn = inter_instance_kp_attn and enable_keypoint_processing
        self.keypoint_cross_attn = keypoint_cross_attn and enable_keypoint_processing

        if enable_keypoint_processing:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            self.inst_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.inst_pos_in_proj = (
                nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            )
            self.inst_out_proj = nn.Linear(kp_dim, d_model) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.memory_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.kp_inst_self_attn = nn.MultiheadAttention(
                embed_dim=kp_dim,
                num_heads=_safe_multinormalize(sa_nhead // grouppose_keypoint_dim_downscale),
                dropout=dropout,
                batch_first=True,
            )
            self.kp_inst_dropout = nn.Dropout(dropout)
            self.kp_inst_norm = nn.LayerNorm(d_model)
            self.kp_norm = nn.LayerNorm(kp_dim)
            self.kp_dropout = nn.Dropout(dropout)

            if self.inter_instance_kp_attn:
                self.inter_inst_kp_attn = nn.MultiheadAttention(
                    embed_dim=kp_dim,
                    num_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    dropout=dropout,
                    batch_first=True,
                )
                self.inter_inst_kp_dropout = nn.Dropout(dropout)
                self.inter_inst_kp_norm = nn.LayerNorm(kp_dim)

            if self.keypoint_cross_attn:
                self.kp_cross_attn = MSDeformAttn(
                    kp_dim,
                    n_levels=num_feature_levels,
                    n_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    n_points=dec_n_points,
                )
                self.kp_cross_attn_dropout = nn.Dropout(dropout)
                self.kp_cross_attn_norm = nn.LayerNorm(kp_dim)

            self.kp_linear1 = nn.Linear(kp_dim, d_model * 4 // grouppose_keypoint_dim_downscale)
            self.kp_dropout2 = nn.Dropout(dropout)
            self.kp_linear3 = nn.Linear(d_model * 4 // grouppose_keypoint_dim_downscale, kp_dim)
            self.kp_dropout4 = nn.Dropout(dropout)
            self.kp_norm5 = nn.LayerNorm(kp_dim)

            self.instance_kp_layer_scale = nn.Parameter(torch.ones(1) * 1e-6)
        self._export = False

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def _pos_embed_for_linear(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        """:meth:`with_pos_embed` for a sum whose only consumers are matmuls under autocast.

        Autocast casts fp32 matmul inputs to its compute dtype, so a full-precision sum that feeds nothing else is
        written once by the add and read once more by the cast. When CUDA autocast is on and the sum is fp32, the
        training path emits it in the compute dtype directly (:class:`_AddInDtype`, bitwise the same values, one kernel
        and half the bytes). Eval, CPU, fp32 training, non-fp32 sums (autocast leaves fp64 alone), ``torch.compile`` and
        tracing keep the plain add. Both callers feed the sum to two matmuls, so the ``tensor``/``pos`` gradients are
        accumulated in the compute dtype before the upcast (one extra rounding, see :class:`_AddInDtype`); the forward
        and the losses are unchanged.
        """
        # An overridden ``with_pos_embed`` (subclass method or instance attribute) defines the sum; honour it.
        if type(self).with_pos_embed is not TransformerDecoderLayer.with_pos_embed or "with_pos_embed" in self.__dict__:
            return self.with_pos_embed(tensor, pos)
        if pos is None:
            return tensor
        if not (self.training and _eager_cuda(tensor) and tensor.shape == pos.shape):
            return tensor + pos
        dtype = _cuda_autocast_dtype()
        if (
            dtype is None
            or tensor.dtype != torch.float32
            or tensor.dtype == dtype
            or pos.dtype not in (tensor.dtype, dtype)
        ):
            return tensor + pos
        return cast(Tensor, _AddInDtype.apply(tensor, pos, dtype))

    def _ffn_hidden(self, tgt: Tensor) -> Tensor:
        """``self.activation(self.linear1(tgt))`` with the ReLU folded into the GEMM epilogue on CUDA.

        :class:`_LinearReLU` is bitwise the two-op result, so it is taken whenever ``linear1`` is the plain
        ``nn.Linear`` the layer builds (exact type, a bias, weight and bias that are exact ``nn.Parameter`` tensors so a
        tensor subclass keeps ``F.linear``'s dispatch, a call nothing overrides or observes, see
        :func:`_module_call_is_plain`) and the activation is ReLU, in eager CUDA execution; other activations, CPU,
        ``torch.compile`` and tracing keep the two ops. Under CUDA autocast fp32 operands are cast to the compute dtype
        first, exactly as autocast casts them for ``linear``; other operand dtypes (autocast leaves fp64 alone) keep the
        two ops.
        """
        linear1 = self.linear1
        if (
            self.activation is F.relu
            and _eager_cuda(tgt)
            and type(linear1) is nn.Linear
            and type(linear1.weight) is nn.Parameter
            and type(linear1.bias) is nn.Parameter
            and _module_call_is_plain(linear1)
        ):
            dtype = _cuda_autocast_dtype()
            if dtype is None:
                if tgt.dtype == linear1.weight.dtype:
                    return cast(Tensor, _LinearReLU.apply(tgt, linear1.weight, linear1.bias))
            elif tgt.dtype == linear1.weight.dtype == linear1.bias.dtype == torch.float32:
                with torch.autocast("cuda", enabled=False):
                    return cast(
                        Tensor,
                        _LinearReLU.apply(tgt.to(dtype), linear1.weight.to(dtype), linear1.bias.to(dtype)),
                    )
        return self.activation(linear1(tgt))

    def _grouped_self_attention_eligible(
        self, tgt: Tensor, tgt_mask: Tensor | None, tgt_key_padding_mask: Tensor | None
    ) -> bool:
        """Return whether :meth:`_grouped_self_attention` can stand in for the ``self_attn`` module call.

        The explicit path reads ``in_proj_weight``/``in_proj_bias``/``out_proj`` and never calls the module, so it is
        limited to the plain ``nn.MultiheadAttention`` ``__init__`` builds (exact type, a call nothing overrides or
        observes, see :func:`_module_call_is_plain`, packed projections with bias held in exact ``nn.Parameter`` tensors
        so a tensor subclass keeps the module call, no ``bias_k``/``bias_v``/``add_zero_attn``), to training on CUDA
        with zero attention dropout and without masks (the only configuration that regroups without changing seeded
        dropout masks), and to eager execution: under ``torch.compile`` or tracing the module call is what Inductor and
        the exporters expect.
        """
        attn = self.self_attn
        if type(attn) is not nn.MultiheadAttention:
            return False
        if not self.training or tgt_mask is not None or tgt_key_padding_mask is not None:
            return False
        if not _eager_cuda(tgt) or tgt.dim() != 3 or tgt.shape[1] % self.group_detr != 0:
            return False
        if not _module_call_is_plain(attn):
            return False
        return bool(
            attn.batch_first
            and attn._qkv_same_embed_dim
            and type(attn.in_proj_weight) is nn.Parameter
            and type(attn.in_proj_bias) is nn.Parameter
            and attn.bias_k is None
            and attn.bias_v is None
            and not attn.add_zero_attn
            and attn.dropout == 0.0
        )

    def _grouped_self_attention(self, tgt: Tensor, query_pos: Tensor | None) -> Tensor:
        """Grouped self-attention with the projections on the ungrouped layout and the regrouping as views.

        Computes what ``self.self_attn(q, k, v)`` computes on ``torch.cat(x.split(queries_per_group, 1), 0)``,
        with the same kernels: the per-token projections commute with the regrouping, and attention is
        independent per (batch, group) pair, so the rows are bitwise the same whether the flattened batch is
        group-major (the module path) or batch-major (a view of the ``[batch, groups * queries_per_group]``
        layout, in both directions). What disappears is the layout work: the three ``cat``s and their
        backward copies, ``MultiheadAttention``'s ``batch_first`` transposes (which route every in-projection
        through ``matmul`` on a non-contiguous tensor: a copy per projection) and, under autocast, one of the
        two casts of the shared query/key input.

        Eligibility requires zero attention dropout because a nonzero mask would be drawn in batch-major rather than
        the module path's group-major order.
        """
        attn = self.self_attn
        bs, num_queries, d_model = tgt.shape
        groups = self.group_detr
        heads = attn.num_heads
        head_dim = d_model // heads
        weight, bias = attn.in_proj_weight, attn.in_proj_bias
        qk_input = self._pos_embed_for_linear(tgt, query_pos)

        def project(x: Tensor, start: int) -> Tensor:
            # ``mm`` then ``add_`` is what the module path runs (``F.linear`` on its transposed inputs takes the
            # matmul route), kept so the projections stay bitwise identical to it.
            out = torch.mm(x.reshape(bs * num_queries, d_model), weight[start : start + d_model].t())
            out = out.add_(bias[start : start + d_model].to(out.dtype))
            return out.view(bs * groups, num_queries // groups, heads, head_dim).transpose(1, 2)

        q = project(qk_input, 0)
        k = project(qk_input, d_model)
        v = project(tgt, 2 * d_model)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn.dropout)  # eligibility implies training
        out = out.transpose(1, 2).reshape(bs, num_queries, d_model)
        return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
        # Keypoint processing parameters
        keypoint_tgt: Tensor | None = None,  # [B, N, total_kp_per_instance, C]
        keypoint_pos: Tensor | None = None,  # [B, N, total_kp_per_instance, C]
        keypoint_class_mask: Tensor | None = None,  # [1 + K, 1 + K]
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        bs, num_queries, _ = tgt.shape

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        if self._grouped_self_attention_eligible(tgt, tgt_mask, tgt_key_padding_mask):
            tgt2 = self._grouped_self_attention(tgt, query_pos)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            v = tgt
            if self.training:
                q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]
                k = q
                v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]

            tgt2 = self.self_attn(
                q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False
            )[0]

            if self.training:
                tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ========== End of Self-Attention =============

        # ========== Begin of Cross-Attention =============
        # A plain MSDeformAttn only reads ``query`` through its two Linear heads, which autocast would cast anyway, so
        # under autocast the positional add can be emitted in the compute dtype directly. Anything that observes or
        # overrides those three calls keeps the original fp32 query input.
        cross_attn = self.cross_attn
        cross_attn_query_is_plain = (
            self.training
            and _eager_cuda(tgt)
            and type(cross_attn) is MSDeformAttn
            and type(cross_attn.sampling_offsets) is nn.Linear
            and type(cross_attn.attention_weights) is nn.Linear
            and _module_call_is_plain(cross_attn, cross_attn.sampling_offsets, cross_attn.attention_weights)
        )
        cross_attn_query = (
            self._pos_embed_for_linear(tgt, query_pos)
            if cross_attn_query_is_plain
            else self.with_pos_embed(tgt, query_pos)
        )
        tgt2 = self.cross_attn(
            cross_attn_query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self._ffn_hidden(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.enable_keypoint_processing:
            if keypoint_tgt is None or keypoint_pos is None:
                raise ValueError("Keypoint processing is enabled but keypoint_tgt/keypoint_pos missing")
            if reference_points is None:
                raise ValueError("Keypoint processing is enabled but reference_points missing")

            tgt_for_kp = self.inst_in_proj(tgt)
            tgt_for_kp_pos = self.inst_pos_in_proj(query_pos)

            # ========== Begin of Keypoint-Instance Self-Attention =============
            _, n_queries, num_kp, kp_dim = keypoint_tgt.shape

            tgt_expanded = tgt_for_kp.unsqueeze(2)  # [B, N, 1, C]
            query_expanded = torch.zeros_like(tgt_for_kp).unsqueeze(2)  # [B, N, 1, C]

            combined_feat = torch.cat([tgt_expanded, keypoint_tgt], dim=2)  # [B, N, 1 + K, C]
            combined_pos = torch.cat([query_expanded, keypoint_pos], dim=2)  # [B, N, 1 + K, C]

            combined_feat = combined_feat.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            combined_pos = combined_pos.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            q = k = combined_feat + combined_pos
            v = combined_feat

            combined_out = self.kp_inst_self_attn(
                q, k, v, attn_mask=_additive_attn_mask(keypoint_class_mask, q.dtype), need_weights=False
            )[0]
            combined_out = combined_out.reshape(bs, num_queries, 1 + num_kp, kp_dim)
            tgt2 = combined_out[:, :, 0, :]
            keypoint_tgt2 = combined_out[:, :, 1:, :]

            tgt = tgt + self.kp_inst_dropout(self.inst_out_proj(tgt2)) * self.instance_kp_layer_scale
            tgt = self.kp_inst_norm(tgt)
            keypoint_tgt = keypoint_tgt + self.kp_dropout(keypoint_tgt2)
            keypoint_tgt = self.kp_norm(keypoint_tgt)

            # ========== End of Keypoint-Instance Self-Attention =============

            # ========== Begin of Cross-Keypoint Attention =============
            if self.inter_instance_kp_attn:
                swapped_keypoint_tgt = keypoint_tgt.transpose(1, 2).reshape(bs * num_kp, num_queries, kp_dim)
                swapped_keypoint_pos = (
                    tgt_for_kp_pos.unsqueeze(1)
                    .expand(bs, num_kp, num_queries, kp_dim)
                    .reshape(
                        bs * num_kp,
                        num_queries,
                        kp_dim,
                    )
                )
                q = swapped_keypoint_tgt + swapped_keypoint_pos
                v = swapped_keypoint_tgt
                swapped_out = self.inter_inst_kp_attn(q, key=q, value=v, need_weights=False)[0]
                swapped_out = swapped_out.view(bs, num_kp, num_queries, kp_dim).transpose(1, 2)
                keypoint_tgt = keypoint_tgt + self.inter_inst_kp_dropout(swapped_out)
                keypoint_tgt = self.inter_inst_kp_norm(keypoint_tgt)

            # ========== End of Cross-Keypoint Attention =============

            # ========== Begin of Keypoint-Specific Cross-Attention =============
            if self.keypoint_cross_attn:
                keypoint_query = self.with_pos_embed(
                    keypoint_tgt, tgt_for_kp_pos.unsqueeze(2).expand(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_query = keypoint_query.reshape(bs, num_queries * num_kp, kp_dim)
                bbox_ref_for_kp = (
                    reference_points.unsqueeze(2)
                    .expand(
                        bs,
                        num_queries,
                        num_kp,
                        reference_points.shape[2],
                        reference_points.shape[3],
                    )
                    .reshape(bs, num_queries * num_kp, reference_points.shape[2], reference_points.shape[3])
                )
                kp_memory = kp_cross_attn_memory if kp_cross_attn_memory is not None else memory
                keypoint_tgt = keypoint_tgt + self.kp_cross_attn_dropout(
                    self.kp_cross_attn(
                        keypoint_query,
                        bbox_ref_for_kp,
                        self.memory_in_proj(kp_memory),
                        spatial_shapes,
                        level_start_index,
                        memory_key_padding_mask,
                        input_spatial_shapes_hw=spatial_shapes_hw,
                    ).reshape(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_tgt = self.kp_cross_attn_norm(keypoint_tgt)

            # ========== End of Keypoint-Specific Cross-Attention =============

            # ========== Begin of Keypoint-Specific FFN =============
            keypoint_tgt = keypoint_tgt + self.kp_dropout4(
                self.kp_linear3(self.kp_dropout2(self.activation(self.kp_linear1(keypoint_tgt))))
            )
            keypoint_tgt = self.kp_norm5(keypoint_tgt)
            # ========== End of Keypoint-Specific FFN =============

            return tgt, keypoint_tgt

        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
        keypoint_tgt: Tensor | None = None,
        keypoint_pos: Tensor | None = None,
        keypoint_class_mask: Tensor | None = None,
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        return self.forward_post(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_hw=spatial_shapes_hw,
            level_start_index=level_start_index,
            keypoint_tgt=keypoint_tgt,
            keypoint_pos=keypoint_pos,
            keypoint_class_mask=keypoint_class_mask,
            kp_cross_attn_memory=kp_cross_attn_memory,
        )


def _get_clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_clones)])


def build_transformer(args: BuilderArgs) -> Transformer:
    two_stage = getattr(args, "two_stage", False)

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,  # type: ignore[attr-defined]
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,  # type: ignore[attr-defined]
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,  # type: ignore[attr-defined]
        bbox_reparam=args.bbox_reparam,
        # Detection-only builder args may omit keypoint-only fields; default to the non-keypoint path.
        use_grouppose_keypoints=getattr(args, "use_grouppose_keypoints", False),
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        grouppose_keypoint_dim_downscale=getattr(args, "grouppose_keypoint_dim_downscale", 1),
        keypoint_cross_attn=getattr(args, "keypoint_cross_attn", True),
        inter_instance_kp_attn=getattr(args, "inter_instance_kp_attn", False),
        num_registers=getattr(args, "num_decoder_registers", 0),
        dual_projector_kp_only=getattr(args, "dual_projector_kp_only", False),
    )


#: Functional activation per name accepted by the transformer layers.
_ACTIVATION_FNS: dict[str, Callable[[Tensor], Tensor]] = {"relu": F.relu, "gelu": F.gelu, "glu": F.glu}


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string."""
    try:
        return _ACTIVATION_FNS[activation]
    except KeyError:
        raise RuntimeError(f"activation should be one of {tuple(_ACTIVATION_FNS)}, not {activation}.") from None
