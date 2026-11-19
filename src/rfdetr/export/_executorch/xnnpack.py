# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Graph preparation that keeps RF-DETR's attention inside the ExecuTorch XNNPACK delegate.

ExecuTorch decomposes ``aten.scaled_dot_product_attention`` into a softmax with guards for fully masked rows
(``eq``/``any``/``logical_not``/``where``), scalar multiplications and broadcast copies. XNNPACK has no kernels for
these, so each attention call splits the graph into several XNNPACK partitions with portable kernels between them.
:func:`decompose_attention` rewrites unmasked attention in the captured program as two batched matrix multiplications
and a softmax, which XNNPACK runs. Without a mask, no row can be fully masked, so for finite inputs the guards never
change a value.

``nn.MultiheadAttention`` slices its packed ``in_proj_weight`` at run time, and XNNPACK takes a linear layer only when
its weight is a constant. :func:`fold_constants` computes these slices at export time, together with every other part of
the program that depends only on weights and constants, except ``aten.full``.

Both functions rewrite the captured program only and leave the model itself unchanged.
"""

from __future__ import annotations

import math
from types import NotImplementedType

import torch
from torch import Tensor

#: Targets that :func:`fold_constants` keeps in the program. ``constant_prop_pass`` skips ``aten.full`` by default
#: because folding it can grow the model a lot (see the comment above its default skip set in ExecuTorch), but it
#: names the skipped ops in the Edge dialect, which never equal the ATen targets of the captured program.
_UNFOLDED_TARGETS = {torch.ops.aten.full.default}


def unmasked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor | NotImplementedType:
    """Scaled dot-product attention as ``softmax(query * scale @ key^T) @ value`` when there is no mask.

    This is a decomposition of ``aten.scaled_dot_product_attention`` for float32 tensors that hold at least one
    element and share their leading dimensions. Any other call returns ``NotImplemented``: a mask, causal masking,
    dropout, grouped-query attention, leading dimensions that differ (the operator broadcasts them), an empty tensor or
    another dtype. The operator then keeps its place in the program, and ExecuTorch applies its default decomposition
    when it lowers the program. The decomposition scales and normalizes in the dtype of its inputs, which makes float16
    and bfloat16 less accurate than the operator, so they keep the default.

    Args:
        query: Query of shape ``(..., L, E)``.
        key: Key of shape ``(..., S, E)``.
        value: Value of shape ``(..., S, Ev)``.
        attn_mask: Attention mask. Not ``None`` keeps the default decomposition.
        dropout_p: Dropout probability. Not ``0`` keeps the default decomposition.
        is_causal: ``True`` keeps the default decomposition.
        scale: Factor for the attention scores. ``None`` uses ``1 / sqrt(E)``, as the operator does.
        enable_gqa: ``True`` keeps the default decomposition.

    Returns:
        Attention output of shape ``(..., L, Ev)``, or ``NotImplemented``.

    Examples:
        >>> q, k, v = torch.randn(3, 2, 4, 5, 8).unbind(0)
        >>> expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        >>> torch.allclose(unmasked_attention(q, k, v), expected, atol=1e-6)
        True
        >>> unmasked_attention(q, k, v, is_causal=True) is NotImplemented
        True
        >>> unmasked_attention(q, k[:1], v[:1]) is NotImplemented
        True
        >>> unmasked_attention(q.half(), k.half(), v.half()) is NotImplemented
        True
    """
    if attn_mask is not None or is_causal or dropout_p != 0.0 or enable_gqa or not _is_decomposable(query, key, value):
        return NotImplemented  # type: ignore[no-any-return]  # typeshed types NotImplemented as Any
    *batch, length, channels = query.shape
    memory_order = _memory_order(query)
    scale = 1 / math.sqrt(channels) if scale is None else scale
    # 3-D tensors lower to XNNPACK batch matrix multiplications without the broadcast copies of a 4-D matmul.
    query = query.reshape(-1, length, channels) * scale
    key = key.reshape(-1, key.shape[-2], channels)
    value = value.reshape(-1, value.shape[-2], value.shape[-1])
    weights = torch.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)
    output = torch.bmm(weights, value).reshape(*batch, length, value.shape[-1])
    if memory_order == sorted(memory_order):
        return output
    # The operator returns its output in the memory order of the query, and the captured graph reshapes the output
    # as views of that memory, so the decomposition returns the same layout. RF-DETR gives query, key and value one
    # layout; the output layout of a call that mixes layouts is not checked.
    restore = sorted(range(len(memory_order)), key=memory_order.__getitem__)
    return output.permute(memory_order).contiguous().permute(restore)


def decompose_attention(exported_program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Rewrite the unmasked attention of *exported_program* with :func:`unmasked_attention`.

    Only the calls for which :func:`unmasked_attention` returns a tensor change; every other attention call stays in the
    program as ``aten.scaled_dot_product_attention``.

    Args:
        exported_program: The captured program.

    Returns:
        The program with each unmasked ``aten.scaled_dot_product_attention`` call decomposed.
    """
    # A table with this one entry decomposes nothing else: `run_decompositions` applies only the table it is given,
    # and an operator whose entry returns NotImplemented keeps its place in the program.
    decomposed: torch.export.ExportedProgram = exported_program.run_decompositions(
        {torch.ops.aten.scaled_dot_product_attention.default: unmasked_attention}
    )
    return decomposed


def fold_constants(exported_program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Compute the parts of *exported_program* that depend only on weights and constants, at export time.

    The program is changed in place and returned. ``aten.full`` calls are not folded.

    Args:
        exported_program: The captured program, which this function changes.

    Returns:
        The program with these parts replaced by constants.

    Raises:
        ImportError: If this ExecuTorch install ships without ``executorch.exir.passes.constant_prop_pass``.
    """
    try:
        from executorch.exir.passes.constant_prop_pass import constant_prop_pass
    except ImportError as exc:
        raise ImportError(
            "constant_prop_pass is unavailable in this ExecuTorch install; upgrade executorch to a version "
            "shipping executorch.exir.passes.constant_prop_pass."
        ) from exc

    exported_program = constant_prop_pass(exported_program, custom_skip_targets=_UNFOLDED_TARGETS)
    # A folded slice (such as one third of the packed in_proj_weight) is a view at an offset into the storage of the
    # full tensor. ExecuTorch's emitter copies a view, but the XNNPACK delegate writes a constant from the start of its
    # whole storage (`get_serialized_buffer_index` in executorch/backends/xnnpack/operators/node_visitor.py, ExecuTorch
    # 1.3.1), so the .pte would hold the packed weight instead of the slice. A copy of each such constant gets its own
    # storage.
    for name, constant in exported_program.constants.items():
        if isinstance(constant, Tensor) and _shares_storage(constant):
            exported_program.constants[name] = constant.detach().clone()
    return exported_program


def _is_decomposable(query: Tensor, key: Tensor, value: Tensor) -> bool:
    """Tell whether :func:`unmasked_attention` handles these tensors.

    The decomposition flattens the leading dimensions of each tensor and multiplies them with ``bmm``, which needs
    leading dimensions that are equal, whereas the operator broadcasts them. It also needs an element in each tensor
    to reshape, and float32 to match the accuracy of the operator.

    Args:
        query: Query of shape ``(..., L, E)``.
        key: Key of shape ``(..., S, E)``.
        value: Value of shape ``(..., S, Ev)``.

    Returns:
        ``True`` for float32 tensors with equal leading dimensions and at least one element each.

    Examples:
        >>> q = torch.zeros(2, 3, 4)
        >>> _is_decomposable(q, q, q), _is_decomposable(q, q[:1], q[:1]), _is_decomposable(q, q[:, :0], q[:, :0])
        (True, False, False)
        >>> _is_decomposable(q.half(), q.half(), q.half())
        False
    """
    if not query.dtype == key.dtype == value.dtype == torch.float32:
        return False
    if min(query.numel(), key.numel(), value.numel()) == 0:
        return False
    return query.shape[:-2] == key.shape[:-2] == value.shape[:-2]


def _memory_order(tensor: Tensor) -> list[int]:
    """Order the dimensions of *tensor* from the outermost to the innermost in memory.

    Args:
        tensor: The tensor to inspect.

    Returns:
        Dimension indices, the dimension with the largest stride first.

    Examples:
        >>> _memory_order(torch.zeros(2, 3, 4)), _memory_order(torch.zeros(2, 4, 3).transpose(1, 2))
        ([0, 1, 2], [0, 2, 1])
    """
    return sorted(range(tensor.dim()), key=lambda dimension: (-tensor.stride(dimension), dimension))


def _shares_storage(tensor: Tensor) -> bool:
    """Tell whether *tensor* is a view into a larger storage.

    Args:
        tensor: The tensor to inspect.

    Returns:
        ``True`` if the storage starts before *tensor* or holds more than its elements.

    Examples:
        >>> packed = torch.zeros(3, 4)
        >>> _shares_storage(packed), _shares_storage(packed[1]), _shares_storage(packed[1].clone())
        (False, True, False)
    """
    return bool(
        tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
    )
