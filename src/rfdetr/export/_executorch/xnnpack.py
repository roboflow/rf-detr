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
and a softmax, which XNNPACK runs. Without a mask, no row can be fully masked, so the guards never change a value.

``nn.MultiheadAttention`` slices its packed ``in_proj_weight`` at run time, and XNNPACK takes a linear layer only when
its weight is a constant. :func:`fold_constants` computes these slices at export time.

Both functions change only the captured program, so the model and a concurrent ``predict()`` are not affected.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def unmasked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor | Any:
    """Scaled dot-product attention as ``softmax(query * scale @ key^T) @ value`` when there is no mask.

    This is a decomposition of ``aten.scaled_dot_product_attention``: for a mask, causal masking, dropout or
    grouped-query attention it returns ``NotImplemented``, and the operator keeps its default decomposition.

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
    """
    if attn_mask is not None or is_causal or dropout_p != 0.0 or enable_gqa:
        return NotImplemented
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
    # as views of that memory, so the decomposition returns the same layout.
    restore = sorted(range(len(memory_order)), key=memory_order.__getitem__)
    return output.permute(memory_order).contiguous().permute(restore)


def decompose_attention(exported_program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Rewrite the unmasked attention of *exported_program* with :func:`unmasked_attention`.

    Args:
        exported_program: The captured program.

    Returns:
        The program with each unmasked ``aten.scaled_dot_product_attention`` call decomposed.
    """
    decomposed: torch.export.ExportedProgram = exported_program.run_decompositions(
        {torch.ops.aten.scaled_dot_product_attention.default: unmasked_attention}
    )
    return decomposed


def fold_constants(exported_program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Compute the parts of *exported_program* that depend only on weights and constants, at export time.

    Args:
        exported_program: The captured program.

    Returns:
        The program with these parts replaced by constants.
    """
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    exported_program = constant_prop_pass(exported_program)
    # A folded slice (such as one third of the packed in_proj_weight) is a view at an offset into the storage of
    # the full tensor, and the .pte serializer writes a constant from the start of its storage. A copy of each
    # such constant gets its own storage, so the serializer writes the slice.
    for name, constant in exported_program.constants.items():
        if isinstance(constant, Tensor) and _shares_storage(constant):
            exported_program.constants[name] = constant.detach().clone()
    return exported_program


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
