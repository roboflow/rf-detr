# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Version-compatible access to torch's compile, trace and CUDA autocast state predicates."""

from __future__ import annotations

import torch

__all__ = ["cuda_autocast_dtype", "is_compiling", "is_tracing"]


def is_compiling() -> bool:
    """Return whether the current execution is inside a ``torch.compile`` graph.

    PyTorch 2.3 added the public ``torch.compiler.is_compiling`` predicate. RF-DETR supports
    PyTorch 2.2, where the equivalent Dynamo predicate remains the compatible fallback. The
    public name is looked up on every call rather than bound once at import: that keeps
    ``torch._dynamo`` untouched wherever the public predicate exists, and lets a test patch
    either torch predicate to drive the compile-only branches on CPU without a real compile.

    Returns:
        Whether Dynamo is compiling the current code path.

    Examples:
        >>> is_compiling()
        False
    """
    predicate = getattr(torch.compiler, "is_compiling", None)
    return predicate() if predicate is not None else torch._dynamo.is_compiling()


def is_tracing() -> bool:
    """Return whether a ``torch.jit.trace`` (ONNX/TorchScript export) is recording the current call.

    Returns:
        Whether a JIT trace is recording the current code path.

    Examples:
        >>> is_tracing()
        False
    """
    return bool(torch.jit.is_tracing())  # type: ignore[attr-defined,no-untyped-call]


def cuda_autocast_dtype() -> torch.dtype | None:
    """Return the CUDA autocast compute dtype, or ``None`` when CUDA autocast is off.

    Works on every supported torch: ``is_autocast_enabled`` takes no device argument on 2.2, and
    ``get_autocast_dtype`` replaced ``get_autocast_gpu_dtype`` later.

    Returns:
        The dtype CUDA autocast casts eligible inputs to, or ``None`` when it is disabled.

    Examples:
        >>> cuda_autocast_dtype() is None
        True
    """
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch 2.2 accepts no device argument.
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return None
    get_dtype = getattr(torch, "get_autocast_dtype", None)
    return get_dtype("cuda") if get_dtype is not None else torch.get_autocast_gpu_dtype()
