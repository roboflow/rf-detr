# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Version-compatible access to torch's compile, trace and CUDA autocast state predicates."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import torch

from rfdetr.utilities.logger import get_logger

__all__ = ["cuda_autocast_dtype", "is_compiling", "is_tracing", "script_or_eager"]

logger = get_logger()


def script_or_eager(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Compile a function with TorchScript where the interpreter allows it, else return it unchanged.

    Python 3.14 makes TorchScript unsupported, so the eager function is returned without trying. Earlier
    versions keep the historical scripted function, but fall back to the eager one when compilation raises:
    hosts that replace ``warnings.warn`` (Databricks notebooks do) break TorchScript's identity-keyed builtin
    lookup, so scripting anything that reaches a torch warning fails, and doing that at import time would
    otherwise make ``import rfdetr`` fail. Both forms compute the same result.

    Args:
        fn: Function to compile; it must also be valid TorchScript source.

    Returns:
        The TorchScript function, or ``fn`` itself when scripting is unsupported or failed.

    Examples:
        >>> from rfdetr.utilities.box_ops import batch_dice_loss
        >>> script_or_eager(batch_dice_loss)(torch.zeros(1, 4), torch.zeros(1, 4)).shape
        torch.Size([1, 1])
    """
    if sys.version_info >= (3, 14):
        return fn
    # Import-time boundary: scripting a loss that reaches a replaced ``warnings.warn`` raises ``RuntimeError``,
    # ``OSError`` (no retrievable source), ``TypeError`` (``functools.wraps`` wrapper) or ``SyntaxError`` (lambda
    # wrapper). Eager is the documented equivalent, so any failure falls back instead of failing ``import rfdetr``.
    try:
        scripted: Callable[..., Any] = torch.jit.script(fn)
    except Exception as error:
        logger.debug("TorchScript compilation of %s failed, using the eager function: %s", fn.__qualname__, error)
        return fn
    return scripted


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
