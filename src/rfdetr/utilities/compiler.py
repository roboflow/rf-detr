# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Version-compatible access to the ``torch.compiler`` state predicates."""

from __future__ import annotations

import torch

__all__ = ["is_compiling"]


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
