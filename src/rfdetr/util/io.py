# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Safe checkpoint loading helpers.

This module provides :func:`_safe_torch_load`, a defense-in-depth wrapper around :func:`torch.load` that prevents
pickle-based remote code execution (CWE-502) when loading checkpoints from external or user-supplied sources.
"""

from __future__ import annotations

import argparse
import pickle
import types
import warnings
from pathlib import Path
from typing import Any

import torch

__all__ = ["_safe_torch_load"]


def _safe_torch_load(path: str | Path, *, trust: bool = False) -> Any:
    """Load a PyTorch checkpoint as safely as possible.

    Tries progressively less restrictive deserialization strategies:

    1. ``weights_only=True`` (strict — only tensors and a small set of
       built-in scalars).
    2. Same as 1, but with ``argparse.Namespace`` and
       ``types.SimpleNamespace`` registered as safe globals so that legacy
       RF-DETR checkpoints that embed an ``args`` namespace can be loaded
       without falling back to pickle.
    3. ``weights_only=False`` (full pickle) — allowed **only** when
       ``trust=True``, with a loud :class:`UserWarning`.  Never used for
       checkpoints received from external sources.

    Args:
        path: Path to the checkpoint file.
        trust: When ``True``, allow pickle deserialization as a last-resort
            fallback and emit a :class:`UserWarning`.  Set this only for
            checkpoint files produced by RF-DETR itself (e.g. during legacy
            checkpoint conversion) that may contain non-tensor Python
            objects that are not covered by the safe-globals list.

    Returns:
        The loaded checkpoint (usually a :class:`dict`).

    Raises:
        RuntimeError: When all safe loading strategies fail and
            ``trust=False``.  The error message suggests passing
            ``trust_checkpoint=True`` so the caller can make an informed
            decision.

    Examples:
        >>> import torch, tempfile, os
        >>> with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fh:
        ...     path = fh.name
        >>> torch.save({"model": {"weight": torch.tensor([1.0])}}, path)
        >>> ckpt = _safe_torch_load(path)
        >>> list(ckpt.keys())
        ['model']
        >>> os.unlink(path)
    """
    # ── Attempt 1: strict safe load ──────────────────────────────────────
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, Exception):
        pass

    # ── Attempt 2: add well-known legacy safe globals ─────────────────────
    # argparse.Namespace and types.SimpleNamespace appear in RF-DETR checkpoints
    # saved by the pre-PTL engine.py training loop.  Registering them as safe
    # globals is semantically safe because they contain only primitive values.
    try:
        torch.serialization.add_safe_globals([argparse.Namespace, types.SimpleNamespace])
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, Exception):
        pass

    # ── Attempt 3 (opt-in): full pickle ───────────────────────────────────
    if trust:
        warnings.warn(
            f"Loading checkpoint {str(path)!r} with weights_only=False. "
            "This allows arbitrary Python objects to be deserialized from the "
            "checkpoint file, which can execute malicious code if the file "
            "comes from an untrusted source. "
            "Only use trust=True for checkpoint files produced by RF-DETR itself.",
            UserWarning,
            stacklevel=3,
        )
        return torch.load(path, map_location="cpu", weights_only=False)

    raise RuntimeError(
        f"Failed to safely load checkpoint {str(path)!r}. "
        "The file likely contains custom Python objects that cannot be "
        "deserialized with weights_only=True. "
        "If you fully trust this checkpoint source, pass trust_checkpoint=True "
        "to the loading function."
    )
