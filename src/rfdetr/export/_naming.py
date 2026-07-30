# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared output-filename stem resolution for the export backends.

Three of the five export backends (ONNX, CoreML, ExecuTorch) resolve their output-artifact stem entirely through
:func:`resolve_export_stem`, choosing between a user-supplied full override (``output_name``), a model variant
identifier (``variant_name``), and a generic default, with consistent path-traversal sanitization. The other two differ:
TFLite takes no ``output_name`` at all (it inherits its stem from the already-resolved ONNX filename), and TensorRT
delegates only the ``output_name`` sanitize step here while still building its own engine path and ``_fp16``/``_fp32``
precision suffix. Centralizing the precedence + sanitization keeps the backends that share it consistent instead of each
re-implementing it.
"""

from __future__ import annotations

import os
import re


def resolve_export_stem(
    variant_name: str | None,
    output_name: str | None,
    *,
    default: str = "inference_model",
) -> tuple[str, bool]:
    """Resolve the output filename stem for an export backend.

    Precedence: ``output_name`` (verbatim custom filename) wins over ``variant_name``
    (model identifier, e.g. ``"rfdetr-small"``) wins over *default*. Both inputs are
    sanitized against path traversal by keeping only the basename and stripping any
    extension the caller may have included (e.g. ``"rfdetr-small.onnx"`` -> ``"rfdetr-small"``).

    Args:
        variant_name: Model variant identifier, or ``None``.
        output_name: User-supplied full filename override, or ``None``. When set, callers
            should suppress any load-bearing detail suffix (precision/backend/etc.) they
            would otherwise append, since the caller asked for this exact name.
        default: Stem to use when both *variant_name* and *output_name* are ``None``.

    Returns:
        A ``(stem, is_custom)`` tuple. ``is_custom`` is ``True`` only when *output_name*
        was used, signalling that detail suffixes should be omitted.

    Raises:
        ValueError: If *output_name* sanitizes to an empty filename stem.

    Examples:
        >>> resolve_export_stem("rfdetr-small", None)
        ('rfdetr-small', False)
        >>> resolve_export_stem("rfdetr-small", "my-model")
        ('my-model', True)
        >>> resolve_export_stem(None, None)
        ('inference_model', False)
        >>> resolve_export_stem(None, None, default="backbone_model")
        ('backbone_model', False)
        >>> resolve_export_stem("sub/dir/rfdetr-nano.onnx", None)
        ('rfdetr-nano', False)
    """
    if output_name:
        stem = _sanitize(output_name)
        if not stem:
            raise ValueError("output_name must resolve to a non-empty filename stem")
        return stem, True
    if variant_name:
        return _sanitize(variant_name), False
    return default, False


def _sanitize(name: str) -> str:
    """Strip directory components and extension from a caller-supplied name."""
    return os.path.splitext(re.split(r"[\\/]", name)[-1])[0]
