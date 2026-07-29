# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared output-filename stem resolution for the export backends.

Every backend (ONNX, CoreML, ExecuTorch, TensorRT) names its output artifact from either a user-supplied full override
(``output_name``) or a model variant identifier (``variant_name``), falling back to a generic default when neither is
given. This module centralizes that precedence + sanitization so the five backends stay consistent instead of re-
implementing it.
"""

from __future__ import annotations

import os


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
        return _sanitize(output_name), True
    if variant_name:
        return _sanitize(variant_name), False
    return default, False


def _sanitize(name: str) -> str:
    """Strip directory components and extension from a caller-supplied name."""
    return os.path.splitext(os.path.basename(name))[0]
