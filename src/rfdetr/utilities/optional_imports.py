# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers for importing optional third-party dependencies with friendly errors."""

from types import ModuleType
from typing import cast


def import_supervision() -> ModuleType:
    """Import the optional ``supervision`` package, raising a friendly hint if it is missing.

    ``supervision`` is an optional dependency: it is required for the ``Detections`` return type of
    inference helpers and for annotation/visualization utilities, but is not installed by the core
    ``rfdetr`` package. This helper defers the import to call time so the rest of the package remains
    usable without it, and turns a bare ``ModuleNotFoundError`` into an actionable installation hint.

    Returns:
        The imported ``supervision`` module.

    Raises:
        ImportError: If ``supervision`` is not installed.
    """
    try:
        import supervision as sv
    except ImportError as exc:
        raise ImportError(
            "This feature requires the 'supervision' package. Install it with "
            "`pip install supervision` (also bundled in the rfdetr[onnx], rfdetr[train], "
            "and rfdetr[visual] extras)."
        ) from exc
    return cast(ModuleType, sv)
