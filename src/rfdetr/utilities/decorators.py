# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecation utilities and decorators."""

import warnings

from deprecate import deprecated, void

__all__ = ["deprecated", "void"]


def _warn_deprecated_module(old: str, new: str) -> None:
    """Emit a DeprecationWarning pointing users to the new module location.

    Args:
        old: Fully-qualified name of the deprecated module (e.g. ``rfdetr.util.logger``).
        new: Fully-qualified name of the replacement (e.g. ``rfdetr.utilities.logger``).
    """
    warnings.warn(
        f"{old} is deprecated; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
