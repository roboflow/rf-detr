# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.visualize.training`` instead."""

import warnings

warnings.warn(
    "rfdetr.lit.plot is deprecated; use rfdetr.visualize.training instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.visualize.training import plot_metrics  # noqa: E402, F401

__all__ = ["plot_metrics"]
