# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility shim — import from ``rfdetr.datasets.aug_configs`` instead."""

import warnings

warnings.warn(
    "rfdetr.datasets.aug_config is deprecated and will be removed in a future release. "
    "Use rfdetr.datasets.aug_configs instead.",
    FutureWarning,
    stacklevel=2,
)

from rfdetr.datasets.aug_configs import (  # noqa: E402
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONFIG,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)

__all__ = [
    "AUG_AGGRESSIVE",
    "AUG_AERIAL",
    "AUG_CONFIG",
    "AUG_CONSERVATIVE",
    "AUG_INDUSTRIAL",
]
