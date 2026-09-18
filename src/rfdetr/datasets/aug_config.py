# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility shim — import from ``rfdetr.datasets.aug_configs`` instead.

Deprecated since v1.9.0, removal scheduled for v1.12.0.

``warnings.warn`` rather than ``pyDeprecate`` on purpose: pyDeprecate exposes function, class and instance decorators
only, none of which can flag a bare module import. A module-level ``warnings.warn`` is the only mechanism that fires
when someone imports the deprecated module path itself.
"""

import warnings

warnings.warn(
    "rfdetr.datasets.aug_config is deprecated since v1.9.0 and will be removed in v1.12.0. "
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
