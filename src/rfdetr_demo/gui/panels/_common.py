# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared constants for GUI panel mixins."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VAST_GPU_FILTERS: tuple[str, ...] = (
    "任意",
    "RTX_4090",
    "RTX_4080",
    "RTX_3090",
    "RTX_A6000",
    "A5000",
    "L40",
    "A100",
)
