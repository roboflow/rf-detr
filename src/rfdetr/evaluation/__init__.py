# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Framework-agnostic evaluation utilities for RF-DETR."""

from rfdetr.evaluation.keypoint_oks import (
    DEFAULT_KEYPOINT_MAX_DETS,
    METRIC_KEY_MAP,
    METRIC_KEY_MAP_50,
    METRIC_KEY_MAP_75,
    METRIC_KEY_MAR,
    MetricKeypointOKS,
)

__all__ = [
    "DEFAULT_KEYPOINT_MAX_DETS",
    "METRIC_KEY_MAP",
    "METRIC_KEY_MAP_50",
    "METRIC_KEY_MAP_75",
    "METRIC_KEY_MAR",
    "MetricKeypointOKS",
]
