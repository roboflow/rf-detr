# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Framework-agnostic evaluation utilities for RF-DETR."""

from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import (
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)

__all__ = [
    "build_matching_data",
    "distributed_merge_matching_data",
    "init_matching_accumulator",
    "merge_matching_data",
    "sweep_confidence_thresholds",
]
