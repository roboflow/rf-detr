# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.models.segmentation_head is deprecated; use rfdetr.models.heads.segmentation."""

import warnings

warnings.warn(
    "rfdetr.models.segmentation_head is deprecated; use rfdetr.models.heads.segmentation instead.",
    DeprecationWarning,
    stacklevel=2,
)
