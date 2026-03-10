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

from rfdetr.models.heads.segmentation import *  # noqa: F401, F403, E402
from rfdetr.models.heads.segmentation import (  # noqa: E402
    DepthwiseConvBlock,
    MLPBlock,
    SegmentationHead,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
