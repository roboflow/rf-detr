# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection and segmentation head subpackage."""

from rfdetr.models.heads.segmentation import (  # noqa: F401
    DepthwiseConvBlock,
    MLPBlock,
    SegmentationHead,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

__all__ = [
    "SegmentationHead",
    "DepthwiseConvBlock",
    "MLPBlock",
    "point_sample",
    "get_uncertain_point_coords_with_randomness",
    "calculate_uncertainty",
]
