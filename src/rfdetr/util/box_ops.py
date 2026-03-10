# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities.box_ops`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.box_ops is deprecated; use rfdetr.utilities.box_ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.utilities.box_ops import (  # noqa: E402, F401
    batch_dice_loss,
    batch_dice_loss_jit,
    batch_sigmoid_ce_loss,
    batch_sigmoid_ce_loss_jit,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    masks_to_boxes,
)

__all__ = [
    "batch_dice_loss",
    "batch_dice_loss_jit",
    "batch_sigmoid_ce_loss",
    "batch_sigmoid_ce_loss_jit",
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "generalized_box_iou",
    "masks_to_boxes",
]
