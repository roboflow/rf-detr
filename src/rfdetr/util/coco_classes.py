# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.assets.coco_classes`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.coco_classes is deprecated; use rfdetr.assets.coco_classes instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402, F401

__all__ = ["COCO_CLASSES"]
