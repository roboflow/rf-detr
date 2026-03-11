# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.assets.coco_classes`` instead."""

from rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.coco_classes", "rfdetr.assets.coco_classes")

from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: F401, E402
