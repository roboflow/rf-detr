# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.visualize.data`` instead."""

from rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.visualize", "rfdetr.visualize.data")

from rfdetr.visualize.data import save_gt_predictions_visualization  # noqa: F401, E402
