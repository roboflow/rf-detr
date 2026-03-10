# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.visualize.data`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.visualize is deprecated; use rfdetr.visualize.data instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.visualize.data import save_gt_predictions_visualization  # noqa: E402

__all__ = ["save_gt_predictions_visualization"]
