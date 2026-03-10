# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities.logger`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.logger is deprecated; use rfdetr.utilities.logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.utilities.logger import get_logger  # noqa: E402

__all__ = ["get_logger"]
