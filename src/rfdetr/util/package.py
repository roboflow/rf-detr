# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities.package`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.package is deprecated; use rfdetr.utilities.package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.utilities.package import get_sha, get_version  # noqa: E402

__all__ = ["get_sha", "get_version"]
