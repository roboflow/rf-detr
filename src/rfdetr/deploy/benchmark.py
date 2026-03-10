# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy.benchmark is deprecated; use rfdetr.export.benchmark."""

import warnings

warnings.warn(
    "rfdetr.deploy.benchmark is deprecated; use rfdetr.export.benchmark instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.export.benchmark import *  # noqa: F401, F403, E402
from rfdetr.export.benchmark import TRTInference, infer_transforms  # noqa: E402
