# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy.export is deprecated; use rfdetr.export.export."""

import warnings

warnings.warn(
    "rfdetr.deploy.export is deprecated; use rfdetr.export.export instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.export.export import *  # noqa: F401, F403, E402
from rfdetr.export.export import export_onnx, make_infer_image, onnx_simplify  # noqa: E402
