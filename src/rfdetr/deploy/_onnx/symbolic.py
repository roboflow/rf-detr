# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy._onnx.symbolic is deprecated; use rfdetr.export._onnx.symbolic."""

import warnings

warnings.warn(
    "rfdetr.deploy._onnx.symbolic is deprecated; use rfdetr.export._onnx.symbolic instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.export._onnx.symbolic import *  # noqa: F401, F403, E402
from rfdetr.export._onnx.symbolic import CustomOpSymbolicRegistry  # noqa: E402
