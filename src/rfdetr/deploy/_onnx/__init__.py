# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy._onnx is deprecated; use rfdetr.export._onnx."""

import warnings

warnings.warn(
    "rfdetr.deploy._onnx is deprecated; use rfdetr.export._onnx instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.export._onnx import *  # noqa: F401, F403, E402
from rfdetr.export._onnx import OnnxOptimizer, CustomOpSymbolicRegistry  # noqa: E402
