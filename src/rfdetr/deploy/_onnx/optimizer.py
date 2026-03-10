# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy._onnx.optimizer is deprecated; use rfdetr.export._onnx.optimizer."""

import warnings

warnings.warn(
    "rfdetr.deploy._onnx.optimizer is deprecated; use rfdetr.export._onnx.optimizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.export._onnx.optimizer import *  # noqa: F401, F403, E402
from rfdetr.export._onnx.optimizer import OnnxOptimizer  # noqa: E402
