# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""TFLite export: ONNX → TFLite conversion via onnx2tf."""

import importlib.util

from rfdetr.export._tflite.converter import export_tflite

_IS_ONNX2TF_AVAILABLE: bool = importlib.util.find_spec("onnx2tf") is not None

__all__ = ["export_tflite", "_IS_ONNX2TF_AVAILABLE"]
