# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""TFLite export: ONNX → TFLite conversion via onnx2tf."""

from rfdetr.export._tflite.exporter import TFLiteExporter, _check_onnx2tf_available

try:
    _check_onnx2tf_available()
    _IS_ONNX2TF_AVAILABLE: bool = True
except ImportError:
    _IS_ONNX2TF_AVAILABLE = False

__all__ = ["TFLiteExporter", "_IS_ONNX2TF_AVAILABLE"]
