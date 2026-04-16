# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""TFLite export: ONNX → TFLite conversion via onnx2tf."""

from rfdetr.export._tflite.converter import export_tflite

__all__ = ["export_tflite"]
