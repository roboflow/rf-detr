# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
onnx optimizer and symbolic registry
"""

from rfdetr.export._onnx import exporter, symbolic
from rfdetr.export._onnx.exporter import OnnxOptimizer
from rfdetr.export._onnx.symbolic import CustomOpSymbolicRegistry

__all__ = [
    "exporter",
    "symbolic",
    "OnnxOptimizer",
    "CustomOpSymbolicRegistry",
]
