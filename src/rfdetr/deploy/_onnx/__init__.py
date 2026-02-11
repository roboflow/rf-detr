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
from rfdetr.deploy._onnx import optimizer, symbolic
from rfdetr.deploy._onnx.optimizer import OnnxOptimizer
from rfdetr.deploy._onnx.symbolic import CustomOpSymbolicRegistry
