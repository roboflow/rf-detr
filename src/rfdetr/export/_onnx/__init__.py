# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX export and its ONNX Runtime helpers.

Import the submodules directly — :mod:`rfdetr.export._onnx.exporter` writes the ``.onnx`` file, and
:mod:`rfdetr.export._onnx.inference` runs one. This package intentionally re-exports nothing, so patching a symbol
reaches the module that actually defines it.
"""
