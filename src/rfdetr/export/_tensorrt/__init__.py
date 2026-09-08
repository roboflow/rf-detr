# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""TensorRT export and its reference runtime.

Import the submodules directly — :mod:`rfdetr.export._tensorrt.exporter` builds an engine from an ONNX file, and
:mod:`rfdetr.export._tensorrt.inference` runs one. This package intentionally re-exports nothing, so patching a symbol
reaches the module that actually defines it.
"""
