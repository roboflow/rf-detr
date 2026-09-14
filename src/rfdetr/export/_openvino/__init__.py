# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OpenVINO IR export: direct PyTorch -> OpenVINO IR (``.xml``/``.bin``) conversion."""

from rfdetr.export._openvino.exporter import _check_openvino_available

try:
    _check_openvino_available()
    _IS_OPENVINO_AVAILABLE: bool = True
except ImportError:
    _IS_OPENVINO_AVAILABLE = False

__all__ = ["_IS_OPENVINO_AVAILABLE"]
