# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LiteRT export: direct PyTorch -> ``.tflite`` conversion via ``torch.export`` + ``litert-torch``."""

from rfdetr.export._litert.exporter import _check_litert_available

try:
    _check_litert_available()
    _IS_LITERT_AVAILABLE: bool = True
except ImportError:
    _IS_LITERT_AVAILABLE = False

__all__ = ["_IS_LITERT_AVAILABLE"]
