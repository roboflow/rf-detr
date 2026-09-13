# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LiteRT export: direct PyTorch -> ``.tflite`` conversion via ``torch.export`` + ``litert-torch``."""

from rfdetr.export._litert.converter import (
    _check_litert_available,
    export_litert,
)

try:
    _check_litert_available()
    _IS_LITERT_AVAILABLE: bool = True
except ImportError:
    _IS_LITERT_AVAILABLE = False

__all__ = ["export_litert", "_IS_LITERT_AVAILABLE"]
