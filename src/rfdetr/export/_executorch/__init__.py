# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ExecuTorch export: PyTorch -> ``.pte`` conversion via ``torch.export``."""

from rfdetr.export._executorch.converter import (
    _check_executorch_available,
    export_executorch,
)

try:
    _check_executorch_available()
    _IS_EXECUTORCH_AVAILABLE: bool = True
except ImportError:
    _IS_EXECUTORCH_AVAILABLE = False

__all__ = ["export_executorch"]
