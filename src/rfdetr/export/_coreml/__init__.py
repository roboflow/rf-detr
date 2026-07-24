# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CoreML export: PyTorch ``torch.export`` -> ``.mlpackage`` via coremltools."""

from rfdetr.export._coreml.converter import _check_coremltools_available, export_coreml
from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._coreml.torch_ops import ensure_coreml_torch_op_patches

try:
    _check_coremltools_available()
    _IS_COREMLTOOLS_AVAILABLE: bool = True
except ImportError:
    _IS_COREMLTOOLS_AVAILABLE = False

__all__ = [
    "export_coreml",
    "_IS_COREMLTOOLS_AVAILABLE",
    "unsupported_coreml_ops",
    "ensure_coreml_torch_op_patches",
]
