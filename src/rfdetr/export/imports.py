# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Install probes for the optional packages the export formats build on.

Each flag answers "is this package installed" without importing it, so a caller can gate on it at collection time (for
example ``pytest.mark.skipif``) and import the package locally where it is used.
"""

from __future__ import annotations

from rfdetr.utilities.package import is_installed

_IS_ONNX_INSTALLED = is_installed("onnx")
_IS_ONNXRUNTIME_INSTALLED = is_installed("onnxruntime")
_IS_OPENVINO_INSTALLED = is_installed("openvino")
_IS_NNCF_INSTALLED = is_installed("nncf")
_IS_LITERT_TORCH_INSTALLED = is_installed("litert_torch")
