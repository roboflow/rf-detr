# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Install probes for the optional packages the export formats build on.

Each flag answers "is this package installed" without importing it, so a caller can gate on it at collection time (for
example ``pytest.mark.skipif``) and import the package locally where it is used. Packages outside the export formats
live in :mod:`rfdetr.utilities.imports`.

The flags are private and exist for the test suite's collection-time gating; no runtime code reads them. An export
format keeps its own import boundary (for example ``_IS_OPENVINO_AVAILABLE`` beside the OpenVINO exporter), so those
names are a separate concept. Add a flag here only for a package some test gates on.
"""

from __future__ import annotations

from rfdetr.utilities.package import is_installed

#: Whether the optional ONNX package is installed.
_IS_ONNX_INSTALLED = is_installed("onnx")
#: Whether the optional ONNX Runtime package is installed.
_IS_ONNXRUNTIME_INSTALLED = is_installed("onnxruntime")
#: Whether the optional ONNX GraphSurgeon package is installed.
_IS_ONNX_GRAPHSURGEON_INSTALLED = is_installed("onnx_graphsurgeon")
#: Whether the optional ONNX Converter Common package (the FP16 graph rewrite) is installed.
_IS_ONNXCONVERTER_COMMON_INSTALLED = is_installed("onnxconverter_common")
#: Whether the optional OpenVINO package is installed.
_IS_OPENVINO_INSTALLED = is_installed("openvino")
#: Whether the optional NNCF package is installed.
_IS_NNCF_INSTALLED = is_installed("nncf")
#: Whether the optional LiteRT Torch package is installed.
_IS_LITERT_TORCH_INSTALLED = is_installed("litert_torch")
