# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public runtimes and pre/post-processing helpers for exported RF-DETR artifacts.

A runtime here loads an exported artifact and runs it, taking already-preprocessed tensors in and returning the model's
raw output tensors: :class:`OpenVINOInference` (``.xml``, NumPy arrays), :class:`TRTInference` (``.trt``, torch tensors
already on the GPU) and :func:`load_executorch_method` (``.pte``). The two helpers around them reproduce
:meth:`rfdetr.detr.RFDETR.predict`'s own steps, so a runtime fed through them matches it: :func:`preprocess_to_nchw`
(PIL image to normalized NCHW array) and :func:`decode_detections` (raw ``dets``/``labels`` to boxes, scores and class
IDs, returned as :class:`DecodedDetections`). The format-specific reference decoders in
``rfdetr.export._onnx.inference`` and ``rfdetr.export._tflite.inference`` stay private.

For multi-backend inference (PyTorch / ONNX / TensorRT) with automatic backend selection, prefer `inference-models
<https://github.com/roboflow/inference/tree/main/inference_models>`_.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfdetr.export._executorch.inference import load_executorch_method
    from rfdetr.export._openvino.inference import OpenVINOInference
    from rfdetr.export._runtime.decode import DecodedDetections, decode_detections
    from rfdetr.export._runtime.preprocess import preprocess_to_nchw
    from rfdetr.export._tensorrt.inference import TRTInference

#: Public name -> private module that defines it, imported on first access.
_LAZY_EXPORTS = {
    "DecodedDetections": "rfdetr.export._runtime.decode",
    "OpenVINOInference": "rfdetr.export._openvino.inference",
    "TRTInference": "rfdetr.export._tensorrt.inference",
    "decode_detections": "rfdetr.export._runtime.decode",
    "load_executorch_method": "rfdetr.export._executorch.inference",
    "preprocess_to_nchw": "rfdetr.export._runtime.preprocess",
}

__all__ = [
    "DecodedDetections",
    "OpenVINOInference",
    "TRTInference",
    "decode_detections",
    "load_executorch_method",
    "preprocess_to_nchw",
]


def __getattr__(name: str) -> Any:
    """Resolve a public runtime or helper on first attribute access.

    Deferred so that importing this module never pulls in an optional runtime dependency the caller
    may not have installed — ``import rfdetr.export.inference`` stays free of ``openvino``, ``tensorrt``
    and ``executorch``.

    Args:
        name: Attribute being looked up on this module.

    Returns:
        The requested class or function.

    Raises:
        AttributeError: If *name* is not one of :data:`__all__`.

    Examples:
        >>> from rfdetr.export import inference
        >>> inference.NotARuntime
        Traceback (most recent call last):
        ...
        AttributeError: module 'rfdetr.export.inference' has no attribute 'NotARuntime'
    """
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
