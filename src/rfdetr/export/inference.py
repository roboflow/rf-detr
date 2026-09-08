# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public inference wrappers for exported RF-DETR artifacts.

This module is session-tier only: a wrapper here loads an exported artifact and runs it, taking already-preprocessed
tensors in and returning the model's raw output tensors. Turning those raw outputs into detections is deliberately
**not** part of this surface — see the reference decoders in ``rfdetr.export._onnx.inference`` and
``rfdetr.export._tflite.inference``, which are private and exist to pin numerical parity with
:meth:`rfdetr.detr.RFDETR.predict`, not to be imported.

For multi-backend inference (PyTorch / ONNX / TensorRT) with automatic backend selection, prefer `inference-models
<https://github.com/roboflow/inference/tree/main/inference_models>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfdetr.export._openvino.inference import OpenVINOInference

__all__ = ["OpenVINOInference"]


def __getattr__(name: str) -> Any:
    """Resolve a public inference wrapper on first attribute access.

    Deferred so that importing this module never pulls in an optional runtime dependency the caller
    may not have installed — ``import rfdetr.export.inference`` stays free of ``openvino``.

    Args:
        name: Attribute being looked up on this module.

    Returns:
        The requested wrapper class.

    Raises:
        AttributeError: If *name* is not one of :data:`__all__`.

    Examples:
        >>> from rfdetr.export import inference
        >>> inference.NotARuntime
        Traceback (most recent call last):
        ...
        AttributeError: module 'rfdetr.export.inference' has no attribute 'NotARuntime'
    """
    if name == "OpenVINOInference":
        from rfdetr.export._openvino.inference import OpenVINOInference

        return OpenVINOInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
