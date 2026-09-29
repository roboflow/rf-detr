# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public access to exported RF-DETR inference.

Use :meth:`rfdetr.detr.RFDETR.from_export` to load an artifact and keep the native :meth:`rfdetr.detr.RFDETR.predict`
input and result contract. The ``OpenVINOInference`` wrapper remains available for callers that supply preprocessed
tensors and need raw outputs.

The reference decoders in ``rfdetr.export._onnx.inference`` and ``rfdetr.export._tflite.inference`` remain private
parity helpers.
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
