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

The three runtimes share no common call contract: :class:`OpenVINOInference` is called with one NCHW NumPy array and
returns a tuple of arrays, :class:`TRTInference` is called with a mapping of binding name to torch tensor on the
engine's device and returns a dict of tensors, and :func:`load_executorch_method` returns a raw ExecuTorch method driven
through its own ``execute([tensor])``. The stable public subset of :class:`TRTInference` is its constructor,
``__call__``, ``engine_device`` and ``synchronize``; its other attributes and methods (engine building, binding and
profiling helpers) are implementation detail that may change without notice, even though the class itself is exported
here.

Use :class:`rfdetr.inference.RFDETRInference` for image inputs and Supervision predictions across native models and
export formats. The TensorRT and OpenVINO classes are deprecated and will be removed in a future release. They remain
available for compatibility and warn when constructed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # ``X as X`` marks an explicit re-export: ``__all__`` below is computed, so linters cannot see these names used.
    from rfdetr.export._executorch.inference import load_executorch_method as load_executorch_method
    from rfdetr.export._openvino.inference import OpenVINOInference as OpenVINOInference
    from rfdetr.export._runtime.decode import DecodedDetections as DecodedDetections
    from rfdetr.export._runtime.decode import decode_detections as decode_detections
    from rfdetr.export._runtime.preprocess import preprocess_to_nchw as preprocess_to_nchw
    from rfdetr.export._tensorrt.inference import TRTInference as TRTInference

#: Public name -> private module that defines it, imported on first access.
_LAZY_EXPORTS = {
    "DecodedDetections": "rfdetr.export._runtime.decode",
    "OpenVINOInference": "rfdetr.export._openvino.inference",
    "TRTInference": "rfdetr.export._tensorrt.inference",
    "decode_detections": "rfdetr.export._runtime.decode",
    "load_executorch_method": "rfdetr.export._executorch.inference",
    "preprocess_to_nchw": "rfdetr.export._runtime.preprocess",
}

__all__ = sorted(_LAZY_EXPORTS)


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


def __dir__() -> list[str]:
    """Include the lazily resolved public names in ``dir()`` and interactive tab-completion.

    Returns:
        The module's own globals plus every name in :data:`__all__`, sorted.

    Examples:
        >>> from rfdetr.export import inference
        >>> "TRTInference" in dir(inference)
        True
    """
    return sorted(set(globals()) | set(__all__))
