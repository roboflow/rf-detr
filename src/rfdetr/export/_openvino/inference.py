# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OpenVINO inference utilities for exported RF-DETR models."""

from __future__ import annotations

import re
import threading
import warnings
from _thread import LockType
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._openvino.exporter import _check_openvino_available
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.utilities.logger import get_logger

logger = get_logger()


@dataclass
class _OpenVINOSession:
    """Hold OpenVINO objects shared by the unified and compatibility APIs."""

    compiled_model: Any
    infer_request: Any
    infer_lock: LockType
    input_layer: Any
    output_layers: list[Any]


def _load_openvino_session(
    model_path: str | Path,
    device: str = "AUTO",
    cache_dir: str | None = None,
    inference_precision: str | None = None,
) -> _OpenVINOSession:
    """Load and compile an OpenVINO IR model for the requested device."""
    _check_openvino_available()
    import openvino as ov

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    core = ov.Core()
    if cache_dir is not None:
        # Set before compilation so OpenVINO can reuse compiled kernels across process starts.
        core.set_property({"CACHE_DIR": cache_dir})
    model = core.read_model(model_path)
    if inference_precision is None:
        compiled_model = core.compile_model(model, device)
    elif inference_precision == "f32":
        compiled_model = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": ov.Type.f32})
    else:
        raise ValueError(f"Unsupported OpenVINO inference precision: {inference_precision!r}.")
    infer_request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)
    output_layers = [compiled_model.output(index) for index in range(len(compiled_model.outputs))]

    logger.info(f"Loaded OpenVINO model from {model_path}")
    logger.info(f"Input shape: {input_layer.partial_shape}")
    logger.info(f"Number of outputs: {len(output_layers)}")
    return _OpenVINOSession(compiled_model, infer_request, threading.Lock(), input_layer, output_layers)


def _infer_openvino(session: _OpenVINOSession, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
    """Run one validated batch and copy outputs from OpenVINO's reusable buffers."""
    if input_data.dtype != np.float32 or not input_data.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"infer() requires a C-contiguous float32 array, got dtype={input_data.dtype} "
            f"contiguous={input_data.flags['C_CONTIGUOUS']}. Construct mean/std with "
            "dtype=np.float32 and finish preprocessing with np.ascontiguousarray(...)."
        )

    with session.infer_lock:
        session.infer_request.infer({session.input_layer: input_data})
        return tuple(
            np.copy(session.infer_request.get_output_tensor(index).data) for index in range(len(session.output_layers))
        )


class OpenVINOInference:
    """Deprecated compatibility facade for raw OpenVINO inference.

    Use :class:`rfdetr.RFDETRInference` to load an artifact and call ``predict()`` for detections. This class will be
    removed in a future release.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        cache_dir: str | None = None,
        inference_precision: str | None = None,
    ) -> None:
        """Initialize the deprecated facade with its previous constructor options."""
        warnings.warn(
            "OpenVINOInference is deprecated and will be removed in a future release. "
            "Use RFDETRInference(model_path).predict(image) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        session = _load_openvino_session(model_path, device, cache_dir, inference_precision)
        self._session = session
        self.compiled_model = session.compiled_model
        self.infer_request = session.infer_request
        self._infer_lock = session.infer_lock
        self.input_layer = session.input_layer
        self.output_layers = session.output_layers

    def infer(self, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Run inference on input data.

        Args:
            input_data: Input tensor in NCHW format (batch, channels, height, width),
                dtype ``float32`` and C-contiguous. Should be ImageNet normalized
                [0.485, 0.456, 0.406] mean, [0.229, 0.224, 0.225] std.

        Returns:
            Tuple of output tensors (typically boxes, labels, and optionally masks/keypoints).
            Each array is a copy, so results stay valid after the next ``infer()`` call.

        Raises:
            ValueError: If *input_data* is not ``float32`` or not C-contiguous. OpenVINO
                accepts a mismatched buffer without erroring and converts to fp32 internally,
                doubling the buffer size shipped across the runtime boundary on every call.
        """
        return _infer_openvino(self._session, input_data)

    def __call__(self, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Alias for infer() to match typical model calling convention."""
        return self.infer(input_data)


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str) -> Any:
    """Load an OpenVINO graph through the shared session functions."""
    if device == "auto":
        target = "AUTO"
    elif device.lower() in {"cpu", "gpu", "npu"} or re.fullmatch(r"(?:gpu|npu)\.[0-9]+", device.lower()):
        target = device.upper()
    else:
        raise ValueError("OpenVINO device must be cpu, gpu, npu, gpu.N, npu.N, or auto.")
    try:
        import openvino as ov
    except ImportError as exc:
        raise ImportError("OpenVINO inference requires openvino.") from exc
    if target != "AUTO":
        available = ov.Core().available_devices
        default_family = target in {"CPU", "GPU", "NPU"}
        if target not in available and not (
            default_family and any(name.startswith(f"{target}.") for name in available)
        ):
            raise RuntimeError(f"OpenVINO device {target} is unavailable. Available devices: {available}.")

    from rfdetr.export._runtime.adapters import ExportRuntime, _input_array

    session = _load_openvino_session(path, device=target, inference_precision="f32")
    if metadata.input_dtype != "float32" or metadata.input_layout != "NCHW":
        raise ValueError("OpenVINO inference wrapper requires a float32 NCHW input.")
    shape = session.input_layer.partial_shape
    for axis, want in enumerate(metadata.input_shape):
        dimension = shape[axis]
        if dimension.is_static and want != -1 and dimension.get_length() != want:
            raise ValueError(f"OpenVINO input axis {axis} disagrees with export metadata.")
    if any(isinstance(key, str) for key in metadata.outputs.values()):
        raise ValueError("OpenVINO output mappings must use positions.")
    positions = [index for index in metadata.outputs.values() if isinstance(index, int)]
    if any(index < 0 or index >= len(session.output_layers) for index in positions):
        raise ValueError("OpenVINO output position is absent from model.")

    def execute(batch: Any) -> tuple[NDArray[Any], ...]:
        """Run the session, which returns owned output arrays."""
        return _infer_openvino(session, _input_array(batch, metadata))

    return ExportRuntime("openvino", metadata, session, target, metadata.input_name, execute, borrowed_outputs=False)
