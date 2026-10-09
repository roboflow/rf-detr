# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OpenVINO inference utilities for exported RF-DETR models."""

from __future__ import annotations

import os
import re
import threading
from _thread import LockType
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from deprecate import TargetMode, deprecated
from numpy.typing import NDArray

from rfdetr.export._openvino.exporter import _check_openvino_available
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.utilities.logger import get_logger

logger = get_logger()


#: Export-side precision spellings accepted by ``inference_precision`` and mapped to OpenVINO's own names, so the
#: vocabulary of ``openvino_precision`` at export time also works here.
_PRECISION_ALIASES: dict[str, str] = {"float32": "f32", "float16": "f16"}

#: Default of ``inference_precision``: ``"f32"`` on every device except the NPU, whose plugin does not support an f32
#: precision hint, so there it leaves the plugin's own precision alone.
_AUTO_PRECISION = "auto"


def _resolve_precision_hint(inference_precision: str | None, device: str) -> str | None:
    """Return the ``INFERENCE_PRECISION_HINT`` value to send for *device*, or ``None`` to send no hint.

    Args:
        inference_precision: The caller's ``inference_precision``: ``"auto"``, ``None`` or an explicit spelling.
        device: OpenVINO device string, e.g. ``"CPU"`` or ``"AUTO:NPU,CPU"``.

    Returns:
        ``None`` for ``None``, and for ``"auto"`` on a device string naming the NPU; ``"f32"`` for ``"auto"``
        elsewhere; an explicit value mapped through :data:`_PRECISION_ALIASES`, unchanged otherwise, so an unsupported
        explicit hint still reaches OpenVINO and is rejected there.

    Examples:
        >>> _resolve_precision_hint("auto", "CPU")
        'f32'
        >>> _resolve_precision_hint("auto", "NPU") is None
        True
        >>> _resolve_precision_hint("auto", "AUTO:NPU,CPU") is None
        True
        >>> _resolve_precision_hint("float16", "NPU")
        'f16'
        >>> _resolve_precision_hint(None, "CPU") is None
        True
    """
    if inference_precision == _AUTO_PRECISION:
        return None if "NPU" in device.upper() else "f32"
    if inference_precision is None:
        return None
    return _PRECISION_ALIASES.get(inference_precision, inference_precision)


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
    inference_precision: str | None = _AUTO_PRECISION,
    config: Mapping[str, Any] | None = None,
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
    properties: dict[str, Any] = {}
    precision_hint = _resolve_precision_hint(inference_precision, device)
    if precision_hint is not None:
        properties["INFERENCE_PRECISION_HINT"] = precision_hint
    properties.update(config or {})
    compiled_model = core.compile_model(model, device, properties)
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


#: Construction warning of :class:`OpenVINOInference`; pyDeprecate fills in the versions.
_OPENVINO_INFERENCE_DEPRECATION = (
    "`OpenVINOInference` was deprecated in v%(deprecated_in)s and will be removed in v%(remove_in)s."
    " Use RFDETRInference(model_path).predict(image) for decoded predictions, with runtime_options for its"
    " cache_dir, inference_precision and config settings."
)


class OpenVINOInference:
    """Deprecated compatibility facade for raw OpenVINO inference.

    Use :class:`rfdetr.RFDETRInference` to load an artifact and call ``predict()`` for detections, passing
    ``cache_dir``, ``inference_precision`` and ``config`` through its ``runtime_options``. Deprecated in v1.12.0 and
    removed in v2.0.0.
    """

    # Deprecate __init__ rather than the class: deprecated_class returns a proxy, and callers using
    # OpenVINOInference.__new__(OpenVINOInference) need the real class.
    @deprecated(  # type: ignore[untyped-decorator]  # pyDeprecate types its wrapper as Callable[..., Any]
        target=TargetMode.NOTIFY,
        deprecated_in="1.12.0",
        remove_in="2.0.0",
        num_warns=-1,
        template_mgs=_OPENVINO_INFERENCE_DEPRECATION,
    )
    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        cache_dir: str | None = None,
        inference_precision: str | None = _AUTO_PRECISION,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the deprecated OpenVINO facade; every construction emits a ``FutureWarning``.

        Args:
            model_path: Path to the OpenVINO IR model (.xml file).
            device: Device the model is compiled for, e.g. ``"AUTO"``, ``"CPU"``, ``"GPU"`` or ``"NPU"``.
            cache_dir: Directory holding the compiled-model cache. When set, OpenVINO reuses the
                compiled kernels across process starts instead of recompiling the model every time.
            inference_precision: OpenVINO ``INFERENCE_PRECISION_HINT``, the precision the device *computes*
                in — independent of the IR's storage precision (``openvino_precision`` at export). Defaults
                to ``"auto"``, i.e. ``"f32"`` on every device except the NPU, whose plugin does not support an
                f32 hint and keeps its own precision. OpenVINO's own CPU default is f16 on ARM and bf16 on x86
                hosts with AMX or AVX512-BF16, and at either precision RF-DETR's logits collapse (no detections
                clear a 0.5 threshold). ``None`` sends no hint and keeps the device default — faster where the
                hardware computes natively in reduced precision (ARM CPU, Intel GPU/NPU), at that accuracy cost.
                Accepted spellings are ``"auto"``, ``"f32"``, ``"f16"`` and ``"bf16"``, plus ``"float32"`` and
                ``"float16"`` as aliases of ``"f32"`` and ``"f16"`` (the vocabulary ``openvino_precision`` uses at
                export). Any other string is not validated here: it is passed through unchanged and OpenVINO
                rejects it when compiling the model (e.g. ``"fp32"``).
            config: Further compile properties for ``compile_model``, e.g. ``{"INFERENCE_NUM_THREADS": 4}``. Applied
                after *inference_precision*, so an ``INFERENCE_PRECISION_HINT`` given here wins.

        Raises:
            ImportError: If OpenVINO is not installed.
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If OpenVINO rejects *inference_precision* or *config* while compiling the model.
        """
        session = _load_openvino_session(model_path, device, cache_dir, inference_precision, config)
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


#: The ``runtime_options`` keys the OpenVINO loader reads, with the meaning they have on :class:`OpenVINOInference`.
_OPENVINO_RUNTIME_OPTIONS = ("cache_dir", "config", "inference_precision")


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str, options: Mapping[str, Any]) -> Any:
    """Load an OpenVINO graph through the shared session functions.

    Args:
        path: OpenVINO IR ``.xml`` file.
        metadata: The artifact's inference metadata.
        device: ``auto``, ``cpu``, ``gpu``, ``npu``, ``gpu.N`` or ``npu.N``.
        options: Any of ``cache_dir``, ``inference_precision`` and ``config``, as on :class:`OpenVINOInference`.

    Returns:
        The loaded runtime.
    """
    from rfdetr.export._runtime.adapters import ExportRuntime, _input_array, _runtime_options

    settings = _runtime_options("OpenVINO", options, _OPENVINO_RUNTIME_OPTIONS)
    if settings.get("cache_dir") is not None:
        settings["cache_dir"] = os.fspath(settings["cache_dir"])
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

    precision = _resolve_precision_hint(settings.pop("inference_precision", _AUTO_PRECISION), target)
    session = _load_openvino_session(path, device=target, inference_precision=precision, **settings)
    if metadata.input_dtype != "float32" or metadata.input_layout != "NCHW":
        raise ValueError("OpenVINO inference wrapper requires a float32 NCHW input.")
    shape = session.input_layer.partial_shape
    # Check the rank before indexing: a dynamic rank has no length, and a rank that differs from the metadata would
    # either index past the graph's last axis or leave its extra axes unchecked.
    expected_rank = len(metadata.input_shape)
    if not shape.rank.is_static:
        raise ValueError(f"OpenVINO input rank must be {expected_rank}, got a dynamic rank.")
    if len(shape) != expected_rank:
        raise ValueError(f"OpenVINO input rank must be {expected_rank}, got {len(shape)}.")
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
