# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX Runtime inference helpers for RF-DETR exported models.

These functions handle session creation, image preprocessing, and detection decoding using ``onnxruntime``, ``numpy``,
``supervision``, and ``Pillow``, alongside the RF-DETR package dependencies. Loading a ``.onnx`` artifact through
``RFDETRInference`` also reads its embedded inference metadata with the ``onnx`` package, so that path needs
``rfdetr[onnx]``, which installs both ``onnx`` and ``onnxruntime``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rfdetr.export._onnx.reference import (  # noqa: F401  (the reference helpers stay importable from here)
    _create_onnx_session,
    _logical_cpu_count,
    _run_inference,
    _run_onnx_raw,
)
from rfdetr.export._runtime.adapters import (
    ExportRuntime,
    _check_input_interface,
    _check_output_interface,
    _input_array,
    _input_tensor,
    _runtime_options,
    _single_input,
    _thread_count_option,
)
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.export._runtime.adapters import ExportRuntime

logger = get_logger()


def _onnx_dtype(type_name: str) -> np.dtype[Any]:
    """Return the NumPy dtype of an ONNX tensor type name such as ``"tensor(float)"``.

    Examples:
        >>> _onnx_dtype("tensor(float)"), _onnx_dtype("tensor(float16)")
        (dtype('float32'), dtype('float16'))
    """
    # ONNX names element types "tensor(<type>)"; only "float" differs from the NumPy name, where it means float64.
    element_type = type_name.removeprefix("tensor(").removesuffix(")")
    return np.dtype("float32" if element_type == "float" else element_type)


def _run_onnx_bound(session: Any, input_name: str, batch: torch.Tensor, outputs: Sequence[Any]) -> list[torch.Tensor]:
    """Run one batch through IO binding on the batch's own device and return the outputs as new tensors there.

    ONNX Runtime reads the input and writes every output in place, so a CUDA batch never goes through host memory.
    Each output takes the batch size for its first axis; the other axes must be static in the graph.

    Args:
        session: The ONNX Runtime session.
        input_name: Name of the graph's single input.
        batch: The input, already in the graph's layout and dtype and contiguous.
        outputs: The session's output descriptions, in graph order.

    Returns:
        One tensor per output, in graph order, on the batch's device.

    Examples:
        Needs an exported ONNX model.

        >>> import onnxruntime  # doctest: +SKIP
        >>> session = onnxruntime.InferenceSession("model.onnx")  # doctest: +SKIP
        >>> batch = torch.zeros(1, 3, 384, 384)  # doctest: +SKIP
        >>> boxes, logits = _run_onnx_bound(session, "input", batch, session.get_outputs())  # doctest: +SKIP
    """
    device = batch.device
    device_id = 0 if device.index is None else device.index
    binding = session.io_binding()
    # Allocate on the CPU: a process-wide default device such as CUDA would make ``.numpy()`` raise.
    input_dtype = torch.empty(0, dtype=batch.dtype, device="cpu").numpy().dtype
    binding.bind_input(input_name, device.type, device_id, input_dtype, tuple(batch.shape), batch.data_ptr())
    results = []
    for output in outputs:
        dtype = _onnx_dtype(output.type)
        tensor = torch.empty(
            (batch.shape[0], *output.shape[1:]), dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype, device=device
        )
        binding.bind_output(output.name, device.type, device_id, dtype, tuple(tensor.shape), tensor.data_ptr())
        results.append(tensor)
    if device.type == "cuda":
        # ONNX Runtime runs on its own CUDA stream, and the batch was written on torch's current stream.
        torch.cuda.current_stream(device).synchronize()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return results


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str, options: Mapping[str, Any]) -> ExportRuntime:
    """Load an ONNX artifact with provider and interface checks; it reads the ``num_threads`` option.

    On the CPU provider an unset ``num_threads`` means every logical core the process may use. ONNX Runtime's own
    default is the physical cores, which leaves the hyperthreads of a typical host idle during a batch-1 call. Pass
    ``num_threads`` to choose another pool size. The CUDA provider keeps ONNX Runtime's default.
    """
    num_threads = _thread_count_option("ONNX", _runtime_options("ONNX", options, ("num_threads",)))
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX inference requires onnxruntime or onnxruntime-gpu.") from exc

    providers = ort.get_available_providers()
    if device == "cpu":
        provider = "CPUExecutionProvider"
    elif device == "auto":
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    elif device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
        provider = "CUDAExecutionProvider"
    else:
        raise ValueError(f"ONNX device {device!r} is unsupported. Use cpu, cuda:N, or auto.")
    if provider not in providers:
        raise RuntimeError(f"ONNX provider {provider} is unavailable. Installed providers: {providers}.")

    device_id = int(device[5:]) if device.startswith("cuda:") else 0
    requested: list[str | tuple[str, dict[str, Any]]] = (
        [(provider, {"device_id": device_id})] if provider == "CUDAExecutionProvider" else [provider]
    )
    if provider == "CPUExecutionProvider" and num_threads is None:
        num_threads = _logical_cpu_count()
    session = _create_onnx_session(
        path, providers=requested, **({} if num_threads is None else {"num_threads": num_threads})
    )
    if provider not in session.get_providers():
        raise RuntimeError(f"ONNX session did not activate requested provider {provider}.")
    if device != "auto" and provider == "CUDAExecutionProvider" and hasattr(session, "disable_fallback"):
        session.disable_fallback()
    input_info = _single_input("ONNX", session.get_inputs())
    dtype = _onnx_dtype(input_info.type)
    # A symbolic or unnamed ONNX dimension is a str or None; only an int fixes the axis.
    shape = [size if isinstance(size, int) else None for size in input_info.shape]
    _check_input_interface("ONNX", metadata, input_info.name, shape, dtype)
    outputs = session.get_outputs()
    output_names = [item.name for item in outputs]
    _check_output_interface("ONNX", metadata, output_names)

    # On the CUDA provider, the batch is prepared on the GPU and bound in place, so it never goes through host memory.
    # That needs torch to see the GPU, every output to lead with the input's batch axis (the same symbol or size, so a
    # scalar or fixed-size auxiliary output stays on the host path) and every other axis to be static, so its buffer can
    # be allocated.
    batch_axis = input_info.shape[0] if input_info.shape else None
    on_device = (
        provider == "CUDAExecutionProvider"
        and torch.cuda.is_available()
        and batch_axis is not None
        and all(
            len(item.shape) >= 1
            and item.shape[0] == batch_axis
            and all(isinstance(size, int) for size in item.shape[1:])
            for item in outputs
        )
    )

    def execute(batch: Any) -> dict[str, Any]:
        """Map one raw ONNX result by graph output name."""
        if on_device:
            return dict(
                zip(output_names, _run_onnx_bound(session, input_info.name, _input_tensor(batch, metadata), outputs))
            )
        return dict(zip(output_names, _run_onnx_raw(session, input_info.name, _input_array(batch, metadata))))

    return ExportRuntime(
        "onnx",
        metadata,
        session=session,
        device_policy=provider,
        execute=execute,
        device=torch.device("cuda", device_id) if on_device else torch.device("cpu"),
        borrowed_outputs=False,
    )
