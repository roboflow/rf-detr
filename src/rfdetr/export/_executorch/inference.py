# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load a ``.pte`` built by :mod:`rfdetr.export._executorch.exporter` and run it via ExecuTorch's Python runtime.

Backend-agnostic: once the program's delegate is registered, the same
``Runtime.get().load_program(...).load_method(...)`` call loads a ``.pte`` regardless of which backend (XNNPACK,
CoreML, QNN, TensorRT) it was delegated to at export time -- only the artifact's own content differs.  The TensorRT
delegate ships outside the ``executorch`` wheel, so :func:`load_export_runtime` imports its package first to register
it; callers of :func:`load_executorch_method` import ``torch_tensorrt_executorch_runtime`` themselves.
Nothing decodes detections here, same division of responsibility as :mod:`rfdetr.export._tensorrt.inference`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from rfdetr.export._runtime.adapters import ExportRuntime, _input_array, _require_apple, _runtime_options
from rfdetr.export._runtime.metadata import ExportMetadata

if TYPE_CHECKING:
    from rfdetr.export._runtime.adapters import ExportRuntime


def load_executorch_method(pte_path: str | Path, *, method_name: str = "forward") -> Any:
    """Load one method of a ``.pte`` program through ExecuTorch's Python runtime.

    Calls :func:`rfdetr.export._executorch.exporter._check_executorch_available` with
    ``require_runtime=True`` first, so a torch/executorch ABI mismatch (see that function's docstring) raises a
    friendly, actionable :class:`ImportError` here rather than surfacing later as an opaque native crash (e.g.
    ``RuntimeError: tensor does not have a device``) on the first ``method.execute(...)`` call.

    For a ``backend="tensorrt"`` program, import ``torch_tensorrt_executorch_runtime`` before calling this, because
    that import registers the TensorRT delegate and this function does not know which delegate the program uses.

    Args:
        pte_path: Path to a ``.pte`` file written by ``RFDETR.export(format="executorch", ...)``.
        method_name: Name of the method to load; RF-DETR's exporter always writes ``"forward"``.

    Returns:
        A loaded ``executorch.runtime.Method``, callable as ``method.execute([input_tensor])``.

    Raises:
        ImportError: If ``executorch`` is not installed, is older than the minimum supported version, or its
            compiled runtime extension cannot load against the installed ``torch`` (see
            :func:`~rfdetr.export._executorch.exporter._check_executorch_available`).
    """
    from rfdetr.export._executorch.exporter import _check_executorch_available

    _check_executorch_available(require_runtime=True)

    from executorch.runtime import Runtime

    return Runtime.get().load_program(str(pte_path)).load_method(method_name)


def _check_tensorrt_device(device: str) -> None:
    """Refuse a device the ExecuTorch TensorRT delegate cannot run on, before the delegate or the program loads.

    The delegate runs its engine on the current CUDA device and takes no device argument. ``cuda:N`` is therefore
    accepted only as an assertion that ``N`` is the current device; selecting another GPU here would change
    process-wide CUDA state for every other caller.

    Args:
        device: Requested device: ``auto``, ``cuda`` or ``cuda:N``.

    Raises:
        ValueError: If *device* has another form, or ``N`` is not the current CUDA device.
        RuntimeError: If no CUDA device is available.

    Examples:
        >>> _check_tensorrt_device("cpu")
        Traceback (most recent call last):
        ...
        ValueError: ExecuTorch TensorRT requires cuda or auto (or cuda:N naming the current CUDA device); got 'cpu'.
    """
    if not (device in {"auto", "cuda"} or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError(
            f"ExecuTorch TensorRT requires cuda or auto (or cuda:N naming the current CUDA device); got {device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ExecuTorch TensorRT requires an available CUDA device.")
    if device.startswith("cuda:"):
        current = torch.cuda.current_device()
        if int(device[5:]) != current:
            raise ValueError(
                f"ExecuTorch TensorRT runs on the current CUDA device, cuda:{current}; got {device!r}. "
                f"Request auto, cuda or cuda:{current}, or restrict the visible GPUs with CUDA_VISIBLE_DEVICES."
            )


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str, options: Mapping[str, Any]) -> ExportRuntime:
    """Load an ExecuTorch method with its embedded delegate policy; it reads no runtime options.

    Each delegate accepts its own *device* values: XNNPACK ``auto`` or ``cpu``, CoreML ``auto``, QNN ``auto`` or
    ``qnn``, and TensorRT ``auto``, ``cuda`` or ``cuda:N`` where ``N`` is the current CUDA device.
    """
    _runtime_options("ExecuTorch", options, ())

    delegate = (metadata.backend or "xnnpack").lower()
    if delegate == "xnnpack":
        if device not in {"auto", "cpu"}:
            raise ValueError("ExecuTorch XNNPACK requires cpu or auto.")
        target = "cpu"
    elif delegate == "coreml":
        _require_apple("ExecuTorch CoreML")
        if device != "auto":
            raise ValueError("ExecuTorch CoreML uses an embedded compute policy; request auto.")
        target = "coreml"
    elif delegate == "qnn":
        if device not in {"auto", "qnn"}:
            raise ValueError("ExecuTorch QNN requires auto or qnn.")
        target = "qnn"
    elif delegate == "tensorrt":
        _check_tensorrt_device(device)
        try:
            # Importing the package registers the TensorRT delegate with ExecuTorch's runtime.
            import torch_tensorrt_executorch_runtime  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "torch_tensorrt_executorch_runtime":
                raise
            raise ImportError(
                "Running an ExecuTorch TensorRT .pte requires the Torch-TensorRT delegate. "
                "Install it with: pip install rfdetr[executorch-tensorrt]"
            ) from exc
        target = "cuda"
    else:
        raise ValueError(f"Unsupported ExecuTorch delegate {delegate!r}.")
    session = load_executorch_method(path)

    def execute(batch: torch.Tensor) -> list[Any]:
        """Clone mutable input storage before each ExecuTorch call."""
        return cast(list[Any], session.execute([torch.from_numpy(_input_array(batch, metadata)).clone()]))

    return ExportRuntime("executorch", metadata, session=session, device_policy=target, execute=execute)
