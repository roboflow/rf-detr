# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Execute exported graphs and return their raw RF-DETR tensors."""

from __future__ import annotations

import asyncio
import inspect
import platform
import re
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from rfdetr.export._runtime.metadata import ExportMetadata


def _input_array(batch: torch.Tensor, metadata: ExportMetadata) -> np.ndarray[Any, Any]:
    """Convert a normalized NCHW batch to the artifact's input interface."""
    if batch.ndim != 4 or batch.shape[1] != metadata.input_shape[1]:
        raise ValueError(f"Expected an NCHW batch with {metadata.input_shape[1]} channels, got {tuple(batch.shape)}.")
    if batch.shape[0] == 0:
        raise ValueError("Export inference requires at least one image.")
    expected = metadata.input_shape
    if metadata.max_batch_size is not None and batch.shape[0] > metadata.max_batch_size:
        raise ValueError(f"Batch size {batch.shape[0]} exceeds export maximum {metadata.max_batch_size}.")
    for axis, (actual, size) in enumerate(zip(batch.shape, expected)):
        if size != -1 and actual != size:
            raise ValueError(f"Input axis {axis} must be {size}, got {actual}. Export a model for this batch or size.")
    if metadata.input_layout == "NHWC":
        batch = batch.permute(0, 2, 3, 1)
    try:
        dtype = np.dtype(metadata.input_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported export input dtype: {metadata.input_dtype!r}.") from exc
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"Export input dtype {dtype} needs quantization parameters, which this adapter cannot infer.")
    return np.ascontiguousarray(batch.detach().cpu().numpy(), dtype=dtype)


def _select_outputs(
    raw: dict[str, Any] | list[Any] | tuple[Any, ...], metadata: ExportMetadata
) -> dict[str, torch.Tensor]:
    """Map verified runtime outputs to the raw RF-DETR names."""
    outputs: dict[str, torch.Tensor] = {}
    for semantic, identifier in metadata.outputs.items():
        try:
            if isinstance(raw, dict):
                value = list(raw.values())[identifier] if isinstance(identifier, int) else raw[identifier]
            elif isinstance(identifier, int):
                value = raw[identifier]
            else:
                raise KeyError(identifier)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Export output {semantic!r} maps to missing runtime output {identifier!r}.") from exc
        if isinstance(value, torch.Tensor):
            output = value.detach().clone().cpu()
        else:
            if hasattr(value, "numpy"):
                value = value.numpy()
            output = torch.from_numpy(np.array(value, copy=True))
        if not output.is_floating_point():
            raise ValueError(
                f"Export output {semantic!r} has non-floating dtype {output.dtype}; quantization is unknown."
            )
        outputs[semantic] = output.float()
    if "pred_boxes" not in outputs or "pred_logits" not in outputs:
        raise ValueError("Export metadata must map pred_boxes and pred_logits.")
    boxes, logits = outputs["pred_boxes"], outputs["pred_logits"]
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or logits.ndim != 3 or boxes.shape[:2] != logits.shape[:2]:
        raise ValueError("Export boxes and logits have incompatible shapes.")
    if metadata.task == "segment" and "pred_masks" not in outputs:
        raise ValueError("Segmentation export has no pred_masks mapping.")
    if metadata.task == "keypoints" and "pred_keypoints" not in outputs:
        raise ValueError("Keypoint export has no pred_keypoints mapping.")
    for semantic in ("pred_masks", "pred_keypoints"):
        if semantic in outputs and outputs[semantic].shape[:2] != boxes.shape[:2]:
            raise ValueError(f"Export output {semantic!r} has a different batch or query count.")
    return outputs


class ExportRuntime:
    """A loaded export with one validated execution policy."""

    def __init__(
        self, backend: str, metadata: ExportMetadata, session: Any, device: str, input_name: str | int
    ) -> None:
        """Keep the selected backend, session, and artifact input interface."""
        self.metadata = metadata
        self.session = session
        self.input_name = input_name
        self.info = {"backend": backend, "device": device}
        self._coreai_bridge: _CoreAILoopBridge | None = None
        self._program: Any = None

    def run(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one batch and return owned CPU tensors."""
        array = _input_array(batch, self.metadata)
        backend = self.info["backend"]
        raw: dict[str, Any] | list[Any] | tuple[Any, ...]
        if backend == "onnx":
            names = list(self.session.get_outputs())
            raw = dict(zip((item.name for item in names), self.session.run(None, {self.input_name: array})))
        elif backend == "tensorrt":
            tensor = torch.from_numpy(array).to(self.session.engine_device)
            raw = self.session({self.input_name: tensor})
        elif backend == "openvino":
            raw = self.session.infer(array)
        elif backend in {"tflite", "litert"}:
            detail = self.session.get_input_details()[0]
            if tuple(detail["shape"]) != tuple(array.shape):
                signature = tuple(detail.get("shape_signature", detail["shape"]))
                if len(signature) != 4 or any(
                    size != -1 and size != actual for size, actual in zip(signature, array.shape)
                ):
                    raise ValueError(f"TFLite input shape {tuple(array.shape)} is outside graph signature {signature}.")
                self.session.resize_tensor_input(self.input_name, array.shape, strict=True)
                self.session.allocate_tensors()
            self.session.set_tensor(self.input_name, array)
            self.session.invoke()
            details = self.session.get_output_details()
            raw = [self.session.get_tensor(detail["index"]) for detail in details]
            if any(isinstance(key, str) for key in self.metadata.outputs.values()):
                raw = {detail["name"]: value for detail, value in zip(details, raw)}
        elif backend == "coreml":
            prediction = self.session.predict({self.input_name: array})
            names = [item.name for item in self.session.get_spec().description.output]
            raw = {name: prediction[name] for name in names}
        elif backend == "executorch":
            raw = self.session.execute([torch.from_numpy(array).clone()])
        elif backend == "coreai":
            assert self._coreai_bridge is not None
            raw = self._coreai_bridge.run(_run_coreai(self.session, self.input_name, array, self.metadata))
        else:
            raise RuntimeError(f"Unknown export runtime {backend!r}.")
        return _select_outputs(raw, self.metadata)


async def _await_result(value: Any) -> Any:
    """Resolve a Core AI operation on runtime versions with async execution."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_coreai(
    session: Any, input_name: str | int, array: np.ndarray[Any, Any], metadata: ExportMetadata
) -> dict[str, Any]:
    """Execute Core AI on its owning loop and copy its name-indexed result into a mapping."""
    from coreai.runtime import NDArray

    result = await _await_result(session({input_name: NDArray(array)}))
    identifiers = list(metadata.outputs.values())
    if any(not isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("Core AI output mappings must use names.")
    names = [identifier for identifier in identifiers if isinstance(identifier, str)]
    outputs: dict[str, Any] = {}
    for name in names:
        value = result[name]
        if hasattr(value, "numpy"):
            value = value.numpy()
        outputs[name] = np.array(value, copy=True)
    return outputs


def _serve_coreai_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Keep one event loop alive for a loaded Core AI model and its predictions."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _stop_coreai_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Stop an unused Core AI loop without keeping its runtime alive."""
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if threading.current_thread() is not thread:
        thread.join(timeout=1)
    if not loop.is_running():
        loop.close()


class _CoreAILoopBridge:
    """Run the synchronous public API on one long-lived Core AI event loop."""

    def __init__(self) -> None:
        """Start the dedicated event loop used by one Core AI session."""
        self.loop = asyncio.new_event_loop()
        self._call_lock = threading.Lock()
        self.thread = threading.Thread(target=_serve_coreai_loop, args=(self.loop,), daemon=True, name="rfdetr-coreai")
        self.thread.start()
        self._finalizer = weakref.finalize(self, _stop_coreai_loop, self.loop, self.thread)

    def run(self, operation: Any) -> Any:
        """Wait for one Core AI operation while its event loop runs on another thread."""
        with self._call_lock:
            return asyncio.run_coroutine_threadsafe(operation, self.loop).result()


def _require_apple(format_name: str) -> None:
    """Reject Apple runtimes on other operating systems."""
    if platform.system() != "Darwin":
        raise RuntimeError(f"{format_name} inference requires macOS and its native runtime.")


def load_runtime(path: str | Path, metadata: ExportMetadata, device: str = "auto") -> ExportRuntime:
    """Load an artifact with the requested runtime device policy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    backend = metadata.format.lower()
    if metadata.task == "backbone":
        raise ValueError("A backbone-only export cannot produce predictions.")
    if backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("ONNX inference requires onnxruntime or onnxruntime-gpu.") from exc
        providers = ort.get_available_providers()
        provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
        if device == "auto":
            provider = "CUDAExecutionProvider" if provider in providers else "CPUExecutionProvider"
        elif device not in {"cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError(f"ONNX device {device!r} is unsupported. Use cpu, cuda:N, or auto.")
        if provider not in providers:
            raise RuntimeError(f"ONNX provider {provider} is unavailable. Installed providers: {providers}.")
        from rfdetr.export._onnx.inference import _create_onnx_session

        device_id = int(device[5:]) if device.startswith("cuda:") else 0
        requested: list[str | tuple[str, dict[str, Any]]] = (
            [(provider, {"device_id": device_id})] if provider == "CUDAExecutionProvider" else [provider]
        )
        session = _create_onnx_session(path, providers=requested)
        if provider not in session.get_providers():
            raise RuntimeError(f"ONNX session did not activate requested provider {provider}.")
        if device != "auto" and provider == "CUDAExecutionProvider" and hasattr(session, "disable_fallback"):
            session.disable_fallback()
        (input_info,) = session.get_inputs()
        if input_info.name != metadata.input_name:
            raise ValueError(f"ONNX input is {input_info.name!r}; metadata says {metadata.input_name!r}.")
        onnx_dtypes = {"float32": "float", "float16": "float16", "float64": "double"}
        expected_type = f"tensor({onnx_dtypes.get(metadata.input_dtype, metadata.input_dtype)})"
        if input_info.type != expected_type:
            raise ValueError(f"ONNX input dtype {input_info.type!r} disagrees with export metadata.")
        shape = metadata.input_shape
        if metadata.input_layout == "NHWC":
            shape = (shape[0], shape[2], shape[3], shape[1])
        if len(input_info.shape) != 4:
            raise ValueError(f"ONNX input rank must be 4, got {len(input_info.shape)}.")
        if any(isinstance(got, int) and want != -1 and got != want for got, want in zip(input_info.shape, shape)):
            raise ValueError(f"ONNX input shape {input_info.shape} disagrees with export metadata {shape}.")
        output_names = {item.name for item in session.get_outputs()}
        missing = {key for key in metadata.outputs.values() if isinstance(key, str)} - output_names
        if missing:
            raise ValueError(f"ONNX output names absent from graph: {sorted(missing)}.")
        return ExportRuntime(backend, metadata, session, provider, input_info.name)
    if backend == "tensorrt":
        if device == "auto":
            device = "cuda:0"
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires an available CUDA device.")
        from rfdetr.export._tensorrt.inference import TRTInference

        session = TRTInference(str(path), device=device, sync_mode=True)
        if len(session.input_names) != 1 or session.input_names[0] != metadata.input_name:
            raise ValueError("TensorRT input binding disagrees with export metadata.")
        binding = session.bindings[session.input_names[0]]
        if len(binding.shape) != 4:
            raise ValueError(f"TensorRT input rank must be 4, got {len(binding.shape)}.")
        if np.dtype(binding.dtype) != np.dtype(metadata.input_dtype):
            raise ValueError("TensorRT input dtype disagrees with export metadata.")
        if any(got != want for got, want in zip(binding.shape[1:], metadata.input_shape[1:])):
            raise ValueError("TensorRT input spatial shape disagrees with export metadata.")
        if metadata.input_shape[0] != -1 and binding.shape[0] != metadata.input_shape[0]:
            raise ValueError("TensorRT fixed batch size disagrees with export metadata.")
        missing = {name for name in metadata.outputs.values() if isinstance(name, str)} - set(session.output_names)
        if missing:
            raise ValueError(f"TensorRT output names absent from engine: {sorted(missing)}.")
        return ExportRuntime(backend, metadata, session, str(session.engine_device), session.input_names[0])
    if backend == "openvino":
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
        from rfdetr.export._openvino.inference import OpenVINOInference

        session = OpenVINOInference(path, device=target, inference_precision="f32")
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
        return ExportRuntime(backend, metadata, session, target, metadata.input_name)
    if backend in {"tflite", "litert"}:
        if device not in {"auto", "cpu"}:
            raise ValueError("TFLite inference supports cpu or auto only.")
        from rfdetr.export._tflite.inference import _create_interpreter

        session = _create_interpreter(path)
        (input_info,) = session.get_input_details()
        if isinstance(metadata.input_name, str) and input_info.get("name") != metadata.input_name:
            raise ValueError("TFLite input name disagrees with export metadata.")
        actual_shape = tuple(int(size) for size in input_info["shape"])
        if len(actual_shape) != 4:
            raise ValueError(f"TFLite input rank must be 4, got {len(actual_shape)}.")
        expected_shape = metadata.input_shape
        if metadata.input_layout == "NHWC":
            expected_shape = (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        if any(want != -1 and want != got for want, got in zip(expected_shape, actual_shape)):
            raise ValueError(f"TFLite input shape {actual_shape} disagrees with metadata {expected_shape}.")
        if np.dtype(input_info["dtype"]) != np.dtype(metadata.input_dtype):
            raise ValueError("TFLite input dtype disagrees with export metadata.")
        output_details = session.get_output_details()
        for name in metadata.outputs.values():
            if isinstance(name, int) and not (0 <= name < len(output_details)):
                raise ValueError(f"TFLite output index {name} is absent from graph.")
            if isinstance(name, str) and name not in {detail.get("name") for detail in output_details}:
                raise ValueError(f"TFLite output name {name!r} is absent from graph.")
        return ExportRuntime(backend, metadata, session, "cpu", input_info["index"])
    if backend == "coreml":
        _require_apple("CoreML")
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError("CoreML inference requires coremltools.") from exc
        units = {"auto": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}
        if device not in units:
            raise ValueError(
                "CoreML accepts auto or cpu. Its GPU and Neural Engine policies also permit CPU execution."
            )
        session = ct.models.MLModel(str(path), compute_units=units[device])
        spec = session.get_spec()
        (input_info,) = spec.description.input
        if metadata.input_name not in {0, input_info.name}:
            raise ValueError("CoreML input name disagrees with export metadata.")
        feature_type = input_info.type
        if feature_type.WhichOneof("Type") != "multiArrayType":
            raise ValueError("CoreML input must be a multiArrayType.")
        array_type = feature_type.multiArrayType
        input_shape = tuple(array_type.shape)
        if len(input_shape) != 4:
            raise ValueError(f"CoreML input rank must be 4, got {len(input_shape)}.")
        expected_shape = metadata.input_shape
        if metadata.input_layout == "NHWC":
            expected_shape = (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        if input_shape != expected_shape:
            raise ValueError(f"CoreML input shape {input_shape} disagrees with export metadata {expected_shape}.")
        coreml_dtypes = {65552: "float16", 65568: "float32", 65600: "float64"}
        if coreml_dtypes.get(array_type.dataType) != metadata.input_dtype:
            raise ValueError("CoreML input dtype disagrees with export metadata.")
        output_names = {item.name for item in spec.description.output}
        for name in metadata.outputs.values():
            if isinstance(name, str) and name not in output_names:
                raise ValueError(f"CoreML output name {name!r} is absent from model.")
            if isinstance(name, int) and not (0 <= name < len(output_names)):
                raise ValueError(f"CoreML output position {name} is absent from model.")
        return ExportRuntime(backend, metadata, session, device, input_info.name)
    if backend == "executorch":
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
        else:
            raise ValueError(f"Unsupported ExecuTorch delegate {delegate!r}.")
        try:
            from executorch.runtime import Runtime
        except ImportError as exc:
            raise ImportError("ExecuTorch inference requires a compatible executorch runtime.") from exc
        program = Runtime.get().load_program(str(path))
        session = program.load_method("forward")
        runtime = ExportRuntime(backend, metadata, session, target, metadata.input_name)
        runtime._program = program
        return runtime
    if backend == "coreai":
        _require_apple("Core AI")
        try:
            from coreai.runtime import AIModel, SpecializationOptions
        except ImportError as exc:
            raise ImportError("Core AI inference requires coreai-core and macOS 27 or later.") from exc
        if device in {"gpu", "ane"}:
            raise ValueError(
                "Core AI gpu and ane select a preferred compute unit, not a strict device. Use auto or cpu."
            )
        if device not in {"auto", "cpu"}:
            raise ValueError("Core AI device must be auto or cpu.")
        if metadata.task == "keypoints" and metadata.input_dtype == "float16" and device == "auto":
            device = "cpu"
        if device == "cpu":
            options = SpecializationOptions.cpu_only()
        else:
            options = SpecializationOptions.default()

        async def load_coreai() -> Any:
            """Load and select the Core AI entry point."""
            loaded = await _await_result(AIModel.load(path, options))
            return await _await_result(loaded.load_function("main"))

        bridge = _CoreAILoopBridge()
        session = bridge.run(load_coreai())
        runtime = ExportRuntime(backend, metadata, session, device, metadata.input_name)
        runtime._coreai_bridge = bridge
        return runtime
    raise ValueError(f"Unsupported export format {metadata.format!r}.")
