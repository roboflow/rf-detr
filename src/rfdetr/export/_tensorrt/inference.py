# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Reference TensorRT runtime for an engine built by :mod:`rfdetr.export._tensorrt.exporter`.

Device-managed rather than session-tier: the caller hands over torch tensors already on the GPU and gets the engine's
output bindings back, with CUDA stream management and synchronization handled here. Nothing decodes detections — that
stays with the caller.

Use :class:`rfdetr.inference.RFDETRInference` for decoded predictions from native models and exported artifacts.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from deprecate import TargetMode, deprecated
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None

from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.export._tensorrt.exporter import Fp16Strategy, fp16_source_graph, resolve_fp16_strategy
from rfdetr.export.prepare import BATCH_AXIS
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _dynamic_batch_advice(max_batch: int) -> str:
    """Name the ``RFDETR.export`` settings for an engine that serves every batch up to *max_batch*.

    The TensorRT exporter refuses ``dynamic_batch=True`` without ``max_batch_size``, so the advice names both, as
    settings the caller can pass exactly as written.

    Args:
        max_batch: The largest batch the re-exported engine has to accept.

    Returns:
        The advice sentence, with a leading space so it can follow the refusal.

    Examples:
        >>> _dynamic_batch_advice(5)
        ' Export with dynamic_batch=True, max_batch_size=5 to serve batches 1 to 5 from one engine.'
    """
    return (
        f" Export with dynamic_batch=True, max_batch_size={max_batch} to serve batches 1 to {max_batch} from one "
        "engine."
    )


def _resolve_engine_device(device: str | torch.device) -> torch.device:
    """Return the CUDA device an engine requested on *device* is loaded and run on.

    A bare ``"cuda"`` is pinned to the current device here, once: TensorRT places the engine on the device current
    at deserialization, and resolving ``"cuda"`` again on a later call would follow the caller's current device
    away from the engine and its buffers.

    Args:
        device: The device the caller asked for.

    Returns:
        A CUDA ``torch.device`` with an explicit index.

    Raises:
        ValueError: If *device* does not parse as a device, or is not a CUDA device; TensorRT runs on nothing else.

    Examples:
        >>> _resolve_engine_device("cuda:1")
        device(type='cuda', index=1)
        >>> _resolve_engine_device("cpu")
        Traceback (most recent call last):
        ...
        ValueError: TensorRT runs on CUDA devices only, got device='cpu'. Pass a CUDA device such as 'cuda:0'.
    """
    try:
        requested = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(
            f"device={device!r} is not a valid device string. Pass a CUDA device such as 'cuda:0'."
        ) from exc
    if requested.type != "cuda":
        raise ValueError(
            f"TensorRT runs on CUDA devices only, got device={device!r}. Pass a CUDA device such as 'cuda:0'."
        )
    return requested if requested.index is not None else torch.device("cuda", torch.cuda.current_device())


#: The shape of every engine input, in input-name order: what identifies one captured graph.
_InputShapes = tuple[tuple[int, ...], ...]


class _CapturedGraph(NamedTuple):
    """One captured launch of an engine at fixed input shapes, and the views of the static input buffers it reads."""

    graph: torch.cuda.CUDAGraph
    inputs: dict[str, Tensor]


@dataclass
class _TensorRTSession:
    """TensorRT engine state shared by module functions and compatibility facades."""

    engine_path: str
    device: str | torch.device
    engine_device: torch.device
    sync_mode: bool
    logger: Any
    engine: Any
    context: Any
    bindings: OrderedDict[str, Any]
    bindings_addr: OrderedDict[str, int]
    input_names: list[str]
    output_names: list[str]
    _declared_shapes: dict[str, tuple[int, ...] | None]
    _input_dtypes: dict[str, torch.dtype]
    stream: Any | None
    time_profile: TimeProfiler
    _engine_host_code_allowed: bool = False
    # The stream a CUDA graph runs on; None unless the session was built with cuda_graph=True, which is how every
    # dispatch tells the graph path from the other two.
    _graph_stream: torch.cuda.Stream | None = None
    # Marks where the caller's stream stands at each call; one event serves every call (see _run_tensorrt_graph).
    _caller_ready: torch.cuda.Event | None = None
    _graphs: dict[_InputShapes, _CapturedGraph] = field(default_factory=dict)
    _static_inputs: dict[str, Tensor] = field(default_factory=dict)


def _load_tensorrt_session(
    engine_path: str = "dino.trt",
    device: str | torch.device = "cuda:0",
    sync_mode: bool = False,
    verbose: bool = False,
    *,
    cuda_graph: bool = False,
    engine_host_code_allowed: bool = False,
) -> _TensorRTSession:
    """Load a TensorRT engine and allocate its bindings on one pinned CUDA device.

    Args:
        engine_path: Path to a ``.trt`` engine.
        device: CUDA device to load and run the engine on.
        sync_mode: Run with ``execute_v2`` instead of launching on a CUDA stream.
        verbose: Log TensorRT at VERBOSE rather than INFO.
        cuda_graph: Capture the engine's launch into a CUDA graph on the first call and replay it afterwards.
        engine_host_code_allowed: Let TensorRT deserialize an engine that carries host code.

    Returns:
        The loaded session.

    Raises:
        ImportError: If TensorRT, or pycuda for ``sync_mode=False`` without *cuda_graph*, is not installed.
        ValueError: If *cuda_graph* is combined with ``sync_mode=True`` or requested for an engine whose optimization
            profile lets an input take more than one shape, or *engine_host_code_allowed* is not a ``bool``.
        RuntimeError: If TensorRT cannot deserialize the engine or create its execution context.
    """
    if not trt:
        raise ImportError("TensorRT is not installed. Please install TensorRT to use TensorRT inference.")
    if cuda_graph and sync_mode:
        raise ValueError(
            "cuda_graph=True cannot be combined with sync_mode=True: sync_mode launches with execute_v2, which "
            "cannot be captured into a CUDA graph. Drop sync_mode to run the graph on its own CUDA stream."
        )
    # A truthy string such as "false" from a config file must not switch a security setting on.
    if not isinstance(engine_host_code_allowed, bool):
        raise ValueError(f"engine_host_code_allowed must be a bool, got {engine_host_code_allowed!r}.")
    engine_device = _resolve_engine_device(device)
    session = _TensorRTSession(
        engine_path=engine_path,
        device=device,
        engine_device=engine_device,
        sync_mode=sync_mode,
        logger=trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO),
        engine=None,
        context=None,
        bindings=OrderedDict(),
        bindings_addr=OrderedDict(),
        input_names=[],
        output_names=[],
        _declared_shapes={},
        _input_dtypes={},
        stream=None,
        time_profile=TimeProfiler(device=engine_device),
        _engine_host_code_allowed=engine_host_code_allowed,
    )
    with torch.cuda.device(engine_device):
        session.engine = _deserialize_tensorrt_engine(session, engine_path)
        if cuda_graph:
            _refuse_varying_graph_shapes(session.engine)
        session.context = session.engine.create_execution_context()
        if session.context is None:
            raise RuntimeError(
                f"TensorRT could not create an execution context for the engine at '{engine_path}'; its error "
                "is in the TensorRT log above."
            )
        session.bindings = _allocate_tensorrt_bindings(session, session.engine, session.context, engine_device)
        session.bindings_addr = OrderedDict((name, binding.ptr) for name, binding in session.bindings.items())
        session.input_names = _get_tensorrt_input_names(session)
        session.output_names = _get_tensorrt_output_names(session)
        _prime_tensorrt_context(session)
    if cuda_graph:
        session._graph_stream = torch.cuda.Stream(device=engine_device)  # type: ignore[no-untyped-call]
        session._caller_ready = torch.cuda.Event()  # type: ignore[no-untyped-call]
    elif not sync_mode:
        if not cuda:
            raise ImportError(
                "pycuda is not installed. Install the `tensorrt-bench` extra "
                "(pip install 'rfdetr[tensorrt-bench]') to use async TensorRT inference."
            )
        session.stream = cuda.Stream()
    return session


def _prime_tensorrt_context(session: _TensorRTSession) -> None:
    """Register the state that never changes again, so the per-call path only touches what does.

    The output buffers are allocated once and never move, so their addresses are registered here instead of on every
    call. The per-input shape memo starts empty rather than at the profile maximum :meth:`get_bindings` just
    declared: an unset entry can only cost one redundant ``set_input_shape`` on the first call, where a pre-filled
    one could skip a declaration the engine actually needs. The torch dtype each input must arrive in is read off
    the engine once, for :meth:`_check_input_memory`.
    """
    session._declared_shapes = dict.fromkeys(session.input_names)
    session._input_dtypes = {
        name: torch.from_numpy(np.empty(0, dtype=session.bindings[name].dtype)).dtype for name in session.input_names
    }
    for name in session.output_names:
        session.context.set_tensor_address(name, int(session.bindings[name].ptr))


def _get_tensorrt_dummy_input(session: _TensorRTSession, batch_size: int) -> dict[str, Tensor]:
    """Build a random input for every engine input, in the dtype and on the device the engine reads it from.

    Args:
        batch_size: Batch to build; a static engine accepts only the batch it was built for.

    Returns:
        One contiguous tensor per engine input, keyed by input name, holding values drawn from ``[0, 1)`` and cast
        to that input's dtype.
    """
    blob: dict[str, Tensor] = {}
    for name, binding in session.bindings.items():
        if session.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            logger.info(f"make dummy input {name} with shape {binding.shape}")
            values = torch.rand(batch_size, *binding.shape[1:], device=session.engine_device)
            blob[name] = values.to(session._input_dtypes[name])
    return blob


def _deserialize_tensorrt_engine(session: _TensorRTSession, path: str) -> Any:
    """Deserialize the engine at *path* onto the current CUDA device.

    Args:
        path: Path to a serialized ``.trt`` engine.

    Returns:
        The deserialized TensorRT engine.

    Raises:
        RuntimeError: If TensorRT cannot deserialize the file. It reports that by returning ``None`` (with the
            reason in its log), for an engine built by another TensorRT version or GPU architecture as well as
            for a truncated or corrupt file, and for an engine that carries host code when
            ``engine_host_code_allowed`` was not set.
    """
    trt.init_libnvinfer_plugins(session.logger, "")
    with open(path, "rb") as f, trt.Runtime(session.logger) as runtime:
        # Set only on request, and before deserializing: TensorRT refuses an engine that carries host code
        # otherwise, and it has to know that before it reads the file, not after.
        if session._engine_host_code_allowed:
            try:
                runtime.engine_host_code_allowed = True
            except AttributeError:
                logger.warning(
                    f"engine_host_code_allowed=True had no effect: TensorRT {trt.__version__} has no such "
                    "switch on its runtime, so an engine that carries host code loads only if that release "
                    "needs none."
                )
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        host_code_hint = (
            ""
            if session._engine_host_code_allowed
            else "If it was exported with trt_version_compatible=True by TensorRT 11, load it with "
            "RFDETRInference(path, runtime_options={'engine_host_code_allowed': True}), or "
            "engine_host_code_allowed=True on TRTInference (only for a file you trust). Otherwise: "
        )
        raise RuntimeError(
            f"TensorRT {trt.__version__} could not deserialize the engine at '{path}'; the reason is in the "
            f"TensorRT log above. {host_code_hint}By default an engine only loads on the kind of GPU and the "
            "TensorRT version that built it, and a truncated or corrupt file fails the same way. Rebuild it on "
            'this machine with RFDETR.export(format="tensorrt"), or export it once for other machines with '
            "trt_hardware_compatibility (other GPUs) or trt_version_compatible (other TensorRT 11 releases)."
        )
    return engine


def _get_tensorrt_input_names(session: _TensorRTSession) -> list[str]:
    names: list[str] = []
    for _, name in enumerate(session.engine):
        if session.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            names.append(name)
    return names


def _get_tensorrt_output_names(session: _TensorRTSession) -> list[str]:
    names: list[str] = []
    for _, name in enumerate(session.engine):
        if session.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            names.append(name)
    return names


def _declare_profile_max_inputs(engine: Any, context: Any) -> None:
    """Declare every dynamic input at its own profile maximum, so the context can resolve the other tensors.

    TensorRT derives a tensor's concrete shape from the input shapes set on the execution context. Declaring the
    inputs first is therefore what makes :meth:`get_bindings` able to ask the context how large each output
    really is, instead of assuming an output's batch is some input's batch.

    Args:
        engine: A deserialized TensorRT engine.
        context: The execution context whose buffers are being allocated.

    Raises:
        ValueError: If TensorRT refuses a maximum it reported itself, which means *context* is not on the
            optimization profile the shape was read from.
    """
    for name in engine:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if engine.get_tensor_shape(name)[BATCH_AXIS] != -1:
            continue
        _, _, profile_max = engine.get_tensor_profile_shape(name, 0)
        max_shape = tuple(int(dim) for dim in profile_max)
        if not context.set_input_shape(name, max_shape):
            raise ValueError(
                f"TensorRT refused input {name!r} at the profile maximum {max_shape} it reported itself; the "
                "execution context is not on optimization profile 0."
            )


def _refuse_varying_graph_shapes(engine: Any) -> None:
    """Refuse, for ``cuda_graph=True``, an engine whose profile lets an input take more than one shape.

    Every captured graph replays through the runtime's one execution context. A call at another shape would
    declare new input shapes on that context between replays, which TensorRT documents as undefined behaviour
    for a captured graph, so only an engine with one shape per input -- on the batch axis too -- may be graphed.
    The decision reads profile 0, the one this runtime runs, not the ``-1`` axes: an engine reports an axis as
    dynamic when any of its profiles varies it, so one whose profile 0 pins every axis still qualifies here. An
    input with a fixed batch is left to :meth:`get_bindings`: it has one shape, or a dynamic axis no mode can size
    a buffer for.

    Args:
        engine: A deserialized TensorRT engine.

    Raises:
        ValueError: If the profile lets a dynamic-batch input take more than one shape, on any axis.
    """
    for name in engine:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if engine.get_tensor_shape(name)[BATCH_AXIS] != -1:
            continue
        min_shape, _, max_shape = (tuple(int(dim) for dim in dims) for dims in engine.get_tensor_profile_shape(name, 0))
        if min_shape != max_shape:
            raise ValueError(
                f"cuda_graph=True needs an engine whose input shapes are fixed, but optimization profile 0 lets "
                f"input {name!r} range from {min_shape} to {max_shape}. Every captured graph replays through one "
                "TensorRT execution context, and a call at another shape would change that context's input "
                "shapes between replays, which TensorRT documents as undefined behaviour. Build the runtime "
                "with sync_mode=True instead of cuda_graph=True, or export the engine without "
                "dynamic_batch=True."
            )


def _allocate_tensorrt_bindings(
    session: _TensorRTSession, engine: Any, context: Any, device: str | torch.device | None = None
) -> OrderedDict[str, Any]:
    """Allocate one device buffer per engine output, and record the shape of every tensor.

    Inputs get no buffer. :meth:`_bind_inputs` runs before every execution and points each input binding at the
    caller's own tensor, so a buffer allocated here would never be read; the binding keeps its resolved shape
    (:meth:`get_dummy_input` builds from it) and a ``0`` address placeholder, because :meth:`run_sync` hands
    ``execute_v2`` the whole address list positionally and the entry has to be there.

    A tensor whose batch axis is dynamic (``-1``, from an engine built with ``dynamic_batch=True``) is allocated at
    the shape the execution context resolves once every dynamic input has been declared at its profile maximum, so
    any batch within the profile fits. Output sizes come from TensorRT itself rather than from an input's batch --
    an engine is free to emit an output whose batch axis does not track its input's. :meth:`run_sync`,
    :meth:`run_async` and :meth:`run_graph` then set the real input shape per call and return the outputs trimmed
    to it.

    Args:
        engine: A deserialized TensorRT engine.
        context: The execution context whose dynamic inputs get declared at their profile maximum.
        device: The device output buffers are allocated on. Defaults to the device recorded in the session.

    Returns:
        One :class:`Binding` per engine tensor, keyed by tensor name, in engine iteration order.

    Raises:
        ValueError: If a tensor still carries an unresolved dimension after the inputs were declared, which
            ``np.empty`` would otherwise report as a bare "negative dimensions are not allowed".
    """
    Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr", "dynamic"))
    bindings = OrderedDict()
    buffer_device = session.engine_device if device is None else device
    _declare_profile_max_inputs(engine, context)

    for name in engine:
        engine_shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        dynamic = engine_shape[BATCH_AXIS] == -1
        # A static tensor is never queried on the context: its shape is fully known on the engine.
        resolved = context.get_tensor_shape(name) if dynamic else engine_shape
        shape = tuple(int(dim) for dim in resolved)
        if -1 in shape:
            axis = shape.index(-1)
            raise ValueError(
                f"Engine tensor {name!r} has an unresolved dimension at axis {axis} (resolved shape {shape}) "
                "after every dynamic input was declared at its profile maximum: either no input carries an "
                f"optimization profile, or the engine makes an axis other than {BATCH_AXIS} dynamic, which this "
                "runtime cannot size a buffer for."
            )
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            bindings[name] = Binding(name, dtype, shape, None, 0, dynamic)
            continue
        data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(buffer_device)
        bindings[name] = Binding(name, dtype, shape, data, data.data_ptr(), dynamic)

    return bindings


def _check_tensorrt_input_memory(session: _TensorRTSession, name: str, tensor: Tensor) -> None:
    """Refuse a tensor the engine would misread: TensorRT reads a dense buffer of its own dtype off the pointer.

    Nothing copies or casts on the caller's behalf, except the copy into a static buffer that ``cuda_graph=True``
    documents. A hidden copy would be timed as inference by :meth:`speed` and the benchmark, and would hide an
    input pipeline that produces the wrong layout.

    Args:
        name: Engine input the tensor is bound to.
        tensor: The caller's tensor for that input.

    Raises:
        ValueError: If *tensor* is on another device than the engine (a CPU tensor on a GPU engine ends in an
            illegal-address fault), has another dtype than the engine's input (TensorRT reads its own element size
            regardless), or is not contiguous (TensorRT would read a ``channels_last`` or sliced tensor as dense
            row-major memory).
    """
    if tensor.device != session.engine_device:
        raise ValueError(
            f"Input {name!r} is on device {tensor.device}, but this engine runs on {session.engine_device}. Move it "
            f"there with .to({str(session.engine_device)!r})."
        )
    expected_dtype = session._input_dtypes[name]
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"Input {name!r} has dtype {tensor.dtype}, but this engine reads {expected_dtype}. Convert it with "
            f".to({expected_dtype})."
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"Input {name!r} is not contiguous (strides {tensor.stride()}), and TensorRT reads it as dense "
            "row-major memory. Pass tensor.contiguous()."
        )


def _describe_tensorrt_profile_refusal(session: _TensorRTSession, name: str, shape: tuple[int, ...]) -> str:
    """Explain why the optimization profile refused *shape* for input *name*, and whether re-exporting fixes it.

    Only called once TensorRT has refused the shape, so the profile query stays off the per-call path.

    Args:
        name: Dynamic engine input whose shape was refused.
        shape: The refused shape.

    Returns:
        The error message, advising a re-export with a larger ``max_batch_size`` only when the batch alone exceeds
        the profile.
    """
    min_shape, _, max_shape = (
        tuple(int(dim) for dim in dims) for dims in session.engine.get_tensor_profile_shape(name, 0)
    )
    message = (
        f"Input {name!r} shape {shape} is outside the engine's optimization profile (min {min_shape}, max {max_shape})."
    )
    # Compared per axis rather than against the maximum alone: an engine may make its image size dynamic too.
    image_fits = len(shape) == len(max_shape) and all(
        low <= dim <= high
        for dim, low, high in zip(
            shape[BATCH_AXIS + 1 :], min_shape[BATCH_AXIS + 1 :], max_shape[BATCH_AXIS + 1 :], strict=True
        )
    )
    if image_fits and shape[BATCH_AXIS] > max_shape[BATCH_AXIS]:
        message += _dynamic_batch_advice(shape[BATCH_AXIS])
    return message


def _bind_tensorrt_inputs(session: _TensorRTSession, blob: Mapping[str, Tensor]) -> None:
    """Point the input bindings at *blob* and, for dynamic engines, declare this call's input shapes.

    Raises:
        ValueError: If a tensor is not memory the engine can read as-is (see :meth:`_check_input_memory`), if a
            dynamic input's shape falls outside the engine's optimization profile -- TensorRT reports that by
            returning ``False`` from ``set_input_shape`` rather than raising -- or if a static engine is handed a
            shape it was not built for. Executing anyway would hand back whatever the output buffers held from the
            previous call, or read past the end of the caller's tensor.
    """
    for name in session.input_names:
        binding = session.bindings[name]
        tensor = blob[name]
        _check_tensorrt_input_memory(session, name, tensor)
        shape = tuple(tensor.shape)
        # Declaring a shape the context already holds is a no-op on TensorRT's side, so skip the round trip
        # when this input ran at the same shape last call -- the common case for a steady batch size.
        if binding.dynamic and shape != session._declared_shapes[name]:
            if not session.context.set_input_shape(name, shape):
                raise ValueError(_describe_tensorrt_profile_refusal(session, name, shape))
            session._declared_shapes[name] = shape
        if not binding.dynamic and shape != binding.shape:
            # Nothing declares a static engine's shape to TensorRT, so an unchecked mismatch is not reported at
            # all: the engine reads binding.shape elements from the blob's raw pointer whatever it holds.
            message = (
                f"Input {name!r} shape {shape} does not match the fixed shape {binding.shape} this engine was "
                "built for."
            )
            # A dynamic engine's profile starts at batch 1, so it would refuse an empty batch too.
            only_batch_differs = (
                len(shape) == len(binding.shape) and shape[BATCH_AXIS + 1 :] == binding.shape[BATCH_AXIS + 1 :]
            )
            if only_batch_differs and shape[BATCH_AXIS] > 0:
                # The engine's batch is the batch_size it was exported with, and the exporter needs
                # batch_size <= max_batch_size, so the bound covers it as well as this batch: re-running the
                # original export with the advised settings then passes its checks.
                message += _dynamic_batch_advice(max(shape[BATCH_AXIS], binding.shape[BATCH_AXIS]))
            raise ValueError(message)
        session.bindings_addr[name] = tensor.data_ptr()


def _collect_tensorrt_outputs(session: _TensorRTSession) -> dict[str, Tensor]:
    """Return the output buffers, trimmed to the batch the engine actually produced."""
    outputs: dict[str, Tensor] = {}
    for name in session.output_names:
        binding = session.bindings[name]
        produced = session.context.get_tensor_shape(name)[BATCH_AXIS] if binding.dynamic else None
        outputs[name] = binding.data if produced is None else binding.data[:produced]
    return outputs


def _run_tensorrt_sync(session: _TensorRTSession, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Run inference synchronously and return the outputs, trimmed to the produced batch.

    Args:
        blob: One tensor per engine input, already on this engine's device.

    Returns:
        One tensor per engine output. A dynamic output is a view into a buffer the next call
        overwrites -- copy it before the next call if it needs to outlive that call.

    Raises:
        ValueError: If an input is refused before launch (see :meth:`_bind_inputs`).
        RuntimeError: If the runtime was built with ``cuda_graph=True``, or TensorRT reports the launch failed.
    """
    if session._graph_stream is not None:
        raise RuntimeError(
            "run_sync cannot run a runtime built with cuda_graph=True: execute_v2 would re-bind the execution "
            "context its captured graph replays through. Call the runtime, or run_graph, instead."
        )
    with torch.cuda.device(session.engine_device):
        _bind_tensorrt_inputs(session, blob)
        # Not migrated to v3 alongside run_async: TensorRT exposes no synchronous v3 call -- execute_async_v3 is
        # the only v3 entry point, and it needs a CUDA stream and an explicit sync per launch. The sync path is
        # deliberately stream-free; loading builds a stream only for the other two modes.
        if not session.context.execute_v2(list(session.bindings_addr.values())):
            raise RuntimeError("TensorRT execute_v2 reported a launch failure.")
        return _collect_tensorrt_outputs(session)


def _run_tensorrt_async(session: _TensorRTSession, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Run inference on this engine's CUDA stream and return the outputs, trimmed to the produced batch.

    Args:
        blob: One tensor per engine input, already on this engine's device.

    Returns:
        One tensor per engine output. A dynamic output is a view into a buffer the next call
        overwrites -- copy it before the next call if it needs to outlive that call.

    Raises:
        ValueError: If an input is refused before launch (see :meth:`_bind_inputs`).
        RuntimeError: If no CUDA stream is available, or TensorRT reports the launch failed.
    """
    with torch.cuda.device(session.engine_device):
        _bind_tensorrt_inputs(session, blob)
        if session.stream is None:
            raise RuntimeError("Async TensorRT inference requires a CUDA stream.")
        # execute_async_v2 (binding lists) is gone from TensorRT 11; the tensor-address API exists since 8.5. Only
        # the inputs are registered here -- the output addresses were set once in _prime_context and never move.
        for name in session.input_names:
            if not session.context.set_tensor_address(name, int(session.bindings_addr[name])):
                raise RuntimeError(f"TensorRT refused the tensor address for input {name!r}.")
        _launch_tensorrt(session, session.stream.handle)
        # Drain the stream before reading the produced shapes: execute_async_v3 only enqueues the work, so until it
        # completes the context still reports the previous call's batch and _collect_outputs would trim to that.
        session.stream.synchronize()
        return _collect_tensorrt_outputs(session)


def _launch_tensorrt(session: _TensorRTSession, stream_handle: int) -> None:
    """Enqueue one execution of the engine on the CUDA stream behind *stream_handle*.

    Args:
        session: The loaded session whose execution context launches.
        stream_handle: Raw handle of the stream to launch on.

    Raises:
        RuntimeError: If TensorRT reports the launch failed.
    """
    if not session.context.execute_async_v3(stream_handle=stream_handle):
        raise RuntimeError("TensorRT execute_async_v3 reported a launch failure.")


def _static_tensorrt_input(session: _TensorRTSession, name: str, like: Tensor) -> Tensor:
    """Return a view shaped like *like* onto the one static buffer input *name* reads from under every graph.

    The buffer is allocated on first use at the binding's shape, which ``cuda_graph=True`` only accepts for an
    engine whose profile fixes it (see :func:`_refuse_varying_graph_shapes`), so the view covers the whole buffer.
    A call copies its input in, replays, and waits before the next call copies its own.

    Args:
        session: The graph-enabled session that owns the buffer.
        name: The engine input.
        like: A validated tensor for that input, which sets the shape, dtype and device of the view.

    Returns:
        A view onto the static buffer of input *name*.
    """
    buffer = session._static_inputs.get(name)
    if buffer is None:
        # A buffer made under torch.inference_mode() (a common way to warm up) is an inference tensor, which no
        # later call outside that mode may copy into.
        with torch.inference_mode(False):
            buffer = session._static_inputs[name] = torch.empty(
                math.prod(session.bindings[name].shape), dtype=like.dtype, device=like.device
            )
    return buffer[: like.numel()].view(like.shape)


def _capture_tensorrt_graph(
    session: _TensorRTSession, blob: Mapping[str, Tensor], stream: torch.cuda.Stream
) -> _CapturedGraph:
    """Capture one launch of the engine at *blob*'s shape, reading from static copies of its inputs.

    Runs after :func:`_bind_tensorrt_inputs`, so the context already holds this call's input shapes. The context's
    input addresses are pointed at the static copies here and stay there, because a graph replays the pointers it
    was captured with. One launch happens before the capture so that TensorRT finishes any lazy set-up, which a
    capture forbids.

    Args:
        session: The graph-enabled session to capture.
        blob: One validated tensor per engine input.
        stream: The stream to launch and capture on.

    Returns:
        The captured graph and the static input buffers it reads.

    Raises:
        RuntimeError: If TensorRT refuses a launch or an address, or the launch cannot be captured into a graph.
    """
    # no_grad: the buffers live as long as the runtime, so an input that requires grad must not leave its autograd
    # history on them, and a view made under no_grad (a warm-up) must stay writable for one that does.
    with torch.cuda.stream(stream), torch.no_grad():
        inputs = {name: _static_tensorrt_input(session, name, blob[name]) for name in session.input_names}
        for name, buffer in inputs.items():
            buffer.copy_(blob[name])
            if not session.context.set_tensor_address(name, int(buffer.data_ptr())):
                raise RuntimeError(f"TensorRT refused the tensor address for input {name!r}.")
        _launch_tensorrt(session, stream.cuda_stream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    caller_stream = torch.cuda.current_stream(session.engine_device)
    try:
        # "thread_local" checks only this thread's CUDA calls for capture safety, as the PyTorch graph path does;
        # the default "global" mode also fails on unsafe calls from other threads, such as another model's malloc.
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            _launch_tensorrt(session, stream.cuda_stream)
    except RuntimeError as err:
        # torch.cuda.graph does not leave the stream it switched to when ending the capture fails, which would
        # leave every later torch call of the caller running on ours.
        torch.cuda.set_stream(caller_stream)
        reason = str(err).rstrip(".")
        raise RuntimeError(
            f"TensorRT could not be captured into a CUDA graph: {reason}. A failed capture can leave CUDA unusable "
            "for the rest of the process, so a process restart may be required; then build the runtime with "
            "sync_mode=True instead of cuda_graph=True to launch the engine directly."
        ) from err
    return _CapturedGraph(graph, inputs)


def _run_tensorrt_graph(session: _TensorRTSession, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Run inference by replaying a CUDA graph and return the outputs, trimmed to the produced batch.

    The first call captures the graph (see :func:`_capture_tensorrt_graph`); loading already refused an engine whose
    input shapes could differ on a later call. Every call then copies its inputs into the graph's static buffers,
    replays it, and waits for the stream once, as :func:`_run_tensorrt_async` does.

    Args:
        session: A session loaded with ``cuda_graph=True``.
        blob: One tensor per engine input, already on this engine's device.

    Returns:
        One tensor per engine output. A dynamic output is a view into a buffer the next call
        overwrites -- copy it before the next call if it needs to outlive that call.

    Raises:
        ValueError: If an input is refused before launch (see :func:`_bind_tensorrt_inputs`).
        RuntimeError: If the session was loaded without ``cuda_graph=True``, TensorRT refuses a launch, or the launch
            cannot be captured into a graph.
    """
    stream = session._graph_stream
    if stream is None or session._caller_ready is None:
        raise RuntimeError("Graph replay requires a runtime built with cuda_graph=True.")
    with torch.cuda.device(session.engine_device):
        _bind_tensorrt_inputs(session, blob)
        # The inputs may still be in flight on the caller's stream, so the graph's stream waits for them first.
        # Unlike Stream.wait_stream, which allocates an event per call, the runtime re-records its own one: a wait
        # orders only against the record made before it, so recording again on the next call cannot loosen it.
        session._caller_ready.record(torch.cuda.current_stream(session.engine_device))
        stream.wait_event(session._caller_ready)
        shapes = tuple(tuple(blob[name].shape) for name in session.input_names)
        captured = session._graphs.get(shapes)
        if captured is None:
            captured = session._graphs[shapes] = _capture_tensorrt_graph(session, blob, stream)
        with torch.cuda.stream(stream), torch.no_grad():
            for name in session.input_names:
                captured.inputs[name].copy_(blob[name], non_blocking=True)
            captured.graph.replay()
        stream.synchronize()
        # A replay enqueues nothing at this call's shapes, so the produced shapes come from the ones
        # _bind_tensorrt_inputs declared, which TensorRT resolves without executing.
        return _collect_tensorrt_outputs(session)


def _run_tensorrt_session(session: _TensorRTSession, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if session._graph_stream is not None:
        return _run_tensorrt_graph(session, blob)
    if session.sync_mode:
        return _run_tensorrt_sync(session, blob)
    return _run_tensorrt_async(session, blob)


def _synchronize_tensorrt_session(session: _TensorRTSession) -> None:
    """Wait for this runtime's work: its graph or async stream, otherwise everything on the engine's device."""
    if session._graph_stream is not None:
        session._graph_stream.synchronize()
        return

    if session.sync_mode:
        if torch.cuda.is_available():
            torch.cuda.synchronize(session.engine_device)
        return

    if session.stream is not None:
        session.stream.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize(session.engine_device)


def _speed_tensorrt_session(session: _TensorRTSession, blob: Mapping[str, Tensor], n: int) -> float:
    """Return the mean wall-clock time of *n* calls to this runtime, in seconds.

    The timed region is whatever ``__call__`` does: input validation (:meth:`_check_input_memory`) and the
    ``torch.cuda.device`` scope run inside it, so the returned mean includes their overhead alongside the
    TensorRT launch itself -- it is not engine-only. With ``cuda_graph=True`` that includes the copy of the inputs
    into the static buffers.

    Args:
        blob: One tensor per engine input, already on this engine's device.
        n: Number of calls to time.

    Returns:
        Mean seconds per call.
    """
    session.time_profile.reset()
    with session.time_profile:
        for _ in range(n):
            _ = _run_tensorrt_session(session, blob)
    return session.time_profile.total / n


def _build_tensorrt_engine(
    onnx_file_path: str,
    engine_file_path: str,
    max_batch_size: int = 32,
    *,
    trt_logger: Any | None = None,
) -> Any:
    """Takes an ONNX file and creates a TensorRT engine to run inference with
    http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57

    FP16 is always requested. Following
    :meth:`~rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine`, TensorRT
    11+ has no FP16 builder flag and takes precision from the graph, so the graph is cast first.

    Args:
        onnx_file_path: Path to the float32 ``.onnx`` model to build from.
        engine_file_path: Path the serialized engine is written to.
        max_batch_size: Unused; retained for call-site compatibility.

    Returns:
        The serialized engine.

    Raises:
        Fp16CastUnsupportedError: If a strongly typed TensorRT needs the graph cast to fp16 and it
            cannot be (already fp16, or explicitly quantized).
        RuntimeError: If TensorRT cannot parse the ONNX file, or parses it but cannot build an engine from
            it -- for example a dynamic-batch ONNX, since this builder declares no optimization profile.
            Nothing is written to *engine_file_path* in either case, so an engine already there is kept.

    Note:
        The builder runs on the current CUDA device and does not change the active device.

    Examples:
        >>> _build_tensorrt_engine("model.onnx", "model.trt")  # doctest: +SKIP
    """
    trt_logger = trt_logger or trt.Logger(trt.Logger.INFO)
    # Strong typing, not the absent flag, is what decides this -- see ``resolve_fp16_strategy``.
    strategy, trt_version = resolve_fp16_strategy(trt)
    use_fp16_flag = strategy is Fp16Strategy.BUILDER_FLAG

    with contextlib.ExitStack() as cleanup:
        # Only the parser reads the cast intermediate; the caller's own path keeps naming its model.
        build_source = onnx_file_path

        if strategy is Fp16Strategy.CAST_GRAPH:
            build_source = cleanup.enter_context(fp16_source_graph(onnx_file_path))
            logger.info(f"TensorRT {trt_version} is strongly typed; benchmarking a cast FP16 graph")
        elif strategy is Fp16Strategy.UNAVAILABLE:
            logger.warning(
                "TensorRT %s does not expose the FP16 builder flag; benchmarking an FP32 engine "
                "instead, so these latencies are not comparable to FP16 numbers.",
                trt_version,
            )

        # TensorRT 11 removed EXPLICIT_BATCH along with the FP16 flag -- explicit batch is the
        # only mode there, so the flag set is empty. Resolved inside the block: on 11 the absent
        # member would otherwise raise after the cast graph is written, leaking it.
        network_flags = (
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
            else 0
        )
        with (
            trt.Builder(trt_logger) as builder,
            builder.create_network(network_flags) as network,
            trt.OnnxParser(network, trt_logger) as parser,
            builder.create_builder_config() as config,
        ):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1024 MiB
            if use_fp16_flag:
                config.set_flag(trt.BuilderFlag.FP16)

            with open(build_source, "rb") as model:
                if not parser.parse(model.read()):
                    parser_errors = "; ".join(str(parser.get_error(error)) for error in range(parser.num_errors))
                    raise RuntimeError(
                        f"TensorRT could not parse the ONNX file '{onnx_file_path}'; the parser reported: "
                        f"{parser_errors}"
                    )

            serialized_engine = builder.build_serialized_network(network, config)
            # TensorRT reports a failed build by returning None; opening the target first would truncate it.
            if serialized_engine is None:
                raise RuntimeError(
                    f"TensorRT could not build an engine from '{onnx_file_path}'; the reason is in the TensorRT "
                    "log above. If the model has a dynamic batch axis, it cannot be built here, because this "
                    "builder declares no optimization profile: build it with "
                    'RFDETR.export(format="tensorrt", dynamic_batch=True, max_batch_size=...) instead.'
                )
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)

            return serialized_engine


#: Construction warning of :class:`TRTInference`; pyDeprecate fills in the versions.
_TRT_INFERENCE_DEPRECATION = (
    "`TRTInference` was deprecated in v%(deprecated_in)s and will be removed in v%(remove_in)s."
    " Use RFDETRInference(engine_path).predict(image) for decoded predictions, with runtime_options for its"
    " cuda_graph, engine_host_code_allowed and verbose settings."
)


class TRTInference:
    """Deprecated compatibility facade for TensorRT inference, deprecated in v1.12.0 and removed in v2.0.0.

    Runs a serialized TensorRT engine on torch tensors that already sit on its CUDA device, through the shared session
    functions behind :class:`rfdetr.inference.RFDETRInference`. Inputs are bound by pointer, never copied, except with
    ``cuda_graph=True``, which copies each into a static buffer on every call. A runtime is not safe to call from
    several threads, in any mode: use one runtime per thread.

    Args:
        engine_path: Path to a ``.trt`` engine. By default it must have been built on this machine's GPU and TensorRT
            version; an engine exported with ``trt_hardware_compatibility`` or ``trt_version_compatible`` also loads
            on the GPUs and TensorRT versions that option covers.
        device: CUDA device to load and run the engine on. A bare ``"cuda"`` is pinned to the current device at
            construction.
        sync_mode: Run with ``execute_v2`` instead of launching on a CUDA stream. The default async mode needs the
            ``tensorrt-bench`` extra (pycuda) for its stream; ``sync_mode=True`` and ``cuda_graph=True`` do not.
        verbose: Log TensorRT at VERBOSE rather than INFO.
        cuda_graph: Capture the engine's launch into a CUDA graph on the first call and replay it on every later
            call, which removes most of the per-call launch cost at small batch sizes. Only an engine whose
            optimization profile 0 fixes the shape of every input qualifies; an engine exported with
            ``dynamic_batch=True`` is refused. The first call pays for the capture: it synchronizes the device, empties
            torch's allocator cache, runs the engine once, and captures it. Cannot be combined with ``sync_mode``.
        engine_host_code_allowed: Let TensorRT deserialize an engine that carries host code, which an engine built by
            TensorRT 11 with ``trt_version_compatible=True`` does. Off by default: loading such an engine runs code it
            contains, so turn it on only for a file you built yourself or otherwise trust.

    Raises:
        ImportError: If TensorRT, or pycuda for ``sync_mode=False`` without *cuda_graph*, is not installed.
        ValueError: If *device* is not a CUDA device, the engine's tensor shapes cannot be resolved from its
            optimization profile, or *cuda_graph* is combined with ``sync_mode=True`` or requested for an engine whose
            optimization profile lets an input take more than one shape.
        RuntimeError: If TensorRT cannot deserialize the engine or create its execution context.
    """

    _runtime_state: _TensorRTSession

    # Deprecate __init__ rather than the class: deprecated_class returns a proxy, and this facade's __getattr__ and
    # __setattr__ forwarding, and callers using TRTInference.__new__(TRTInference), need the real class.
    @deprecated(  # type: ignore[untyped-decorator]  # pyDeprecate types its wrapper as Callable[..., Any]
        target=TargetMode.NOTIFY,
        deprecated_in="1.12.0",
        remove_in="2.0.0",
        num_warns=-1,
        template_mgs=_TRT_INFERENCE_DEPRECATION,
    )
    def __init__(
        self,
        engine_path: str = "dino.trt",
        device: str | torch.device = "cuda:0",
        sync_mode: bool = False,
        verbose: bool = False,
        *,
        cuda_graph: bool = False,
        engine_host_code_allowed: bool = False,
    ) -> None:
        """Create the legacy facade; every construction emits a ``FutureWarning``."""
        self._runtime_state = _load_tensorrt_session(
            engine_path,
            device,
            sync_mode,
            verbose,
            cuda_graph=cuda_graph,
            engine_host_code_allowed=engine_host_code_allowed,
        )

    def __getattr__(self, name: str) -> Any:
        """Expose the runtime session's existing public attributes and helpers."""
        state = object.__getattribute__(self, "_runtime_state")
        return getattr(state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep legacy attribute updates on the shared session state."""
        if name in {"_runtime_state", "engine_device"} or "_runtime_state" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._runtime_state, name, value)

    @property
    def engine_device(self) -> torch.device:
        """Return the resolved CUDA device without allowing the facade to retarget its engine."""
        return self._runtime_state.engine_device

    def __call__(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Run the engine through the shared TensorRT session function."""
        return _run_tensorrt_session(self._runtime_state, blob)

    def run_sync(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Run synchronous TensorRT inference for compatibility with existing callers."""
        return _run_tensorrt_sync(self._runtime_state, blob)

    def run_async(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Run asynchronous TensorRT inference for compatibility with existing callers."""
        return _run_tensorrt_async(self._runtime_state, blob)

    def run_graph(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Replay the captured CUDA graph for runtimes built with ``cuda_graph=True``."""
        return _run_tensorrt_graph(self._runtime_state, blob)

    def synchronize(self) -> None:
        """Wait for TensorRT work started through this facade."""
        _synchronize_tensorrt_session(self._runtime_state)

    def speed(self, blob: Mapping[str, Tensor], n: int) -> float:
        """Measure mean inference time through this facade."""
        return _speed_tensorrt_session(self._runtime_state, blob, n)

    def get_dummy_input(self, batch_size: int) -> dict[str, Tensor]:
        """Build a dummy input matching this engine's input bindings."""
        return _get_tensorrt_dummy_input(self._runtime_state, batch_size)

    def load_engine(self, path: str) -> Any:
        """Deserialize an engine on the session's current CUDA device."""
        return _deserialize_tensorrt_engine(self._runtime_state, path)

    def get_input_names(self) -> list[str]:
        """Return this engine's input tensor names."""
        return _get_tensorrt_input_names(self._runtime_state)

    def get_output_names(self) -> list[str]:
        """Return this engine's output tensor names."""
        return _get_tensorrt_output_names(self._runtime_state)

    @staticmethod
    def _refuse_varying_graph_shapes(engine: Any) -> None:
        """Refuse, for ``cuda_graph=True``, an engine whose profile lets an input take more than one shape."""
        _refuse_varying_graph_shapes(engine)

    def get_bindings(
        self, engine: Any, context: Any, device: str | torch.device | None = None
    ) -> OrderedDict[str, Any]:
        """Allocate TensorRT output bindings for an engine and execution context."""
        return _allocate_tensorrt_bindings(self._runtime_state, engine, context, device)

    @staticmethod
    def _declare_profile_max_inputs(engine: Any, context: Any) -> None:
        """Declare dynamic inputs at their optimization profile maximum."""
        _declare_profile_max_inputs(engine, context)

    def build_engine(self, onnx_file_path: str, engine_file_path: str, max_batch_size: int = 32) -> Any:
        """Build a TensorRT engine through the legacy benchmark builder."""
        return _build_tensorrt_engine(onnx_file_path, engine_file_path, max_batch_size, trt_logger=self.logger)


class TimeProfiler(contextlib.ContextDecorator):
    """Accumulate wall-clock time across ``with`` blocks, waiting for queued CUDA work at each edge.

    Args:
        device: CUDA device to wait for. ``None`` waits for the current device.

    Examples:
        >>> profiler = TimeProfiler()
        >>> with profiler:
        ...     pass
        >>> profiler.total >= 0.0
        True
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = None if device is None else torch.device(device)
        self.total = 0.0
        self.start = 0.0

    def __enter__(self) -> "TimeProfiler":
        self.start = self.time()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.total += self.time() - self.start

    def reset(self) -> None:
        self.total = 0.0

    def time(self) -> float:
        """Return ``time.perf_counter()`` once the profiled device has finished its queued work."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


#: The ``runtime_options`` keys the TensorRT loader reads, each a ``bool`` that defaults to ``False``.
_TENSORRT_RUNTIME_OPTIONS = ("cuda_graph", "engine_host_code_allowed", "verbose")


def load_export_runtime(path: str | Path, metadata: ExportMetadata, device: str, options: Mapping[str, Any]) -> Any:
    """Load a TensorRT engine while preserving device tensors through execution.

    The engine runs with ``execute_v2`` by default, which needs no pycuda stream. ``cuda_graph=True`` captures the
    launch into a CUDA graph on its own stream instead; :func:`_load_tensorrt_session` refuses it for an engine whose
    optimization profile lets an input shape vary.

    Args:
        path: Serialized ``.trt`` or ``.engine`` file.
        metadata: The artifact's inference metadata.
        device: ``auto`` (``cuda:0``), ``cuda`` or ``cuda:N``.
        options: Any of ``cuda_graph``, ``engine_host_code_allowed`` and ``verbose``, as on :class:`TRTInference`.

    Returns:
        The loaded runtime.

    Raises:
        ValueError: If *options* holds another key or a non-``bool`` value, or the engine disagrees with *metadata*.
        RuntimeError: If no CUDA device is available.
    """
    from rfdetr.export._runtime.adapters import ExportRuntime, _runtime_options

    settings = _runtime_options("TensorRT", options, _TENSORRT_RUNTIME_OPTIONS)
    for key, value in settings.items():
        # A truthy string such as "false" from a config file must not switch a setting on.
        if not isinstance(value, bool):
            raise ValueError(f"TensorRT runtime option {key} must be a bool, got {value!r}.")
    cuda_graph = settings.get("cuda_graph", False)
    if device == "auto":
        device = "cuda:0"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("TensorRT requires an available CUDA device.")

    session = _load_tensorrt_session(
        str(path),
        device=device,
        sync_mode=not cuda_graph,
        verbose=settings.get("verbose", False),
        cuda_graph=cuda_graph,
        engine_host_code_allowed=settings.get("engine_host_code_allowed", False),
    )
    if len(session.input_names) != 1 or session.input_names[0] != metadata.input_name:
        raise ValueError("TensorRT input binding disagrees with export metadata.")
    binding = session.bindings[session.input_names[0]]
    if len(binding.shape) != 4:
        raise ValueError(f"TensorRT input rank must be 4, got {len(binding.shape)}.")
    if np.dtype(binding.dtype) != np.dtype(metadata.input_dtype):
        raise ValueError("TensorRT input dtype disagrees with export metadata.")
    if metadata.input_layout != "NCHW":
        raise ValueError("TensorRT inference requires an NCHW input.")
    if any(got != want for got, want in zip(binding.shape[1:], metadata.input_shape[1:])):
        raise ValueError("TensorRT input spatial shape disagrees with export metadata.")
    if metadata.input_shape[0] != -1 and binding.shape[0] != metadata.input_shape[0]:
        raise ValueError("TensorRT fixed batch size disagrees with export metadata.")
    missing = {name for name in metadata.outputs.values() if isinstance(name, str)} - set(session.output_names)
    if missing:
        raise ValueError(f"TensorRT output names absent from engine: {sorted(missing)}.")
    input_dtype = torch.from_numpy(np.empty(0, dtype=binding.dtype)).dtype

    def execute(batch: torch.Tensor) -> dict[str, Tensor]:
        """Feed the engine a contiguous tensor on its own CUDA device."""
        tensor = batch.to(device=session.engine_device, dtype=input_dtype).contiguous()
        # execute_v2 has no stream argument; wait for torch's input work before it reads the pointer. A graph
        # replay needs no host wait: _run_tensorrt_graph orders its stream after the caller's with an event.
        if not cuda_graph and session.engine_device.type == "cuda":
            torch.cuda.current_stream(session.engine_device).synchronize()
        return _run_tensorrt_session(session, {session.input_names[0]: tensor})

    return ExportRuntime(
        "tensorrt",
        metadata,
        session,
        str(session.engine_device),
        session.input_names[0],
        execute,
        device=session.engine_device,
    )
