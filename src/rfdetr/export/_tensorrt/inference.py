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

For production TensorRT inference prefer the ``inference-models`` library, which covers RF-DETR across PyTorch, ONNX and
TensorRT with automatic backend selection.
"""

from __future__ import annotations

import contextlib
import time
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None

from rfdetr.export._tensorrt.exporter import Fp16Strategy, fp16_source_graph, resolve_fp16_strategy
from rfdetr.export.prepare import BATCH_AXIS
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class TRTInference:
    """Run a serialized TensorRT engine on torch tensors that already sit on its CUDA device.

    TensorRT places an engine, and every execution context created from it, on the CUDA device that is current when the
    engine is deserialized, and each launch must find that device current again. The runtime makes *device* current
    around both rather than relying on the caller's current device. Inputs are bound by pointer, never copied: each must
    be a contiguous tensor of the engine's input dtype on that device. That is the layout of an engine with linear
    (row-major) device I/O, which is what ``RFDETR.export(format="tensorrt")`` builds; vectorized formats such as
    ``chw32`` are not supported.

    Args:
        engine_path: Path to a ``.trt`` engine built on this machine's GPU and TensorRT version.
        device: CUDA device to load and run the engine on. A bare ``"cuda"`` is pinned to the current device at
            construction.
        sync_mode: Run with ``execute_v2`` instead of launching on a CUDA stream; the stream needs the
            ``tensorrt-bench`` extra (pycuda).
        verbose: Log TensorRT at VERBOSE rather than INFO.

    Raises:
        ImportError: If TensorRT, or pycuda for ``sync_mode=False``, is not installed.
        ValueError: If *device* is not a CUDA device, or the engine's tensor shapes cannot be resolved from its
            optimization profile (see :meth:`get_bindings`).
        RuntimeError: If TensorRT cannot deserialize the engine or create its execution context.
    """

    def __init__(
        self,
        engine_path: str = "dino.trt",
        device: str | torch.device = "cuda:0",
        sync_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        if not trt:
            raise ImportError("TensorRT is not installed. Please install TensorRT to use TRTInference.")

        self.engine_path = engine_path
        self.device = device
        self._engine_device = self._resolve_engine_device(device)
        self.sync_mode = sync_mode

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        with torch.cuda.device(self._engine_device):
            self.engine = self.load_engine(engine_path)

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(
                    f"TensorRT could not create an execution context for the engine at '{engine_path}'; its error "
                    "is in the TensorRT log above."
                )

            self.bindings = self.get_bindings(self.engine, self.context, self._engine_device)
            self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

            self.input_names = self.get_input_names()
            self.output_names = self.get_output_names()
            self._prime_context()
        self.stream = None

        if not self.sync_mode:
            if not cuda:
                raise ImportError(
                    "pycuda is not installed. Install the `tensorrt-bench` extra "
                    "(pip install 'rfdetr[tensorrt-bench]') to use TRTInference with async mode."
                )

            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler(device=self._engine_device)

    @staticmethod
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
            ValueError: If *device* is not a CUDA device; TensorRT runs on nothing else.

        Examples:
            >>> TRTInference._resolve_engine_device("cuda:1")
            device(type='cuda', index=1)
            >>> TRTInference._resolve_engine_device("cpu")
            Traceback (most recent call last):
            ...
            ValueError: TensorRT runs on CUDA devices only, got device='cpu'. Pass a CUDA device such as 'cuda:0'.
        """
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError(
                f"TensorRT runs on CUDA devices only, got device={device!r}. Pass a CUDA device such as 'cuda:0'."
            )
        return requested if requested.index is not None else torch.device("cuda", torch.cuda.current_device())

    def _prime_context(self) -> None:
        """Register the state that never changes again, so the per-call path only touches what does.

        The output buffers are allocated once and never move, so their addresses are registered here instead of on every
        call. The per-input shape memo starts empty rather than at the profile maximum :meth:`get_bindings` just
        declared: an unset entry can only cost one redundant ``set_input_shape`` on the first call, where a pre-filled
        one could skip a declaration the engine actually needs. The torch dtype each input must arrive in is read off
        the engine once, for :meth:`_check_input_memory`.
        """
        self._declared_shapes: dict[str, tuple[int, ...] | None] = dict.fromkeys(self.input_names)
        self._input_dtypes = {
            name: torch.from_numpy(np.empty(0, dtype=self.bindings[name].dtype)).dtype for name in self.input_names
        }
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.bindings[name].ptr))

    def get_dummy_input(self, batch_size: int) -> dict[str, Tensor]:
        """Build a random input for every engine input, in the dtype and on the device the engine reads it from.

        Args:
            batch_size: Batch to build; a static engine accepts only the batch it was built for.

        Returns:
            One contiguous tensor per engine input, keyed by input name, holding values drawn from ``[0, 1)`` and cast
            to that input's dtype.
        """
        blob: dict[str, Tensor] = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.info(f"make dummy input {name} with shape {binding.shape}")
                values = torch.rand(batch_size, *binding.shape[1:], device=self._engine_device)
                blob[name] = values.to(self._input_dtypes[name])
        return blob

    def load_engine(self, path: str) -> Any:
        """Deserialize the engine at *path* onto the current CUDA device.

        Args:
            path: Path to a serialized ``.trt`` engine.

        Returns:
            The deserialized TensorRT engine.

        Raises:
            RuntimeError: If TensorRT cannot deserialize the file. It reports that by returning ``None`` (with the
                reason in its log), for an engine built by another TensorRT version or GPU architecture as well as
                for a truncated or corrupt file.
        """
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not deserialize the engine at '{path}'; the reason is in the "
                "TensorRT log above. By default an engine only loads on the TensorRT version and GPU architecture "
                "that built it, and a truncated or corrupt file fails the same way. Rebuild it on this machine with "
                'RFDETR.export(format="tensorrt").'
            )
        return engine

    def get_input_names(self) -> list[str]:
        names: list[str] = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self) -> list[str]:
        names: list[str] = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    @staticmethod
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

    def get_bindings(
        self, engine: Any, context: Any, device: str | torch.device | None = None
    ) -> OrderedDict[str, Any]:
        """Allocate one device buffer per engine output, and record the shape of every tensor.

        Inputs get no buffer. :meth:`_bind_inputs` runs before every execution and points each input binding at the
        caller's own tensor, so a buffer allocated here would never be read; the binding keeps its resolved shape
        (:meth:`get_dummy_input` builds from it) and a ``0`` address placeholder, because :meth:`run_sync` hands
        ``execute_v2`` the whole address list positionally and the entry has to be there.

        A tensor whose batch axis is dynamic (``-1``, from an engine built with ``dynamic_batch=True``) is allocated at
        the shape the execution context resolves once every dynamic input has been declared at its profile maximum, so
        any batch within the profile fits. Output sizes come from TensorRT itself rather than from an input's batch --
        an engine is free to emit an output whose batch axis does not track its input's. :meth:`run_sync` /
        :meth:`run_async` then set the real input shape per call and return the outputs trimmed to it.

        Args:
            engine: A deserialized TensorRT engine.
            context: The execution context whose dynamic inputs get declared at their profile maximum.
            device: The device output buffers are allocated on. Defaults to the device this instance runs the engine on.

        Returns:
            One :class:`Binding` per engine tensor, keyed by tensor name, in engine iteration order.

        Raises:
            ValueError: If a tensor still carries an unresolved dimension after the inputs were declared, which
                ``np.empty`` would otherwise report as a bare "negative dimensions are not allowed".
        """
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr", "dynamic"))
        bindings = OrderedDict()
        buffer_device = self._engine_device if device is None else device
        self._declare_profile_max_inputs(engine, context)

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

    def _check_input_memory(self, name: str, tensor: Tensor) -> None:
        """Refuse a tensor the engine would misread: TensorRT reads a dense buffer of its own dtype off the pointer.

        Nothing copies or casts on the caller's behalf. A hidden copy would be timed as inference by :meth:`speed` and
        the benchmark, and would hide an input pipeline that produces the wrong layout.

        Args:
            name: Engine input the tensor is bound to.
            tensor: The caller's tensor for that input.

        Raises:
            ValueError: If *tensor* is on another device than the engine (a CPU tensor on a GPU engine ends in an
                illegal-address fault), has another dtype than the engine's input (TensorRT reads its own element size
                regardless), or is not contiguous (TensorRT would read a ``channels_last`` or sliced tensor as dense
                row-major memory).
        """
        if tensor.device != self._engine_device:
            raise ValueError(
                f"Input {name!r} is on device {tensor.device}, but this engine runs on {self._engine_device}. Move it "
                f"there with .to({str(self._engine_device)!r})."
            )
        expected_dtype = self._input_dtypes[name]
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

    def _describe_profile_refusal(self, name: str, shape: tuple[int, ...]) -> str:
        """Explain why the optimization profile refused *shape* for input *name*, and whether re-exporting fixes it.

        Only called once TensorRT has refused the shape, so the profile query stays off the per-call path.

        Args:
            name: Dynamic engine input whose shape was refused.
            shape: The refused shape.

        Returns:
            The error message, advising a larger ``max_batch_size`` only when the batch is the one axis above its
            profile range.
        """
        min_shape, _, max_shape = (
            tuple(int(dim) for dim in dims) for dims in self.engine.get_tensor_profile_shape(name, 0)
        )
        message = (
            f"Input {name!r} shape {shape} is outside the engine's optimization profile "
            f"(min {min_shape}, max {max_shape})."
        )
        # Compared per axis rather than against the maximum alone: an engine may make its image size dynamic too.
        image_fits = len(shape) == len(max_shape) and all(
            low <= dim <= high
            for dim, low, high in zip(
                shape[BATCH_AXIS + 1 :], min_shape[BATCH_AXIS + 1 :], max_shape[BATCH_AXIS + 1 :], strict=True
            )
        )
        if image_fits and shape[BATCH_AXIS] > max_shape[BATCH_AXIS]:
            message += " Export with a larger max_batch_size."
        return message

    def _bind_inputs(self, blob: Mapping[str, Tensor]) -> None:
        """Point the input bindings at *blob* and, for dynamic engines, declare this call's input shapes.

        Raises:
            ValueError: If a tensor is not memory the engine can read as-is (see :meth:`_check_input_memory`), if a
                dynamic input's shape falls outside the engine's optimization profile -- TensorRT reports that by
                returning ``False`` from ``set_input_shape`` rather than raising -- or if a static engine is handed a
                shape it was not built for. Executing anyway would hand back whatever the output buffers held from the
                previous call, or read past the end of the caller's tensor.
        """
        for name in self.input_names:
            binding = self.bindings[name]
            tensor = blob[name]
            self._check_input_memory(name, tensor)
            shape = tuple(tensor.shape)
            # Declaring a shape the context already holds is a no-op on TensorRT's side, so skip the round trip
            # when this input ran at the same shape last call -- the common case for a steady batch size.
            if binding.dynamic and shape != self._declared_shapes[name]:
                if not self.context.set_input_shape(name, shape):
                    raise ValueError(self._describe_profile_refusal(name, shape))
                self._declared_shapes[name] = shape
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
                    message += " Export with dynamic_batch=True to serve a range of batch sizes from one engine."
                raise ValueError(message)
            self.bindings_addr[name] = tensor.data_ptr()

    def _collect_outputs(self) -> dict[str, Tensor]:
        """Return the output buffers, trimmed to the batch the engine actually produced."""
        outputs: dict[str, Tensor] = {}
        for name in self.output_names:
            binding = self.bindings[name]
            produced = self.context.get_tensor_shape(name)[BATCH_AXIS] if binding.dynamic else None
            outputs[name] = binding.data if produced is None else binding.data[:produced]
        return outputs

    def run_sync(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Run inference synchronously and return the outputs, trimmed to the produced batch.

        Args:
            blob: One tensor per engine input, already on this engine's device.

        Returns:
            One tensor per engine output. A dynamic output is a view into a buffer the next call
            overwrites -- copy it before the next call if it needs to outlive that call.

        Raises:
            ValueError: If an input is refused before launch (see :meth:`_bind_inputs`).
            RuntimeError: If TensorRT reports the launch failed.
        """
        with torch.cuda.device(self._engine_device):
            self._bind_inputs(blob)
            # Not migrated to v3 alongside run_async: TensorRT exposes no synchronous v3 call -- execute_async_v3 is
            # the only v3 entry point, and it needs a CUDA stream and an explicit sync per launch. The sync path is
            # deliberately stream-free; __init__ only builds a stream, and only then requires pycuda, for async mode.
            if not self.context.execute_v2(list(self.bindings_addr.values())):
                raise RuntimeError("TensorRT execute_v2 reported a launch failure.")
            return self._collect_outputs()

    def run_async(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
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
        with torch.cuda.device(self._engine_device):
            self._bind_inputs(blob)
            if self.stream is None:
                raise RuntimeError("Async TensorRT inference requires a CUDA stream.")
            # execute_async_v2 (binding lists) is gone from TensorRT 11; the tensor-address API exists since 8.5. Only
            # the inputs are registered here -- the output addresses were set once in _prime_context and never move.
            for name in self.input_names:
                if not self.context.set_tensor_address(name, int(self.bindings_addr[name])):
                    raise RuntimeError(f"TensorRT refused the tensor address for input {name!r}.")
            if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                raise RuntimeError("TensorRT execute_async_v3 reported a launch failure.")
            # Drain the stream before reading the produced shapes: execute_async_v3 only enqueues the work, so until it
            # completes the context still reports the previous call's batch and _collect_outputs would trim to that.
            self.stream.synchronize()
            return self._collect_outputs()

    def __call__(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if self.sync_mode:
            return self.run_sync(blob)
        else:
            return self.run_async(blob)

    def synchronize(self) -> None:
        """Wait for this runtime's work: its stream in async mode, otherwise everything on the engine's device."""
        if self.sync_mode:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self._engine_device)
            return

        if self.stream is not None:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize(self._engine_device)

    def speed(self, blob: Mapping[str, Tensor], n: int) -> float:
        self.time_profile.reset()
        with self.time_profile:
            for _ in range(n):
                _ = self(blob)
        return self.time_profile.total / n

    def build_engine(self, onnx_file_path: str, engine_file_path: str, max_batch_size: int = 32) -> Any:
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
            The serialized engine, or ``None`` if the ONNX file failed to parse.

        Raises:
            Fp16CastUnsupportedError: If a strongly typed TensorRT needs the graph cast to fp16 and it
                cannot be (already fp16, or explicitly quantized).
            RuntimeError: If TensorRT parses the model but cannot build an engine from it, for example a
                dynamic-batch ONNX, since this builder declares no optimization profile. Nothing is written to
                *engine_file_path* then, so an engine already there is kept.

        Examples:
            >>> TRTInference.build_engine(trt_inference, "model.onnx", "model.trt")  # doctest: +SKIP
        """
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
                trt.Builder(self.logger) as builder,
                builder.create_network(network_flags) as network,
                trt.OnnxParser(network, self.logger) as parser,
                builder.create_builder_config() as config,
            ):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1024 MiB
                if use_fp16_flag:
                    config.set_flag(trt.BuilderFlag.FP16)

                with open(build_source, "rb") as model:
                    if not parser.parse(model.read()):
                        logger.error("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            logger.error(parser.get_error(error))
                        return None

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
        self.device = device
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
