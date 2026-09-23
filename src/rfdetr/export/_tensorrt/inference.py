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
    """TensorRT inference engine."""

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
        self.sync_mode = sync_mode

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.device)
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

        # self.time_profile = TimeProfiler()
        self.time_profile = TimeProfiler()

    def _prime_context(self) -> None:
        """Register the context state that never changes again, so the per-call path only touches what does.

        The output buffers are allocated once and never move, so their addresses are registered here instead of on every
        call. The per-input shape memo starts empty rather than at the profile maximum :meth:`get_bindings` just
        declared: an unset entry can only cost one redundant ``set_input_shape`` on the first call, where a pre-filled
        one could skip a declaration the engine actually needs.
        """
        self._declared_shapes: dict[str, tuple[int, ...] | None] = dict.fromkeys(self.input_names)
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.bindings[name].ptr))

    def get_dummy_input(self, batch_size: int) -> dict[str, Tensor]:
        blob: dict[str, Tensor] = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.info(f"make dummy input {name} with shape {binding.shape}")
                blob[name] = torch.rand(batch_size, *binding.shape[1:]).float().to(self.device)
        return blob

    def load_engine(self, path: str) -> Any:
        """Load engine."""
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

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
            device: The device output buffers are allocated on. Defaults to this instance's own device.

        Returns:
            One :class:`Binding` per engine tensor, keyed by tensor name, in engine iteration order.

        Raises:
            ValueError: If a tensor still carries an unresolved dimension after the inputs were declared, which
                ``np.empty`` would otherwise report as a bare "negative dimensions are not allowed".
        """
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr", "dynamic"))
        bindings = OrderedDict()
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
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr(), dynamic)

        return bindings

    def _bind_inputs(self, blob: Mapping[str, Tensor]) -> None:
        """Point the input bindings at *blob* and, for dynamic engines, declare this call's input shapes.

        Raises:
            ValueError: If a dynamic input's shape falls outside the engine's optimization profile -- TensorRT
                reports that by returning ``False`` from ``set_input_shape`` rather than raising -- or if a static
                engine is handed a shape it was not built for. Executing anyway would hand back whatever the output
                buffers held from the previous call, or read past the end of the caller's tensor.
        """
        for name in self.input_names:
            binding = self.bindings[name]
            shape = tuple(blob[name].shape)
            # Declaring a shape the context already holds is a no-op on TensorRT's side, so skip the round trip
            # when this input ran at the same shape last call -- the common case for a steady batch size.
            if binding.dynamic and shape != self._declared_shapes[name]:
                if not self.context.set_input_shape(name, shape):
                    raise ValueError(
                        f"Input {name!r} shape {shape} is outside the engine's optimization profile (batch up to "
                        f"{binding.shape[BATCH_AXIS]}, spatial {tuple(binding.shape[BATCH_AXIS + 1 :])}). "
                        "Export with a larger max_batch_size."
                    )
                self._declared_shapes[name] = shape
            if not binding.dynamic and shape != binding.shape:
                # Nothing declares a static engine's shape to TensorRT, so an unchecked mismatch is not reported at
                # all: the engine reads binding.shape elements from the blob's raw pointer whatever it holds.
                raise ValueError(
                    f"Input {name!r} shape {shape} does not match the fixed shape {binding.shape} this engine was "
                    "built for. Export with dynamic_batch=True to serve a range of batch sizes from one engine."
                )
            self.bindings_addr[name] = blob[name].data_ptr()

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
            RuntimeError: If TensorRT reports the launch failed.
        """
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
            RuntimeError: If no CUDA stream is available, or TensorRT reports the launch failed.
        """
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
        if self.sync_mode:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return

        if self.stream is not None:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

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
                with open(engine_file_path, "wb") as f:
                    f.write(serialized_engine)

                return serialized_engine


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self) -> None:
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
