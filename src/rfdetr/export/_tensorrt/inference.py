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

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class TRTInference:
    """TensorRT inference engine."""

    def __init__(
        self,
        engine_path: str = "dino.trt",
        device: str | torch.device = "cuda:0",
        sync_mode: bool = False,
        max_batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        if not trt:
            raise ImportError("TensorRT is not installed. Please install TensorRT to use TRTInference.")

        self.engine_path = engine_path
        self.device = device
        self.sync_mode = sync_mode
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
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

    def get_dummy_input(self, batch_size: int) -> dict[str, Tensor]:
        blob: dict[str, Tensor] = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.info(f"make dummy input {name} with shape {binding.shape}")
                blob[name] = torch.rand(batch_size, *binding.shape[1:]).float().to("cuda:0")
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

    def get_bindings(
        self, engine: Any, context: Any, max_batch_size: int = 32, device: str | torch.device | None = None
    ) -> OrderedDict[str, Any]:
        """Build binddings."""
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                raise NotImplementedError

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_sync(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def run_async(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        if self.stream is None:
            raise RuntimeError("Async TensorRT inference requires a CUDA stream.")
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        outputs = {n: self.bindings[n].data for n in self.output_names}
        self.stream.synchronize()
        return outputs

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
        """
        explicit_batch_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with (
            trt.Builder(self.logger) as builder,
            builder.create_network(explicit_batch_flag) as network,
            trt.OnnxParser(network, self.logger) as parser,
            builder.create_builder_config() as config,
        ):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1024 MiB
            config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_file_path, "rb") as model:
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
