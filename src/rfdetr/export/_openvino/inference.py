# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OpenVINO inference utilities for exported RF-DETR models."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._openvino.exporter import _check_openvino_available
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class OpenVINOInference:
    """Inference wrapper for OpenVINO IR models.

    Import it from its public path, :mod:`rfdetr.export.inference` — this module is private and its
    location is not part of the public API.

    Session-tier by design: it takes already-preprocessed NCHW tensors and returns the model's raw
    output tensors. Decoding those into detections is the caller's job (see
    :doc:`the export guide </learn/export>`).

    A single instance is safe to call from multiple threads: ``infer()`` is guarded by an
    internal lock, since OpenVINO's ``InferRequest.infer()`` is not thread-safe on a shared
    request object (concurrent calls would silently corrupt each other's output buffers).
    The lock serializes calls made through the same instance; for parallel throughput, create
    one ``OpenVINOInference`` per worker thread instead.

    Example:
        .. code-block:: python

            from rfdetr.export.inference import OpenVINOInference

            model = OpenVINOInference("output/inference_model.xml")
            # Prepare input image (NCHW format, ImageNet normalized)
            outputs = model.infer(image_array)
            boxes, labels = outputs
    """

    def __init__(self, model_path: str | Path, device: str = "AUTO", cache_dir: str | None = None) -> None:
        """Initialize OpenVINO inference session.

        Args:
            model_path: Path to the OpenVINO IR model (.xml file).
            device: Device the model is compiled for, e.g. ``"AUTO"``, ``"CPU"``, ``"GPU"`` or ``"NPU"``.
            cache_dir: Directory holding the compiled-model cache. When set, OpenVINO reuses the
                compiled kernels across process starts instead of recompiling the model every time.

        Raises:
            ImportError: If OpenVINO is not installed.
            FileNotFoundError: If the model file doesn't exist.
        """
        _check_openvino_available()
        import openvino as ov

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize OpenVINO runtime
        core = ov.Core()
        if cache_dir is not None:
            # Must be set before compilation so compiled kernels are reused across process starts.
            core.set_property({"CACHE_DIR": cache_dir})
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.infer_request = self.compiled_model.create_infer_request()
        # Guards infer_request.infer() + get_output_tensor(): both touch the same shared
        # buffers, which are not safe for concurrent access from multiple threads.
        self._infer_lock = threading.Lock()

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]

        logger.info(f"Loaded OpenVINO model from {model_path}")
        logger.info(f"Input shape: {self.input_layer.partial_shape}")
        logger.info(f"Number of outputs: {len(self.output_layers)}")

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
        if input_data.dtype != np.float32 or not input_data.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"infer() requires a C-contiguous float32 array, got dtype={input_data.dtype} "
                f"contiguous={input_data.flags['C_CONTIGUOUS']}. Construct mean/std with "
                "dtype=np.float32 and finish preprocessing with np.ascontiguousarray(...)."
            )

        with self._infer_lock:
            # Run inference
            self.infer_request.infer({self.input_layer: input_data})

            # Copy outputs: `get_output_tensor(i).data` is a view onto the reused infer-request
            # buffers, which the next `infer()` call overwrites in place.
            return tuple(np.copy(self.infer_request.get_output_tensor(i).data) for i in range(len(self.output_layers)))

    def __call__(self, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Alias for infer() to match typical model calling convention."""
        return self.infer(input_data)
