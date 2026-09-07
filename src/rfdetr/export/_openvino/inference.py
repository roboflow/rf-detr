# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OpenVINO inference utilities for exported RF-DETR models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class OpenVINOInference:
    """Inference wrapper for OpenVINO IR models.

    Example:
        .. code-block:: python

            from rfdetr.export._openvino.inference import OpenVINOInference

            model = OpenVINOInference("output/inference_model.xml")
            # Prepare input image (NCHW format, ImageNet normalized)
            outputs = model.infer(image_array)
            boxes, labels = outputs
    """

    def __init__(self, model_path: str | Path, device: str = "AUTO", cache_dir: str | None = None):
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
        try:
            import openvino as ov
        except ImportError:
            logger.error(
                'OpenVINO is not installed. Please run `pip install "rfdetr[openvino]"` and try again.',
            )
            raise

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

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]

        logger.info(f"Loaded OpenVINO model from {model_path}")
        logger.info(f"Input shape: {self.input_layer.partial_shape}")
        logger.info(f"Number of outputs: {len(self.output_layers)}")

    def infer(self, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Run inference on input data.

        Args:
            input_data: Input tensor in NCHW format (batch, channels, height, width).
                Should be ImageNet normalized [0.485, 0.456, 0.406] mean,
                [0.229, 0.224, 0.225] std.

        Returns:
            Tuple of output tensors (typically boxes, labels, and optionally masks/keypoints).
            Each array is a copy, so results stay valid after the next ``infer()`` call.
        """
        # Run inference
        self.infer_request.infer({self.input_layer: input_data})

        # Copy outputs: `get_output_tensor(i).data` is a view onto the reused infer-request
        # buffers, which the next `infer()` call overwrites in place.
        return tuple(np.copy(self.infer_request.get_output_tensor(i).data) for i in range(len(self.output_layers)))

    def __call__(self, input_data: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Alias for infer() to match typical model calling convention."""
        return self.infer(input_data)
