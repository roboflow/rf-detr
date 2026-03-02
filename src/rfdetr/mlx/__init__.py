# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MLX backend for RF-DETR inference on Apple Silicon.

Provides native Metal-accelerated inference using MLX, achieving up to 6x
speedup over PyTorch MPS on Apple Silicon hardware (M1-M4).

Usage::

    from rfdetr import RFDETRNano

    model = RFDETRNano()
    model.optimize_for_inference(backend="mlx")
    detections = model.predict(image)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfdetr.mlx.inference import MLXInferenceModel, MLXSegInferenceModel


def is_mlx_available() -> bool:
    """Check whether MLX is available on this system.

    Returns:
        True if running on macOS with MLX installed, False otherwise.
    """
    try:
        import platform

        import mlx.core  # noqa: F401

        return platform.system() == "Darwin"
    except ImportError:
        return False


def build_mlx_inference(
    model_config: object,
    pytorch_model: object,
) -> "MLXInferenceModel":
    """Build a compiled MLX inference model from a PyTorch RF-DETR model.

    Converts PyTorch weights to MLX format, builds the MLX model graph,
    casts to FP16, and compiles the full forward pass for Metal execution.

    Args:
        model_config: RF-DETR model configuration (e.g., RFDETRNanoConfig).
        pytorch_model: The rfdetr.main.Model instance with loaded weights.

    Returns:
        Compiled MLX inference model ready for predict() calls.

    Raises:
        RuntimeError: If MLX is not available on this system.
    """
    if not is_mlx_available():
        raise RuntimeError(
            "MLX is not available. MLX requires macOS on Apple Silicon. Install with: pip install 'rfdetr[mlx]'"
        )

    from rfdetr.mlx.inference import MLXInferenceModel

    return MLXInferenceModel.from_pytorch(model_config, pytorch_model)


def build_mlx_seg_inference(
    model_config: object,
    pytorch_model: object,
) -> "MLXSegInferenceModel":
    """Build a compiled MLX segmentation inference model from a PyTorch RF-DETR seg model.

    Converts PyTorch weights (backbone, decoder, and segmentation head) to MLX
    format, builds the MLX model graph, casts to FP16, and compiles the full
    forward pass for Metal execution.

    Args:
        model_config: RF-DETR segmentation model configuration (e.g., RFDETRSegNanoConfig).
        pytorch_model: The rfdetr.main.Model instance with loaded weights.

    Returns:
        Compiled MLX segmentation inference model ready for predict() calls.

    Raises:
        RuntimeError: If MLX is not available on this system.
    """
    if not is_mlx_available():
        raise RuntimeError(
            "MLX is not available. MLX requires macOS on Apple Silicon. Install with: pip install 'rfdetr[mlx]'"
        )

    from rfdetr.mlx.inference import MLXSegInferenceModel

    return MLXSegInferenceModel.from_pytorch(model_config, pytorch_model)
