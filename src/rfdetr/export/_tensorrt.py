# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""TensorRT export helper: build a serialized engine from ONNX in-process.

The engine is built with the TensorRT Python API (via `polygraphy`), so no
``trtexec`` binary on ``PATH`` is required — only ``pip install rfdetr[trt]``.

For TensorRT *inference*, use the ``inference-models`` library which provides
multi-backend RF-DETR support (PyTorch, ONNX, TensorRT) with automatic backend
selection::

    from inference_models import AutoModel

    model = AutoModel.from_pretrained("rfdetr-small")

See https://github.com/roboflow/inference/tree/main/inference_models for details.
"""

from __future__ import annotations

from pathlib import Path

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# polygraphy ships in the ``rfdetr[trt]`` extra alongside ``tensorrt``. Import it
# lazily at module scope (guarded) so importing this module never fails on hosts
# without TensorRT, and so tests can monkeypatch these names without polygraphy
# installed.
try:
    from polygraphy.backend.trt import (
        CreateConfig,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )
except ImportError:  # pragma: no cover - exercised via the guard in build_engine
    CreateConfig = None
    engine_from_network = None
    network_from_onnx_path = None
    save_engine = None


def build_engine(onnx_path: str, *, fp16: bool = True, verbose: bool = False, dry_run: bool = False) -> str:
    """Build a serialized TensorRT engine from an ONNX model, in-process.

    Uses the TensorRT Python API through ``polygraphy`` — no ``trtexec`` subprocess.
    Workspace size is left to the TensorRT default (it auto-sizes to the available
    device memory), which meets or exceeds the historical 4 GiB cap.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        fp16: Enable FP16 precision when building the engine.
        verbose: Emit extra progress logging.
        dry_run: Log the intended build and return the engine path without
            building anything (no TensorRT / GPU required).

    Returns:
        Path to the generated ``.trt`` engine file.

    Raises:
        ImportError: If ``polygraphy``/``tensorrt`` are not installed.

    Examples:
        >>> build_engine("output/inference_model.onnx", dry_run=True)  # doctest: +SKIP
        'output/inference_model.trt'
    """
    # Swap only the final suffix so paths with an earlier ".onnx" segment (or no
    # ".onnx" at all) are not corrupted and never alias the input path.
    engine_path = str(Path(onnx_path).with_suffix(".trt"))

    if dry_run:
        logger.info(f"[dry-run] Would build TensorRT engine (fp16={fp16}): {onnx_path} -> {engine_path}")
        return engine_path

    if engine_from_network is None:
        raise ImportError("TensorRT export requires the 'trt' extra. Install with: pip install rfdetr[trt]")

    if verbose:
        logger.info(f"Building TensorRT engine (fp16={fp16}) from {onnx_path}")

    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=CreateConfig(fp16=fp16),
    )
    save_engine(engine, path=engine_path)

    logger.info(f"Successfully built TensorRT engine: {engine_path}")
    return engine_path
