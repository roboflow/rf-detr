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
``trtexec`` binary on ``PATH`` is required — only ``pip install rfdetr[tensorrt]``.

For TensorRT *inference*, use the ``inference-models`` library which provides
multi-backend RF-DETR support (PyTorch, ONNX, TensorRT) with automatic backend
selection::

    from inference_models import AutoModel

    model = AutoModel.from_pretrained("rfdetr-small")

See https://github.com/roboflow/inference/tree/main/inference_models for details.
"""

from __future__ import annotations

import os

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# polygraphy ships in the ``rfdetr[tensorrt]`` extra alongside ``tensorrt``. Import it
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

    _IS_TENSORRT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via the guard in build_engine
    CreateConfig = None
    engine_from_network = None
    network_from_onnx_path = None
    save_engine = None

    _IS_TENSORRT_AVAILABLE = False


def build_engine(
    onnx_path: str,
    *,
    fp16: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
    output_name: str | None = None,
) -> str:
    """Build a serialized TensorRT engine from an ONNX model, in-process.

    Uses the TensorRT Python API through ``polygraphy`` — no ``trtexec`` subprocess.
    Workspace size is left to the TensorRT default (it auto-sizes to the available
    device memory), which meets or exceeds the historical 4 GiB cap.

    Args:
        onnx_path: Path to the source ``.onnx`` file. Its stem (typically the model variant name,
            e.g. ``"rfdetr-medium"``) is reused for the engine filename unless *output_name* is given.
        fp16: Enable FP16 precision when building the engine.  Automatically downgraded to FP32 (with a
            warning) on TensorRT builds that do not expose the FP16 builder flag — the engine filename
            reflects the precision actually built, not just the requested value (except under
            *dry_run*, where no build/probe happens so the requested value is used as-is).
        verbose: Emit extra progress logging.
        dry_run: Log the intended build and return the engine path without
            building anything (no TensorRT / GPU required).
        output_name: Full filename override (without extension). Takes precedence over the ONNX
            stem and suppresses the ``_fp16``/``_fp32`` suffix — the engine is named
            ``{output_name}.trt`` verbatim, written alongside *onnx_path*.

    Returns:
        Path to the generated ``.trt`` engine file.

    Raises:
        ImportError: If ``polygraphy``/``tensorrt`` are not installed.

    Examples:
        >>> build_engine("output/rfdetr-medium.onnx", dry_run=True)  # doctest: +SKIP
        'output/rfdetr-medium_fp16.trt'
    """
    onnx_stem = os.path.splitext(onnx_path)[0]

    def _engine_path(*, fp16_used: bool) -> str:
        if output_name:
            stem = os.path.splitext(os.path.basename(output_name))[0]
            return os.path.join(os.path.dirname(onnx_path), f"{stem}.trt")
        # Precision materially changes the engine (fp16 vs fp32 accuracy/speed), so it is always
        # encoded — unless a custom name was requested. Swapping only the final suffix (rather than
        # rebuilding the whole path) keeps any earlier ".onnx"-like segment intact and never aliases
        # the input path; a string-level split (not pathlib) preserves separators verbatim (pathlib
        # rewrites "/" to "\\" on Windows).
        return f"{onnx_stem}_{'fp16' if fp16_used else 'fp32'}.trt"

    engine_path = _engine_path(fp16_used=fp16)

    if dry_run:
        logger.info(f"[dry-run] Would build TensorRT engine (fp16={fp16}): {onnx_path} -> {engine_path}")
        return engine_path

    if engine_from_network is None:
        raise ImportError("TensorRT export requires the 'tensorrt' extra. Install with: pip install rfdetr[tensorrt]")

    if fp16:
        # Some TensorRT builds (e.g. lean/partial wheels) do not expose the FP16 builder flag. polygraphy aborts
        # when asked to set an unavailable flag, so probe for it up front and fall back to an FP32 engine instead
        # of crashing the whole export.
        try:
            import tensorrt as trt

            if not hasattr(trt.BuilderFlag, "FP16"):
                logger.warning(
                    "TensorRT %s does not expose the FP16 builder flag; building an FP32 engine instead. "
                    "Pass fp16=False to silence this warning.",
                    getattr(trt, "__version__", "unknown"),
                )
                fp16 = False
                engine_path = _engine_path(fp16_used=fp16)
        except ImportError:
            pass  # a missing/broken tensorrt import is surfaced by the polygraphy build chain below

    if verbose:
        logger.info(f"Building TensorRT engine (fp16={fp16}) from {onnx_path}")

    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=CreateConfig(fp16=fp16),
    )
    save_engine(engine, path=engine_path)

    logger.info(f"Successfully built TensorRT engine: {engine_path}")
    return engine_path
