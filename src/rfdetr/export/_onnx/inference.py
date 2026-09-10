# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX Runtime inference helpers for RF-DETR exported models.

These functions handle session creation, image preprocessing, and detection decoding without requiring PyTorch or the
RF-DETR training stack — only ``onnxruntime``, ``numpy``, ``supervision``, and ``Pillow`` are needed at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from supervision import Detections

from rfdetr.export._runtime.decode import decode_detections
from rfdetr.export._runtime.preprocess import preprocess_to_nchw
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _create_onnx_session(model_path: str | Path, providers: list[str] | None = None) -> Any:
    """Load an ONNX model and create an ONNX Runtime inference session.

    Imports ``onnxruntime`` at call time so that the rest of the package remains usable without it installed.  Input and
    output names / shapes are logged at DEBUG level for troubleshooting.

    When ``providers`` is ``None``, the session auto-selects the best available backend: CUDA if ``onnxruntime-gpu`` is
    installed, otherwise CPU (with a warning).  Pass an explicit list to pin the backend — useful for benchmarking
    CPU vs CUDA side-by-side.

    Args:
        model_path: Path to the ``.onnx`` model file.
        providers: Ordered list of ORT execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.  When ``None`` (default), the best available
            provider is selected automatically.

    Returns:
        An ``onnxruntime.InferenceSession`` ready for inference.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.

    Examples:
        .. code-block:: python

            sess = _create_onnx_session("model.onnx")
            print(sess.get_inputs()[0].name)
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX Runtime inference requires 'onnxruntime'. Install it: `pip install onnxruntime`"
        ) from exc

    if providers is None:
        _preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _available = ort.get_available_providers()
        providers = [p for p in _preferred if p in _available] or ["CPUExecutionProvider"]
        if providers[0] == "CPUExecutionProvider":
            logger.warning(
                "CUDAExecutionProvider not available — running ONNX inference on CPU. "
                "Install onnxruntime-gpu for GPU acceleration: `pip install onnxruntime-gpu`"
            )
    session = ort.InferenceSession(str(model_path), providers=providers)
    logger.debug("ONNX Runtime providers in use: %s", session.get_providers())
    for inp in session.get_inputs():
        logger.debug("Input  : name=%s  shape=%s  type=%s", inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        logger.debug("Output : name=%s  shape=%s  type=%s", out.name, out.shape, out.type)
    return session


def _run_inference(
    session: Any,
    image_path: str | Path,
    threshold: float = 0.3,
    num_select: int | None = None,
    background_class_id: int | None = -1,
) -> tuple[Detections, PILImage.Image]:
    """Preprocess one image, run ONNX Runtime inference, and decode detections.

    Reads input shape from the session (NCHW ``float32``), resizes and normalises the image with ImageNet statistics,
    invokes the model, then decodes the ``dets`` / ``labels`` output tensors into a :class:`supervision.Detections`
    object with pixel-space ``xyxy`` boxes.

    **Input contract** (must match ``RFDETR.predict()`` preprocessing exactly):

    - Image is opened with Pillow and converted to ``"RGB"`` (3-channel) or ``"L"``
      (1-channel greyscale) depending on the model's channel count. PIL is used only to decode and
      convert — never to resize.
    - Resize follows ``RFDETR.predict()``'s exact convention — bilinear, half-pixel centers,
      ``antialias=False`` — applied by :func:`~rfdetr.export._runtime.preprocess.preprocess_to_nchw` via
      ``torchvision`` when importable
      (bit-exact) or the pure-NumPy ``_bilinear_resize_half_pixel`` fallback. PIL's own BILINEAR/BICUBIC
      filters apply adaptive antialiasing when downscaling and would shift pixel values away from predict(),
      degrading confidence.
    - Pixel values are scaled to ``[0, 1]`` then normalised with ImageNet
      statistics: ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.
    - The tensor is kept as ``[1, C, H, W]`` (NCHW) — unlike the TFLite helper
      which uses NHWC because ``onnx2tf`` transposes at export time.  ONNX RT consumes the native ONNX NCHW layout
      directly.

    Args:
        session: ONNX Runtime ``InferenceSession`` returned by
            ``_create_onnx_session``.
        image_path: Path to the input image (any format supported by Pillow).
            RGB images are used as-is; RGBA / palette images are converted.
        threshold: Confidence threshold; detections below this are discarded.
        num_select: Maximum query/class pairs selected before thresholding. ``None`` uses the exported model's query
            count, matching shipped RF-DETR configurations; pass an explicit value for custom exports.
        background_class_id: Exported class slot to exclude before selection. The default ``-1`` preserves the common
            final-slot background convention. Pass ``None`` for sparse COCO checkpoints, whose final slot is class 90,
            or ``0`` for legacy background-first keypoint checkpoints.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains pixel-space ``xyxy`` boxes and ``pil_img`` is
        the original PIL image at its original resolution.

    Examples:
        .. code-block:: python

            sess = _create_onnx_session("model.onnx")
            dets, img = _run_inference(sess, "photo.jpg", threshold=0.3)
            print(dets.confidence)
    """
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_name = inputs[0].name
    # ONNX NCHW: [batch, channels, height, width]
    _, channels, height, width = inputs[0].shape

    with PILImage.open(image_path) as pil_img:
        inp_tensor = preprocess_to_nchw(pil_img, height, width, channels)

    raw_outputs = session.run(None, {input_name: inp_tensor})

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to output reordering.
    output_names = [out.name for out in outputs]
    boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), None)
    logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), None)
    if boxes_idx is None or logits_idx is None:
        # Fall back to shape-based matching: boxes (*, 4) and logits (*, num_classes+1).
        logger.warning(
            "Name-based ONNX output matching failed (available names: %s). Falling back to shape-based matching.",
            output_names,
        )
        shape_boxes_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] == 4
        ]
        shape_logits_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] != 4
        ]
        if len(shape_boxes_candidates) == 1 and len(shape_logits_candidates) == 1:
            boxes_idx = shape_boxes_candidates[0]
            logits_idx = shape_logits_candidates[0]
        elif len(raw_outputs) == 2:
            # Ambiguous shapes (e.g. num_classes==3 → logits dim==4 == boxes dim).
            # ONNX preserves output order: index 0 = dets (boxes), index 1 = labels (logits).
            logger.warning(
                "Shape-based ONNX output matching is ambiguous (both outputs have last dim==4, "
                "which happens when num_classes==3).  Falling back to positional order: "
                "output 0 = boxes ('dets'), output 1 = logits ('labels').  "
                "If detections look wrong, inspect output names with _create_onnx_session() "
                "and set LOG_LEVEL=DEBUG."
            )
            boxes_idx = 0
            logits_idx = 1
        else:
            available_shapes = [list(arr_out.shape) for arr_out in raw_outputs]
            raise ValueError(
                f"Shape-based ONNX output matching failed. Expected exactly one rank-3 tensor with "
                f"last dim == 4 (boxes) and one rank-3 tensor with last dim != 4 (logits). "
                f"Available output shapes: {available_shapes}"
            )

    boxes_cwh = raw_outputs[boxes_idx][0]  # (Q, 4) normalised cxcywh
    # Background placement is checkpoint-dependent and cannot be inferred from the tensor width alone.
    logits = raw_outputs[logits_idx][0]

    decoded = decode_detections(
        boxes_cwh,
        logits,
        pil_img.size,
        threshold=threshold,
        num_select=num_select,
        background_class_id=background_class_id,
    )

    detections = Detections(xyxy=decoded.xyxy, confidence=decoded.confidence, class_id=decoded.class_id.astype(int))
    return detections, pil_img


# Benchmarking helper — not part of production inference API; subject to removal.
def _onnx_runtime(
    onnx_path: Path | str,
    image: PILImage.Image,
    providers: list[str],
    warmup: int = 20,
    runs: int = 100,
) -> tuple[float, float, str]:
    """Benchmark ONNX Runtime inference for one image and provider list.

    Creates a fresh ``InferenceSession`` with the requested providers, preprocesses ``image`` once using ImageNet
    normalisation, then runs timed inference with ``time.perf_counter``.  GPU timings may underestimate real latency
    if the CUDA execution provider is configured for asynchronous execution; for accurate GPU timing use CUDA events.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        image: Input image (any size); resized to the model's expected spatial resolution.
        providers: Ordered list of ORT execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        warmup: Number of un-timed warm-up runs before measurement begins.
        runs: Number of timed runs used to compute statistics.

    Returns:
        A ``(mean_ms, std_ms, provider_label)`` tuple where ``provider_label`` is the first active provider with
        ``"ExecutionProvider"`` stripped, e.g. ``"CUDA"`` or ``"CPU"``.

    Examples:
        .. code-block:: python

            mean_ms, std_ms, label = _onnx_runtime("model.onnx", image, ["CPUExecutionProvider"])
            print(f"{label}: {mean_ms:.1f} ms ± {std_ms:.1f}")
    """
    import time

    sess = _create_onnx_session(onnx_path, providers=providers)
    active = sess.get_providers()[0]
    if active != providers[0]:
        raise RuntimeError(
            f"Requested provider {providers[0]!r} not active — ORT fell back to {active!r}. "
            "Install onnxruntime-gpu: `pip install onnxruntime-gpu`"
        )
    input_meta = sess.get_inputs()[0]
    _, channels, height, width = input_meta.shape
    inp = preprocess_to_nchw(image, height, width, channels)
    feed = {input_meta.name: inp}

    for _ in range(warmup):
        sess.run(None, feed)
    timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr_t = np.array(timings)
    provider_label = sess.get_providers()[0].replace("ExecutionProvider", "")
    return float(arr_t.mean()), float(arr_t.std()), provider_label
