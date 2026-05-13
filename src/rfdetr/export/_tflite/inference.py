# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""TFLite inference helpers for RF-DETR exported models.

These functions handle interpreter creation, image preprocessing, and
detection decoding without requiring PyTorch or the RF-DETR training stack —
only ``tflite-runtime`` (or ``tensorflow``), ``numpy``, ``supervision``, and
``Pillow`` are needed at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image as PILImage

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _create_interpreter(model_path: str | Path) -> Any:
    """Load a TFLite model, allocate tensors, and log I/O shapes.

    Tries ``tflite_runtime`` first (lightweight; preferred on edge devices),
    then falls back to ``tensorflow.lite`` (pre-installed on Colab / full TF
    environments).

    Args:
        model_path: Path to the ``.tflite`` model file.

    Returns:
        An allocated TFLite interpreter ready for inference.
    """
    try:
        import tflite_runtime.interpreter as _tflite

        _Interpreter = _tflite.Interpreter  # noqa: N806
    except ImportError:
        try:
            import tensorflow as _tf

            _Interpreter = _tf.lite.Interpreter  # noqa: N806
        except ImportError as exc:
            raise ImportError(
                "TFLite inference requires either 'tflite-runtime' or 'tensorflow'. "
                "Install one: `pip install tflite-runtime`  OR  `pip install tensorflow`"
            ) from exc

    interp = _Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    logger.debug("Input  : %s  %s", inp_det[0]["shape"], inp_det[0]["dtype"].__name__)
    for od in out_det:
        logger.debug("Output : %s  name=%s", od["shape"], od.get("name", "<unnamed>"))
    return interp


def _run_inference(
    interp: Any,
    image_path: str | Path,
    threshold: float = 0.3,
) -> tuple[sv.Detections, PILImage.Image]:
    """Preprocess one image, run TFLite inference, and decode detections.

    Reads input shape from the interpreter (NHWC ``float32``), resizes and
    normalises the image with ImageNet statistics, invokes the model, then
    decodes the ``dets`` / ``labels`` output tensors into a
    :class:`supervision.Detections` object with pixel-space ``xyxy`` boxes.

    Args:
        interp: Allocated TFLite interpreter returned by ``_create_interpreter``.
        image_path: Path to the input image (any format supported by Pillow).
        threshold: Confidence threshold; detections below this are discarded.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains
        pixel-space ``xyxy`` boxes and ``pil_img`` is the original PIL image
        at its original resolution.
    """
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    _, height, width, channels = inp_det[0]["shape"]

    expected_dtype = np.float32
    actual_dtype = inp_det[0]["dtype"]
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"_run_inference only supports float32 input tensors, but model expects {actual_dtype.__name__}. "
            "Export the model with float32 quantization or implement input quantization manually."
        )

    _imagenet_mean = [0.485, 0.456, 0.406]
    _imagenet_std = [0.229, 0.224, 0.225]
    mean = np.array([_imagenet_mean[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([_imagenet_std[i % 3] for i in range(channels)], dtype=np.float32)

    pil_img = PILImage.open(image_path)
    pil_mode = "L" if channels == 1 else "RGB"
    arr = np.array(pil_img.convert(pil_mode).resize((width, height)), dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" → (height, width); TFLite needs (height, width, 1)
        arr = arr[:, :, np.newaxis]
    inp_tensor = (arr - mean) / std

    interp.set_tensor(inp_det[0]["index"], inp_tensor[np.newaxis])
    interp.invoke()

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to onnx2tf output reordering.
    available_output_names = [str(od.get("name", "<unnamed>")) for od in out_det]
    boxes_idx = next((i for i, od in enumerate(out_det) if "dets" in str(od.get("name", ""))), None)
    logits_idx = next((i for i, od in enumerate(out_det) if "labels" in str(od.get("name", ""))), None)
    if boxes_idx is None or logits_idx is None:
        # onnx2tf sometimes renames outputs to generic "Identity", "Identity_N" instead
        # of preserving the original ONNX node names. Fall back to shape-based
        # matching for the detection outputs only: boxes (*, 4) and logits
        # (*, num_classes+1). Segmentation exports may include additional outputs
        # such as masks; unnamed extra outputs are not resolved by this fallback.
        logger.debug(
            "Name-based output matching failed (available: %s). Falling back to shape-based matching.",
            available_output_names,
        )
        shape_boxes_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 3 and od["shape"][-1] == 4]
        shape_logits_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 3 and od["shape"][-1] != 4]
        if len(shape_boxes_candidates) == 1 and len(shape_logits_candidates) == 1:
            boxes_idx = shape_boxes_candidates[0]
            logits_idx = shape_logits_candidates[0]
        elif len(out_det) == 2:
            # Ambiguous shapes (e.g. num_classes==3 → logits dim==4 == boxes dim).
            # onnx2tf preserves ONNX output order: index 0 = dets (boxes), index 1 = labels (logits).
            logger.debug("Shape-based matching ambiguous. Using positional order (0=boxes, 1=logits).")
            boxes_idx = 0
            logits_idx = 1
        else:
            available_shapes = [list(od["shape"]) for od in out_det]
            raise ValueError(
                f"Shape-based TFLite output matching failed. Expected exactly one rank-3 tensor with "
                f"last dim == 4 (boxes) and one rank-3 tensor with last dim != 4 (logits). "
                f"Available output shapes: {available_shapes}"
            )
    boxes_cwh = interp.get_tensor(out_det[boxes_idx]["index"])[0]  # (Q, 4) normalized cxcywh
    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = interp.get_tensor(out_det[logits_idx]["index"])[0, :, :-1]  # (Q, num_classes)

    # RF-DETR uses per-class sigmoid (not softmax) — mirrors PostProcess.forward in postprocess.py.
    logger.debug(
        "Logits stats: shape=%s min=%.3f max=%.3f mean=%.3f",
        logits.shape,
        float(logits.min()),
        float(logits.max()),
        float(logits.mean()),
    )
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    scores = scores_all.max(axis=-1)
    cls = scores_all.argmax(axis=-1)
    logger.debug(
        "Scores stats: min=%.3f max=%.3f — detections above threshold %.2f: %d",
        float(scores.min()),
        float(scores.max()),
        threshold,
        int((scores > threshold).sum()),
    )
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cwh[keep].T
    ow, oh = pil_img.size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    return sv.Detections(xyxy=xyxy, confidence=scores[keep], class_id=cls[keep].astype(int)), pil_img
