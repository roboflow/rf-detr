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


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Args:
        x: Input array of arbitrary shape.

    Returns:
        Array of the same shape with softmax applied along the last axis.

    Examples:
        >>> import numpy as np
        >>> np.round(_softmax(np.array([1.0, 2.0, 3.0])), 8).tolist()
        [0.09003057, 0.24472847, 0.66524096]
    """
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _create_interpreter(model_path: str | Path) -> Any:
    """Load a TFLite model, allocate tensors, and log I/O shapes.

    Tries ``tflite_runtime`` first (lightweight; preferred on edge devices),
    then falls back to ``tensorflow.lite`` (pre-installed on Colab / full TF
    environments).

    Args:
        model_path: Path to the ``.tflite`` model file.

    Returns:
        An allocated TFLite interpreter ready for inference.

    Examples:
        >>> interp = _create_interpreter("model_float32.tflite")  # doctest: +SKIP
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
                "Install one: pip install tflite-runtime  OR  pip install tensorflow"
            ) from exc

    interp = _Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    logger.debug("Input  : %s  %s", inp_det[0]["shape"], inp_det[0]["dtype"].__name__)
    for od in out_det:
        logger.debug("Output : %s  name=%s", od["shape"], od["name"])
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

    Examples:
        >>> interp = _create_interpreter("model.tflite")  # doctest: +SKIP
        >>> dets, img = _run_inference(interp, "image.jpg", threshold=0.3)  # doctest: +SKIP
    """
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    _, H, W, C = inp_det[0]["shape"]  # noqa: N806

    _imagenet_mean = [0.485, 0.456, 0.406]
    _imagenet_std = [0.229, 0.224, 0.225]
    mean = np.array([_imagenet_mean[i % 3] for i in range(C)], dtype=np.float32)
    std = np.array([_imagenet_std[i % 3] for i in range(C)], dtype=np.float32)

    pil_img = PILImage.open(image_path)
    pil_mode = "L" if C == 1 else "RGB"
    arr = np.array(pil_img.convert(pil_mode).resize((W, H)), dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" → (H, W); TFLite needs (H, W, 1)
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
        missing_outputs = []
        if boxes_idx is None:
            missing_outputs.append("dets")
        if logits_idx is None:
            missing_outputs.append("labels")
        missing = ", ".join(missing_outputs)
        available = ", ".join(available_output_names)
        raise ValueError(
            f"Expected TFLite output tensor(s) {missing!r} not found. Available output tensor names: [{available}]"
        )
    boxes_cwh = interp.get_tensor(out_det[boxes_idx]["index"])[0]  # (Q, 4) normalized cxcywh
    logits = interp.get_tensor(out_det[logits_idx]["index"])[0]  # (Q, num_classes+1)

    probs = _softmax(logits[:, :-1])  # drop background (last logit)
    scores = probs.max(axis=-1)
    cls = probs.argmax(axis=-1)
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cwh[keep].T
    ow, oh = pil_img.size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    return sv.Detections(xyxy=xyxy, confidence=scores[keep], class_id=cls[keep].astype(int)), pil_img
