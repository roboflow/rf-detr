# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ONNX Runtime inference helpers for RF-DETR exported models.

These functions handle session creation, image preprocessing, and detection
decoding without requiring PyTorch or the RF-DETR training stack — only
``onnxruntime``, ``numpy``, ``supervision``, and ``Pillow`` are needed at
inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image as PILImage

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _create_onnx_session(model_path: str | Path) -> Any:
    """Load an ONNX model and create an ONNX Runtime inference session.

    Imports ``onnxruntime`` at call time so that the rest of the package
    remains usable without it installed.  Input and output names / shapes
    are logged at DEBUG level for troubleshooting.

    Args:
        model_path: Path to the ``.onnx`` model file.

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

    session = ort.InferenceSession(str(model_path))
    for inp in session.get_inputs():
        logger.debug("Input  : name=%s  shape=%s  type=%s", inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        logger.debug("Output : name=%s  shape=%s  type=%s", out.name, out.shape, out.type)
    return session


def _run_inference(
    session: Any,
    image_path: str | Path,
    threshold: float = 0.3,
) -> tuple[sv.Detections, PILImage.Image]:
    """Preprocess one image, run ONNX Runtime inference, and decode detections.

    Reads input shape from the session (NCHW ``float32``), resizes and
    normalises the image with ImageNet statistics, invokes the model, then
    decodes the ``dets`` / ``labels`` output tensors into a
    :class:`supervision.Detections` object with pixel-space ``xyxy`` boxes.

    **Input contract** (must match ``RFDETR.predict()`` preprocessing exactly):

    - Image is opened as-is and converted to ``"RGB"`` (3-channel) or ``"L"``
      (1-channel greyscale) depending on the model's channel count.
    - Resize uses ``PIL.Image.Resampling.BILINEAR`` — matching
      ``torchvision.transforms.functional.resize()`` which defaults to
      ``InterpolationMode.BILINEAR``.  Using PIL's default (``BICUBIC``) would
      produce slightly different pixel values and can degrade confidence.
    - Pixel values are scaled to ``[0, 1]`` then normalised with ImageNet
      statistics: ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.
    - The tensor is kept as ``[1, C, H, W]`` (NCHW) — unlike the TFLite helper
      which uses NHWC because ``onnx2tf`` transposes at export time.  ONNX RT
      consumes the native ONNX NCHW layout directly.

    Args:
        session: ONNX Runtime ``InferenceSession`` returned by
            ``_create_onnx_session``.
        image_path: Path to the input image (any format supported by Pillow).
            RGB images are used as-is; RGBA / palette images are converted.
        threshold: Confidence threshold; detections below this are discarded.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains
        pixel-space ``xyxy`` boxes and ``pil_img`` is the original PIL image
        at its original resolution.

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

    _imagenet_mean = [0.485, 0.456, 0.406]
    _imagenet_std = [0.229, 0.224, 0.225]
    mean = np.array([_imagenet_mean[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([_imagenet_std[i % 3] for i in range(channels)], dtype=np.float32)

    pil_img = PILImage.open(image_path)
    pil_mode = "L" if channels == 1 else "RGB"
    # Use BILINEAR resampling to match torchvision.transforms.functional.resize()
    # which defaults to InterpolationMode.BILINEAR.  PIL's default (None → BICUBIC)
    # produces different pixel values and can cause a measurable confidence drop.
    arr = (
        np.array(
            pil_img.convert(pil_mode).resize((width, height), PILImage.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    if arr.ndim == 2:  # "L" → (height, width); needs (height, width, 1)
        arr = arr[:, :, np.newaxis]

    # Normalise HWC, then transpose to CHW for ONNX (NCHW)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    inp_tensor = arr[np.newaxis].astype(np.float32)  # (1, C, H, W)

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
    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = raw_outputs[logits_idx][0, :, :-1]  # (Q, num_classes)

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
