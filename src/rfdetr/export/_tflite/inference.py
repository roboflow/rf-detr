# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""TFLite inference helpers for RF-DETR exported models.

These functions handle interpreter creation, image preprocessing, and decoding of detection and segmentation-mask
outputs without requiring PyTorch or the RF-DETR training stack: only ``tflite-runtime`` (or ``tensorflow``), ``numpy``,
``supervision``, and ``Pillow`` are needed at inference time.
"""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from supervision import Detections

from rfdetr.export._resize import _bilinear_resize_half_pixel
from rfdetr.export._topk import _select_topk_multiclass
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def _create_interpreter(model_path: str | Path) -> Any:
    """Load a TFLite model, allocate tensors, and log I/O shapes.

    Tries ``tflite_runtime`` first (lightweight; preferred on edge devices), then falls back to ``tensorflow.lite``
    (pre-installed on Colab / full TF environments).

    Args:
        model_path: Path to the ``.tflite`` model file.

    Returns:
        An allocated TFLite interpreter ready for inference.
    """
    _Interpreter = None  # noqa: N806
    _tried: list[str] = []
    for _pkg, _attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        with contextlib.suppress(ImportError):
            _Interpreter = getattr(importlib.import_module(_pkg), _attr)  # noqa: N806
            break
        _tried.append(_pkg.split(".")[0])
    if _Interpreter is None:
        _tried_str = ", ".join(f"'{p}'" for p in _tried)
        raise ImportError(
            f"TFLite inference requires 'ai_edge_litert', 'tflite-runtime', or 'tensorflow' "
            f"(tried: {_tried_str}). "
            "Install one: `pip install ai_edge_litert`  OR  `pip install tflite-runtime`"
        )

    interp = _Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    logger.debug("Input  : %s  %s", inp_det[0]["shape"], inp_det[0]["dtype"].__name__)
    for od in out_det:
        logger.debug("Output : %s  name=%s", od["shape"], od.get("name", "<unnamed>"))
    return interp


def _decode_masks(mask_logits: NDArray[np.floating[Any]], out_size: tuple[int, int]) -> NDArray[np.bool_]:
    """Upsample mask logits to image size and threshold at zero.

    Matches ``PostProcess.forward``: bilinear upsample with ``align_corners=False`` followed by ``> 0``.
    Uses ``torch.nn.functional.interpolate`` when torch is importable for bit-exact parity, and falls
    back to the pure-NumPy ``_bilinear_resize_half_pixel`` otherwise.

    Args:
        mask_logits: Raw mask logits of shape ``(K, Hm, Wm)``.
        out_size: Target ``(width, height)`` in pixels.

    Returns:
        Boolean mask array of shape ``(K, height, width)``.

    Raises:
        ValueError: If *mask_logits* is not rank-3.

    Note:
        ``out_size`` follows PIL convention ``(width, height)``; the returned array uses
        NumPy/PyTorch convention ``(K, height, width)``.
    """
    if mask_logits.ndim != 3:
        raise ValueError(
            f"_decode_masks expects rank-3 (K, Hm, Wm); got shape {mask_logits.shape}. "
            "This usually means the rank-4 mask-output heuristic in _run_inference matched the wrong tensor."
        )
    width, height = out_size
    if mask_logits.shape[0] == 0:
        return np.zeros((0, height, width), dtype=np.bool_)
    try:
        import torch
        import torch.nn.functional as _F  # noqa: N812

        with torch.no_grad():
            t = torch.from_numpy(mask_logits.astype(np.float32)).unsqueeze(0)
            t = _F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
        resized: NDArray[np.float32] = np.asarray(t.squeeze(0).numpy(), dtype=np.float32)
    except ImportError:
        resized = _bilinear_resize_half_pixel(mask_logits.astype(np.float32), height, width)
    return resized > 0.0


def _preprocess_image(
    pil_img: PILImage.Image,
    hw: tuple[int, int],
    channels: int = 3,
) -> NDArray[np.float32]:
    """Resize and ImageNet-normalise an image to match ``RFDETR.predict()``.

    Uses ``torchvision.transforms.functional`` when importable for bit-exact parity, and falls back
    to the pure-NumPy ``_bilinear_resize_half_pixel`` for torch-free deployments. Both paths resize
    with predict()'s convention: bilinear, half-pixel centers, ``antialias=False``.

    Args:
        pil_img: Source PIL image at native resolution.
        hw: Target ``(height, width)`` from the interpreter's input shape.
        channels: Channel count (3 for RGB, 1 for grayscale).

    Returns:
        Float32 array of shape ``(1, height, width, channels)`` in NHWC.

    Note:
        The NumPy fallback matches the torchvision path up to float32 op-order noise (~5e-5 in
        normalised space). For bit-exact parity with ``RFDETR.predict()``, ensure ``torch`` and
        ``torchvision`` are importable.
    """
    height, width = hw
    pil_mode = "L" if channels == 1 else "RGB"
    pil_rgb = pil_img.convert(pil_mode)

    with contextlib.suppress(ImportError):
        # Match PyTorch.predict() exactly: torchvision to_tensor -> resize(antialias=False) -> normalize.
        # antialias=False mirrors detr.py's predict(); torchvision's float-tensor default is True.
        import torch
        import torchvision.transforms.functional as _F  # noqa: N812

        with torch.no_grad():
            t = _F.to_tensor(pil_rgb)
            t = _F.resize(t, list(hw), antialias=False)
            mean_list = [_IMAGENET_MEAN[i % 3] for i in range(channels)]
            std_list = [_IMAGENET_STD[i % 3] for i in range(channels)]
            t = _F.normalize(t, mean_list, std_list)
        nchw_float = np.asarray(t.unsqueeze(0).cpu().numpy(), dtype=np.float32)
        # NCHW -> NHWC for the TFLite interpreter.
        return np.asarray(nchw_float.transpose(0, 2, 3, 1), dtype=np.float32)

    # Torch-free fallback: same antialias-free half-pixel bilinear as predict(), in NumPy.
    # PIL resize is not an option here: both BILINEAR and BICUBIC apply an adaptive antialias
    # filter when downscaling and diverge from predict() by up to ~1.7 in normalised space.
    arr = np.asarray(pil_rgb, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" -> (height, width); TFLite needs (height, width, 1).
        arr = arr[:, :, np.newaxis]
    arr = _bilinear_resize_half_pixel(arr.transpose(2, 0, 1), height, width).transpose(1, 2, 0)

    mean = np.array([_IMAGENET_MEAN[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([_IMAGENET_STD[i % 3] for i in range(channels)], dtype=np.float32)

    return np.asarray(((arr - mean) / std)[np.newaxis], dtype=np.float32)


def _run_inference(
    interp: Any,
    image_path: str | Path,
    threshold: float = 0.3,
    num_select: int | None = None,
) -> tuple[Detections, PILImage.Image]:
    """Preprocess one image, run TFLite inference, and decode detections.

    Reads input shape from the interpreter (NHWC ``float32``), resizes and normalises the image with ImageNet
    statistics, invokes the model, then decodes the ``dets`` / ``labels`` output tensors into a
    :class:`supervision.Detections` object with pixel-space ``xyxy`` boxes. For segmentation exports the ``masks``
    output is also decoded into ``Detections.mask``.

    Args:
        interp: Allocated TFLite interpreter returned by ``_create_interpreter``.
        image_path: Path to the input image (any format supported by Pillow).
        threshold: Confidence threshold; detections below this are discarded.
        num_select: Maximum query/class pairs selected before thresholding. ``None`` uses the exported model's query
            count, matching shipped RF-DETR configurations; pass an explicit value for custom exports.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains pixel-space ``xyxy`` boxes (and ``mask`` for
        segmentation models) and ``pil_img`` is the original PIL image at its original resolution.
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

    with PILImage.open(image_path) as pil_img:
        inp_tensor = _preprocess_image(pil_img, (int(height), int(width)), int(channels))

    interp.set_tensor(inp_det[0]["index"], inp_tensor)
    interp.invoke()

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to onnx2tf output reordering.
    available_output_names = [str(od.get("name", "<unnamed>")) for od in out_det]
    boxes_idx = next((i for i, od in enumerate(out_det) if "dets" in str(od.get("name", ""))), None)
    logits_idx = next((i for i, od in enumerate(out_det) if "labels" in str(od.get("name", ""))), None)
    if boxes_idx is None or logits_idx is None:
        # onnx2tf sometimes renames outputs to generic "Identity", "Identity_N"
        # instead of preserving the original ONNX node names. Fall back to
        # shape-based matching: boxes are the rank-3 tensor with last dim 4,
        # logits the rank-3 tensor with last dim != 4. A rank-4 mask output,
        # if present, is matched separately below.
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

    # Sanity-check: normalized cxcywh boxes must be in [0, 1].  When num_classes==3
    # the logits tensor also has last-dim 4, making shape-based and positional matching
    # ambiguous — onnx2tf may output [labels, dets] rather than [dets, labels].
    # A max > 2.0 or min < -2.0 reliably signals the tensors are swapped (logits routinely
    # reach ±3–10; normalized coords are in [0, 1] by definition).  The min check handles
    # the case where all logits are negative (e.g. max ≈ -2.96) — without it the swap is
    # never triggered and logit values are misinterpreted as box coords.
    if float(boxes_cwh.max()) > 2.0 or float(boxes_cwh.min()) < -2.0:
        logger.debug(
            "Box tensor max=%.2f exceeds [0,1] — swapping boxes/logits assignment "
            "(num_classes==%d likely caused ambiguous positional fallback).",
            float(boxes_cwh.max()),
            interp.get_tensor(out_det[logits_idx]["index"]).shape[-1] - 1,
        )
        boxes_idx, logits_idx = logits_idx, boxes_idx
        boxes_cwh = interp.get_tensor(out_det[boxes_idx]["index"])[0]

    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = interp.get_tensor(out_det[logits_idx]["index"])[0, :, :-1]  # (Q, num_classes)

    # RF-DETR uses per-class sigmoid (not softmax) — mirrors PostProcess.forward in postprocess.py.
    if logits.size:
        logger.debug(
            "Logits stats: shape=%s min=%.3f max=%.3f mean=%.3f",
            logits.shape,
            float(logits.min()),
            float(logits.max()),
            float(logits.mean()),
        )
    else:
        logger.debug("Logits stats: empty shape=%s", logits.shape)
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    # Flatten (Q, C) to Q*C query/class pairs and take the top-scoring ones before thresholding —
    # mirrors PostProcess._select_topk. A per-query argmax (the previous approach) keeps at most
    # one class per query, silently dropping legitimate detections whenever a query scores above
    # threshold on more than one class; see _topk.py for why that happens routinely here.
    selection_cap = logits.shape[0] if num_select is None else num_select
    scores, cls, query_idx = _select_topk_multiclass(scores_all, threshold, num_select=selection_cap)
    if scores_all.size:
        logger.debug(
            "Scores stats: min=%.3f max=%.3f — detections above threshold %.2f: %d",
            float(scores_all.min()),
            float(scores_all.max()),
            threshold,
            int(scores.shape[0]),
        )
    else:
        logger.debug("Scores stats: empty — detections above threshold %.2f: %d", threshold, int(scores.shape[0]))

    cx, cy, bw, bh = boxes_cwh[query_idx].T
    ow, oh = pil_img.size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    # Segmentation exports add a rank-4 mask output; decode it when present.
    mask_idx = next((i for i, od in enumerate(out_det) if "masks" in str(od.get("name", ""))), None)
    if mask_idx is None:
        rank4_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 4]
        if len(rank4_candidates) == 1:
            mask_idx = rank4_candidates[0]
        elif len(rank4_candidates) >= 2:
            logger.warning(
                "Ambiguous rank-4 outputs (%d candidates); skipping mask decode. "
                "Name your mask output to contain 'masks' to disambiguate.",
                len(rank4_candidates),
            )
    masks = None
    if mask_idx is not None and query_idx.shape[0] > 0:
        raw_masks = interp.get_tensor(out_det[mask_idx]["index"])[0]  # (Q, Hm, Wm)
        # Fancy-index by query_idx, NOT a boolean mask: a query can now contribute more than one
        # detection (see _select_topk_multiclass), so its mask must be gathered once per detection,
        # repeats included, rather than once per unique query.
        masks = _decode_masks(raw_masks[query_idx], (ow, oh))

    detections = Detections(xyxy=xyxy, confidence=scores, class_id=cls.astype(int), mask=masks)
    return detections, pil_img
