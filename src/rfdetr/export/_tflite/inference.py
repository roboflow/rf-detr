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
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from supervision import Detections

from rfdetr.export._resize import _bilinear_resize_half_pixel
from rfdetr.export._runtime.decode import decode_detections
from rfdetr.export._runtime.preprocess import preprocess_to_nchw
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
_RANK4_OUTPUT_KINDS: tuple[str, ...] = ("masks", "keypoints")


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

    Thin NHWC adapter over :func:`~rfdetr.export._runtime.preprocess.preprocess_to_nchw`, which uses
    ``torchvision.transforms.functional`` when importable for bit-exact parity and falls back to the pure-NumPy
    ``_bilinear_resize_half_pixel`` for torch-free deployments. Both paths resize with predict()'s convention:
    bilinear, half-pixel centers, ``antialias=False``.

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
    nchw = preprocess_to_nchw(pil_img, height, width, channels)
    # NCHW -> NHWC for the TFLite interpreter, which consumes the layout onnx2tf transposed to at export time.
    return np.asarray(nchw.transpose(0, 2, 3, 1), dtype=np.float32)


def _run_inference(
    interp: Any,
    image_path: str | Path,
    threshold: float = 0.3,
    num_select: int | None = None,
    background_class_id: int | None = -1,
    rank4_output: Literal["masks", "keypoints"] | None = None,
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
        background_class_id: Exported class slot to exclude before selection. The default ``-1`` preserves the common
            final-slot background convention. Pass ``None`` for sparse COCO checkpoints, whose final slot is class 90,
            or ``0`` for legacy background-first keypoint checkpoints.
        rank4_output: What a rank-4 output holds when no output names itself ``masks``. Segmentation and keypoint
            exports each add exactly one rank-4 tensor, and RF-DETR's own TFLite files reach the interpreter with
            the ONNX output names replaced by ``StatefulPartitionedCall:N``, so the kind is usually not readable
            from the graph. The default ``None`` decodes a mask only from an output that names itself. Pass
            ``"masks"`` for a name-stripped segmentation export or ``"keypoints"`` to suppress anonymous-mask
            decoding for a keypoint export.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains pixel-space ``xyxy`` boxes (and ``mask`` for
        segmentation models) and ``pil_img`` is the original PIL image at its original resolution.

    Raises:
        ValueError: If *rank4_output* is neither ``None`` nor one of ``"masks"``/``"keypoints"``, if the model's
            input tensor is not ``float32``, or if the ``dets``/``labels`` outputs cannot be matched by name or
            shape.
    """
    if rank4_output is not None and rank4_output not in _RANK4_OUTPUT_KINDS:
        raise ValueError(f"rank4_output must be one of {_RANK4_OUTPUT_KINDS} or None; got {rank4_output!r}")

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

    # Background placement is checkpoint-dependent and cannot be inferred from the tensor width alone.
    logits = interp.get_tensor(out_det[logits_idx]["index"])[0]

    decoded = decode_detections(
        boxes_cwh,
        logits,
        pil_img.size,
        threshold=threshold,
        num_select=num_select,
        background_class_id=background_class_id,
    )
    query_idx = decoded.query_index

    # Segmentation exports add a rank-4 mask output; decode it when present. Keypoint exports add a rank-4
    # output too (pred_keypoints), and the ONNX output names rarely survive the conversion, so an anonymous
    # rank-4 tensor is only taken for a mask when the caller declares the export a segmentation one.
    mask_idx = next((i for i, od in enumerate(out_det) if "masks" in str(od.get("name", ""))), None)
    if mask_idx is None and rank4_output == "masks":
        rank4_candidates = [
            i for i, od in enumerate(out_det) if len(od["shape"]) == 4 and "keypoints" not in str(od.get("name", ""))
        ]
        if len(rank4_candidates) == 1:
            mask_idx = rank4_candidates[0]
            logger.debug(
                "Rank-4 output %s carries no kind in its name; decoding it as a caller-declared segmentation mask.",
                str(out_det[mask_idx].get("name", "<unnamed>")),
            )
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
        masks = _decode_masks(raw_masks[query_idx], pil_img.size)

    detections = Detections(
        xyxy=decoded.xyxy, confidence=decoded.confidence, class_id=decoded.class_id.astype(int), mask=masks
    )
    return detections, pil_img
