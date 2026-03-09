# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Parity tests: COCOEvalCallback vs legacy coco_extended_metrics (PTL Ch2/T5).

Asserts that the new ``COCOEvalCallback`` produces mAP50 and F1 values within
the tolerances specified in MIGRATION_PT_LIGHTNING.md Chapter 2:

    |Δ mAP50| ≤ 0.005
    |Δ F1|    ≤ 0.01

The intermediate scenario (2 classes, mixed TP/FP, varying confidence)
mirrors the ``intermediate_scenario_cocoeval`` fixture in
``tests/util/test_metrics.py`` so the same data drives both paths.

Module-scoped fixtures (``detection_parity``, ``segmentation_parity``,
``perfect_parity``, ``degenerate_parity``) compute each scenario once and
share the ``(legacy_dict, new_dict)`` result across all tests in the module,
avoiding repeated COCOeval calls.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr.engine import (
    build_matching_data,
    coco_extended_metrics,
    init_matching_accumulator,
    merge_matching_data,
    sweep_confidence_thresholds,
)
from rfdetr.lit.callbacks.coco_eval import COCOEvalCallback

# ---------------------------------------------------------------------------
# Tolerances (per MIGRATION_PT_LIGHTNING.md Chapter 2 milestone)
# ---------------------------------------------------------------------------

_MAP50_TOL = 0.005
_F1_TOL = 0.01

# ---------------------------------------------------------------------------
# Scenario constants — match tests/util/test_metrics.py exactly
# ---------------------------------------------------------------------------

_BOX_SIZE = 200
_BOX_SPACING = 250
_ROW_SPACING = 260

# Smaller constants for segmentation scenario (reduces mask memory ~40x)
_SEGM_BOX_SIZE = 40
_SEGM_BOX_SPACING = 50
_SEGM_ROW_SPACING = 55


# ---------------------------------------------------------------------------
# Shared scenario builder helpers
# ---------------------------------------------------------------------------


def _make_contained_pred_box(gt_box: list[float], target_iou: float) -> list[float]:
    """Return a prediction box centred inside *gt_box* with the given IoU.

    Uses the contained-box formula IoU = pred_area / gt_area, valid when the
    prediction is a square fully inside a square GT.

    Args:
        gt_box: COCO-format ``[x, y, w, h]`` for the ground-truth box.
        target_iou: Desired IoU between prediction and GT.

    Returns:
        COCO-format ``[x, y, w, h]`` for the prediction box.
    """
    x, y, gt_size, _ = gt_box
    if target_iou >= 1.0:
        return list(gt_box)
    pred_size = gt_size * math.sqrt(target_iou)
    offset = (gt_size - pred_size) / 2
    return [x + offset, y + offset, pred_size, pred_size]


def _build_intermediate_scenario() -> dict:
    """Build intermediate scenario raw data: 2 classes, mixed TP/FP.

    Exactly mirrors the ``intermediate_scenario_cocoeval`` fixture in
    ``tests/util/test_metrics.py``:

    - Class 1: 10 GTs, 10 TP predictions (IoU ≥ 0.525, varying confidence).
    - Class 2: 10 GTs, 10 TP predictions (IoU ≥ 0.525) + 10 FP predictions
      (IoU = 0, lower confidence).

    Returns:
        Dict with keys:

        - ``image_id`` (int)
        - ``image_width``, ``image_height`` (int)
        - ``categories`` (list of ``{"id": int, "name": str}``)
        - ``gt_abs_xywh``: list of ``(cat_id, [x, y, w, h])``
        - ``pred_abs_xywh``: list of ``(cat_id, [x, y, w, h], score)``
    """
    image_id = 1
    n_boxes = 10

    class1_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class1_confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    class2_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class2_confs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    class2_fp_confs = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    image_width = n_boxes * _BOX_SPACING
    image_height = 3 * _ROW_SPACING

    gt_abs_xywh: list[tuple[int, list[float]]] = []
    pred_abs_xywh: list[tuple[int, list[float], float]] = []

    # Row 0: Class 1 — 10 GTs each paired with one TP prediction
    for i, (iou, conf) in enumerate(zip(class1_ious, class1_confs)):
        gt_box = [float(i * _BOX_SPACING), 0.0, float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((1, gt_box))
        pred_abs_xywh.append((1, pred_box, conf))

    # Row 1: Class 2 — 10 GTs each paired with one TP prediction
    for i, (iou, conf) in enumerate(zip(class2_ious, class2_confs)):
        gt_box = [float(i * _BOX_SPACING), float(_ROW_SPACING), float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((2, gt_box))
        pred_abs_xywh.append((2, pred_box, conf))

    # Row 2: Class 2 — 10 FP predictions (no GTs on this row, IoU = 0)
    for i, conf in enumerate(class2_fp_confs):
        fp_box = [float(i * _BOX_SPACING), float(2 * _ROW_SPACING), float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_abs_xywh.append((2, fp_box, conf))

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
        "gt_abs_xywh": gt_abs_xywh,
        "pred_abs_xywh": pred_abs_xywh,
    }


def _build_perfect_scenario() -> dict:
    """Two classes, 5 GTs each with IoU=0.96 TP predictions, score=1.0."""
    image_id = 1
    image_width = 5 * _BOX_SPACING
    image_height = 2 * _ROW_SPACING

    gt_abs_xywh: list[tuple[int, list[float]]] = []
    pred_abs_xywh: list[tuple[int, list[float], float]] = []

    for cat_id, row_y in [(1, 0), (2, _ROW_SPACING)]:
        for i in range(5):
            gt_box = [float(i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
            pred_box = _make_contained_pred_box(gt_box, target_iou=0.96)
            gt_abs_xywh.append((cat_id, gt_box))
            pred_abs_xywh.append((cat_id, pred_box, 1.0))

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
        "gt_abs_xywh": gt_abs_xywh,
        "pred_abs_xywh": pred_abs_xywh,
    }


def _build_degenerate_scenario() -> dict:
    """Two classes: GTs on left, FP predictions on right (IoU=0), score=1.0."""
    image_id = 1
    pred_x_offset = 5 * _BOX_SPACING + 100  # gap ensures zero IoU
    image_width = pred_x_offset + 5 * _BOX_SPACING
    image_height = 2 * _ROW_SPACING

    gt_abs_xywh: list[tuple[int, list[float]]] = []
    pred_abs_xywh: list[tuple[int, list[float], float]] = []

    for cat_id, row_y in [(1, 0), (2, _ROW_SPACING)]:
        for i in range(5):
            gt_box = [float(i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
            fp_box = [float(pred_x_offset + i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
            gt_abs_xywh.append((cat_id, gt_box))
            pred_abs_xywh.append((cat_id, fp_box, 1.0))

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
        "gt_abs_xywh": gt_abs_xywh,
        "pred_abs_xywh": pred_abs_xywh,
    }


# ---------------------------------------------------------------------------
# Legacy path runner
# ---------------------------------------------------------------------------


def _run_legacy(scenario: dict) -> dict:
    """Run legacy ``COCOeval`` + ``coco_extended_metrics`` on *scenario* data.

    Args:
        scenario: Raw scenario dict from :func:`_build_intermediate_scenario`.

    Returns:
        Dict with keys ``"map50"``, ``"f1"``, ``"precision"``, ``"recall"``
        (all floats).
    """
    images = [
        {
            "id": scenario["image_id"],
            "width": scenario["image_width"],
            "height": scenario["image_height"],
        }
    ]
    annotations = []
    ann_id = 1
    for cat_id, box in scenario["gt_abs_xywh"]:
        annotations.append(
            {
                "id": ann_id,
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
        )
        ann_id += 1

    predictions = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        predictions.append(
            {
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "bbox": box,
                "score": score,
            }
        )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": scenario["categories"],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = coco_extended_metrics(coco_eval)
    return {
        "map50": float(coco_eval.stats[1]),
        "f1": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


# ---------------------------------------------------------------------------
# New callback path runner
# ---------------------------------------------------------------------------


def _run_callback(scenario: dict) -> dict:
    """Drive :class:`COCOEvalCallback` with *scenario* data and return logged metrics.

    Converts COCO-format boxes to the formats expected by the callback:

    - Predictions: absolute xyxy (already in absolute coords, just reformat).
    - Targets: normalised CxCyWH + ``orig_size``.

    Args:
        scenario: Raw scenario dict from :func:`_build_intermediate_scenario`.

    Returns:
        Dict mapping metric key (e.g. ``"val/mAP_50"``) to float value.
    """
    W = scenario["image_width"]
    H = scenario["image_height"]

    # Predictions: COCO [x, y, w, h] absolute → absolute xyxy tensors
    pred_boxes_xyxy: list[list[float]] = []
    pred_scores_list: list[float] = []
    pred_labels_list: list[int] = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        x, y, w, h = box
        pred_boxes_xyxy.append([x, y, x + w, y + h])
        pred_scores_list.append(score)
        pred_labels_list.append(cat_id)

    preds = [
        {
            "boxes": torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
            "scores": torch.tensor(pred_scores_list, dtype=torch.float32),
            "labels": torch.tensor(pred_labels_list, dtype=torch.long),
        }
    ]

    # Targets: COCO [x, y, w, h] absolute → normalised CxCyWH for the callback
    gt_boxes_norm: list[list[float]] = []
    gt_labels_list: list[int] = []
    for cat_id, box in scenario["gt_abs_xywh"]:
        x, y, w, h = box
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        wn = w / W
        hn = h / H
        gt_boxes_norm.append([cx, cy, wn, hn])
        gt_labels_list.append(cat_id)

    targets = [
        {
            "boxes": torch.tensor(gt_boxes_norm, dtype=torch.float32),
            "labels": torch.tensor(gt_labels_list, dtype=torch.long),
            "orig_size": torch.tensor([H, W]),  # [H, W] as expected by _convert_targets
        }
    ]

    cb = COCOEvalCallback(max_dets=500)
    trainer = MagicMock()
    module = MagicMock()
    logged: dict[str, float] = {}
    module.log.side_effect = lambda key, val: logged.__setitem__(key, float(val))

    cb.setup(trainer, module, stage="validate")
    cb.on_validation_batch_end(trainer, module, {"results": preds, "targets": targets}, None, 0)
    cb.on_validation_epoch_end(trainer, module)

    return logged


# ---------------------------------------------------------------------------
# Segmentation scenario helpers
# ---------------------------------------------------------------------------


def _box_to_bool_mask(box_xywh: list[float], H: int, W: int) -> torch.Tensor:
    """Return a [H, W] boolean mask with True inside the [x, y, w, h] box.

    Args:
        box_xywh: COCO-format ``[x, y, w, h]`` (float, absolute pixels).
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        Bool tensor of shape ``[H, W]``.
    """
    x, y, w, h = box_xywh
    mask = torch.zeros(H, W, dtype=torch.bool)
    y1 = max(0, int(round(y)))
    y2 = min(H, int(round(y + h)))
    x1 = max(0, int(round(x)))
    x2 = min(W, int(round(x + w)))
    mask[y1:y2, x1:x2] = True
    return mask


def _build_segmentation_scenario() -> dict:
    """Build segmentation scenario: same TP/FP structure as the detection scenario.

    Uses smaller boxes (``_SEGM_BOX_SIZE=40``) to keep boolean mask tensors
    memory-efficient (~500×165 image).

    Structure mirrors ``_build_intermediate_scenario()``:

    - Class 1: 10 GTs + 10 TP predictions (IoU ≥ 0.525, varying confidence).
    - Class 2: 10 GTs + 10 TP predictions (IoU ≥ 0.525) + 10 FP predictions
      (IoU = 0, lower confidence).

    Returns:
        Same keys as :func:`_build_intermediate_scenario`.
    """
    image_id = 1
    n_boxes = 10

    class1_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class1_confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    class2_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class2_confs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    class2_fp_confs = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    image_width = n_boxes * _SEGM_BOX_SPACING
    image_height = 3 * _SEGM_ROW_SPACING

    gt_abs_xywh: list[tuple[int, list[float]]] = []
    pred_abs_xywh: list[tuple[int, list[float], float]] = []

    for i, (iou, conf) in enumerate(zip(class1_ious, class1_confs)):
        gt_box = [float(i * _SEGM_BOX_SPACING), 0.0, float(_SEGM_BOX_SIZE), float(_SEGM_BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((1, gt_box))
        pred_abs_xywh.append((1, pred_box, conf))

    for i, (iou, conf) in enumerate(zip(class2_ious, class2_confs)):
        gt_box = [
            float(i * _SEGM_BOX_SPACING),
            float(_SEGM_ROW_SPACING),
            float(_SEGM_BOX_SIZE),
            float(_SEGM_BOX_SIZE),
        ]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((2, gt_box))
        pred_abs_xywh.append((2, pred_box, conf))

    for i, conf in enumerate(class2_fp_confs):
        fp_box = [
            float(i * _SEGM_BOX_SPACING),
            float(2 * _SEGM_ROW_SPACING),
            float(_SEGM_BOX_SIZE),
            float(_SEGM_BOX_SIZE),
        ]
        pred_abs_xywh.append((2, fp_box, conf))

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
        "gt_abs_xywh": gt_abs_xywh,
        "pred_abs_xywh": pred_abs_xywh,
    }


def _run_legacy_segm(scenario: dict) -> dict:
    """Run legacy ``COCOeval(iouType='segm')`` on *scenario* data.

    GT annotations use polygon segmentation (box corners); predictions use
    RLE-encoded binary box masks.

    Args:
        scenario: Raw scenario dict from :func:`_build_segmentation_scenario`.

    Returns:
        Dict with keys ``"map50"``, ``"f1"``, ``"precision"``, ``"recall"``.
    """
    W = scenario["image_width"]
    H = scenario["image_height"]

    images = [{"id": scenario["image_id"], "width": W, "height": H}]
    annotations = []
    ann_id = 1
    for cat_id, box in scenario["gt_abs_xywh"]:
        x, y, w, h = box
        annotations.append(
            {
                "id": ann_id,
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "bbox": box,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                "area": w * h,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    predictions = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        x, y, w, h = box
        mask_np = np.zeros((H, W), dtype=np.uint8)
        y1 = max(0, int(round(y)))
        y2 = min(H, int(round(y + h)))
        x1 = max(0, int(round(x)))
        x2 = min(W, int(round(x + w)))
        mask_np[y1:y2, x1:x2] = 1
        rle = mask_utils.encode(np.asfortranarray(mask_np))
        rle["counts"] = rle["counts"].decode("utf-8")
        predictions.append(
            {
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "segmentation": rle,
                "score": score,
            }
        )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": scenario["categories"],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = coco_extended_metrics(coco_eval)
    return {
        "map50": float(coco_eval.stats[1]),
        "f1": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


def _run_callback_segm(scenario: dict) -> dict:
    """Drive :class:`COCOEvalCallback` (segmentation=True) with mask tensors.

    Predictions include stacked [N, H, W] boolean masks; targets include
    normalised CxCyWH boxes and [M, H, W] boolean masks.

    Args:
        scenario: Raw scenario dict from :func:`_build_segmentation_scenario`.

    Returns:
        Dict mapping metric key to float value (same format as
        :func:`_run_callback`).
    """
    W = scenario["image_width"]
    H = scenario["image_height"]

    pred_boxes_xyxy: list[list[float]] = []
    pred_scores_list: list[float] = []
    pred_labels_list: list[int] = []
    pred_mask_list: list[torch.Tensor] = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        x, y, w, h = box
        pred_boxes_xyxy.append([x, y, x + w, y + h])
        pred_scores_list.append(score)
        pred_labels_list.append(cat_id)
        pred_mask_list.append(_box_to_bool_mask(box, H, W))

    preds = [
        {
            "boxes": torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
            "scores": torch.tensor(pred_scores_list, dtype=torch.float32),
            "labels": torch.tensor(pred_labels_list, dtype=torch.long),
            "masks": torch.stack(pred_mask_list),  # [N, H, W]
        }
    ]

    gt_boxes_norm: list[list[float]] = []
    gt_labels_list: list[int] = []
    gt_mask_list: list[torch.Tensor] = []
    for cat_id, box in scenario["gt_abs_xywh"]:
        x, y, w, h = box
        gt_boxes_norm.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
        gt_labels_list.append(cat_id)
        gt_mask_list.append(_box_to_bool_mask(box, H, W))

    targets = [
        {
            "boxes": torch.tensor(gt_boxes_norm, dtype=torch.float32),
            "labels": torch.tensor(gt_labels_list, dtype=torch.long),
            "masks": torch.stack(gt_mask_list),  # [M, H, W]
            "orig_size": torch.tensor([H, W]),
        }
    ]

    cb = COCOEvalCallback(max_dets=500, segmentation=True)
    trainer = MagicMock()
    module = MagicMock()
    logged: dict[str, float] = {}
    module.log.side_effect = lambda key, val: logged.__setitem__(key, float(val))

    cb.setup(trainer, module, stage="validate")
    cb.on_validation_batch_end(trainer, module, {"results": preds, "targets": targets}, None, 0)
    cb.on_validation_epoch_end(trainer, module)

    return logged


# ---------------------------------------------------------------------------
# Module-scoped fixtures — compute each scenario once, share across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detection_parity():
    """``(legacy_dict, new_dict)`` for the intermediate detection scenario.

    Runs ``_build_intermediate_scenario`` → ``_run_legacy`` → ``_run_callback``
    exactly once for the entire test module; all detection parity tests share
    the result.
    """
    scenario = _build_intermediate_scenario()
    return _run_legacy(scenario), _run_callback(scenario)


@pytest.fixture(scope="module")
def segmentation_parity():
    """``(legacy_dict, new_dict)`` for the segmentation scenario.

    Runs ``_build_segmentation_scenario`` → ``_run_legacy_segm`` → ``_run_callback_segm``
    exactly once for the entire test module.
    """
    scenario = _build_segmentation_scenario()
    return _run_legacy_segm(scenario), _run_callback_segm(scenario)


@pytest.fixture(scope="module")
def perfect_parity():
    """``(legacy_dict, new_dict)`` for the all-TP boundary scenario."""
    scenario = _build_perfect_scenario()
    return _run_legacy(scenario), _run_callback(scenario)


@pytest.fixture(scope="module")
def degenerate_parity():
    """``(legacy_dict, new_dict)`` for the all-FP boundary scenario."""
    scenario = _build_degenerate_scenario()
    return _run_legacy(scenario), _run_callback(scenario)


# ---------------------------------------------------------------------------
# Parity tests — detection mAP50
# ---------------------------------------------------------------------------


class TestMAPParityDetection:
    """val/mAP_50 from COCOEvalCallback agrees with legacy COCOeval.

    Tolerance: ``|Δ mAP50| ≤ 0.005``.
    """

    def test_map50_within_tolerance(self, detection_parity) -> None:
        """|Δ mAP50| ≤ 0.005 on the intermediate scenario."""
        legacy, new = detection_parity
        delta = abs(legacy["map50"] - new["val/mAP_50"])
        assert delta <= _MAP50_TOL, (
            f"mAP50 parity failed: legacy={legacy['map50']:.4f}, "
            f"new={new['val/mAP_50']:.4f}, delta={delta:.4f} > {_MAP50_TOL}"
        )

    def test_map50_95_is_logged(self, detection_parity) -> None:
        """val/mAP_50_95 is always logged.

        Note: torchmetrics returns -1.0 as a sentinel when ``map`` cannot be
        computed with non-default ``max_detection_thresholds`` (e.g. 500).
        The assertion only verifies the key is present; use ``val/mAP_50`` for
        numeric comparisons.
        """
        _, new = detection_parity
        assert "val/mAP_50_95" in new

    def test_mar_logged_and_nonnegative(self, detection_parity) -> None:
        """val/mAR is logged and non-negative."""
        _, new = detection_parity
        assert "val/mAR" in new
        assert new["val/mAR"] >= 0.0


# ---------------------------------------------------------------------------
# Parity tests — detection F1 / precision / recall
# ---------------------------------------------------------------------------


class TestF1ParityDetection:
    """F1, precision, recall from COCOEvalCallback agree with legacy path.

    Tolerance: ``|Δ| ≤ 0.01`` for all three metrics.
    """

    @pytest.mark.parametrize(
        "legacy_key, new_key",
        [
            pytest.param("f1", "val/F1", id="f1"),
            pytest.param("precision", "val/precision", id="precision"),
            pytest.param("recall", "val/recall", id="recall"),
        ],
    )
    def test_metric_within_tolerance(self, detection_parity, legacy_key, new_key) -> None:
        """|Δ metric| ≤ 0.01 on the intermediate scenario."""
        legacy, new = detection_parity
        delta = abs(legacy[legacy_key] - new[new_key])
        assert delta <= _F1_TOL, (
            f"{legacy_key} parity failed: legacy={legacy[legacy_key]:.4f}, "
            f"new={new[new_key]:.4f}, delta={delta:.4f} > {_F1_TOL}"
        )


# ---------------------------------------------------------------------------
# Sanity-check parity at the boundary scenarios
# ---------------------------------------------------------------------------


class TestBoundaryScenarioParity:
    """Both paths report consistent values on boundary (all-TP, all-FP) scenarios."""

    @pytest.mark.parametrize(
        "legacy_key, new_key",
        [
            pytest.param("map50", "val/mAP_50", id="map50"),
            pytest.param("f1", "val/F1", id="f1"),
        ],
    )
    def test_perfect_scenario_near_one(self, perfect_parity, legacy_key, new_key) -> None:
        """Both paths report metric ≥ 0.99 on the perfect scenario."""
        legacy, new = perfect_parity
        assert legacy[legacy_key] >= 0.99, f"Legacy {legacy_key} unexpectedly low: {legacy[legacy_key]:.4f}"
        assert new[new_key] >= 0.99, f"Callback {new_key} unexpectedly low: {new[new_key]:.4f}"

    def test_degenerate_scenario_f1_is_zero(self, degenerate_parity) -> None:
        """Both paths report F1 = 0.0 on the degenerate (all-FP) scenario."""
        legacy, new = degenerate_parity
        assert legacy["f1"] == pytest.approx(0.0), f"Legacy F1 should be 0, got {legacy['f1']:.4f}"
        assert new["val/F1"] == pytest.approx(0.0), f"Callback F1 should be 0, got {new['val/F1']:.4f}"


# ---------------------------------------------------------------------------
# Parity tests — segmentation mAP50
# ---------------------------------------------------------------------------


class TestMAPParitySegmentation:
    """val/segm_mAP_50 from COCOEvalCallback agrees with legacy COCOeval(segm).

    Tolerance: ``|Δ mask mAP50| ≤ 0.005``.
    """

    def test_segm_map50_within_tolerance(self, segmentation_parity) -> None:
        """|Δ mask mAP50| ≤ 0.005 on the segmentation scenario."""
        legacy, new = segmentation_parity
        delta = abs(legacy["map50"] - new["val/segm_mAP_50"])
        assert delta <= _MAP50_TOL, (
            f"Segm mAP50 parity failed: legacy={legacy['map50']:.4f}, "
            f"new={new['val/segm_mAP_50']:.4f}, delta={delta:.4f} > {_MAP50_TOL}"
        )

    def test_segm_map50_95_is_logged(self, segmentation_parity) -> None:
        """val/segm_mAP_50_95 is always logged in segmentation mode."""
        _, new = segmentation_parity
        assert "val/segm_mAP_50_95" in new


# ---------------------------------------------------------------------------
# Parity tests — segmentation F1 / precision / recall
# ---------------------------------------------------------------------------


class TestF1ParitySegmentation:
    """F1, precision, recall from COCOEvalCallback(segm) agree with legacy path.

    Tolerance: ``|Δ| ≤ 0.01`` for all three metrics.

    The callback computes F1 via ``build_matching_data(iou_type='segm')`` and
    ``sweep_confidence_thresholds``; the legacy path uses
    ``COCOeval(iouType='segm')`` + ``coco_extended_metrics()``.
    """

    @pytest.mark.parametrize(
        "legacy_key, new_key",
        [
            pytest.param("f1", "val/F1", id="f1"),
            pytest.param("precision", "val/precision", id="precision"),
            pytest.param("recall", "val/recall", id="recall"),
        ],
    )
    def test_segm_metric_within_tolerance(self, segmentation_parity, legacy_key, new_key) -> None:
        """|Δ metric| ≤ 0.01 on the segmentation scenario."""
        legacy, new = segmentation_parity
        delta = abs(legacy[legacy_key] - new[new_key])
        assert delta <= _F1_TOL, (
            f"Segm {legacy_key} parity failed: legacy={legacy[legacy_key]:.4f}, "
            f"new={new[new_key]:.4f}, delta={delta:.4f} > {_F1_TOL}"
        )


# ---------------------------------------------------------------------------
# Test-loop callback path runner
# ---------------------------------------------------------------------------


def _run_callback_test(scenario: dict) -> dict:
    """Drive :class:`COCOEvalCallback` via the *test-loop* hooks and return logged metrics.

    Identical data pipeline to :func:`_run_callback`; only the PTL hook names
    differ (``on_test_batch_end`` / ``on_test_epoch_end``) and the metric
    prefix is ``test/`` instead of ``val/``.

    Args:
        scenario: Raw scenario dict from :func:`_build_intermediate_scenario`.

    Returns:
        Dict mapping metric key (e.g. ``"test/mAP_50"``) to float value.
    """
    W = scenario["image_width"]
    H = scenario["image_height"]

    pred_boxes_xyxy: list[list[float]] = []
    pred_scores_list: list[float] = []
    pred_labels_list: list[int] = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        x, y, w, h = box
        pred_boxes_xyxy.append([x, y, x + w, y + h])
        pred_scores_list.append(score)
        pred_labels_list.append(cat_id)

    preds = [
        {
            "boxes": torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
            "scores": torch.tensor(pred_scores_list, dtype=torch.float32),
            "labels": torch.tensor(pred_labels_list, dtype=torch.long),
        }
    ]

    gt_boxes_norm: list[list[float]] = []
    gt_labels_list: list[int] = []
    for cat_id, box in scenario["gt_abs_xywh"]:
        x, y, w, h = box
        gt_boxes_norm.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
        gt_labels_list.append(cat_id)

    targets = [
        {
            "boxes": torch.tensor(gt_boxes_norm, dtype=torch.float32),
            "labels": torch.tensor(gt_labels_list, dtype=torch.long),
            "orig_size": torch.tensor([H, W]),
        }
    ]

    cb = COCOEvalCallback(max_dets=500)
    trainer = MagicMock()
    module = MagicMock()
    logged: dict[str, float] = {}
    module.log.side_effect = lambda key, val: logged.__setitem__(key, float(val))

    cb.setup(trainer, module, stage="test")
    cb.on_test_batch_end(trainer, module, {"results": preds, "targets": targets}, None, 0)
    cb.on_test_epoch_end(trainer, module)

    return logged


# ---------------------------------------------------------------------------
# Module-scoped fixture — test vs val path on identical data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detection_test_vs_val_parity():
    """``(val_logged, test_logged)`` for the intermediate detection scenario.

    Runs both :func:`_run_callback` and :func:`_run_callback_test` on the same
    data once for the module; all test-vs-val parity tests share the result.
    """
    scenario = _build_intermediate_scenario()
    return _run_callback(scenario), _run_callback_test(scenario)


# ---------------------------------------------------------------------------
# Parity tests — test loop vs val loop (same data, same algorithm)
# ---------------------------------------------------------------------------


class TestTestVsValParity:
    """Test-loop metrics must agree with val-loop metrics on identical data.

    Both loops share the same underlying computation
    (``MeanAveragePrecision`` + ``sweep_confidence_thresholds``); the only
    difference is the logged key prefix.  Running them on identical data
    verifies prefix routing correctness and that no extra state or logic
    diverges between the two paths.

    Tolerance: exact match (same algorithm, same data, deterministic
    computation), with a small floating-point tolerance of ``1e-6``.
    """

    @pytest.mark.parametrize(
        "val_key, test_key",
        [
            pytest.param("val/mAP_50", "test/mAP_50", id="mAP50"),
            pytest.param("val/mAP_75", "test/mAP_75", id="mAP75"),
            pytest.param("val/mAR", "test/mAR", id="mAR"),
            pytest.param("val/F1", "test/F1", id="F1"),
            pytest.param("val/precision", "test/precision", id="precision"),
            pytest.param("val/recall", "test/recall", id="recall"),
        ],
    )
    def test_metric_identical_in_test_and_val(self, detection_test_vs_val_parity, val_key, test_key) -> None:
        """Test-loop metric must match val-loop metric exactly on identical data."""
        val_logged, test_logged = detection_test_vs_val_parity
        assert test_key in test_logged, f"Expected {test_key!r} in test logged keys: {list(test_logged)}"
        assert val_key in val_logged, f"Expected {val_key!r} in val logged keys: {list(val_logged)}"
        assert test_logged[test_key] == pytest.approx(val_logged[val_key], abs=1e-6), (
            f"{test_key}={test_logged[test_key]:.6f} != {val_key}={val_logged[val_key]:.6f}"
        )

    def test_test_prefix_logged_not_val(self, detection_test_vs_val_parity) -> None:
        """Test-loop must emit test/ keys, not val/ keys."""
        _, test_logged = detection_test_vs_val_parity
        val_bleed = [k for k in test_logged if k.startswith("val/")]
        assert not val_bleed, f"Found val/ keys in test loop: {val_bleed}"

    def test_val_prefix_logged_not_test(self, detection_test_vs_val_parity) -> None:
        """Val-loop must emit val/ keys, not test/ keys."""
        val_logged, _ = detection_test_vs_val_parity
        test_bleed = [k for k in val_logged if k.startswith("test/")]
        assert not test_bleed, f"Found test/ keys in val loop: {test_bleed}"
