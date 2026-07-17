# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO val2017 inference benchmarks asserting pretrained-weight accuracy on CPU and GPU.

Each model family (detection, segmentation) is covered by **two independent code paths**:

``RFDETR.predict()`` path (public API)
    Loads images as PIL, calls ``RFDETR.predict()`` in batches, accumulates predictions into
    ``torchmetrics.MeanAveragePrecision`` and a confidence-threshold sweep for macro-F1.  Exercises the
    end-to-end public inference surface — preprocessing, backbone, decoder, postprocessing — without any
    PTL machinery.  Tests: :func:`test_inference_detection_rfdetr_predict`,
    :func:`test_inference_segmentation_rfdetr_predict`.

PTL training-stack path (``Trainer.validate``)
    Copies pretrained weights into :class:`~rfdetr.training.RFDETRModelModule`, runs ``Trainer.validate``
    with a :class:`~rfdetr.training.RFDETRDataModule`, and reads ``val/mAP_50`` / ``val/F1`` from the
    callback metrics.  Exercises ``validation_step``, ``on_after_batch_transfer``, and
    :class:`~rfdetr.training.COCOEvalCallback` — the same code path used during training.  Tests:
    :func:`test_inference_detection_ptl_predict`, :func:`test_inference_segmentation_ptl_predict`.

Both paths run on CPU (nano models) and GPU (small and larger models, ``@pytest.mark.gpu``).

API contract tests (return type, shape) live in ``tests/models/test_predict.py`` and do not require a COCO
download.
"""

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch
from faster_coco_eval import COCO
from pytorch_lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from rfdetr import (
    RFDETRKeypointPreview,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.evaluation.coco_eval import CocoEvaluator
from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import (
    build_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

# All tests in this file download COCO val2017 (~1 GB); exclude from CPU CI with -m "not coco17".
pytestmark = pytest.mark.coco17

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bbox_dict(
    boxes: "list[list[float]] | np.ndarray",
    labels: "list[int] | np.ndarray",
    scores: "list[float] | np.ndarray | None" = None,
    iscrowd: "list[int] | np.ndarray | None" = None,
) -> dict[str, torch.Tensor]:
    """Build a torchmetrics-compatible bounding-box dict from raw list or array data.

    Handles empty inputs transparently — an empty *boxes* list produces a ``(0, 4)`` tensor.

    Args:
        boxes: Bounding boxes in xyxy format, shape (N, 4).
        labels: Integer class labels, length N.
        scores: Per-detection confidence scores, length N.  Present in prediction dicts only.
        iscrowd: Crowd flags (0/1), length N.  Present in target dicts only.

    Returns:
        Dict always containing ``boxes`` (N, 4) float32 and ``labels`` (N,) int64; optionally
        ``scores`` (N,) float32 and/or ``iscrowd`` (N,) uint8.
    """
    result: dict[str, torch.Tensor] = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    if scores is not None:
        result["scores"] = torch.tensor(scores, dtype=torch.float32)
    if iscrowd is not None:
        result["iscrowd"] = torch.tensor(iscrowd, dtype=torch.uint8)
    return result


def _coco_ann_to_target(coco_gt: "COCO", img_id: int) -> dict[str, torch.Tensor]:
    """Build a torchmetrics target dict from COCO ground-truth annotations for one image.

    Args:
        coco_gt: Loaded ``pycocotools.coco.COCO`` object.
        img_id: COCO image ID.

    Returns:
        Dict with ``boxes`` (M, 4) xyxy float, ``labels`` (M,) int64, ``iscrowd`` (M,) uint8.
    """
    anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    gt_boxes: list[list[float]] = []
    gt_labels: list[int] = []
    iscrowd: list[int] = []
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        gt_boxes.append([bx, by, bx + bw, by + bh])
        gt_labels.append(ann["category_id"])
        iscrowd.append(int(ann.get("iscrowd", 0)))
    return _bbox_dict(gt_boxes, gt_labels, iscrowd=iscrowd)


def _score_rfdetr_predict(
    rfdetr_obj: RFDETR,
    images_root: Path,
    annotations_path: Path,
    num_samples: int,
    batch_size: int,
) -> tuple[float, float]:
    """Run ``RFDETR.predict()`` on a COCO val subset and return ``(mAP@50, macro-F1)``.

    Loads images from disk as PIL images, calls ``rfdetr_obj.predict()`` in batches, converts
    :class:`~supervision.Detections` to torchmetrics format, and computes bbox mAP@50 via
    ``MeanAveragePrecision`` and macro-F1 via a confidence-threshold sweep.

    Args:
        rfdetr_obj: Pretrained :class:`~rfdetr.detr.RFDETR` instance.
        images_root: Directory containing COCO val images (``val2017/``).
        annotations_path: Path to ``instances_val2017.json``.
        num_samples: Number of images to evaluate (first N by sorted image ID).
        batch_size: Number of images per ``predict()`` call.

    Returns:
        Tuple ``(mAP@50, macro_f1)`` computed over the evaluated subset.
    """
    coco_gt = COCO(str(annotations_path))
    img_ids = sorted(coco_gt.getImgIds())[:num_samples]

    map_metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=False,
        max_detection_thresholds=[1, 10, 500],
        backend="faster_coco_eval",
    )
    f1_local = init_matching_accumulator()

    for start in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[start : start + batch_size]
        images: list[PIL.Image.Image] = []
        for img_id in batch_ids:
            with PIL.Image.open(images_root / f"{img_id:012d}.jpg") as im:
                images.append(im.convert("RGB"))
        detections_batch = rfdetr_obj.predict(images, threshold=0.001, include_source_image=False)
        if not isinstance(detections_batch, list):
            detections_batch = [detections_batch]
        preds = [_bbox_dict(det.xyxy, det.class_id, scores=det.confidence) for det in detections_batch]
        targets = [_coco_ann_to_target(coco_gt, img_id) for img_id in batch_ids]

        map_metric.update(preds, targets)
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type="bbox")
        merge_matching_data(f1_local, batch_matching)

    metrics = map_metric.compute()
    map50 = float(metrics["map_50"])

    f1_val = 0.0
    if f1_local:
        sorted_ids = sorted(f1_local.keys())
        per_class_list = [f1_local[cid] for cid in sorted_ids]
        classes_with_gt = [i for i, cid in enumerate(sorted_ids) if f1_local[cid]["total_gt"] > 0]
        f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
        best = max(f1_results, key=lambda x: x["macro_f1"])
        f1_val = float(best["macro_f1"])

    return map50, f1_val


def _build_train_config(coco_root: Path, tmp_path: Path, batch_size: int) -> TrainConfig:
    """Build a minimal :class:`~rfdetr.config.TrainConfig` for COCO inference runs.

    Loggers and EMA are disabled; the config is only used for validation.

    Args:
        coco_root: Directory containing ``val2017/`` and ``annotations/``.
        tmp_path: Temporary directory used as ``output_dir``.
        batch_size: DataLoader batch size.

    Returns:
        Minimal :class:`~rfdetr.config.TrainConfig` suitable for validation.
    """
    return TrainConfig(
        dataset_file="coco",
        dataset_dir=str(coco_root),
        output_dir=str(tmp_path),
        batch_size=batch_size,
        num_workers=0 if not torch.cuda.is_available() else min(os.cpu_count(), 4),
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        use_ema=False,
        run_test=False,
        compute_val_loss=False,
    )


def _build_datamodule(
    model_config: ModelConfig,
    train_config: TrainConfig,
    num_samples: Optional[int] = None,
) -> RFDETRDataModule:
    """Set up an :class:`~rfdetr.training.RFDETRDataModule` for validation.

    Calls ``setup("validate")`` so ``_dataset_val`` is ready.  When *num_samples* is set the dataset is wrapped in a
    :class:`torch.utils.data.Subset`.

    Args:
        model_config: Architecture config (``segmentation_head`` controls mask loading).
        train_config: Training config.
        num_samples: If set, truncate the val dataset to this many samples.

    Returns:
        Datamodule with ``_dataset_val`` populated.
    """
    dm = RFDETRDataModule(model_config, train_config)
    dm.setup("validate")
    if num_samples is not None:
        dm._dataset_val = torch.utils.data.Subset(
            dm._dataset_val,
            list(range(min(num_samples, len(dm._dataset_val)))),
        )
    return dm


def _build_ptl_module(rfdetr_obj: RFDETR, train_config: TrainConfig) -> RFDETRModelModule:
    """Copy pretrained weights from *rfdetr_obj* into a fresh :class:`~rfdetr.training.RFDETRModelModule`.

    Constructs the module with the same architecture (no pretrain download), loads weights from
    ``rfdetr_obj.model.model``, and asserts PTL lineage and weight-copy correctness before returning.

    Args:
        rfdetr_obj: A pretrained :class:`~rfdetr.detr.RFDETR` instance.
        train_config: Shared :class:`~rfdetr.config.TrainConfig` (must have a
            valid ``output_dir``).

    Returns:
        Weight-synced :class:`~rfdetr.training.RFDETRModelModule` ready for ``Trainer.validate`` or ``Trainer.predict``.
    """
    module = RFDETRModelModule(rfdetr_obj.model_config, train_config)
    module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    module.model.eval()

    assert isinstance(module, RFDETRModelModule), f"Expected RFDETRModelModule, got {type(module).__name__}"
    assert isinstance(module, LightningModule), (
        "module must be a pytorch_lightning.LightningModule — this confirms evaluation runs through the PTL stack"
    )

    _first_key = next(iter(rfdetr_obj.model.model.state_dict()))
    assert torch.equal(
        rfdetr_obj.model.model.state_dict()[_first_key].cpu(),
        module.model.state_dict()[_first_key].cpu(),
    ), f"Weight copy failed: '{_first_key}' differs between legacy model and PTL module"

    return module


def _select_fixed_person_images(
    images_root: Path,
    annotations_path: Path,
    max_images: int = 8,
) -> tuple[list[str], list[int]]:
    """Load a deterministic subset of COCO person-keypoint validation images.

    Args:
        images_root: Directory containing COCO validation images.
        annotations_path: COCO person-keypoints annotations JSON path.
        max_images: Maximum number of keypoint-bearing images to load.

    Returns:
        RGB image paths and their corresponding COCO image IDs.

    Raises:
        RuntimeError: If no usable person-keypoint images are available.
    """
    with annotations_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    image_id_to_name = {int(item["id"]): str(item["file_name"]) for item in payload["images"]}
    person_image_ids = sorted(
        {
            int(annotation["image_id"])
            for annotation in payload["annotations"]
            if int(annotation.get("num_keypoints", 0)) > 0 and int(annotation.get("iscrowd", 0)) == 0
        }
    )
    selected_ids = person_image_ids[:max_images]
    if not selected_ids:
        raise RuntimeError("No keypoint-bearing COCO validation images were found.")

    image_paths: list[str] = []
    for image_id in selected_ids:
        image_path = images_root / image_id_to_name[image_id]
        image_paths.append(str(image_path))

    return image_paths, selected_ids


def _predict_keypoint_preview_batches(
    model: RFDETRKeypointPreview,
    image_paths: Sequence[str],
    batch_size: int,
    threshold: float = 0.5,
) -> list[sv.KeyPoints]:
    """Run keypoint-preview inference in fixed-size batches.

    Args:
        model: Loaded keypoint-preview model.
        image_paths: COCO image paths to evaluate.
        batch_size: Number of RGB images to pass to each ``predict()`` call.
        threshold: Minimum confidence score passed to ``RFDETRKeypointPreview.predict()``.

    Returns:
        Per-image keypoint detections in the same order as ``image_paths``.

    Raises:
        RuntimeError: If batched prediction unexpectedly returns a single detection object.
    """
    predictions: list[sv.KeyPoints] = []
    for start_idx in range(0, len(image_paths), batch_size):
        batch_paths = list(image_paths[start_idx : start_idx + batch_size])
        batch_images: list[PIL.Image.Image] = []
        for image_path in batch_paths:
            with PIL.Image.open(image_path) as image:
                batch_images.append(image.convert("RGB"))

        batch_predictions = model.predict(batch_images, threshold=threshold, include_source_image=False)
        if not isinstance(batch_predictions, list):
            raise RuntimeError("Expected batched keypoint preview inference to return list[KeyPoints].")
        predictions.extend(batch_predictions)
    return predictions


def _detections_to_coco_predictions(
    detections_batch: list[sv.KeyPoints],
    image_ids: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Convert batched supervision keypoints into the COCO evaluator format.

    Args:
        detections_batch: Per-image prediction batch returned by RF-DETR.
        image_ids: COCO image IDs matching ``detections_batch`` order.

    Returns:
        COCO evaluator prediction dictionary keyed by image ID.
    """
    predictions: dict[int, dict[str, torch.Tensor]] = {}
    for image_id, key_points in zip(image_ids, detections_batch):
        xyxy = key_points.data.get("xyxy")
        if xyxy is None or key_points.detection_confidence is None or key_points.class_id is None:
            raise ValueError("Expected keypoint preview predictions to populate detection details.")
        if key_points.keypoint_confidence is None:
            raise ValueError("Expected keypoint preview predictions to populate per-keypoint confidence.")
        keypoints = np.concatenate((key_points.xy, key_points.keypoint_confidence[:, :, np.newaxis]), axis=2)
        predictions[image_id] = {
            "boxes": torch.as_tensor(xyxy, dtype=torch.float32),
            "scores": torch.as_tensor(key_points.detection_confidence, dtype=torch.float32),
            "labels": torch.as_tensor(key_points.class_id, dtype=torch.int64),
            "keypoints": torch.as_tensor(keypoints, dtype=torch.float32),
        }
    return predictions


@pytest.fixture(scope="session")
def keypoint_preview_predictions(
    download_coco_val_keypoints: tuple[Path, Path],
) -> tuple[list[sv.KeyPoints], list[int], Path]:
    """Run one deterministic keypoint-preview inference pass for the COCO benchmark tests."""
    images_root, annotations_path = download_coco_val_keypoints
    image_paths, image_ids = _select_fixed_person_images(images_root, annotations_path)
    model = RFDETRKeypointPreview(device="cuda" if torch.cuda.is_available() else "cpu")
    predictions = _predict_keypoint_preview_batches(model, image_paths, batch_size=8)
    return predictions, image_ids, annotations_path


# ---------------------------------------------------------------------------
# Inference — RFDETR.predict() (CPU nano) / Trainer.validate() (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.66, 0.66, 200, 6, id="det-nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, 6, id="det-small", marks=pytest.mark.gpu),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, 4, id="det-medium", marks=pytest.mark.gpu),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, 2, id="det-large", marks=pytest.mark.gpu),
    ],
)
def test_inference_detection_rfdetr_predict(
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts mAP@50 and macro-F1 thresholds on COCO val for detection models via ``RFDETR.predict()``.

    Loads a pretrained detection model, calls ``RFDETR.predict()`` in batches on *num_samples* COCO val images,
    scores via ``torchmetrics.MeanAveragePrecision`` and a confidence-threshold sweep.  Runs on CPU (nano) and GPU
    (small/medium/large) — GPU params use a smaller *num_samples* to stay within the CI timeout.

    Args:
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Detection model class to instantiate with pretrained weights.
        threshold_map: Minimum bbox mAP@50 required.
        threshold_f1: Minimum macro-F1 (best across confidence sweep) required.
        num_samples: Number of COCO val images to evaluate.
        batch_size: Number of images per batch.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    model = model_cls(device=device_str)
    map_val, f1_val = _score_rfdetr_predict(model, images_root, annotations_path, num_samples, batch_size)

    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 0.64, 200, 6, id="seg-nano"),
        pytest.param(RFDETRSegSmall, 0.66, 0.67, 100, 6, id="seg-small", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegMedium, 0.68, 0.68, 100, 4, id="seg-medium", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegLarge, 0.70, 0.69, 100, 2, id="seg-large", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegXLarge, 0.72, 0.70, 100, 2, id="seg-xlarge", marks=pytest.mark.gpu),
        pytest.param(RFDETRSeg2XLarge, 0.73, 0.71, 100, 2, id="seg-2xlarge", marks=pytest.mark.gpu),
    ],
)
def test_inference_segmentation_rfdetr_predict(
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts bbox mAP@50 and macro-F1 thresholds on COCO val for segmentation models via ``RFDETR.predict()``.

    Loads a pretrained segmentation model, calls ``RFDETR.predict()`` in batches on *num_samples* COCO val images,
    scores via ``torchmetrics.MeanAveragePrecision`` and a confidence-threshold sweep.  Masks are not required — only
    bbox IoU is used for scoring.  Runs on CPU (nano) and GPU (small and larger variants).

    Args:
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Segmentation model class to instantiate with pretrained weights.
        threshold_map: Minimum bbox mAP@50 required.
        threshold_f1: Minimum macro-F1 (best across confidence sweep) required.
        num_samples: Number of COCO val images to evaluate.
        batch_size: Number of images per batch.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    model = model_cls(device=device_str)
    map_val, f1_val = _score_rfdetr_predict(model, images_root, annotations_path, num_samples, batch_size)

    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


@pytest.mark.coco17
def test_keypoint_preview_pretrained_inference_thresholded(
    keypoint_preview_predictions: tuple[list[sv.KeyPoints], list[int], Path],
) -> None:
    """Pretrained preview inference should emit thresholded person keypoints."""
    predictions, _, _ = keypoint_preview_predictions
    assert predictions, "Expected at least one inference result."

    total_detections = 0
    total_keypoint_sets = 0
    confidences: list[np.ndarray] = []

    for key_points in predictions:
        total_detections += len(key_points)
        assert key_points.detection_confidence is not None
        confidences.append(key_points.detection_confidence)
        assert key_points.keypoint_confidence is not None
        assert key_points.xy.ndim == 3
        assert key_points.xy.shape[1:] == (17, 2)
        assert key_points.keypoint_confidence.shape == (len(key_points), 17)
        assert np.isfinite(key_points.xy).all()
        assert np.isfinite(key_points.keypoint_confidence).all()
        total_keypoint_sets += key_points.xy.shape[0]

    assert total_detections > 0, "Expected at least one detection above threshold=0.5."
    assert total_keypoint_sets > 0, "Expected at least one emitted keypoint set."

    all_confidences = np.concatenate(confidences) if confidences else np.array([], dtype=np.float32)
    assert all_confidences.size > 0
    assert float(np.mean(all_confidences)) >= 0.5


@pytest.mark.gpu
@pytest.mark.coco17
@pytest.mark.parametrize(
    ("threshold_keypoint_map", "num_samples", "batch_size"),
    [
        pytest.param(0.71, 500, 2, id="keypoint-preview"),
    ],
)
def test_inference_keypoint_preview_rfdetr_predict(
    download_coco_val_keypoints: tuple[Path, Path],
    threshold_keypoint_map: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """``RFDETRKeypointPreview.predict()`` meets the keypoint COCO AP threshold."""
    images_root, annotations_path = download_coco_val_keypoints
    image_paths, image_ids = _select_fixed_person_images(images_root, annotations_path, max_images=num_samples)
    assert len(image_ids) >= num_samples, f"Expected at least {num_samples} keypoint-bearing images."

    model = RFDETRKeypointPreview(device="cuda" if torch.cuda.is_available() else "cpu")
    predictions = _predict_keypoint_preview_batches(model, image_paths, batch_size=batch_size, threshold=0.0)
    coco_gt = COCO(str(annotations_path))
    coco_gt.label2cat = {1: 1}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"])
    evaluator.update(_detections_to_coco_predictions(predictions, image_ids))
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    keypoint_ap_50_95 = float(evaluator.coco_eval["keypoints"].stats[0])
    assert keypoint_ap_50_95 >= threshold_keypoint_map, (
        f"keypoint AP@50:95 {keypoint_ap_50_95:.4f} < {threshold_keypoint_map}"
    )


# ---------------------------------------------------------------------------
# Inference — Trainer.validate() via PTL stack (CPU + GPU, COCO val2017)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.66, 0.66, 200, 6, id="det-nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, 6, id="det-small", marks=pytest.mark.gpu),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, 4, id="det-medium", marks=pytest.mark.gpu),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, 2, id="det-large", marks=pytest.mark.gpu),
    ],
)
def test_inference_detection_ptl_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts mAP@50 and macro-F1 thresholds on COCO val for detection models via the PTL training stack.

    Loads a pretrained detection model, copies weights into a :class:`~rfdetr.training.RFDETRModelModule`, and asserts
    mAP and F1 via ``Trainer.validate``.  Exercises the PTL validation loop (``validation_step`` + callbacks) rather
    than the public ``RFDETR.predict()`` API.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Detection model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` required.
        threshold_f1: Minimum ``val/F1`` (best macro-F1 across confidence sweep) required.
        num_samples: Number of val samples used for ``Trainer.validate``.
        batch_size: DataLoader batch size.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent
    accelerator = "auto" if torch.cuda.is_available() else "cpu"

    model = model_cls(device=device_str)
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    module = _build_ptl_module(model, tc)
    trainer = build_trainer(tc, model.model_config, accelerator=accelerator)

    dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
    (metrics,) = trainer.validate(module, datamodule=dm)
    map_val = metrics["val/mAP_50"]
    f1_val = metrics["val/F1"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 0.64, 200, 6, id="seg-nano"),
        pytest.param(RFDETRSegSmall, 0.66, 0.67, 100, 6, id="seg-small", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegMedium, 0.68, 0.68, 100, 4, id="seg-medium", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegLarge, 0.70, 0.69, 100, 2, id="seg-large", marks=pytest.mark.gpu),
        pytest.param(RFDETRSegXLarge, 0.72, 0.70, 100, 2, id="seg-xlarge", marks=pytest.mark.gpu),
        pytest.param(RFDETRSeg2XLarge, 0.73, 0.71, 100, 2, id="seg-2xlarge", marks=pytest.mark.gpu),
    ],
)
def test_inference_segmentation_ptl_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts bbox mAP@50 and macro-F1 thresholds on COCO val for segmentation models via the PTL training stack.

    Same structure as :func:`test_inference_detection_ptl_predict` but for segmentation variants.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Segmentation model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` (bbox) required.
        threshold_f1: Minimum ``val/F1`` (best macro-F1 across confidence sweep) required.
        num_samples: Number of val samples used for ``Trainer.validate``.
        batch_size: DataLoader batch size.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent
    accelerator = "auto" if torch.cuda.is_available() else "cpu"

    model = model_cls(device=device_str)
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    module = _build_ptl_module(model, tc)
    trainer = build_trainer(tc, model.model_config, accelerator=accelerator)

    dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
    (metrics,) = trainer.validate(module, datamodule=dm)
    map_val = metrics["val/mAP_50"]
    f1_val = metrics["val/F1"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"
