# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO val2017 inference benchmarks covering both the public ``RFDETR.predict()`` API and the PTL training stack.

API contract tests (return type of ``predict()``) live in ``tests/models/test_predict.py`` and do not require a COCO
download.

Test functions:

- :func:`test_inference_detection_rfdetr_predict` — calls ``RFDETR.predict()`` on COCO val images, scores via
  ``torchmetrics.MeanAveragePrecision``, and asserts mAP@50 and macro-F1 thresholds for detection models.
- :func:`test_inference_segmentation_rfdetr_predict` — same for segmentation models (bbox mAP; masks not required).
- :func:`test_inference_detection_ptl_predict` — exercises the PTL predict loop (50 samples via
  ``trainer.predict()``), then asserts mAP@50 and F1 via ``Trainer.validate`` on the full sample set.
- :func:`test_inference_segmentation_ptl_predict` — same for segmentation models.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image
import pytest
import torch
from pycocotools.coco import COCO
from pytorch_lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from rfdetr import (
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
from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import (
    build_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

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
        class_metrics=True,
        max_detection_thresholds=[1, 10, 500],
        backend="faster_coco_eval",
    )
    f1_local = init_matching_accumulator()

    for start in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[start : start + batch_size]
        images = [PIL.Image.open(images_root / f"{img_id:012d}.jpg").convert("RGB") for img_id in batch_ids]
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
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts mAP@50 and macro-F1 thresholds on COCO val for detection models.

    CPU (nano): uses ``RFDETR.predict()`` directly with PIL images scored via
    ``torchmetrics.MeanAveragePrecision`` and a confidence-threshold sweep.

    GPU (small/medium/large): uses ``Trainer.validate()`` via a PTL DataLoader —
    the same path as :func:`test_inference_detection_ptl_predict` but without the
    preceding ``trainer.predict()`` pass.  This avoids serial PIL-loop overhead for
    large models and keeps the GPU test suite within the CI time budget.

    Args:
        tmp_path: Pytest-provided temporary directory (GPU path only).
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

    if torch.cuda.is_available():
        coco_root = images_root.parent
        tc = _build_train_config(coco_root, tmp_path, batch_size)
        module = _build_ptl_module(model, tc)
        dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
        trainer = build_trainer(tc, model.model_config, accelerator="auto")
        (metrics,) = trainer.validate(module, datamodule=dm)
        map_val = metrics["val/mAP_50"]
        f1_val = metrics["val/F1"]
    else:
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
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """Asserts bbox mAP@50 and macro-F1 thresholds on COCO val for segmentation models.

    Same dual-path structure as :func:`test_inference_detection_rfdetr_predict`:
    CPU (nano) uses ``RFDETR.predict()``, GPU variants use ``Trainer.validate()``.
    Masks are not required; only bbox IoU is used for scoring.

    Args:
        tmp_path: Pytest-provided temporary directory (GPU path only).
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

    if torch.cuda.is_available():
        coco_root = images_root.parent
        tc = _build_train_config(coco_root, tmp_path, batch_size)
        module = _build_ptl_module(model, tc)
        dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
        trainer = build_trainer(tc, model.model_config, accelerator="auto")
        (metrics,) = trainer.validate(module, datamodule=dm)
        map_val = metrics["val/mAP_50"]
        f1_val = metrics["val/F1"]
    else:
        map_val, f1_val = _score_rfdetr_predict(model, images_root, annotations_path, num_samples, batch_size)

    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


# ---------------------------------------------------------------------------
# Inference — trainer.predict() (GPU, COCO val2017)
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
    """``trainer.predict()`` runs through the PTL predict loop for detection models.

    Loads a pretrained detection model, copies weights into a :class:`~rfdetr.training.RFDETRModelModule`, runs
    ``trainer.predict()`` on a small subset (50 samples) to exercise
    :meth:`~rfdetr.training.RFDETRModelModule.predict_step`, then runs ``Trainer.validate`` on the full *num_samples* to
    assert mAP and F1.

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

    # Run trainer.predict() on a small slice — exercises RFDETRModelModule.predict_step.
    predict_dm = _build_datamodule(model.model_config, tc, num_samples=50)
    predictions = trainer.predict(module, dataloaders=predict_dm.val_dataloader())
    assert predictions is not None, "trainer.predict() returned None"
    assert len(predictions) > 0, "trainer.predict() returned empty list"

    # Verify mAP and F1 via Trainer.validate on the full num_samples.
    val_dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
    (metrics,) = trainer.validate(module, datamodule=val_dm)
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
    """``trainer.predict()`` runs through the PTL predict loop for segmentation models.

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

    # Run trainer.predict() on a small slice — exercises RFDETRModelModule.predict_step.
    predict_dm = _build_datamodule(model.model_config, tc, num_samples=50)
    predictions = trainer.predict(module, dataloaders=predict_dm.val_dataloader())
    assert predictions is not None, "trainer.predict() returned None"
    assert len(predictions) > 0, "trainer.predict() returned empty list"

    # Verify mAP and F1 via Trainer.validate on the full num_samples.
    val_dm = _build_datamodule(model.model_config, tc, num_samples=num_samples)
    (metrics,) = trainer.validate(module, datamodule=val_dm)
    map_val = metrics["val/mAP_50"]
    f1_val = metrics["val/F1"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"
