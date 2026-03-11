# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO val2017 inference benchmarks for the PTL training stack.

For every detection and segmentation model variant, this module:

1. Loads pretrained weights via the :class:`~rfdetr.detr.RFDETR` wrapper.
2. Verifies :meth:`~rfdetr.detr.RFDETR.predict` returns valid
   :class:`supervision.Detections` objects.
3. Copies the same weights into a fresh :class:`~rfdetr.training.RFDETRModule`.
4. Evaluates via ``Trainer.validate`` and asserts mAP thresholds.

Test functions:

- :func:`test_inference_detection_rfdetr_predict` — ``RFDETR.predict()``
  returns valid detections for detection models (Nano/Small/Medium/Large).
- :func:`test_inference_segmentation_rfdetr_predict` — ``RFDETR.predict()``
  returns detections with masks for segmentation models (Nano through 2XLarge).
- :func:`test_inference_detection_ptl_predict` — ``trainer.predict()`` exercises
  the PTL predict loop (50 samples) then asserts mAP via ``Trainer.validate``.
- :func:`test_inference_segmentation_ptl_predict` — same for segmentation models.
"""

import os
from pathlib import Path
from typing import Optional

import pytest
import supervision as sv
import torch
from pytorch_lightning import LightningModule

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
from rfdetr.training import RFDETRDataModule, RFDETRModule, build_trainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
        num_workers=os.cpu_count() or 1,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        use_ema=False,
        run_test=False,
    )


def _build_datamodule(
    model_config: ModelConfig,
    train_config: TrainConfig,
    num_samples: Optional[int] = None,
) -> RFDETRDataModule:
    """Set up an :class:`~rfdetr.training.RFDETRDataModule` for validation.

    Calls ``setup("validate")`` so ``_dataset_val`` is ready.  When
    *num_samples* is set the dataset is wrapped in a
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


def _build_ptl_module(rfdetr_obj: RFDETR, train_config: TrainConfig) -> RFDETRModule:
    """Copy pretrained weights from *rfdetr_obj* into a fresh :class:`~rfdetr.training.RFDETRModule`.

    Constructs the module with the same architecture (no pretrain download),
    loads weights from ``rfdetr_obj.model.model``, and asserts PTL lineage and
    weight-copy correctness before returning.

    Args:
        rfdetr_obj: A pretrained :class:`~rfdetr.detr.RFDETR` instance.
        train_config: Shared :class:`~rfdetr.config.TrainConfig` (must have a
            valid ``output_dir``).

    Returns:
        Weight-synced :class:`~rfdetr.training.RFDETRModule` ready for
        ``Trainer.validate`` or ``Trainer.predict``.
    """
    module = RFDETRModule(rfdetr_obj.model_config, train_config)
    module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    module.model.eval()

    assert isinstance(module, RFDETRModule), f"Expected RFDETRModule, got {type(module).__name__}"
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
# Inference — RFDETR.predict() (GPU, COCO val2017)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.67, 2000, 6, id="det-nano"),
        pytest.param(RFDETRSmall, 0.72, 500, 6, id="det-small"),
        pytest.param(RFDETRMedium, 0.73, 500, 4, id="det-medium"),
        pytest.param(RFDETRLarge, 0.74, 500, 2, id="det-large"),
    ],
)
def test_inference_detection_rfdetr_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """``RFDETR.predict()`` returns valid ``sv.Detections`` for detection models.

    Loads a pretrained detection model, runs ``predict()`` on a sample of COCO
    val images, and asserts:

    - Return type is a list of :class:`supervision.Detections`.
    - ``Trainer.validate`` on the same weights meets the mAP threshold.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Detection model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` required.
        num_samples: Number of val images used for ``Trainer.validate``.
        batch_size: DataLoader batch size for ``Trainer.validate``.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent

    rfdetr = model_cls(device=device_str)

    # Verify RFDETR.predict() API returns sv.Detections.
    # predict() accepts str paths or PIL Images, not pathlib.Path objects.
    sample_images = [str(p) for p in sorted(images_root.glob("*.jpg"))[:4]]
    assert sample_images, "No COCO val images found."
    detections = rfdetr.predict(sample_images, threshold=0.3)
    assert isinstance(detections, list), "predict() should return a list for multiple images"
    assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"

    # Verify mAP via Trainer.validate on the same pretrained weights.
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)
    module = _build_ptl_module(rfdetr, tc)
    accelerator = "auto" if torch.cuda.is_available() else "cpu"
    trainer = build_trainer(tc, rfdetr.model_config, accelerator=accelerator)
    results = trainer.validate(module, datamodule=dm)
    map_val = results[0]["val/mAP_50"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 500, 6, id="seg-nano"),
        pytest.param(RFDETRSegSmall, 0.66, 100, 6, id="seg-small"),
        pytest.param(RFDETRSegMedium, 0.68, 100, 4, id="seg-medium"),
        pytest.param(RFDETRSegLarge, 0.70, 100, 2, id="seg-large"),
        pytest.param(RFDETRSegXLarge, 0.72, 100, 2, id="seg-xlarge"),
        pytest.param(RFDETRSeg2XLarge, 0.73, 100, 2, id="seg-2xlarge"),
    ],
)
def test_inference_segmentation_rfdetr_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """``RFDETR.predict()`` returns valid ``sv.Detections`` with masks for segmentation models.

    Same structure as :func:`test_inference_detection_rfdetr_predict` but for
    segmentation variants.  Also asserts that the returned detections contain a
    non-``None`` ``mask`` field.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Segmentation model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` (bbox) required.
        num_samples: Number of val images used for ``Trainer.validate``.
        batch_size: DataLoader batch size for ``Trainer.validate``.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent

    rfdetr = model_cls(device=device_str)

    # Verify RFDETR.predict() returns sv.Detections with masks.
    # predict() accepts str paths or PIL Images, not pathlib.Path objects.
    sample_images = [str(p) for p in sorted(images_root.glob("*.jpg"))[:4]]
    assert sample_images, "No COCO val images found."
    detections = rfdetr.predict(sample_images, threshold=0.3)
    assert isinstance(detections, list), "predict() should return a list for multiple images"
    assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"
    assert any(d.mask is not None for d in detections if len(d) > 0), (
        "Segmentation model predict() should return detections with masks"
    )

    # Verify mAP via Trainer.validate on the same pretrained weights.
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)
    module = _build_ptl_module(rfdetr, tc)
    accelerator = "auto" if torch.cuda.is_available() else "cpu"
    trainer = build_trainer(tc, rfdetr.model_config, accelerator=accelerator)
    results = trainer.validate(module, datamodule=dm)
    map_val = results[0]["val/mAP_50"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"


# ---------------------------------------------------------------------------
# Inference — trainer.predict() (GPU, COCO val2017)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.67, 2000, 6, id="det-nano"),
        pytest.param(RFDETRSmall, 0.72, 500, 6, id="det-small"),
        pytest.param(RFDETRMedium, 0.73, 500, 4, id="det-medium"),
        pytest.param(RFDETRLarge, 0.74, 500, 2, id="det-large"),
    ],
)
def test_inference_detection_ptl_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """``trainer.predict()`` runs through the PTL predict loop for detection models.

    Loads a pretrained detection model, copies weights into a
    :class:`~rfdetr.training.RFDETRModule`, runs ``trainer.predict()`` on a
    small subset (50 samples) to exercise :meth:`~rfdetr.training.RFDETRModule.predict_step`,
    then runs ``Trainer.validate`` on the full *num_samples* to assert mAP.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Detection model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` required.
        num_samples: Number of val samples used for ``Trainer.validate``.
        batch_size: DataLoader batch size.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent
    accelerator = "auto" if torch.cuda.is_available() else "cpu"

    rfdetr = model_cls(device=device_str)
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    module = _build_ptl_module(rfdetr, tc)
    trainer = build_trainer(tc, rfdetr.model_config, accelerator=accelerator)

    # Run trainer.predict() on a small slice — exercises RFDETRModule.predict_step.
    predict_dm = _build_datamodule(rfdetr.model_config, tc, num_samples=50)
    predictions = trainer.predict(module, dataloaders=predict_dm.val_dataloader())
    assert predictions is not None, "trainer.predict() returned None"
    assert len(predictions) > 0, "trainer.predict() returned empty list"

    # Verify mAP via Trainer.validate on the full num_samples.
    val_dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)
    results = trainer.validate(module, datamodule=val_dm)
    map_val = results[0]["val/mAP_50"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 500, 6, id="seg-nano"),
        pytest.param(RFDETRSegSmall, 0.66, 100, 6, id="seg-small"),
        pytest.param(RFDETRSegMedium, 0.68, 100, 4, id="seg-medium"),
        pytest.param(RFDETRSegLarge, 0.70, 100, 2, id="seg-large"),
        pytest.param(RFDETRSegXLarge, 0.72, 100, 2, id="seg-xlarge"),
        pytest.param(RFDETRSeg2XLarge, 0.73, 100, 2, id="seg-2xlarge"),
    ],
)
def test_inference_segmentation_ptl_predict(
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    num_samples: int,
    batch_size: int,
) -> None:
    """``trainer.predict()`` runs through the PTL predict loop for segmentation models.

    Same structure as :func:`test_inference_detection_ptl_predict` but for
    segmentation variants.

    Args:
        tmp_path: Pytest-provided temporary directory.
        download_coco_val: Fixture providing ``(images_root, annotations_path)``.
        model_cls: Segmentation model class to instantiate with pretrained weights.
        threshold_map: Minimum ``val/mAP_50`` (bbox) required.
        num_samples: Number of val samples used for ``Trainer.validate``.
        batch_size: DataLoader batch size.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, _ = download_coco_val
    coco_root = images_root.parent
    accelerator = "auto" if torch.cuda.is_available() else "cpu"

    rfdetr = model_cls(device=device_str)
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    module = _build_ptl_module(rfdetr, tc)
    trainer = build_trainer(tc, rfdetr.model_config, accelerator=accelerator)

    # Run trainer.predict() on a small slice — exercises RFDETRModule.predict_step.
    predict_dm = _build_datamodule(rfdetr.model_config, tc, num_samples=50)
    predictions = trainer.predict(module, dataloaders=predict_dm.val_dataloader())
    assert predictions is not None, "trainer.predict() returned None"
    assert len(predictions) > 0, "trainer.predict() returned empty list"

    # Verify mAP via Trainer.validate on the full num_samples.
    val_dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)
    results = trainer.validate(module, datamodule=val_dm)
    map_val = results[0]["val/mAP_50"]
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
