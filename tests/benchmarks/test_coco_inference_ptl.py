# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PTL inference benchmarks: RFDETRModule evaluation via native Trainer.validate.

For every detection and segmentation model variant, this module:

1. Loads pretrained weights via the ``RFDETR`` wrapper.
2. Copies the same weights into a fresh :class:`~rfdetr.training.module.RFDETRModule`.
3. Asserts the module is a genuine ``pytorch_lightning.LightningModule``.
4. Evaluates via ``Trainer.validate`` (native PTL path).
5. Asserts COCO baseline thresholds.
"""

import os
from pathlib import Path
from typing import Optional

import pytest
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
from rfdetr.training import RFDETRDataModule, build_trainer
from rfdetr.training.module import RFDETRModule

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_train_config(coco_root: Path, tmp_path: Path, batch_size: int) -> TrainConfig:
    """Build a minimal TrainConfig pointing at the COCO val2017 dataset.

    Args:
        coco_root: Directory containing ``val2017/`` and ``annotations/``.
        tmp_path: Temporary directory used as ``output_dir``.
        batch_size: DataLoader batch size.

    Returns:
        Fully-specified :class:`~rfdetr.config.TrainConfig` with optional
        loggers and EMA disabled (inference-only benchmarks have no use for them).
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
    """Build and set up an :class:`~rfdetr.training.datamodule.RFDETRDataModule`.

    Calls ``setup("validate")`` so ``_dataset_val`` is ready.  If
    *num_samples* is provided, the validation dataset is truncated to a
    :class:`torch.utils.data.Subset` to speed up benchmark runs.

    ``get_coco_api_from_dataset`` unwraps ``Subset`` up to 10 levels, so the
    COCO ground-truth API is still fully functional after truncation.

    Args:
        model_config: Architecture configuration (``segmentation_head`` controls
            whether masks are loaded).
        train_config: Training hyperparameter configuration.
        num_samples: If set, limit the validation dataset to this many samples.

    Returns:
        Configured :class:`~rfdetr.training.datamodule.RFDETRDataModule` with
        ``_dataset_val`` populated.
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
    """Copy pretrained weights from *rfdetr_obj* into a fresh RFDETRModule.

    Builds an :class:`~rfdetr.training.module.RFDETRModule` with the same
    architecture as *rfdetr_obj* (no pretrain download), loads the weights
    from ``rfdetr_obj.model.model``, and runs PTL-identity assertions before
    returning.

    Args:
        rfdetr_obj: A pretrained :class:`~rfdetr.detr.RFDETR` instance.
        train_config: Shared :class:`~rfdetr.config.TrainConfig` used as the
            module's training config context (must have a valid ``output_dir``).

    Returns:
        Weight-synced :class:`~rfdetr.training.module.RFDETRModule` ready for
        ``Trainer.validate``.
    """
    ptl_module = RFDETRModule(rfdetr_obj.model_config, train_config)
    ptl_module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    ptl_module.model.eval()

    # Assert PTL lineage — the evaluated object must be a genuine
    # LightningModule, not a raw nn.Module or a duck-typed stand-in.
    assert isinstance(ptl_module, RFDETRModule), f"Expected RFDETRModule, got {type(ptl_module).__name__}"
    assert isinstance(ptl_module, LightningModule), (
        "ptl_module must be a pytorch_lightning.LightningModule — this confirms evaluation runs through the PTL stack"
    )

    # Spot-check that the weight copy succeeded.
    # Compare on CPU so the check is device-agnostic (legacy model may be on CUDA,
    # the freshly-constructed RFDETRModule always starts on CPU).
    _first_key = next(iter(rfdetr_obj.model.model.state_dict()))
    assert torch.equal(
        rfdetr_obj.model.model.state_dict()[_first_key].cpu(),
        ptl_module.model.state_dict()[_first_key].cpu(),
    ), f"Weight copy failed: '{_first_key}' differs between legacy model and PTL module"

    return ptl_module


# ---------------------------------------------------------------------------
# Detection benchmark — native PTL validation
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.67, 0.66, 2000, 6, id="nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, 6, id="small"),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, 4, id="medium"),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, 2, id="large"),
    ],
)
def test_ptl_detection_validation(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """PTL-native Trainer.validate must meet COCO detection acceptance thresholds.

    Exercises the full PTL validation stack (``validation_step`` ->
    ``COCOEvalCallback``) and asserts mAP@50 and F1 acceptance thresholds.
    """
    images_root, _ = download_coco_val
    coco_root = images_root.parent

    rfdetr = model_cls(device="cuda" if torch.cuda.is_available() else "cpu")
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)

    ptl_module = _build_ptl_module(rfdetr, tc)
    trainer = build_trainer(tc, rfdetr.model_config, accelerator="auto", precision="32-true")
    results = trainer.validate(ptl_module, datamodule=dm)
    ptl_map = results[0]["val/mAP_50"]
    ptl_f1 = results[0]["val/f1"]

    test_id = request.node.callspec.id
    print(f"[{test_id}] ptl mAP@50={ptl_map:.4f}  F1={ptl_f1:.4f}")

    assert ptl_map >= threshold_map, f"PTL mAP@50 {ptl_map:.4f} < {threshold_map}"
    assert ptl_f1 >= threshold_f1, f"PTL F1 {ptl_f1:.4f} < {threshold_f1}"


# ---------------------------------------------------------------------------
# Segmentation benchmark — native PTL validation
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_segm_map", "threshold_segm_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 0.64, 500, 6, id="nano"),
        pytest.param(RFDETRSegSmall, 0.66, 0.67, 100, 6, id="small"),
        pytest.param(RFDETRSegMedium, 0.68, 0.68, 100, 4, id="medium"),
        pytest.param(RFDETRSegLarge, 0.70, 0.69, 100, 2, id="large"),
        pytest.param(RFDETRSegXLarge, 0.72, 0.70, 100, 2, id="xlarge"),
        pytest.param(RFDETRSeg2XLarge, 0.73, 0.71, 100, 2, id="2xlarge"),
    ],
)
def test_ptl_segmentation_validation(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_segm_map: float,
    threshold_segm_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """PTL-native Trainer.validate must meet COCO segmentation acceptance thresholds.

    Exercises the full PTL validation stack for segmentation models and asserts
    segm_mAP@50 and segm_F1 acceptance thresholds.
    """
    images_root, _ = download_coco_val
    coco_root = images_root.parent

    rfdetr = model_cls(device="cuda" if torch.cuda.is_available() else "cpu")
    tc = _build_train_config(coco_root, tmp_path, batch_size)
    dm = _build_datamodule(rfdetr.model_config, tc, num_samples=num_samples)

    ptl_module = _build_ptl_module(rfdetr, tc)
    trainer = build_trainer(tc, rfdetr.model_config, accelerator="auto", precision="32-true")
    results = trainer.validate(ptl_module, datamodule=dm)
    ptl_segm_map = results[0]["val/segm_mAP_50"]
    ptl_segm_f1 = results[0]["val/segm_f1"]

    test_id = request.node.callspec.id
    print(f"[{test_id}] ptl segm_mAP@50={ptl_segm_map:.4f}  segm_F1={ptl_segm_f1:.4f}")

    assert ptl_segm_map >= threshold_segm_map, f"PTL segm_mAP@50 {ptl_segm_map:.4f} < {threshold_segm_map}"
    assert ptl_segm_f1 >= threshold_segm_f1, f"PTL segm_F1 {ptl_segm_f1:.4f} < {threshold_segm_f1}"
