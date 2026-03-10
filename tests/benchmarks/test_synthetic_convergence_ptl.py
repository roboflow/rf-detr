# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PTL-native convergence tests.

Two GPU convergence benchmarks, one per training entry-point:

* :func:`test_ptl_native_convergence` — exercises the native PTL stack
  (``RFDETRModule`` + ``RFDETRDataModule`` + ``Trainer.fit``).  Uses
  ``Trainer.validate`` before and after training so only Lightning elements
  appear in the test.

* :func:`test_ptl_training_improves_performance` — exercises
  :meth:`~rfdetr.detr.RFDETR.train` with the PTL stack and validates
  via ``Trainer.validate``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule

from rfdetr import RFDETRNano
from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.lit import RFDETRDataModule, RFDETRModule, build_trainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke test (CPU-friendly, no GPU required)
# ---------------------------------------------------------------------------


def test_train_fast_dev_run(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """Smoke-test the full PTL stack on a real synthetic dataset with fast_dev_run.

    Uses ``build_trainer(tc, mc, fast_dev_run=2)`` and
    ``trainer.fit(module, datamodule=datamodule)`` with a real model and real
    data (no mocking).  Only asserts the pipeline runs without error;
    convergence is tested by the GPU-only tests below.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(synthetic_shape_dataset_dir / "train" / "_annotations.coco.json") as f:
        num_classes = len(json.load(f)["categories"])

    mc = RFDETRBaseConfig(num_classes=num_classes, pretrain_weights=None)
    tc = TrainConfig(
        dataset_dir=str(synthetic_shape_dataset_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        num_workers=0,
        use_ema=False,
        run_test=False,
        tensorboard=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        drop_path=0.0,
        grad_accum_steps=1,
    )

    module = RFDETRModule(mc, tc)
    datamodule = RFDETRDataModule(mc, tc)
    trainer = build_trainer(tc, mc, accelerator="auto", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)


# ---------------------------------------------------------------------------
# GPU convergence tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_ptl_native_convergence(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """Native PTL stack converges: RFDETRModule + RFDETRDataModule + Trainer.fit.

    Uses ``Trainer.validate`` before and after ``Trainer.fit`` so that only
    Lightning elements are exercised.

    Assertions:
        - ``val/mAP_50`` before training ≤ 5 %.
        - ``val/mAP_50`` after 10 epochs ≥ 35 %.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    with open(dataset_dir / "train" / "_annotations.coco.json") as f:
        num_classes = len(json.load(f)["categories"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = "cpu" if device == "cpu" else "auto"

    mc = RFDETRBaseConfig(num_classes=num_classes, pretrain_weights=None, amp=False)
    tc = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=10,
        batch_size=4,
        grad_accum_steps=1,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
        lr=1e-3,
        warmup_epochs=1.0,
        use_ema=True,
        multi_scale=False,
        run_test=False,
        tensorboard=False,
    )

    module = RFDETRModule(mc, tc)
    datamodule = RFDETRDataModule(mc, tc)

    # Pre-training baseline — untrained model should have near-zero mAP.
    pre_trainer = build_trainer(tc, mc, accelerator=accelerator)
    pre_results = pre_trainer.validate(module, datamodule=datamodule)
    map_before = pre_results[0]["val/mAP_50"]
    assert map_before <= 0.05, f"Untrained val mAP {map_before:.3f} should be ≤ 5 %."

    # Train via native PTL Trainer.fit.
    trainer = build_trainer(tc, mc, accelerator=accelerator)
    trainer.fit(module, datamodule=datamodule)

    # Post-training validation — model should have converged.
    post_results = trainer.validate(module, datamodule=datamodule)
    map_after = post_results[0]["val/mAP_50"]
    assert map_after >= 0.35, f"val mAP {map_after:.3f} should reach at least 0.35 after PTL training."


