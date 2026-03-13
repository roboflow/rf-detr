# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end benchmarks for training convergence via the PTL stack.

Smoke test (CPU-friendly):

* :func:`test_train_fast_dev_run` — ``Trainer.fit`` completes without error on a synthetic dataset.

Training convergence (GPU, synthetic dataset, no pretrained weights):

* :func:`test_train_convergence_native_ptl` — ``RFDETRModule`` + ``Trainer.fit`` reaches ≥ 35 % mAP@50.
* :func:`test_train_convergence_rfdetr_api` — ``RFDETR.train()`` reaches ≥ 35 % mAP@50.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule

from rfdetr import RFDETRNano
from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.training import RFDETRDataModule, RFDETRModule, build_trainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ptl_module_from(rfdetr_obj: RFDETR, dataset_dir: Path, output_dir: Path) -> RFDETRModule:
    """Build an :class:`~rfdetr.training.RFDETRModule` from an RFDETR instance.

    Creates the module with the same architecture as *rfdetr_obj*, copies its
    current weights, and asserts PTL lineage before returning.

    Args:
        rfdetr_obj: A (possibly trained) :class:`~rfdetr.detr.RFDETR` instance.
        dataset_dir: Dataset directory forwarded to :class:`~rfdetr.config.TrainConfig`.
        output_dir: Output directory forwarded to :class:`~rfdetr.config.TrainConfig`.

    Returns:
        Weight-synced :class:`~rfdetr.training.RFDETRModule` in eval mode.
    """
    train_config = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
    )
    model_config = rfdetr_obj.model_config.model_copy(update={"pretrain_weights": None})
    module = RFDETRModule(model_config, train_config)
    module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    module.model.eval()

    assert isinstance(module, RFDETRModule), f"Expected RFDETRModule, got {type(module).__name__}"
    assert isinstance(module, LightningModule), "Module must be a pytorch_lightning.LightningModule"
    return module


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
# Training convergence (GPU, synthetic dataset)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_train_convergence_native_ptl(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """Native PTL stack converges: ``RFDETRModule`` + ``RFDETRDataModule`` + ``Trainer.fit``.

    Uses ``Trainer.validate`` before and after ``Trainer.fit`` so only Lightning
    elements are exercised — no ``engine.evaluate`` or legacy paths.

    Assertions:
        - ``val/mAP_50`` before training ≤ 5 %.
        - ``val/mAP_50`` after 10 epochs ≥ 35 %.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    with open(dataset_dir / "train" / "_annotations.coco.json") as f:
        num_classes = len(json.load(f)["categories"])

    accelerator = "auto" if torch.cuda.is_available() else "cpu"

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
    assert map_after >= 0.35, f"val mAP {map_after:.3f} should reach at least 0.35 after Trainer.fit."


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_train_convergence_rfdetr_api(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """``RFDETR.train()`` entry-point converges on synthetic data.

    Exercises the public ``model.train()`` API end-to-end.  Pre- and
    post-training mAP are measured via ``Trainer.validate`` so the assertion
    is identical to :func:`test_train_convergence_native_ptl`.

    Assertions:
        - ``val/mAP_50`` before training ≤ 5 %.
        - ``val/mAP_50`` after 10 epochs ≥ 35 %.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    with open(dataset_dir / "train" / "_annotations.coco.json") as f:
        num_classes = len(json.load(f)["categories"])

    accelerator = "auto" if torch.cuda.is_available() else "cpu"
    device = None if torch.cuda.is_available() else "cpu"

    model = RFDETRNano(num_classes=num_classes, pretrain_weights=None, amp=False)
    # Use the model's own config so RFDETRDataModule uses the correct resolution.
    # RFDETRNano (patch_size=16, num_windows=2) requires block_size=32 divisibility;
    # its resolution=384 satisfies this, while RFDETRBaseConfig resolution=560 does not.
    mc = model.model_config
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

    datamodule = RFDETRDataModule(mc, tc)

    # Pre-training baseline via a temporary PTL module.
    pre_module = _make_ptl_module_from(model, dataset_dir, output_dir)
    pre_trainer = build_trainer(tc, mc, accelerator=accelerator)
    pre_results = pre_trainer.validate(pre_module, datamodule=datamodule)
    map_before = pre_results[0]["val/mAP_50"]
    assert map_before <= 0.05, f"Untrained val mAP {map_before:.3f} should be ≤ 5 %."

    # Train via the public RFDETR.train() API.
    train_kwargs = dict(
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
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)

    # Post-training: copy trained weights into a fresh module and validate.
    post_module = _make_ptl_module_from(model, dataset_dir, output_dir)
    post_trainer = build_trainer(tc, mc, accelerator=accelerator)
    post_results = post_trainer.validate(post_module, datamodule=datamodule)
    map_after = post_results[0]["val/mAP_50"]
    assert map_after >= 0.35, f"val mAP {map_after:.3f} should reach at least 0.35 after RFDETR.train()."
