# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PTL-native convergence tests: RFDETR.train() + compat.evaluate throughout.

Two GPU convergence benchmarks, one per training entry-point:

* :func:`test_ptl_native_convergence` — exercises the native PTL stack
  (``RFDETRModule`` + ``RFDETRDataModule`` + ``Trainer.fit``).  Uses
  ``Trainer.validate`` before and after training so only Lightning elements
  appear in the test.

* :func:`test_ptl_training_improves_performance` — exercises
  :meth:`~rfdetr.detr.RFDETR.train`.  Mirrors
  :func:`~tests.benchmarks.test_synthetic_convergence.test_synthetic_training_improves_performance`
  exactly: same args, same dataloader construction, same assertions — only
  the training call and the evaluation wrapper differ.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule

from rfdetr import RFDETRNano
from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.lit import RFDETRDataModule, RFDETRModule, build_trainer
from rfdetr.lit.compat import evaluate as compat_evaluate
from rfdetr.main import populate_args
from rfdetr.util import misc as utils

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ptl_module_from(rfdetr_obj, dataset_dir: Path, output_dir: Path) -> RFDETRModule:
    """Build a temporary RFDETRModule from a trained/untrained RFDETR instance.

    Creates an :class:`~rfdetr.lit.module.RFDETRModule` with the same
    architecture as *rfdetr_obj*, then copies its current weights (from
    ``rfdetr_obj.model.model``).  No pretrain weights are downloaded; the
    module's weights are entirely derived from *rfdetr_obj*.

    Args:
        rfdetr_obj: A (possibly trained) :class:`~rfdetr.detr.RFDETR` instance.
        dataset_dir: Dataset directory forwarded to :class:`~rfdetr.config.TrainConfig`.
        output_dir: Output directory forwarded to :class:`~rfdetr.config.TrainConfig`.

    Returns:
        A weight-synced :class:`~rfdetr.lit.module.RFDETRModule` ready for
        ``compat.evaluate``.
    """
    train_config = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
    )
    model_config = rfdetr_obj.model_config.model_copy(
        update={
            # Disable pretrain loading so this helper has no network/disk side effects.
            "pretrain_weights": None,
        },
    )
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
    Lightning elements are exercised — no ``compat_evaluate`` or
    ``engine.evaluate`` calls.

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


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_ptl_training_improves_performance(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """RFDETR.train() converges — mirrors test_synthetic_training_improves_performance.

    Structure is intentionally identical to
    :func:`~tests.benchmarks.test_synthetic_convergence.test_synthetic_training_improves_performance`:
    same args, same dataloader construction, same assertions.  The only
    differences are:

    * :meth:`~rfdetr.detr.RFDETR.train` (PTL stack) replaces the legacy ``model.train()``.
    * :func:`~rfdetr.lit.compat.evaluate` replaces ``engine.evaluate``.

    Same thresholds (mAP ≥ 35 %, F1 ≥ 35 %, loss drop to 70 %) apply.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFDETRNano(pretrain_weights=None, num_classes=4, device=str(device))

    args = populate_args(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        class_names=["square", "triangle", "circle"],
        batch_size=4,
        grad_accum_steps=1,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
        device=str(device),
        amp=False,
        use_ema=True,
        square_resize_div_64=True,
        epochs=10,
    )
    if not hasattr(args, "segmentation_head"):
        args.segmentation_head = False
    if not hasattr(args, "fp16_eval"):
        args.fp16_eval = False
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500

    train_dataset = build_dataset(image_set="train", args=args, resolution=args.resolution)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler=torch.utils.data.SequentialSampler(train_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    train_ds = get_coco_api_from_dataset(train_dataset)

    val_dataset = build_dataset(image_set="val", args=args, resolution=args.resolution)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        args.batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)

    # Pre-training baseline via PTL compat path.
    ptl_pre = _make_ptl_module_from(model, dataset_dir, output_dir)
    with torch.no_grad():
        base_stats_val, _ = compat_evaluate(ptl_pre, val_loader, base_ds, device)
        base_stats_train, _ = compat_evaluate(ptl_pre, train_loader, train_ds, device)
    del ptl_pre

    Path(output_dir / "base_stats_val.json").write_text(json.dumps(base_stats_val, indent=2))
    Path(output_dir / "base_stats_train.json").write_text(json.dumps(base_stats_train, indent=2))
    base_map = base_stats_val["results_json"]["map"]
    base_f1_score = base_stats_val["results_json"]["f1_score"]
    base_loss_bbox = base_stats_train["loss_bbox"]
    base_loss_giou = base_stats_train["loss_giou"]

    assert math.isfinite(base_loss_bbox), f"Base loss {base_loss_bbox:.3f} must be finite."
    assert math.isfinite(base_loss_giou), f"Base loss {base_loss_giou:.3f} must be finite."
    assert math.isfinite(base_map), f"Base mAP50 {base_map:.3f} must be finite."
    assert math.isfinite(base_f1_score), f"Base F1 {base_f1_score:.3f} must be finite."
    assert base_map <= 0.05, f"Base mAP50 {base_map:.3f} should be low before training."
    assert base_f1_score <= 0.05, f"Base F1 {base_f1_score:.3f} should be low before training."

    # Train via the PyTorch Lightning stack.
    model.train(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=10,
        batch_size=4,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
        lr=1e-3,
        warmup_epochs=1.0,
        use_ema=True,
        multi_scale=False,
        run_test=False,
        device=str(device),
    )

    # Post-training evaluation via PTL compat path.
    ptl_post = _make_ptl_module_from(model, dataset_dir, output_dir)
    with torch.no_grad():
        final_stats_val, _ = compat_evaluate(ptl_post, val_loader, base_ds, device)
        final_stats_train, _ = compat_evaluate(ptl_post, train_loader, train_ds, device)

    Path(output_dir / "final_stats_val.json").write_text(json.dumps(final_stats_val, indent=2))
    Path(output_dir / "final_stats_train.json").write_text(json.dumps(final_stats_train, indent=2))
    final_map = final_stats_val["results_json"]["map"]
    final_f1_score = final_stats_val["results_json"]["f1_score"]
    final_loss_bbox = final_stats_train["loss_bbox"]
    final_loss_giou = final_stats_train["loss_giou"]

    threshold_map = 0.35
    threshold_f1_score = 0.35
    threshold_loss = 0.7
    assert math.isfinite(final_loss_bbox), f"Final loss {final_loss_bbox:.3f} must be finite."
    assert math.isfinite(final_loss_giou), f"Final loss {final_loss_giou:.3f} must be finite."
    assert math.isfinite(final_map), f"Final mAP {final_map:.3f} must be finite."
    assert math.isfinite(final_f1_score), f"Final F1 {final_f1_score:.3f} must be finite."
    assert final_map >= threshold_map, (
        f"Final mAP {final_map:.3f} should reach at least {threshold_map} after PTL training."
    )
    assert final_f1_score >= threshold_f1_score, (
        f"Final F1 {final_f1_score:.3f} should reach at least {threshold_f1_score} after PTL training."
    )
    assert final_loss_bbox <= base_loss_bbox * threshold_loss, (
        f"Loss {base_loss_bbox:.3f} -> {final_loss_bbox:.3f} should drop to at least {threshold_loss * 100}%."
    )
    assert final_loss_giou <= base_loss_giou * threshold_loss, (
        f"Loss {base_loss_giou:.3f} -> {final_loss_giou:.3f} should drop to at least {threshold_loss * 100}%."
    )
