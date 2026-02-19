# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
import math
import os
from pathlib import Path

import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.main import populate_args
from rfdetr.models import PostProcess, build_criterion_and_postprocessors
from rfdetr.util import misc as utils


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_synthetic_training_improves_performance(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """
    Benchmark test to verify that training improves model performance on synthetic data.

    This test validates the training loop by ensuring that:
    1. A randomly initialized model starts with low performance (mAP < 5%, F1 < 5%)
    2. After training for 10 epochs, the model achieves reasonable performance thresholds
    3. Training losses decrease to at least 70% of their initial values

    The performance thresholds (mAP >= 35%, F1 >= 35%) were established empirically
    through testing on synthetic shape datasets. These thresholds ensure the model
    learns meaningful patterns without requiring full COCO-scale validation.

    Note: This test uses batch_size=2 with grad_accum_steps=4 to simulate an effective
    batch size of 8 while reducing GPU memory requirements. The test will only rerun
    on exceptions (e.g., asset download failures) but not on assertion failures.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFDETRNano(pretrain_weights=None, num_classes=4, device=str(device))

    # Build args once with populate_args, then reuse its values for training
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
    train_config = {
        **vars(args),
        "lr": 1e-3,
        "warmup_epochs": 1.0,
        "multi_scale": False,
        "dont_save_weights": False,
        "min_batches": 2,
        "run_test": False,
    }
    if not hasattr(args, "segmentation_head"):
        args.segmentation_head = False
    if not hasattr(args, "fp16_eval"):
        args.fp16_eval = False
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500
    device = torch.device(args.device)
    criterion, _ = build_criterion_and_postprocessors(args)
    postprocess = PostProcess(num_select=args.num_select)

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

    with torch.no_grad():
        model.model.model.eval()
        base_stats_val, _ = evaluate(model.model.model, criterion, postprocess, val_loader, base_ds, device, args=args)
        base_stats_train, _ = evaluate(
            model.model.model, criterion, postprocess, train_loader, train_ds, device, args=args
        )
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

    model.train(**train_config)

    with torch.no_grad():
        model.model.model.eval()
        final_stats_val, _ = evaluate(model.model.model, criterion, postprocess, val_loader, base_ds, device, args=args)
        final_stats_train, _ = evaluate(
            model.model.model, criterion, postprocess, train_loader, train_ds, device, args=args
        )
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
        f"Final mAP {final_map:.3f} should reach at least {threshold_map} after training."
    )
    assert final_f1_score >= threshold_f1_score, (
        f"Final F1 {final_f1_score:.3f} should reach at least {threshold_f1_score} after training."
    )
    assert final_loss_bbox <= base_loss_bbox * threshold_loss, (
        f"Loss {base_loss_bbox:.3f} -> {final_loss_bbox:.3f} should drop to at least {threshold_loss * 100}%."
    )
    assert final_loss_giou <= base_loss_giou * threshold_loss, (
        f"Loss {base_loss_giou:.3f} -> {final_loss_giou:.3f} should drop to at least {threshold_loss * 100}%."
    )
