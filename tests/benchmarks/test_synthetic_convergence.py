# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
from pathlib import Path

import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset
from rfdetr.engine import evaluate
from rfdetr.main import populate_args
from rfdetr.models import PostProcess, build_criterion_and_postprocessors
from rfdetr.util import misc as utils


@pytest.mark.slow
def test_synthetic_training_improves_map50(
    tmp_path: Path,
    synthetic_dataset_dir: Path,
) -> None:
    torch.manual_seed(7)
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_dataset_dir

    model = RFDETRNano(pretrain_weights=None, num_classes=4, device="cpu")

    base_args = populate_args(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        num_classes=4,
        class_names=["square", "triangle", "circle"],
        batch_size=2,
        grad_accum_steps=1,
        epochs=1,
        num_workers=0,
        amp=False,
        use_ema=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        print_freq=200,
        device="cpu",
    )

    args = base_args
    if not hasattr(args, "segmentation_head"):
        args.segmentation_head = False
    if not hasattr(args, "fp16_eval"):
        args.fp16_eval = False
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500
    device = torch.device(args.device)
    criterion, _ = build_criterion_and_postprocessors(args)
    postprocess = PostProcess(num_select=args.num_select)

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
        base_stats, _ = evaluate(model.model.model, criterion, postprocess, val_loader, base_ds, device, args=args)
    base_map50 = base_stats["coco_eval_bbox"][1]
    base_val_loss = base_stats["loss"]

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=2,
        batch_size=2,
        grad_accum_steps=1,
        lr=1e-4,
        num_workers=0,
        amp=False,
        use_ema=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        dont_save_weights=False,
        min_batches=2,
        run_test=False,
    )

    with torch.no_grad():
        trained_stats, _ = evaluate(model.model.model, criterion, postprocess, val_loader, base_ds, device, args=args)
    trained_map50 = trained_stats["coco_eval_bbox"][1]
    trained_val_loss = trained_stats["loss"]

    diagnostics = {
        "base_map50": float(base_map50),
        "trained_map50": float(trained_map50),
        "base_val_loss": float(base_val_loss),
        "trained_val_loss": float(trained_val_loss),
    }
    (output_dir / "synthetic_benchmark.json").write_text(json.dumps(diagnostics, indent=2))

    assert trained_map50 > base_map50 + 0.01
    assert trained_val_loss < base_val_loss * 0.98
