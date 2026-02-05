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
    synthetic_shape_dataset_dir: Path,
) -> None:
    torch.manual_seed(7)
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
        batch_size=2,
        grad_accum_steps=1,
        num_workers=0,
        device=str(device),
        amp=False,
        use_ema=True,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        square_resize_div_64=True,
        print_freq=20,
        epochs=10,
    )
    train_config = {
        **vars(args),
        "lr": 1e-4,
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
        base_stats_train, _ = evaluate(model.model.model, criterion, postprocess, train_loader, train_ds, device, args=args)
    base_map50 = base_stats_val["coco_eval_bbox"][1]
    base_loss = base_stats_train["loss"]

    assert torch.isfinite(torch.tensor(base_loss)), f"Base loss {base_loss:.3f} must be finite."
    assert torch.isfinite(torch.tensor(base_map50)), f"Base mAP50 {base_map50:.3f} must be finite."
    assert base_map50 <= 0.05, f"Base mAP50 {base_map50:.3f} should be low before training."

    model.train(**train_config)

    with torch.no_grad():
        model.model.model.eval()
        final_stats_val, _ = evaluate(model.model.model, criterion, postprocess, val_loader, base_ds, device, args=args)
        final_stats_train, _ = evaluate(model.model.model, criterion, postprocess, train_loader, train_ds, device, args=args)
    final_map50 = final_stats_val["coco_eval_bbox"][1]
    final_loss = final_stats_train["loss"]

    diagnostics = {
        "base_map50": float(base_map50),
        "final_map50": float(final_map50),
        "base_train_loss": float(base_loss),
        "final_train_loss": float(final_loss),
    }
    print(f"{diagnostics=}")
    (output_dir / "synthetic_benchmark.json").write_text(json.dumps(diagnostics, indent=2))

    assert torch.isfinite(torch.tensor(final_map50)), f"Final mAP50 {final_map50:.3f} must be finite."
    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss:.3f} must be finite."
    assert final_map50 >= 0.4, f"Final mAP50 {final_map50:.3f} should reach at least 0.5 after training."
    assert final_loss <= base_loss * 0.9, f"Loss {base_loss:.3f} -> {final_loss:.3f} should drop by at least 10%."
