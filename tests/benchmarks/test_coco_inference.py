# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch

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
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.detr import RFDETR
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.misc import collate_fn


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.67, 0.66, None, 6, id="nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, 6, id="small"),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, 4, id="medium"),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, 2, id="large"),
    ],
)
def test_coco_detection_inference_benchmark(
    request: pytest.FixtureRequest,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """
    Benchmark test for object detection model inference on COCO validation set.

    This test validates that pretrained detection models maintain their expected
    performance levels on the COCO val2017 dataset. It ensures that:
    1. Models load correctly with pretrained weights
    2. Inference produces valid predictions
    3. Performance metrics (mAP@50 and F1 score) meet minimum thresholds

    The performance thresholds (mAP@50 and F1 score) were established by running
    inference on the complete COCO val2017 dataset with each model variant. These
    thresholds represent the expected baseline performance and help detect regressions
    in model quality or inference pipeline changes.

    Note: To reduce test time, some model variants use a subset of the validation
    set (500 samples). The nano model runs on the full dataset as a comprehensive check.
    Batch sizes are adjusted per model size to avoid GPU OOM: large models use batch_size=2,
    medium models use batch_size=4, and small models use batch_size=6.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    rfdetr = model_cls(device=device)
    config = rfdetr.model_config
    args = rfdetr.model.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500

    transforms = make_coco_transforms_square_div_64(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetection(images_root, annotations_path, transforms=transforms)
    if num_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(num_samples, len(val_dataset)))))
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    rfdetr.model.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            rfdetr.model.model,
            criterion,
            postprocess,
            data_loader,
            base_ds,
            torch.device(device),
            args=args,
        )

    # Dump results JSON for debugging
    # Use env var COCO_BENCHMARK_DEBUG_DIR to specify a permanent folder, otherwise use temp
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    debug_path = Path(debug_dir) / f"coco_inference_stats_detection_{test_id}_nb-spl-{num_samples or 'all'}.json"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dumped stats to {debug_path}")

    results = stats["results_json"]
    map_val = results["map"]
    f1_val = results["f1_score"]

    print(f"COCO val2017 [{test_id}]: mAP@50={map_val:.4f}, F1={f1_val:.4f}")
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


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
def test_coco_segmentation_inference_benchmark(
    request: pytest.FixtureRequest,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_segm_map: float,
    threshold_segm_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """
    Benchmark test for instance segmentation model inference on COCO validation set.

    This test validates that pretrained segmentation models maintain their expected
    performance levels on the COCO val2017 dataset. It ensures that:
    1. Segmentation models load correctly with pretrained weights
    2. Inference produces valid predictions for both bounding boxes and masks
    3. Segmentation performance metrics (mask mAP@50 and F1 score) meet minimum thresholds

    The performance thresholds (mask mAP@50 and F1 score) were established by running
    inference on the complete COCO val2017 dataset with each model variant. These
    thresholds represent the expected baseline segmentation performance and help detect
    regressions in model quality or the segmentation inference pipeline.

    Note: To reduce test time, model variants use subsets of the validation set
    (100-500 samples depending on model size). The nano model uses 500 samples
    for a more comprehensive check. Batch sizes are adjusted per model size to avoid
    GPU OOM: large models use batch_size=2, medium models use batch_size=4, and small
    models use batch_size=6.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    rfdetr = model_cls(device=device)
    config = rfdetr.model_config
    args = rfdetr.model.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500

    # Add segmentation-specific args if missing
    if not hasattr(args, "mask_ce_loss_coef"):
        args.mask_ce_loss_coef = 5.0
    if not hasattr(args, "mask_dice_loss_coef"):
        args.mask_dice_loss_coef = 5.0
    if not hasattr(args, "mask_point_sample_ratio"):
        args.mask_point_sample_ratio = 16

    transforms = make_coco_transforms_square_div_64(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    # Enable mask loading for segmentation models
    val_dataset = CocoDetection(
        images_root,
        annotations_path,
        transforms=transforms,
        include_masks=True,
    )
    if num_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(num_samples, len(val_dataset)))))
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    rfdetr.model.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            rfdetr.model.model,
            criterion,
            postprocess,
            data_loader,
            base_ds,
            torch.device(device),
            args=args,
        )

    # Dump results JSON for debugging
    # Use env var COCO_BENCHMARK_DEBUG_DIR to specify a permanent folder, otherwise use temp
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    debug_path = Path(debug_dir) / f"coco_inference_stats_segmentation_{test_id}_nb-spl-{num_samples or 'all'}.json"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dumped stats to {debug_path}")

    # Check bbox results
    results_bbox = stats["results_json"]
    bbox_map_val = results_bbox["map"]
    bbox_f1_val = results_bbox["f1_score"]

    # Check segmentation results
    results_segm = stats["results_json_masks"]
    segm_map_val = results_segm["map"]
    segm_f1_val = results_segm["f1_score"]

    print(f"COCO val2017 Segmentation [{test_id}]:")
    print(f"  BBox mAP@50={bbox_map_val:.4f}, F1={bbox_f1_val:.4f}")
    print(f"  Segm mAP@50={segm_map_val:.4f}, F1={segm_f1_val:.4f}")

    # Assert segmentation metrics
    assert segm_map_val >= threshold_segm_map, f"Segm mAP@50 {segm_map_val:.4f} < {threshold_segm_map}"
    assert segm_f1_val >= threshold_segm_f1, f"Segm F1 {segm_f1_val:.4f} < {threshold_segm_f1}"
