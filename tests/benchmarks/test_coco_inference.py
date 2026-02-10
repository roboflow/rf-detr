# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Optional

import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util import misc as utils

_MODEL_CLASSES = {
    "nano": RFDETRNano,
}


@pytest.mark.gpu
def test_coco_inference_benchmark(
    download_coco_val: tuple[Path, Path],
    model_size: str = "nano",
    threshold_map: float = 0.65,
    threshold_f1: float = 0.65,
    num_samples: Optional[int] = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    model_cls = _MODEL_CLASSES[model_size]
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
        batch_size=4,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
    )
    base_ds = get_coco_api_from_dataset(val_dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    rfdetr.model.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            rfdetr.model.model, criterion, postprocess,
            data_loader, base_ds, torch.device(device), args=args,
        )

    results = stats["results_json"]
    map_val = results["map"]
    f1_val = results["f1_score"]

    print(f"COCO val2017 [{model_size}]: mAP@50={map_val:.4f}, F1={f1_val:.4f}")

    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"
