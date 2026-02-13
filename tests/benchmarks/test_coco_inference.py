# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import importlib.util
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Optional

import pytest
import torch

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.detr import RFDETR
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util import misc as utils

_PLUS_AVAILABLE = importlib.util.find_spec("rfdetr_plus") is not None
if _PLUS_AVAILABLE:
    try:
        from rfdetr import RFDETR2XLarge, RFDETRXLarge

        RFDETRXLarge_PML = partial(RFDETRXLarge, accept_platform_model_license=True)
        RFDETR2XLarge_PML = partial(RFDETR2XLarge, accept_platform_model_license=True)
    except ImportError:
        _PLUS_AVAILABLE = False
        RFDETRXLarge_PML = None
        RFDETR2XLarge_PML = None
else:
    RFDETRXLarge_PML = None
    RFDETR2XLarge_PML = None

_PLUS_SKIP = pytest.mark.skipif(not _PLUS_AVAILABLE, reason="requires rfdetr_plus models")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples"),
    [
        pytest.param(RFDETRNano, 0.67, 0.67, None, id="nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, id="small"),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, id="medium"),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, id="large"),
        pytest.param(
            RFDETRXLarge_PML, 0.77, 0.74, 500, id="xlarge", marks=_PLUS_SKIP,
        ),
        pytest.param(
            RFDETR2XLarge_PML, 0.78, 0.74, 500, id="2xlarge", marks=_PLUS_SKIP,
        ),
    ],
)
def test_coco_inference_benchmark(
    request: pytest.FixtureRequest,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: Optional[int],
) -> None:
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
        batch_size=6,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    rfdetr.model.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            rfdetr.model.model, criterion, postprocess,
            data_loader, base_ds, torch.device(device), args=args,
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
