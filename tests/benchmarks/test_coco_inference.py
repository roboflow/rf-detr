# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch
from torchvision.datasets import CocoDetection

from rfdetr.config import RFDETRNanoConfig
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms
from rfdetr.engine import evaluate
from rfdetr.main import Model
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util import misc as utils


class CocoDetectionWithTargets(CocoDetection):
    def __init__(self, img_folder: Path, ann_file: Path, transforms) -> None:
        super().__init__(root=str(img_folder), annFile=str(ann_file))
        self._transforms = transforms
        self.prepare = ConvertCoco(include_masks=False)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def _project_coco_paths(project_root: Path) -> tuple[Path, Path, Path]:
    images_root = project_root / "val2017"
    annotations_path = project_root / "annotations" / "instances_val2017.json"
    weights_path = project_root / "rf-detr-nano.pth"
    return images_root, annotations_path, weights_path


def _require_paths(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        pytest.skip(f"COCO assets missing: {missing_list}")


def test_coco_inference_benchmark() -> None:
    project_root = Path(__file__).resolve().parents[2]
    images_root, annotations_path, weights_path = _project_coco_paths(project_root)
    _require_paths(images_root, annotations_path, weights_path)

    config = RFDETRNanoConfig(pretrain_weights=str(weights_path), device="cpu")
    model_wrapper = Model(**config.dict())
    args = model_wrapper.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500

    transforms = make_coco_transforms(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetectionWithTargets(images_root, annotations_path, transforms=transforms)
    print(f"Loaded {len(val_dataset)} images for evaluation.")

    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)

    criterion, postprocess = build_criterion_and_postprocessors(args)
    device = torch.device("cpu")
    model_wrapper.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(model_wrapper.model, criterion, postprocess, data_loader, base_ds, device, args=args)

    assert "results_json" in stats, "COCO evaluation did not return JSON metrics."
