# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

_COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

_MODEL_CONFIGS = {
    "nano": RFDETRNanoConfig,
}


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


def _download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / url.rsplit("/", 1)[-1]
    print(f"Downloading {url} ...")
    urlretrieve(url, str(zip_path))
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    zip_path.unlink()


@pytest.fixture(scope="session")
def download_coco_val() -> tuple[Path, Path]:
    """Download COCO val2017 images and annotations if not already present."""
    images_root = DATA_DIR / "val2017"
    annotations_path = DATA_DIR / "annotations" / "instances_val2017.json"

    if not images_root.exists():
        _download_and_extract(_COCO_URLS["val2017"], DATA_DIR)
    if not annotations_path.exists():
        _download_and_extract(_COCO_URLS["annotations"], DATA_DIR)

    return images_root, annotations_path


@pytest.mark.gpu
def test_coco_inference_benchmark(
    download_coco_val: tuple[Path, Path],
    model_size: str = "nano",
    threshold_map: float = 0.30,
    threshold_map50: float = 0.45,
    threshold_f1: float = 0.30,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_root, annotations_path = download_coco_val

    config_cls = _MODEL_CONFIGS[model_size]
    config = config_cls(device=device.type)
    model_wrapper = Model(**config.dict())
    args = model_wrapper.args

    transforms = make_coco_transforms(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetectionWithTargets(images_root, annotations_path, transforms=transforms)
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

    model_wrapper.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            model_wrapper.model, criterion, postprocess,
            data_loader, base_ds, device, args=args,
        )

    results = stats["results_json"]
    map_val = results["map"]
    map50_val = results["map50"]
    f1_val = results["f1_score"]

    assert map_val >= threshold_map, f"mAP {map_val:.4f} < {threshold_map}"
    assert map50_val >= threshold_map50, f"mAP50 {map50_val:.4f} < {threshold_map50}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"

    print(f"COCO val2017 [{model_size}]: mAP={map_val:.4f}, mAP50={map50_val:.4f}, F1={f1_val:.4f}")
