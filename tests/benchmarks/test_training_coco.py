# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO benchmark coverage for short keypoint-preview training on a deterministic subset."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset

from rfdetr import RFDETRKeypointPreview
from rfdetr.config import TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.utilities.reproducibility import seed_all


def _to_float(value: float | torch.Tensor) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def _build_subset_annotations(
    payload: dict,
    image_ids: list[int],
) -> dict:
    image_id_set = set(image_ids)
    images = [image for image in payload["images"] if int(image["id"]) in image_id_set]
    annotations = [
        annotation
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in image_id_set
        and int(annotation.get("iscrowd", 0)) == 0
        and int(annotation.get("num_keypoints", 0)) > 0
    ]
    categories = [category for category in payload["categories"] if int(category["id"]) == 1]
    return {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _build_coco_keypoint_subset_from_val(
    *,
    images_root: Path,
    annotations_path: Path,
    output_root: Path,
    train_images: int,
    val_images: int,
) -> Path:
    with annotations_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    person_image_ids = sorted(
        {
            int(annotation["image_id"])
            for annotation in payload["annotations"]
            if int(annotation.get("iscrowd", 0)) == 0 and int(annotation.get("num_keypoints", 0)) > 0
        }
    )
    required = train_images + val_images
    if len(person_image_ids) < required:
        raise RuntimeError(f"Need at least {required} keypoint images, found {len(person_image_ids)}.")

    train_ids = person_image_ids[:train_images]
    val_ids = person_image_ids[train_images : train_images + val_images]
    image_by_id = {int(image["id"]): image for image in payload["images"]}

    train_dir = output_root / "train2017"
    val_dir = output_root / "val2017"
    annotations_dir = output_root / "annotations"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for image_id in train_ids:
        file_name = str(image_by_id[image_id]["file_name"])
        shutil.copy2(images_root / file_name, train_dir / file_name)
    for image_id in val_ids:
        file_name = str(image_by_id[image_id]["file_name"])
        shutil.copy2(images_root / file_name, val_dir / file_name)

    train_payload = _build_subset_annotations(payload, train_ids)
    val_payload = _build_subset_annotations(payload, val_ids)

    train_annotations = annotations_dir / "person_keypoints_train2017.json"
    val_annotations = annotations_dir / "person_keypoints_val2017.json"
    train_annotations.write_text(json.dumps(train_payload), encoding="utf-8")
    val_annotations.write_text(json.dumps(val_payload), encoding="utf-8")

    return output_root


def _build_subset_datamodule(
    model: RFDETRKeypointPreview,
    train_config: TrainConfig,
    train_subset_size: int = 24,
    val_subset_size: int = 8,
) -> RFDETRDataModule:
    datamodule = RFDETRDataModule(model.model_config, train_config)
    datamodule.setup("fit")
    if datamodule._dataset_train is None or datamodule._dataset_val is None:
        raise RuntimeError("Expected both training and validation datasets to be initialized.")

    train_count = min(train_subset_size, len(datamodule._dataset_train))
    val_count = min(val_subset_size, len(datamodule._dataset_val))
    datamodule._dataset_train = Subset(datamodule._dataset_train, list(range(train_count)))
    datamodule._dataset_val = Subset(datamodule._dataset_val, list(range(val_count)))
    return datamodule


@pytest.mark.gpu
@pytest.mark.coco17
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_keypoint_training_subset_reports_loss_and_metric(
    tmp_path: Path,
    download_coco_val_keypoints: tuple[Path, Path],
) -> None:
    """Short deterministic fine-tuning should report finite loss and keypoint AP on the fixed subset."""
    seed_all(7)
    images_root, annotations_path = download_coco_val_keypoints
    subset_root = _build_coco_keypoint_subset_from_val(
        images_root=images_root,
        annotations_path=annotations_path,
        output_root=tmp_path / "coco_keypoint_subset",
        train_images=1000,
        val_images=200,
    )
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(subset_root),
        output_dir=str(tmp_path / "train_output"),
        epochs=5,
        batch_size=4,
        num_workers=0,
        grad_accum_steps=2,
        use_ema=False,
        run_test=False,
        compute_val_loss=True,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    model = RFDETRKeypointPreview()
    datamodule = _build_subset_datamodule(model, train_config)

    module = RFDETRModelModule(model.model_config, train_config)
    module.model.load_state_dict(model.model.model.state_dict())
    module.model.eval()

    trainer = build_trainer(train_config, model.model_config, accelerator="gpu")
    (pre_metrics,) = trainer.validate(module, datamodule=datamodule)
    pre_loss = _to_float(pre_metrics["val/loss"])
    pre_map = _to_float(pre_metrics["val/keypoint_map_50_95"])
    assert torch.isfinite(torch.tensor(pre_loss)), f"Expected finite pre-training val/loss, got {pre_loss:.6f}"
    assert torch.isfinite(torch.tensor(pre_map)), f"Expected finite pre-training keypoint AP, got {pre_map:.6f}"

    trainer.fit(module, datamodule=datamodule)
    (post_metrics,) = trainer.validate(module, datamodule=datamodule)
    post_loss = _to_float(post_metrics["val/loss"])
    post_map = _to_float(post_metrics["val/keypoint_map_50_95"])
    assert torch.isfinite(torch.tensor(post_loss)), f"Expected finite post-training val/loss, got {post_loss:.6f}"
    assert torch.isfinite(torch.tensor(post_map)), f"Expected finite post-training keypoint AP, got {post_map:.6f}"
    assert post_map >= pre_map - 0.02, (
        "Expected keypoint AP to improve or remain close on subset "
        f"(map before={pre_map:.6f}, map after={post_map:.6f}, "
        f"loss before={pre_loss:.6f}, loss after={post_loss:.6f})."
    )


@pytest.mark.gpu
@pytest.mark.coco17
def test_keypoint_training_full_coco_release_qualification(
    tmp_path: Path,
    download_coco_train_val_keypoints: Path,
) -> None:
    """Opt-in release gate: train on full COCO train2017 and validate on COCO val2017 keypoints."""
    if os.getenv("RFDETR_RUN_FULL_COCO_KEYPOINT_TRAINING") != "1":
        pytest.skip("Set RFDETR_RUN_FULL_COCO_KEYPOINT_TRAINING=1 to run the full COCO keypoint training gate.")

    seed_all(7)
    epochs = int(os.getenv("RFDETR_FULL_COCO_KEYPOINT_EPOCHS", "1"))
    batch_size = int(os.getenv("RFDETR_FULL_COCO_KEYPOINT_BATCH_SIZE", "2"))
    grad_accum_steps = int(os.getenv("RFDETR_FULL_COCO_KEYPOINT_GRAD_ACCUM_STEPS", "8"))
    num_workers = int(os.getenv("RFDETR_FULL_COCO_KEYPOINT_NUM_WORKERS", "4"))
    minimum_map = float(os.getenv("RFDETR_FULL_COCO_KEYPOINT_MIN_MAP", "0.0"))

    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(download_coco_train_val_keypoints),
        output_dir=str(tmp_path / "full_coco_keypoint_train"),
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        grad_accum_steps=grad_accum_steps,
        use_ema=False,
        run_test=False,
        compute_val_loss=True,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    model = RFDETRKeypointPreview()
    datamodule = RFDETRDataModule(model.model_config, train_config)
    module = RFDETRModelModule(model.model_config, train_config)
    module.model.load_state_dict(model.model.model.state_dict())

    trainer = build_trainer(train_config, model.model_config, accelerator="gpu")
    trainer.fit(module, datamodule=datamodule)
    (metrics,) = trainer.validate(module, datamodule=datamodule)

    val_loss = _to_float(metrics["val/loss"])
    keypoint_map = _to_float(metrics["val/keypoint_map_50_95"])
    assert torch.isfinite(torch.tensor(val_loss)), f"Expected finite full-COCO val/loss, got {val_loss:.6f}"
    assert torch.isfinite(torch.tensor(keypoint_map)), f"Expected finite full-COCO keypoint AP, got {keypoint_map:.6f}"
    assert keypoint_map >= minimum_map, f"keypoint AP@50:95 {keypoint_map:.4f} < {minimum_map:.4f}"
