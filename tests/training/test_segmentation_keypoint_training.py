# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-side coverage for a model with both a segmentation head and a keypoint head."""

import json
import types
from pathlib import Path

import torch
from PIL import Image

from rfdetr import RFDETRSegNano
from rfdetr.config import RFDETRSegNanoConfig, SegmentationKeypointTrainConfig
from rfdetr.datasets.coco import ConvertCoco, build_roboflow_from_coco
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

_IMAGE = Image.new("RGB", (64, 48))

#: One instance with a polygon mask and two keypoints, one box-only instance without either.
_FULL_ANNOTATION = {
    "id": 1,
    "image_id": 1,
    "category_id": 0,
    "bbox": [8.0, 8.0, 24.0, 24.0],
    "area": 576.0,
    "iscrowd": 0,
    "segmentation": [[8.0, 8.0, 32.0, 8.0, 32.0, 32.0, 8.0, 32.0]],
    "keypoints": [10.0, 12.0, 2.0, 20.0, 22.0, 2.0],
    "num_keypoints": 2,
}
_BOX_ONLY_ANNOTATION = {
    "id": 2,
    "image_id": 1,
    "category_id": 0,
    "bbox": [36.0, 4.0, 20.0, 20.0],
    "area": 400.0,
    "iscrowd": 0,
}


def _seg_keypoint_model_config() -> RFDETRSegNanoConfig:
    """Return a CPU segmentation model config that also carries a two-keypoint head, without pretrained weights.

    Examples:
        >>> config = _seg_keypoint_model_config()
        >>> config.segmentation_head, config.use_grouppose_keypoints, config.num_keypoints_per_class
        (True, True, [2])
    """
    return RFDETRSegNanoConfig(
        pretrain_weights=None,
        device="cpu",
        num_classes=1,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[2],
    )


def _write_masks_and_keypoints_coco(path: Path) -> None:
    """Write a one-image Roboflow-style COCO split whose annotations carry polygon masks and keypoints.

    Examples:
        >>> import tempfile
        >>> output = Path(tempfile.mkdtemp()) / "valid" / "_annotations.coco.json"
        >>> _write_masks_and_keypoints_coco(output)
        >>> sorted(json.loads(output.read_text())["categories"][0])
        ['id', 'keypoints', 'name', 'skeleton', 'supercategory']
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image_path = path.parent / "image.png"
    Image.new("RGB", (64, 48), color=(255, 255, 255)).save(image_path)
    data = {
        "images": [{"id": 1, "file_name": image_path.name, "width": 64, "height": 48}],
        "annotations": [_FULL_ANNOTATION, _BOX_ONLY_ANNOTATION],
        "categories": [
            {"id": 0, "name": "obj", "supercategory": "obj", "keypoints": ["a", "b"], "skeleton": []},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class TestGetTrainConfigOfRealModel:
    """``RFDETR.get_train_config`` on a real model carrying both heads."""

    def test_segmentation_model_with_keypoint_head_gets_combined_config(self, tmp_path: Path) -> None:
        """A real segmentation model built with a keypoint head trains with ``SegmentationKeypointTrainConfig``."""
        model = RFDETRSegNano(
            pretrain_weights=None,
            device="cpu",
            num_classes=1,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[2],
        )

        config = model.get_train_config(dataset_dir=str(tmp_path), output_dir=str(tmp_path))

        assert type(config) is SegmentationKeypointTrainConfig


class TestConvertCocoMasksWithKeypoints:
    """``ConvertCoco`` with mask and keypoint output on the same annotations."""

    def test_masks_and_keypoints_stay_aligned_with_boxes(self) -> None:
        """Every kept box has one mask and one ``(K, 3)`` keypoint row, in annotation order."""
        converter = ConvertCoco(include_masks=True, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = converter(_IMAGE, {"image_id": 1, "annotations": [_FULL_ANNOTATION, _BOX_ONLY_ANNOTATION]})

        assert target["boxes"].shape == (2, 4)
        assert target["masks"].shape == (2, 48, 64)
        assert target["keypoints"].shape == (2, 2, 3)

    def test_annotation_with_polygon_and_keypoints_keeps_both(self) -> None:
        """A fully annotated instance has a non-empty mask and its keypoint coordinates, untouched."""
        converter = ConvertCoco(include_masks=True, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = converter(_IMAGE, {"image_id": 1, "annotations": [_FULL_ANNOTATION]})

        assert target["masks"].dtype == torch.bool
        assert target["masks"][0].any()
        assert target["keypoints"][0].tolist() == [[10.0, 12.0, 2.0], [20.0, 22.0, 2.0]]

    def test_box_only_annotation_gets_empty_mask_and_zero_keypoints(self) -> None:
        """An instance with neither a polygon nor keypoints is kept with an empty mask and zeroed keypoints."""
        converter = ConvertCoco(include_masks=True, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = converter(_IMAGE, {"image_id": 1, "annotations": [_FULL_ANNOTATION, _BOX_ONLY_ANNOTATION]})

        assert not target["masks"][1].any()
        assert target["keypoints"][1].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


class TestRoboflowCocoMasksWithKeypoints:
    """A Roboflow COCO split read for a model with both heads."""

    def test_dataset_item_carries_masks_and_keypoints_for_the_same_rows(self, tmp_path: Path) -> None:
        """One item yields as many masks as keypoint rows, each on the resized image grid."""
        _write_masks_and_keypoints_coco(tmp_path / "valid" / "_annotations.coco.json")
        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            square_resize_div_64=False,
            segmentation_head=True,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[2],
            aug_config={},
            augmentation_backend="cpu",
        )

        dataset = build_roboflow_from_coco("val", args, resolution=64)
        image, target = dataset[0]

        assert target["boxes"].shape[0] == target["masks"].shape[0] == target["keypoints"].shape[0] == 2
        assert target["masks"].shape[-2:] == image.shape[-2:]
        assert target["keypoints"].shape[1:] == (2, 3)


class TestBuildTrainerWithBothHeads:
    """``build_trainer`` wiring for a model with both heads and the combined train config."""

    def test_combined_config_builds_a_trainer_that_evaluates_masks_and_keypoints(self, tmp_path: Path) -> None:
        """The evaluation callback tracks masks and keypoints, and checkpoints follow the segmentation metric."""
        train_config = SegmentationKeypointTrainConfig(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
            use_ema=False,
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
        )

        trainer = build_trainer(train_config, _seg_keypoint_model_config())

        eval_callback = next(cb for cb in trainer.callbacks if isinstance(cb, COCOEvalCallback))
        best_callback = next(cb for cb in trainer.callbacks if isinstance(cb, BestModelCallback))
        assert eval_callback._use_segm_metrics is True
        assert best_callback.monitor == "val/segm_mAP_50_95"
