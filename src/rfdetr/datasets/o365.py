# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""Dataset file for Object365."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rfdetr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.utilities.logger import get_logger

Image.MAX_IMAGE_PIXELS = None

logger = get_logger()


def build_o365_raw(image_set: str, args: Any, resolution: int) -> CocoDetection:
    root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
    PATHS = {  # noqa: N806
        "train": (root, root / "zhiyuan_objv2_train_val_wo_5k.json"),
        "val": (root, root / "zhiyuan_objv2_minival5k.json"),
    }
    img_folder, ann_file = PATHS[image_set]

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_backend = augmentation_backend

    if augmentation_backend == "auto":
        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            try:
                import kornia.augmentation  # type: ignore[import-not-found]

                resolved_backend = "gpu"
            except ImportError:
                resolved_backend = "cpu"
        else:
            resolved_backend = "cpu"
    elif augmentation_backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("augmentation_backend='gpu' requires a CUDA device")
        try:
            import kornia.augmentation  # noqa: F401 # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("GPU augmentation requires kornia. Install with: pip install 'rfdetr[kornia]'") from e
        resolved_backend = "gpu"

    if resolved_backend != "cpu":
        logger.warning(
            "O365 dataset does not support custom aug_config in Phase 1 GPU augmentation; "
            "Albumentations augmentation is skipped and normalization runs on GPU. "
            "Pass augmentation_backend='cpu' for full CPU augmentation pipeline with O365."
        )
    gpu_postprocess = resolved_backend != "cpu"

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    return dataset


def build_o365(image_set: str, args: Any, resolution: int) -> CocoDetection:
    if image_set == "train":
        train_ds = build_o365_raw("train", args, resolution=resolution)
        return train_ds
    if image_set == "val":
        val_ds = build_o365_raw("val", args, resolution=resolution)
        return val_ds
    raise ValueError("Unknown image_set: {}".format(image_set))
