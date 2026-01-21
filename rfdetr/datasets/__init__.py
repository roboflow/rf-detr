# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .o365 import build_o365
from .coco import build_roboflow_from_coco
from .yolo import build_roboflow_yolo


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def detect_roboflow_format(dataset_dir: Path) -> str:
    """Detect if a Roboflow dataset is in COCO or YOLO format.
    
    Args:
        dataset_dir: Path to the Roboflow dataset root directory
        
    Returns:
        'coco' if COCO format detected, 'yolo' if YOLO format detected
        
    Raises:
        ValueError: If neither format is detected
    """
    # Check for COCO format: look for _annotations.coco.json in train folder
    coco_annotation = dataset_dir / "train" / "_annotations.coco.json"
    if coco_annotation.exists():
        return "coco"
    
    # Check for YOLO format: look for data.yaml and train/images folder
    yolo_data_file = dataset_dir / "data.yaml"
    yolo_images_dir = dataset_dir / "train" / "images"
    if yolo_data_file.exists() and yolo_images_dir.exists():
        return "yolo"
    
    raise ValueError(
        f"Could not detect dataset format in {dataset_dir}. "
        f"Expected either COCO format (train/_annotations.coco.json) "
        f"or YOLO format (data.yaml + train/images/)"
    )


def build_roboflow(image_set, args, resolution):
    """Build a Roboflow dataset, auto-detecting COCO or YOLO format.
    
    This function detects the dataset format and delegates to the
    appropriate builder function.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    
    dataset_format = detect_roboflow_format(root)
    
    if dataset_format == "coco":
        return build_roboflow_from_coco(image_set, args, resolution)
    else:  # yolo
        return build_roboflow_yolo(image_set, args, resolution)


def build_dataset(image_set, args, resolution):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args, resolution)
    if args.dataset_file == 'o365':
        return build_o365(image_set, args, resolution)
    if args.dataset_file == 'roboflow':
        return build_roboflow(image_set, args, resolution)
    raise ValueError(f'dataset {args.dataset_file} not supported')
