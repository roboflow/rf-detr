# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utils to convert a yolo dataset to coco format."""
import os
import supervision as sv

YOLO_YAML_FILE = "data.yaml"
NECESSARY_SPLIT_DIRS = ["train", "valid"]
NECESSARY_DATA_SUBDIRS = ["images", "labels"]
OPTIONAL_SPLIT_DIR = "test"

def is_valid_yolo_format(dataset_dir: str) -> bool:
    """
    Checks if the specified dataset directory is in yolo format.

    We accept a dataset to be in yolo format if the following conditions are met:
    - The dataset_dir contains a data.yaml file
    - The dataset_dir contains "train" and "valid" subdirectories, each containing "images" and "labels" subdirectories
    - The "test" subdirectory is optional

    Returns a boolean indicating whether the dataset is in correct yolo format.
    """
    contains_data_yaml = os.path.exists(os.path.join(dataset_dir, YOLO_YAML_FILE))
    contains_necessary_split_dirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir)) for split_dir in NECESSARY_SPLIT_DIRS
    )
    contains_necessary_data_subdirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir, data_subdir))
        for split_dir in NECESSARY_SPLIT_DIRS
        for data_subdir in NECESSARY_DATA_SUBDIRS
    )
    return contains_data_yaml and contains_necessary_split_dirs and contains_necessary_data_subdirs


def convert_to_coco(dataset_dir: str) -> str:
    """
    Converts the specified dataset directory from yolo format to coco format.

    The converted dataset will be saved in a new directory with the same name as the original dataset directory, but with
    the suffix "_coco".

    Returns a string containing the path to the converted dataset directory.
    """
    coco_dataset_dir = f"{dataset_dir}_coco"

    if os.path.exists(coco_dataset_dir):
        raise ValueError(f"Directory {coco_dataset_dir} already exists. Please remove or rename it before converting the dataset.")
    else:
        os.makedirs(coco_dataset_dir)

    for split_dir in NECESSARY_SPLIT_DIRS:
        sv.DetectionDataset.from_yolo(
            images_directory_path=os.path.join(dataset_dir, split_dir, "images"),
            annotations_directory_path=os.path.join(dataset_dir, split_dir, "labels"),
            data_yaml_path=os.path.join(dataset_dir, YOLO_YAML_FILE),
        ).as_coco(
            images_directory_path=os.path.join(coco_dataset_dir, split_dir),
            annotations_path=os.path.join(coco_dataset_dir, split_dir, "_annotations.coco.json"),
        )
    
    if os.path.exists(os.path.join(dataset_dir, OPTIONAL_SPLIT_DIR)):
        sv.DetectionDataset.from_yolo(
            images_directory_path=os.path.join(dataset_dir, OPTIONAL_SPLIT_DIR, "images"),
            annotations_directory_path=os.path.join(dataset_dir, OPTIONAL_SPLIT_DIR, "labels"),
            data_yaml_path=os.path.join(dataset_dir, YOLO_YAML_FILE),
        ).as_coco(
            images_directory_path=os.path.join(coco_dataset_dir, OPTIONAL_SPLIT_DIR),
            annotations_path=os.path.join(coco_dataset_dir, OPTIONAL_SPLIT_DIR, "_annotations.coco.json"),
        )

    return coco_dataset_dir