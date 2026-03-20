# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Multi-dataset YAML configuration and combined dataset builder."""

import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.utils.data
import yaml

from rfdetr.config import DatasetEntry, MultiDatasetConfig
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class ClassMappingDataset(torch.utils.data.Dataset):
    """Wrapper that remaps class labels and optionally pads boxes for OBB.

    Args:
        dataset: Underlying dataset to wrap.
        class_mapping: Mapping from source class names to target class indices.
            If None, no remapping is performed.
        pad_to_obb: If True, pad 4-dim boxes to 5-dim with angle=0.
        source_class_names: List of source class names for index-to-name mapping.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        class_mapping: Optional[Dict[str, int]] = None,
        pad_to_obb: bool = False,
        source_class_names: Optional[List[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.class_mapping = class_mapping
        self.pad_to_obb = pad_to_obb
        self.source_class_names = source_class_names

        # Build index-to-index mapping if class_mapping and source names are available
        self._idx_mapping: Optional[Dict[int, int]] = None
        if class_mapping is not None and source_class_names is not None:
            self._idx_mapping = {}
            for i, name in enumerate(source_class_names):
                if name in class_mapping:
                    self._idx_mapping[i] = class_mapping[name]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        img, target = self.dataset[idx]

        if self._idx_mapping is not None and "labels" in target:
            labels = target["labels"]
            new_labels = []
            keep_mask = []
            for i, label in enumerate(labels.tolist()):
                if label in self._idx_mapping:
                    new_labels.append(self._idx_mapping[label])
                    keep_mask.append(i)

            if len(keep_mask) < len(labels):
                keep_indices = torch.tensor(keep_mask, dtype=torch.long)
                target = target.copy()
                target["labels"] = torch.tensor(new_labels, dtype=torch.int64)
                if "boxes" in target:
                    target["boxes"] = target["boxes"][keep_indices]
                if "area" in target:
                    target["area"] = target["area"][keep_indices]
                if "iscrowd" in target:
                    target["iscrowd"] = target["iscrowd"][keep_indices]
                if "obb_corners" in target:
                    target["obb_corners"] = target["obb_corners"][keep_indices]
            else:
                target = target.copy()
                target["labels"] = torch.tensor(new_labels, dtype=torch.int64)

        if self.pad_to_obb and "boxes" in target:
            boxes = target["boxes"]
            if boxes.shape[-1] == 4 and boxes.numel() > 0:
                # Pad with angle=0 for axis-aligned boxes
                target = target.copy()
                target["boxes"] = torch.cat(
                    [boxes, torch.zeros((*boxes.shape[:-1], 1), dtype=boxes.dtype)],
                    dim=-1,
                )

        return img, target


def parse_multi_dataset_config(config_path: str) -> MultiDatasetConfig:
    """Parse a multi-dataset YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed MultiDatasetConfig.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Multi-dataset config not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return MultiDatasetConfig(**raw)


def _build_single_dataset(
    entry: DatasetEntry,
    split: str,
    args: Any,
    resolution: int,
) -> torch.utils.data.Dataset:
    """Build a single dataset from a DatasetEntry.

    Args:
        entry: Dataset entry configuration.
        split: Split identifier ("train", "val", "test").
        args: Namespace with general config.
        resolution: Target resolution.

    Returns:
        Dataset instance.
    """
    # Create a modified args with dataset-specific overrides
    entry_args = types.SimpleNamespace(**vars(args))
    entry_args.dataset_dir = entry.path
    entry_args.oriented = entry.oriented
    if entry.aug_config is not None:
        entry_args.aug_config = entry.aug_config

    if entry.format == "coco":
        from rfdetr.datasets.coco import build_roboflow_from_coco

        return build_roboflow_from_coco(split, entry_args, resolution)
    elif entry.format == "yolo":
        from rfdetr.datasets.yolo import build_roboflow_from_yolo

        return build_roboflow_from_yolo(split, entry_args, resolution)
    elif entry.format == "dota":
        from rfdetr.datasets.dota import build_dota

        return build_dota(split, entry_args, resolution)
    else:
        raise ValueError(f"Unknown dataset format: {entry.format}")


def build_multi_dataset(
    image_set: str,
    args: Any,
    resolution: int,
) -> torch.utils.data.Dataset:
    """Build a combined dataset from a multi-dataset YAML config.

    The YAML config file is specified by args.dataset_dir. It defines multiple
    datasets per split, each with its own format, weight, class mapping, and
    augmentation config.

    Args:
        image_set: Split identifier ("train", "val", "test").
        args: Namespace with dataset_dir pointing to the YAML config file.
        resolution: Target resolution.

    Returns:
        Combined ConcatDataset with optional class mapping and OBB padding.
    """
    config = parse_multi_dataset_config(args.dataset_dir)

    # Get the entries for the requested split
    if image_set == "train":
        entries = config.train
    elif image_set == "val":
        entries = config.val
    elif image_set == "test":
        if config.test is None:
            raise ValueError("No test split defined in multi-dataset config")
        entries = config.test
    else:
        raise ValueError(f"Unknown split: {image_set}")

    if not entries:
        raise ValueError(f"No datasets defined for split '{image_set}'")

    # Check if any dataset is oriented (to know if we need to pad)
    any_oriented = any(e.oriented for e in entries)

    datasets: List[torch.utils.data.Dataset] = []
    weights_per_dataset: List[float] = []

    for entry in entries:
        dataset = _build_single_dataset(entry, image_set, args, resolution)

        # Get source class names for mapping
        source_class_names = None
        if entry.class_mapping is not None:
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                source_class_names = [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]

        # Wrap with class mapping and OBB padding
        pad_to_obb = any_oriented and not entry.oriented
        wrapped = ClassMappingDataset(
            dataset,
            class_mapping=entry.class_mapping,
            pad_to_obb=pad_to_obb,
            source_class_names=source_class_names,
        )
        datasets.append(wrapped)
        weights_per_dataset.append(entry.weight)

    if len(datasets) == 1:
        return datasets[0]

    combined = torch.utils.data.ConcatDataset(datasets)

    # Store weights for weighted sampling (used by module_data.py)
    sample_weights: List[float] = []
    for ds, w in zip(datasets, weights_per_dataset):
        sample_weights.extend([w] * len(ds))
    combined.sample_weights = sample_weights  # type: ignore[attr-defined]

    logger.info(
        "Built multi-dataset with %d datasets, %d total samples for split '%s'",
        len(datasets),
        len(combined),
        image_set,
    )
    return combined
