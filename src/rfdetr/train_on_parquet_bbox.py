# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Dedicated YAML-driven RF-DETR parquet training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, ClassVar, Literal, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rfdetr.detr import RFDETR
from rfdetr.variants import (
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)

VariantName = Literal[
    "nano",
    "small",
    "medium",
    "large",
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
]

MODEL_CLASSES: dict[str, Type[RFDETR]] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
    "seg-nano": RFDETRSegNano,
    "seg-small": RFDETRSegSmall,
    "seg-medium": RFDETRSegMedium,
    "seg-large": RFDETRSegLarge,
    "seg-xlarge": RFDETRSegXLarge,
}


class ParquetDatasetConfig(BaseModel):
    """Pydantic config for DatasetRecordWithBBox parquet training data."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    source: Literal["parquet"] = "parquet"
    dataset_dir: str | None = None
    dataset_repo_id: str | None = None
    max_objects: int | None = Field(default=400, ge=1)

    @model_validator(mode="after")
    def validate_source(self) -> "ParquetDatasetConfig":
        """Validate that exactly one source locator is provided."""
        if bool(self.dataset_dir) == bool(self.dataset_repo_id):
            raise ValueError("Exactly one of dataset_dir or dataset_repo_id must be provided.")
        return self


class RFDETRParquetTrainingConfig(BaseModel):
    """Top-level YAML config for RF-DETR parquet training."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True)

    variant: VariantName = "large"
    model: dict[str, Any] = Field(default_factory=dict)
    dataset: ParquetDatasetConfig = Field(default_factory=lambda: ParquetDatasetConfig(dataset_dir="./dataset_root"))
    label_mapping: dict[int, str] = Field(default_factory=lambda: {0: "text", 1: "table"})
    train: dict[str, Any] = Field(
        default_factory=lambda: {
            "output_dir": "./output/rfdetr-parquet",
            "epochs": 50,
            "batch_size": 4,
            "grad_accum_steps": 4,
            "num_workers": 4,
            "tensorboard": True,
            "progress_bar": "tqdm",
        }
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_labelmapping_key(cls, values: Any) -> Any:
        """Accept labelmapping as a compatibility alias for label_mapping."""
        if isinstance(values, dict) and "labelmapping" in values and "label_mapping" not in values:
            values = dict(values)
            values["label_mapping"] = values.pop("labelmapping")
        return values

    @model_validator(mode="after")
    def validate_nested_configs(self) -> "RFDETRParquetTrainingConfig":
        """Validate labels plus selected RF-DETR model and train configs."""
        normalized_labels = self.normalized_label_mapping()
        variant_cls = MODEL_CLASSES[self.variant]
        model_kwargs = dict(self.model)
        model_kwargs.setdefault("num_classes", len(normalized_labels))
        variant_cls._model_config_class(**model_kwargs)

        train_kwargs = self.to_train_kwargs(dataset_dir=self.dataset.dataset_dir or ".")
        variant_cls._train_config_class(**train_kwargs)
        return self

    @classmethod
    def read_yaml(cls, path: str | Path) -> "RFDETRParquetTrainingConfig":
        """Read config from YAML."""
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.model_validate(data)

    def write_yaml(self, path: str | Path) -> None:
        """Write config to YAML."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)

    def normalized_label_mapping(self) -> dict[int, str]:
        """Return label mapping with integer keys and validated names."""
        if not self.label_mapping:
            raise ValueError("label_mapping must not be empty.")
        normalized: dict[int, str] = {}
        for raw_category_id, raw_name in self.label_mapping.items():
            category_id = int(raw_category_id)
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"label_mapping[{category_id}] must be a non-empty string.")
            normalized[category_id] = raw_name
        return dict(sorted(normalized.items()))

    def class_names(self) -> list[str]:
        """Return class names ordered by category id."""
        return [name for _, name in self.normalized_label_mapping().items()]

    def to_model_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the selected RF-DETR model constructor."""
        kwargs = dict(self.model)
        kwargs.setdefault("num_classes", len(self.normalized_label_mapping()))
        return kwargs

    def to_train_kwargs(self, *, dataset_dir: str) -> dict[str, Any]:
        """Return kwargs for RFDETR.train()."""
        kwargs = dict(self.train)
        kwargs.update(
            {
                "dataset_file": "parquet_bbox",
                "dataset_dir": dataset_dir,
                "dataset_repo_id": self.dataset.dataset_repo_id,
                "parquet_max_objects": self.dataset.max_objects,
                "parquet_label_mapping": self.normalized_label_mapping(),
                "class_names": self.class_names(),
                "limit_val_batches": 0,
                "num_sanity_val_steps": 0,
            }
        )
        return kwargs

    def resolve_dataset_dir(self) -> str:
        """Resolve local dataset dir or Hugging Face dataset repo id to a local path."""
        if self.dataset.dataset_dir:
            return str(Path(self.dataset.dataset_dir).expanduser().resolve())
        if not self.dataset.dataset_repo_id:
            raise ValueError("dataset_repo_id is required when dataset_dir is not set.")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Hugging Face dataset repos require huggingface_hub. "
                "Install training dependencies with `pip install 'rfdetr[train]'`."
            ) from exc
        return snapshot_download(repo_id=self.dataset.dataset_repo_id, repo_type="dataset")

    def train_model(self) -> None:
        """Instantiate the selected model and run training."""
        dataset_dir = self.resolve_dataset_dir()
        model_cls = MODEL_CLASSES[self.variant]
        model = model_cls(**self.to_model_kwargs())
        model.train(**self.to_train_kwargs(dataset_dir=dataset_dir))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train.yaml", help="Path to YAML training config.")
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write a default parquet training config to --config and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dedicated parquet training entry point."""
    args = parse_args()
    config_path = Path(args.config)
    if args.write_default_config:
        RFDETRParquetTrainingConfig().write_yaml(config_path)
        return
    RFDETRParquetTrainingConfig.read_yaml(config_path).train_model()


if __name__ == "__main__":
    main()
