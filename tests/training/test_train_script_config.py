# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the dedicated parquet training script config."""

import pytest
from rfdetr.train import RFDETRParquetTrainingConfig


def test_default_config_writes_and_reads(tmp_path):
    """Default config should round-trip through YAML."""
    path = tmp_path / "train.yaml"
    RFDETRParquetTrainingConfig().write_yaml(path)

    config = RFDETRParquetTrainingConfig.read_yaml(path)

    assert config.variant == "large"
    assert config.dataset.dataset_dir == "./dataset_root"
    assert config.label_mapping == {0: "text", 1: "table"}


def test_repo_config_maps_to_train_kwargs():
    """Repo-backed config should map to parquet RF-DETR train kwargs."""
    config = RFDETRParquetTrainingConfig.model_validate(
        {
            "variant": "nano",
            "dataset": {"source": "parquet", "dataset_repo_id": "org/dataset"},
            "label_mapping": {0: "text", 1: "table"},
            "train": {"output_dir": "out", "epochs": 1, "batch_size": 1, "tensorboard": False},
        }
    )

    kwargs = config.to_train_kwargs(dataset_dir="/tmp/snapshot")

    assert kwargs["dataset_file"] == "parquet_bbox"
    assert kwargs["dataset_dir"] == "/tmp/snapshot"
    assert kwargs["dataset_repo_id"] == "org/dataset"
    assert kwargs["parquet_label_mapping"] == {0: "text", 1: "table"}
    assert kwargs["class_names"] == ["text", "table"]
    assert kwargs["limit_val_batches"] == 0
    assert kwargs["num_sanity_val_steps"] == 0


def test_labelmapping_alias_is_accepted():
    """The config accepts labelmapping as a compatibility alias."""
    config = RFDETRParquetTrainingConfig.model_validate(
        {
            "dataset": {"source": "parquet", "dataset_dir": "./dataset"},
            "labelmapping": {2: "figure", 0: "text"},
        }
    )

    assert config.normalized_label_mapping() == {0: "text", 2: "figure"}
    assert config.class_names() == ["text", "figure"]


def test_missing_dataset_source_fails():
    """Exactly one dataset source must be provided."""
    with pytest.raises(ValueError, match="Exactly one"):
        RFDETRParquetTrainingConfig.model_validate(
            {
                "dataset": {"source": "parquet"},
                "label_mapping": {0: "text"},
            }
        )


def test_empty_label_mapping_fails():
    """The explicit label mapping is required."""
    with pytest.raises(ValueError, match="label_mapping must not be empty"):
        RFDETRParquetTrainingConfig.model_validate(
            {
                "dataset": {"source": "parquet", "dataset_dir": "./dataset"},
                "label_mapping": {},
            }
        )


def test_label_mapping_injects_num_classes_when_absent():
    """Model kwargs should default num_classes from explicit labels."""
    config = RFDETRParquetTrainingConfig.model_validate(
        {
            "variant": "nano",
            "dataset": {"source": "parquet", "dataset_dir": "./dataset"},
            "label_mapping": {0: "text", 1: "table", 2: "figure"},
        }
    )

    assert config.to_model_kwargs()["num_classes"] == 3
