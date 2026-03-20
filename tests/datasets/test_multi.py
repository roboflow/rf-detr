# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for multi-dataset YAML configuration and builder."""

from pathlib import Path

import pytest
import torch
import yaml

from rfdetr.datasets.multi import ClassMappingDataset, parse_multi_dataset_config


class TestMultiDatasetConfig:
    """Tests for MultiDatasetConfig parsing."""

    def test_parse_basic_config(self, tmp_path: Path) -> None:
        """Parse a basic multi-dataset YAML config."""
        config = {
            "num_classes": 5,
            "class_names": ["a", "b", "c", "d", "e"],
            "train": [
                {"path": "/data/ds1", "format": "dota", "oriented": True, "weight": 1.0},
            ],
            "val": [
                {"path": "/data/ds1", "format": "dota", "oriented": True},
            ],
        }
        config_path = tmp_path / "dataset.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parsed = parse_multi_dataset_config(str(config_path))
        assert parsed.num_classes == 5
        assert len(parsed.train) == 1
        assert parsed.train[0].format == "dota"
        assert parsed.train[0].oriented is True
        assert parsed.train[0].weight == 1.0

    def test_parse_multi_dataset_config(self, tmp_path: Path) -> None:
        """Parse a config with multiple datasets per split."""
        config = {
            "num_classes": 3,
            "train": [
                {"path": "/data/ds1", "format": "dota", "oriented": True, "weight": 1.0},
                {"path": "/data/ds2", "format": "yolo", "oriented": False, "weight": 0.5},
            ],
            "val": [
                {"path": "/data/ds1", "format": "dota", "oriented": True},
            ],
        }
        config_path = tmp_path / "dataset.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parsed = parse_multi_dataset_config(str(config_path))
        assert len(parsed.train) == 2
        assert parsed.train[1].weight == 0.5
        assert parsed.train[1].oriented is False

    def test_class_mapping_config(self, tmp_path: Path) -> None:
        """Parse config with class mapping."""
        config = {
            "num_classes": 2,
            "train": [
                {
                    "path": "/data/ds1",
                    "format": "dota",
                    "class_mapping": {"plane": 0, "ship": 1},
                },
            ],
            "val": [
                {"path": "/data/ds1", "format": "dota"},
            ],
        }
        config_path = tmp_path / "dataset.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parsed = parse_multi_dataset_config(str(config_path))
        assert parsed.train[0].class_mapping == {"plane": 0, "ship": 1}

    def test_missing_config_raises(self) -> None:
        """Missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_multi_dataset_config("/nonexistent/path.yml")


class TestClassMappingDataset:
    """Tests for ClassMappingDataset wrapper."""

    def _make_dummy_dataset(self, num_items: int = 5) -> torch.utils.data.Dataset:
        """Create a simple in-memory dataset for testing."""

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return num_items

            def __getitem__(self, idx: int):
                img = torch.rand(3, 32, 32)
                target = {
                    "boxes": torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                    "area": torch.tensor([200.0]),
                    "iscrowd": torch.tensor([0]),
                }
                return img, target

        return DummyDataset()

    def test_no_mapping(self) -> None:
        """Without mapping, labels should pass through unchanged."""
        ds = self._make_dummy_dataset()
        wrapped = ClassMappingDataset(ds)
        _, target = wrapped[0]
        assert target["labels"].tolist() == [0]

    def test_pad_to_obb(self) -> None:
        """Padding should add angle=0 column to 4-dim boxes."""
        ds = self._make_dummy_dataset()
        wrapped = ClassMappingDataset(ds, pad_to_obb=True)
        _, target = wrapped[0]
        assert target["boxes"].shape == (1, 5)
        assert target["boxes"][0, 4].item() == 0.0

    def test_class_mapping_with_index(self) -> None:
        """Class mapping should remap labels via source class names."""
        ds = self._make_dummy_dataset()
        wrapped = ClassMappingDataset(
            ds,
            class_mapping={"cat": 5},
            source_class_names=["cat", "dog"],
        )
        _, target = wrapped[0]
        # Label 0 maps to "cat" -> 5
        assert target["labels"].tolist() == [5]

    def test_length_preserved(self) -> None:
        """Length should match underlying dataset."""
        ds = self._make_dummy_dataset(num_items=10)
        wrapped = ClassMappingDataset(ds)
        assert len(wrapped) == 10
