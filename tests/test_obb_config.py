# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for oriented bounding box configuration fields."""

from rfdetr.config import DatasetEntry, ModelConfig, MultiDatasetConfig, TrainConfig


class TestOrientedConfig:
    """Tests for oriented bbox config fields."""

    def test_model_config_oriented_default(self) -> None:
        """ModelConfig.oriented should default to False."""
        mc = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[2, 5, 8, 11],
            dec_layers=3,
            projector_scale=["P4"],
            hidden_dim=256,
            patch_size=14,
            num_windows=4,
            sa_nheads=8,
            ca_nheads=16,
            dec_n_points=2,
            resolution=560,
            positional_encoding_size=37,
        )
        assert mc.oriented is False

    def test_model_config_oriented_true(self) -> None:
        """ModelConfig.oriented can be set to True."""
        mc = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[2, 5, 8, 11],
            dec_layers=3,
            projector_scale=["P4"],
            hidden_dim=256,
            patch_size=14,
            num_windows=4,
            sa_nheads=8,
            ca_nheads=16,
            dec_n_points=2,
            resolution=560,
            positional_encoding_size=37,
            oriented=True,
        )
        assert mc.oriented is True

    def test_train_config_loss_angle_coef(self) -> None:
        """TrainConfig should have loss_angle_coef field."""
        tc = TrainConfig(dataset_dir="/tmp/test")
        assert tc.loss_angle_coef == 1.0

    def test_train_config_dataset_file_dota(self) -> None:
        """TrainConfig should accept 'dota' as dataset_file."""
        tc = TrainConfig(dataset_dir="/tmp/test", dataset_file="dota")
        assert tc.dataset_file == "dota"

    def test_train_config_dataset_file_multi(self) -> None:
        """TrainConfig should accept 'multi' as dataset_file."""
        tc = TrainConfig(dataset_dir="/tmp/test", dataset_file="multi")
        assert tc.dataset_file == "multi"


class TestDatasetEntry:
    """Tests for DatasetEntry config."""

    def test_defaults(self) -> None:
        """Default values should be sensible."""
        entry = DatasetEntry(path="/data/ds1")
        assert entry.format == "dota"
        assert entry.oriented is True
        assert entry.weight == 1.0
        assert entry.class_mapping is None
        assert entry.aug_config is None

    def test_all_fields(self) -> None:
        """All fields can be set."""
        entry = DatasetEntry(
            path="/data/ds1",
            format="yolo",
            oriented=False,
            weight=0.5,
            class_mapping={"cat": 0, "dog": 1},
            aug_config={"HorizontalFlip": {"p": 0.5}},
        )
        assert entry.format == "yolo"
        assert entry.oriented is False
        assert entry.weight == 0.5


class TestMultiDatasetConfig:
    """Tests for MultiDatasetConfig."""

    def test_basic(self) -> None:
        """Basic multi-dataset config should parse."""
        config = MultiDatasetConfig(
            num_classes=5,
            train=[DatasetEntry(path="/data/ds1")],
            val=[DatasetEntry(path="/data/ds1")],
        )
        assert config.num_classes == 5
        assert len(config.train) == 1
        assert config.test is None

    def test_with_test_split(self) -> None:
        """Test split should be optional."""
        config = MultiDatasetConfig(
            num_classes=3,
            train=[DatasetEntry(path="/data/ds1")],
            val=[DatasetEntry(path="/data/ds1")],
            test=[DatasetEntry(path="/data/ds1")],
        )
        assert config.test is not None
        assert len(config.test) == 1
