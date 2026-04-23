# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for YAML config files in configs/ — PTL Ch4/T6.

Verifies that every example YAML config file:
  - exists on disk,
  - parses as valid YAML,
  - contains a ``model`` section with ``model_config`` and ``train_config``,
  - references the expected model class_path, and
  - segmentation configs use SegmentationTrainConfig.
"""

import pathlib

import pytest
import yaml

CONFIGS_DIR = pathlib.Path(__file__).parent.parent.parent / "configs"

DETECTION_CONFIGS = [
    "rfdetr_nano",
    "rfdetr_small",
    "rfdetr_medium",
    "rfdetr_base",
    "rfdetr_large",
]

SEGMENTATION_CONFIGS = [
    "rfdetr_seg_nano",
    "rfdetr_seg_small",
    "rfdetr_seg_medium",
    "rfdetr_seg_large",
    "rfdetr_seg_xlarge",
    "rfdetr_seg_2xlarge",
]

ALL_CONFIGS = DETECTION_CONFIGS + SEGMENTATION_CONFIGS

# Maps filename stem → expected model_config class_path.
EXPECTED_MODEL_CLASS = {
    "rfdetr_nano": "rfdetr.config.RFDETRNanoConfig",
    "rfdetr_small": "rfdetr.config.RFDETRSmallConfig",
    "rfdetr_medium": "rfdetr.config.RFDETRMediumConfig",
    "rfdetr_base": "rfdetr.config.RFDETRBaseConfig",
    "rfdetr_large": "rfdetr.config.RFDETRLargeConfig",
    "rfdetr_seg_nano": "rfdetr.config.RFDETRSegNanoConfig",
    "rfdetr_seg_small": "rfdetr.config.RFDETRSegSmallConfig",
    "rfdetr_seg_medium": "rfdetr.config.RFDETRSegMediumConfig",
    "rfdetr_seg_large": "rfdetr.config.RFDETRSegLargeConfig",
    "rfdetr_seg_xlarge": "rfdetr.config.RFDETRSegXLargeConfig",
    "rfdetr_seg_2xlarge": "rfdetr.config.RFDETRSeg2XLargeConfig",
}


def _load(name: str) -> dict:
    """Parse a config file by stem name and return its dict."""
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestConfigFilesExist:
    """Every expected YAML config file must be present on disk."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_file_exists(self, name):
        """configs/{name}.yaml must exist."""
        assert (CONFIGS_DIR / f"{name}.yaml").exists(), f"Missing config file: {name}.yaml"


# ---------------------------------------------------------------------------
# YAML validity
# ---------------------------------------------------------------------------


class TestConfigFilesValidYAML:
    """Each file must be parseable as YAML and produce a mapping."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_is_valid_yaml(self, name):
        """yaml.safe_load must succeed and return a dict."""
        data = _load(name)
        assert isinstance(data, dict), f"{name}.yaml did not parse to a dict"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestConfigStructure:
    """Each YAML must have a model section with model_config and train_config."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_model_section(self, name):
        """Top-level 'model' key must be present."""
        assert "model" in _load(name), f"{name}.yaml missing 'model' section"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_model_config(self, name):
        """model.model_config must be present."""
        assert "model_config" in _load(name)["model"], f"{name}.yaml missing model.model_config"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_train_config(self, name):
        """model.train_config must be present."""
        assert "train_config" in _load(name)["model"], f"{name}.yaml missing model.train_config"


# ---------------------------------------------------------------------------
# Class paths
# ---------------------------------------------------------------------------


class TestConfigClassPaths:
    """model_config class_path must match the expected model variant."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_config_class_path(self, name):
        """model.model_config.class_path must match the variant."""
        got = _load(name)["model"]["model_config"]["class_path"]
        want = EXPECTED_MODEL_CLASS[name]
        assert got == want, f"{name}.yaml: expected class_path {want!r}, got {got!r}"

    @pytest.mark.parametrize("name", SEGMENTATION_CONFIGS)
    def test_seg_uses_segmentation_train_config(self, name):
        """Segmentation configs must use SegmentationTrainConfig."""
        got = _load(name)["model"]["train_config"]["class_path"]
        assert got == "rfdetr.config.SegmentationTrainConfig", (
            f"{name}.yaml: train_config must use SegmentationTrainConfig, got {got!r}"
        )

    @pytest.mark.parametrize("name", DETECTION_CONFIGS)
    def test_det_uses_train_config(self, name):
        """Detection configs must use TrainConfig (not a subclass)."""
        got = _load(name)["model"]["train_config"]["class_path"]
        assert got == "rfdetr.config.TrainConfig", f"{name}.yaml: train_config must use TrainConfig, got {got!r}"
