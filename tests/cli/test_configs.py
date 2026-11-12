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
    """Parse a config file by stem name and return its dict.

    >>> _load("rfdetr_small")["model"]["train_config"]["class_path"]
    'rfdetr.config.TrainConfig'
    """
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestConfigFilesExist:
    """Every expected YAML config file must be present on disk."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_file_exists(self, name: str) -> None:
        """configs/{name}.yaml must exist."""
        assert (CONFIGS_DIR / f"{name}.yaml").exists(), f"Missing config file: {name}.yaml"


# ---------------------------------------------------------------------------
# YAML validity
# ---------------------------------------------------------------------------


class TestConfigFilesValidYAML:
    """Each file must be parseable as YAML and produce a mapping."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_is_valid_yaml(self, name: str) -> None:
        """yaml.safe_load must succeed and return a dict."""
        data = _load(name)
        assert isinstance(data, dict), f"{name}.yaml did not parse to a dict"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestConfigStructure:
    """Each YAML must have a model section with model_config and train_config."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_model_section(self, name: str) -> None:
        """Top-level 'model' key must be present."""
        assert "model" in _load(name), f"{name}.yaml missing 'model' section"

    @pytest.mark.parametrize("key", ["model_config", "train_config"])
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_section_has_key(self, name: str, key: str) -> None:
        """The model section must contain both 'model_config' and 'train_config'."""
        assert key in _load(name)["model"], f"{name}.yaml missing model.{key}"


# ---------------------------------------------------------------------------
# Class paths
# ---------------------------------------------------------------------------


class TestConfigClassPaths:
    """model_config and train_config class_path must match the expected model variant."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_config_class_path(self, name: str) -> None:
        """model.model_config.class_path must match the variant."""
        got = _load(name)["model"]["model_config"]["class_path"]
        want = EXPECTED_MODEL_CLASS[name]
        assert got == want, f"{name}.yaml: expected class_path {want!r}, got {got!r}"

    @pytest.mark.parametrize(
        ("name", "want"),
        [(name, "rfdetr.config.SegmentationTrainConfig") for name in SEGMENTATION_CONFIGS]
        + [(name, "rfdetr.config.TrainConfig") for name in DETECTION_CONFIGS],
    )
    def test_train_config_class_path(self, name: str, want: str) -> None:
        """Segmentation configs must use SegmentationTrainConfig; detection configs TrainConfig (not a subclass)."""
        got = _load(name)["model"]["train_config"]["class_path"]
        assert got == want, f"{name}.yaml: train_config must use {want}, got {got!r}"
