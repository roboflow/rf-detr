# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from rfdetr.config import (
    ModelConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    SegmentationTrainConfig,
)
from rfdetr.detr import RFDETR


@pytest.fixture
def sample_model_config() -> dict[str, object]:
    return {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [1, 2, 3],
        "dec_layers": 3,
        "projector_scale": ["P3"],
        "hidden_dim": 256,
        "patch_size": 14,
        "num_windows": 2,
        "sa_nheads": 8,
        "ca_nheads": 8,
        "dec_n_points": 4,
        "resolution": 384,
        "positional_encoding_size": 256,
    }


class TestModelConfigValidation:
    def test_rejects_unknown_fields(self, sample_model_config) -> None:
        sample_model_config["unknown"] = "value"

        with pytest.raises(ValidationError, match=r"Unknown parameter\(s\): 'unknown'"):
            ModelConfig(**sample_model_config)

    def test_rejects_unknown_attribute_assignment(self, sample_model_config) -> None:
        config = ModelConfig(**sample_model_config)

        with pytest.raises(ValueError, match=r"Unknown attribute: 'unknown'\."):
            setattr(config, "unknown", "value")


class TestSegmentationTrainConfigNumSelect:
    """Unit tests for SegmentationTrainConfig.num_select default and per-model values."""

    def test_defaults_to_none(self) -> None:
        config = SegmentationTrainConfig(dataset_dir="/tmp")
        assert config.num_select is None

    def test_explicit_value_is_accepted(self) -> None:
        config = SegmentationTrainConfig(dataset_dir="/tmp", num_select=42)
        assert config.num_select == 42

    @pytest.mark.parametrize(
        "config_class, expected_num_select",
        [
            (RFDETRSegNanoConfig, 100),
            (RFDETRSegSmallConfig, 100),
            (RFDETRSegMediumConfig, 200),
            (RFDETRSegLargeConfig, 200),
            (RFDETRSegXLargeConfig, 300),
            (RFDETRSeg2XLargeConfig, 300),
        ],
    )
    def test_model_config_has_variant_specific_num_select(self, config_class, expected_num_select) -> None:
        assert config_class().num_select == expected_num_select


def _make_rfdetr_stub(model_config):
    """Build a minimal RFDETR-like object for testing train_from_config() without loading weights."""
    stub = RFDETR.__new__(RFDETR)
    stub.model_config = model_config
    stub.model = MagicMock()
    stub.model.class_names = []
    stub.callbacks = {
        "on_fit_epoch_end": [],
        "on_train_end": [],
    }
    return stub


class TestSegmentationNumSelectMerge:
    """
    Verify that SegmentationTrainConfig with num_select=None does not override the
    model-specific num_select during train_from_config() merging.

    The bug: when num_select=None in SegmentationTrainConfig, the merge loop in
    train_from_config() removes num_select from model_config and then spreads the
    None value, producing num_select=None in all_kwargs.

    The fix: None values in train_config should not override model_config values.
    """

    @pytest.mark.parametrize(
        "model_config_cls, expected_num_select",
        [
            (RFDETRSegNanoConfig, 100),
            (RFDETRSegSmallConfig, 100),
            (RFDETRSegXLargeConfig, 300),
        ],
    )
    def test_none_num_select_preserves_model_value(self, model_config_cls, expected_num_select, tmp_path) -> None:
        """num_select should come from the model config when not set in the train config."""
        stub = _make_rfdetr_stub(model_config_cls())
        train_config = SegmentationTrainConfig(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            dataset_file="coco",
            tensorboard=False,
        )
        assert train_config.num_select is None

        stub.train_from_config(train_config)

        call_kwargs = stub.model.train.call_args.kwargs
        assert call_kwargs["num_select"] == expected_num_select, (
            f"Expected num_select={expected_num_select} from {model_config_cls.__name__}, "
            f"got {call_kwargs['num_select']}. "
            "SegmentationTrainConfig.num_select=None must not override the model's value."
        )

    def test_explicit_num_select_overrides_model_value(self, tmp_path) -> None:
        """When the user sets num_select explicitly it should win over the model default."""
        stub = _make_rfdetr_stub(RFDETRSegNanoConfig())  # model default: 100
        train_config = SegmentationTrainConfig(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            dataset_file="coco",
            tensorboard=False,
            num_select=42,
        )

        stub.train_from_config(train_config)

        call_kwargs = stub.model.train.call_args.kwargs
        assert call_kwargs["num_select"] == 42

    def test_segmentation_raises_when_square_resize_disabled(self, tmp_path) -> None:
        """Segmentation training must not silently override square_resize_div_64=False; it must raise ValueError."""
        stub = _make_rfdetr_stub(RFDETRSegNanoConfig())
        train_config = SegmentationTrainConfig(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            dataset_file="coco",
            tensorboard=False,
            square_resize_div_64=False,
        )

        with pytest.raises(ValueError, match="square_resize_div_64"):
            stub.train_from_config(train_config)
