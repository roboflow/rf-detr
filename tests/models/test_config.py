# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from rfdetr.config import ModelConfig


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
