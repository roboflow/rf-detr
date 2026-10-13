# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``RFDETRModelModule(load_encoder_weights=False)`` builds without fetching upstream encoder weights."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule


class TestLoadEncoderWeights:
    """``load_encoder_weights`` maps onto the builder's ``force_no_pretrain`` default."""

    @pytest.mark.parametrize(("load_encoder_weights", "force_no_pretrain"), [(True, False), (False, True)])
    def test_flag_reaches_the_builder(self, load_encoder_weights: bool, force_no_pretrain: bool, tmp_path) -> None:
        """The module hands the inverted flag to ``build_model_from_config`` through its defaults."""
        model_config = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=MagicMock()) as build,
            patch("rfdetr.training.module_model.build_criterion_from_config", return_value=(MagicMock(), MagicMock())),
        ):
            RFDETRModelModule(model_config, train_config, load_encoder_weights=load_encoder_weights)

        assert build.call_args.kwargs["defaults"].force_no_pretrain is force_no_pretrain
