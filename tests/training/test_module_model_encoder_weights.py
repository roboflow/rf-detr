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


@pytest.mark.parametrize(("load_encoder_weights", "force_no_pretrain"), [(True, False), (False, True)])
def test_encoder_weights_flag_reaches_the_builder(
    load_encoder_weights: bool, force_no_pretrain: bool, tmp_path
) -> None:
    model_config = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
    train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=MagicMock()) as build,
        patch("rfdetr.training.module_model.build_criterion_from_config", return_value=(MagicMock(), MagicMock())),
    ):
        RFDETRModelModule(model_config, train_config, load_encoder_weights=load_encoder_weights)

    assert build.call_args.kwargs["defaults"].force_no_pretrain is force_no_pretrain
