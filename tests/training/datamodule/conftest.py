# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pytest fixtures shared by the RFDETRDataModule tests in tests/training/datamodule/.

Plain builders live in :mod:`tests.training.helpers`; this module only wraps them as fixtures.
"""

import pytest

from tests.training.helpers import _base_model_config, _build_datamodule, _datamodule_train_config


@pytest.fixture
def fixture_training_setup():
    """Return the default model config, train config, and DataModule together.

    Examples:
        >>> model_config, train_config, datamodule = fixture_training_setup()  # doctest: +SKIP
        # A pytest fixture; request it by name instead of calling it.
    """
    model_config = _base_model_config()
    train_config = _datamodule_train_config()
    datamodule = _build_datamodule(model_config, train_config)
    return model_config, train_config, datamodule
