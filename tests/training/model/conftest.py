# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pytest fixtures shared by the RFDETRModelModule tests in tests/training/model/.

Plain builders live in :mod:`tests.training.helpers`; this module only wraps them as fixtures.
"""

import pytest

from tests.training.helpers import _build_module


@pytest.fixture
def build_module(tmp_path):
    """Factory fixture — returns (module, fake_model, fake_criterion, fake_postprocess).

    build_model and build_criterion_and_postprocessors are mocked automatically. tmp_path is injected automatically so
    test methods do not need to declare it.

    Examples:
        >>> module, model, criterion, postprocess = build_module()  # doctest: +SKIP
        # Needs pytest's tmp_path fixture; cannot run standalone.
    """
    return lambda model_config=None, train_config=None: _build_module(model_config, train_config, tmp_path)
