# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for the models test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_torch_safe_globals():
    """Reset torch serialization safe globals after each test.

    Prevents cross-test state contamination caused by ``_safe_torch_load``'s Attempt 2 path, which calls
    ``torch.serialization.add_safe_globals``. Without this reset, globals registered by one test bleed into subsequent
    tests and can mask trust-gate failures.
    """
    yield
    try:
        torch.serialization.clear_safe_globals()
    except AttributeError:
        pass  # torch <2.4 does not have clear_safe_globals
