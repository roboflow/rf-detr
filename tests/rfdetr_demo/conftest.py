# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared pytest fixtures for rfdetr_demo tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def frame_bgr() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)
