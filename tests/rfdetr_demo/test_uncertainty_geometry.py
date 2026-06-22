# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for uncertainty geometry helpers."""

from __future__ import annotations

import numpy as np

from rfdetr_demo.inference.uncertainty.geometry import (
    clamp_ellipse_axes,
    resolve_max_ellipse_axis,
)


def test_resolve_max_ellipse_axis_user_value() -> None:
    assert resolve_max_ellipse_axis(48.0, frame_width=1920, frame_height=1080) == 48.0


def test_resolve_max_ellipse_axis_auto_cap() -> None:
    cap = resolve_max_ellipse_axis(None, frame_width=1920, frame_height=1080)
    assert 24.0 <= cap <= 36.0


def test_clamp_ellipse_axes_respects_max() -> None:
    eigenvalues = np.array([100.0, 50.0], dtype=np.float64)
    ax, ay = clamp_ellipse_axes(eigenvalues, sigma=2.0, max_axis=10.0)
    assert ax <= 10
    assert ay <= 10
    assert ax >= 1
    assert ay >= 1
