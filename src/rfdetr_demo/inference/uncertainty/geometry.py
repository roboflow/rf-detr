# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Geometry helpers for keypoint uncertainty ellipses."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from rfdetr_demo.inference.uncertainty.constants import (
    COCO17_KEYPOINT_NAMES,
    DEFAULT_UNCERTAINTY_MAX_AXIS_PX,
)


def joint_index_to_bgr(joint_index: int, num_joints: int = len(COCO17_KEYPOINT_NAMES)) -> tuple[int, int, int]:
    """Map a keypoint index to a distinct BGR color on the hue wheel."""
    if num_joints <= 0:
        raise ValueError(f"num_joints must be positive, got {num_joints}")
    hue = int((180 * joint_index) / num_joints) % 180
    hsv_pixel = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def resolve_max_ellipse_axis(
    user_value: float | None,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    """Return a pixel cap for uncertainty ellipse axes (subdued, human-scale motion)."""
    if user_value is not None and user_value > 0:
        return float(user_value)
    frame_min = min(frame_width, frame_height)
    auto_cap = max(24.0, 0.03 * frame_min)
    return min(DEFAULT_UNCERTAINTY_MAX_AXIS_PX, auto_cap)


def clamp_ellipse_axes(
    eigenvalues: npt.NDArray[np.float64],
    *,
    sigma: float,
    max_axis: float,
) -> tuple[int, int]:
    """Convert eigenvalues to capped integer ellipse radii in pixels."""
    axes = sigma * np.sqrt(np.maximum(eigenvalues, 0.0))
    axes = np.minimum(axes, max_axis)
    return max(1, round(float(axes[0]))), max(1, round(float(axes[1])))


def covariance_trace(covariance: npt.NDArray[np.float32]) -> float:
    """Return the trace of a 2x2 covariance matrix."""
    return float(covariance[0, 0] + covariance[1, 1])


def decompose_covariance(
    covariance: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float] | None:
    """Eigendecompose a 2x2 covariance matrix for ellipse drawing."""
    if not np.isfinite(covariance).all():
        return None
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        return None
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    return eigenvalues, eigenvectors, angle
