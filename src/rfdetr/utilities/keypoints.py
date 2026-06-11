# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint utility functions shared by inference and visualization."""

from __future__ import annotations

import numpy as np

__all__ = ["precision_cholesky_to_pixel_covariance"]


def precision_cholesky_to_pixel_covariance(
    precision_cholesky: np.ndarray,
    source_shape: np.ndarray,
) -> np.ndarray:
    """Convert RF-DETR keypoint precision parameters into pixel covariances.

    The keypoint head predicts lower-triangular precision-Cholesky parameters in
    normalized image coordinates. This helper inverts those precision matrices
    and scales them to pixel coordinates for Supervision covariance annotators.

    Args:
        precision_cholesky: Lower-triangular precision parameters with shape
            ``(N, K, 3)``. Each triplet is ``(log_l11, l21, log_l22)``.
        source_shape: Per-detection ``(height, width)`` rows with shape
            ``(N, 2)``.

    Returns:
        Pixel-space covariance matrices with shape ``(N, K, 2, 2)``.

    Raises:
        ValueError: If ``precision_cholesky`` or ``source_shape`` has an
            incompatible shape.

    Example:
        >>> precision = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        >>> shape = np.array([[10.0, 20.0]], dtype=np.float32)
        >>> precision_cholesky_to_pixel_covariance(precision, shape)[0, 0]
        array([[400.,   0.],
               [  0., 100.]], dtype=float32)
    """
    if precision_cholesky.ndim != 3 or precision_cholesky.shape[2] != 3:
        raise ValueError(f"precision_cholesky must have shape (N, K, 3), got {precision_cholesky.shape}.")
    if source_shape.shape != (precision_cholesky.shape[0], 2):
        raise ValueError(f"source_shape must have shape ({precision_cholesky.shape[0]}, 2), got {source_shape.shape}.")

    covariances = np.full((*precision_cholesky.shape[:2], 2, 2), float("nan"), dtype=np.float32)
    for detection_index, detection_precision in enumerate(precision_cholesky):
        height, width = source_shape[detection_index]
        scale = np.diag([width, height]).astype(np.float64)
        for keypoint_index, params in enumerate(detection_precision):
            if not np.isfinite(params).all():
                continue
            log_l11 = float(params[0])
            l21 = float(params[1])
            log_l22 = float(params[2])
            l11 = float(np.exp(log_l11))
            l22 = float(np.exp(log_l22))
            precision = np.array(
                [[l11 * l11, l11 * l21], [l11 * l21, l21 * l21 + l22 * l22]],
                dtype=np.float64,
            )
            try:
                covariance = np.linalg.inv(precision)
            except np.linalg.LinAlgError:
                continue
            pixel_covariance = scale @ covariance @ scale
            if np.isfinite(pixel_covariance).all():
                covariances[detection_index, keypoint_index] = pixel_covariance
    return covariances
