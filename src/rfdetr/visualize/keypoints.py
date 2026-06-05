# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private keypoint visualization helpers for RF-DETR demos and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from supervision import KeyPoints


def _copy_key_points(key_points: KeyPoints) -> KeyPoints:
    """Return a mutable copy of a Supervision ``KeyPoints`` object."""
    return KeyPoints(
        xy=key_points.xy.copy(),
        keypoint_confidence=(
            key_points.keypoint_confidence.copy() if key_points.keypoint_confidence is not None else None
        ),
        detection_confidence=(
            key_points.detection_confidence.copy() if key_points.detection_confidence is not None else None
        ),
        visible=key_points.visible.copy() if key_points.visible is not None else None,
        class_id=key_points.class_id.copy() if key_points.class_id is not None else None,
        data=dict(key_points.data),
    )


def _precision_cholesky_to_pixel_covariance(
    precision_cholesky: np.ndarray,
    source_shape: np.ndarray,
) -> np.ndarray:
    """Convert RF-DETR keypoint precision parameters into pixel covariances.

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
        >>> _precision_cholesky_to_pixel_covariance(precision, shape)[0, 0]
        array([[400.,   0.],
               [  0., 100.]], dtype=float32)
    """
    if precision_cholesky.ndim != 3 or precision_cholesky.shape[2] != 3:
        raise ValueError(f"precision_cholesky must have shape (N, K, 3), got {precision_cholesky.shape}.")
    if source_shape.shape != (precision_cholesky.shape[0], 2):
        raise ValueError(f"source_shape must have shape ({precision_cholesky.shape[0]}, 2), got {source_shape.shape}.")

    covariances = np.full((*precision_cholesky.shape[:2], 2, 2), np.nan, dtype=np.float32)
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


def _key_points_for_display(
    key_points: KeyPoints,
    *,
    keypoint_threshold: float = 0.0,
) -> KeyPoints:
    """Build ``KeyPoints`` for visualization from RF-DETR keypoint predictions.

    Args:
        key_points: RF-DETR keypoint prediction output.
        keypoint_threshold: Per-keypoint confidence threshold used to hide
            low-confidence points through ``key_points.visible``.

    Returns:
        A Supervision ``KeyPoints`` object ready for keypoint annotators.

    Raises:
        ValueError: If keypoints are present without per-point confidence.
    """
    key_points = _copy_key_points(key_points)

    if len(key_points) == 0:
        return KeyPoints.empty()
    if key_points.keypoint_confidence is None:
        raise ValueError("Expected RF-DETR keypoints to include per-keypoint confidence.")

    raw_precision = key_points.data.get("keypoint_precision_cholesky")
    raw_source_shape = key_points.data.get("source_shape")
    if raw_precision is not None and raw_source_shape is not None:
        precision = np.asarray(raw_precision, dtype=np.float32)
        source_shape = np.asarray(raw_source_shape, dtype=np.float32)
        if precision.shape[:2] == key_points.xy.shape[:2] and source_shape.shape == (len(key_points), 2):
            key_points.data["covariance"] = _precision_cholesky_to_pixel_covariance(
                precision_cholesky=precision,
                source_shape=source_shape,
            )

    visible = key_points.keypoint_confidence >= keypoint_threshold
    key_points.visible = visible if key_points.visible is None else key_points.visible & visible
    return key_points


def _keypoint_prediction_records(
    key_points: KeyPoints,
    *,
    image: str | Path | None = None,
    keypoint_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Build flat keypoint prediction rows for notebook or terminal display.

    Args:
        key_points: RF-DETR keypoint prediction output.
        image: Optional image identifier added to each output row. Paths are
            represented by their file name.
        keypoint_threshold: Per-keypoint confidence threshold used when
            ``key_points.visible`` is not already populated.

    Returns:
        A list of row dictionaries, one per visible/non-zero keypoint.
    """
    if key_points.keypoint_confidence is None:
        return []

    image_name = Path(image).name if image is not None else None
    visible = (
        key_points.visible if key_points.visible is not None else key_points.keypoint_confidence >= keypoint_threshold
    )
    class_names = key_points.data.get("class_name")
    records: list[dict[str, Any]] = []
    for detection_index, xy in enumerate(key_points.xy):
        for keypoint_index, (point, confidence, is_visible) in enumerate(
            zip(xy, key_points.keypoint_confidence[detection_index], visible[detection_index], strict=True)
        ):
            if not is_visible or np.allclose(point, 0):
                continue
            class_name = None
            if isinstance(class_names, np.ndarray) and detection_index < len(class_names):
                class_name = str(class_names[detection_index])
            records.append(
                {
                    "image": image_name,
                    "detection_index": detection_index,
                    "class_id": int(key_points.class_id[detection_index]) if key_points.class_id is not None else None,
                    "class_name": class_name,
                    "detection_confidence": (
                        float(key_points.detection_confidence[detection_index])
                        if key_points.detection_confidence is not None
                        else None
                    ),
                    "keypoint_index": keypoint_index,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "keypoint_confidence": float(confidence),
                }
            )
    return records
