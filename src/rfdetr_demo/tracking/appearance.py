# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lightweight CPU appearance descriptors (color histograms) for ReID.

The descriptor is an HSV hue-saturation histogram of a person's torso region,
usable to re-associate tracks after long occlusions without an embedding model.
OpenCV is imported lazily so the tracking package stays importable without it;
only the ReID code path requires ``cv2``.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

# COCO-17 torso joints: left/right shoulder, left/right hip.
_TORSO_JOINTS: tuple[int, ...] = (5, 6, 11, 12)


def _visible_torso_points(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    """Return visible torso joint coordinates for one detection, or None."""
    xy = key_points.xy[detection_index]
    if key_points.visible is not None:
        visible = key_points.visible[detection_index]
    else:
        visible = ~np.all(np.isclose(xy, 0), axis=1)
    joints = [index for index in _TORSO_JOINTS if index < len(xy) and visible[index]]
    if len(joints) < 2:
        return None
    return xy[joints]


def appearance_roi(
    key_points: sv.KeyPoints,
    detection_index: int,
    person_box: np.ndarray,
) -> np.ndarray | None:
    """Return an ``[x1, y1, x2, y2]`` torso ROI, preferring keypoints over the box.

    Falls back to the central band of ``person_box`` (horizontally centered,
    vertical 20-65%) when torso joints are not visible, which keeps most of the
    background out of the color histogram.

    Args:
        key_points: Detections for the current frame.
        detection_index: Row to build an ROI for.
        person_box: Full-person ``[x1, y1, x2, y2]`` fallback box.

    Returns:
        A float ROI box, or None when it would be degenerate.
    """
    points = _visible_torso_points(key_points, detection_index)
    if points is not None:
        x1 = float(points[:, 0].min())
        y1 = float(points[:, 1].min())
        x2 = float(points[:, 0].max())
        y2 = float(points[:, 1].max())
        if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
            return np.array([x1, y1, x2, y2], dtype=np.float64)

    bx1, by1, bx2, by2 = (float(value) for value in person_box)
    width = bx2 - bx1
    height = by2 - by1
    if width < 2.0 or height < 2.0:
        return None
    return np.array(
        [
            bx1 + 0.25 * width,
            by1 + 0.20 * height,
            bx2 - 0.25 * width,
            by1 + 0.65 * height,
        ],
        dtype=np.float64,
    )


def appearance_histogram(
    frame_bgr: np.ndarray,
    roi: np.ndarray,
    *,
    hue_bins: int = 8,
    saturation_bins: int = 8,
) -> np.ndarray | None:
    """Return an L1-normalized HSV hue-saturation histogram for an ROI crop.

    Args:
        frame_bgr: Full frame in BGR order (as delivered by OpenCV capture).
        roi: ``[x1, y1, x2, y2]`` region to describe.
        hue_bins: Number of hue bins.
        saturation_bins: Number of saturation bins.

    Returns:
        A flattened ``hue_bins * saturation_bins`` descriptor summing to 1, or
        None when the crop is empty or fully unsaturated.
    """
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "Appearance ReID requires OpenCV. Install it via the demo extras "
            "(e.g. `uv sync --all-groups`) or set RFDETR_TRACK_REID=0."
        ) from error

    height, width = frame_bgr.shape[:2]
    x1 = max(0, min(width, int(round(float(roi[0])))))
    y1 = max(0, min(height, int(round(float(roi[1])))))
    x2 = max(0, min(width, int(round(float(roi[2])))))
    y2 = max(0, min(height, int(round(float(roi[3])))))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [hue_bins, saturation_bins],
        [0, 180, 0, 256],
    )
    total = float(histogram.sum())
    if total <= 0.0:
        return None
    return (histogram.flatten() / total).astype(np.float64)


def histogram_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Return histogram-intersection similarity in ``[0, 1]``.

    Both inputs are assumed L1-normalized; missing descriptors score 0.
    """
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(np.minimum(left, right).sum())
