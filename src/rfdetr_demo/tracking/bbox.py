# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bounding-box geometry and NMS helpers for person tracking."""

from __future__ import annotations

import numpy as np
import supervision as sv


def keypoints_xyxy(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    """Return axis-aligned bbox [x1,y1,x2,y2] for one detection."""
    xy = key_points.xy[detection_index]
    visible = key_points.visible
    if visible is not None:
        mask = visible[detection_index]
        points = xy[mask]
    else:
        points = xy[~np.all(np.isclose(xy, 0), axis=1)]
    if len(points) == 0:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float64,
    )


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union for two axis-aligned boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def hungarian_maximize(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return row/col pairs maximizing total cost (greedy fallback for small matrices)."""
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-cost)
        return list(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
    except ImportError:
        pairs: list[tuple[int, int]] = []
        remaining_rows = set(range(cost.shape[0]))
        remaining_cols = set(range(cost.shape[1]))
        while remaining_rows and remaining_cols:
            best: tuple[int, int, float] | None = None
            for row in remaining_rows:
                for col in remaining_cols:
                    value = float(cost[row, col])
                    if best is None or value > best[2]:
                        best = (row, col, value)
            if best is None or best[2] <= 0:
                break
            pairs.append((best[0], best[1]))
            remaining_rows.remove(best[0])
            remaining_cols.remove(best[1])
        return pairs


def detection_bbox(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    """Return axis-aligned bbox for one detection, preferring ``data['xyxy']``."""
    xyxy = key_points.data.get("xyxy") if key_points.data else None
    if xyxy is not None and len(xyxy) > detection_index:
        box = np.asarray(xyxy[detection_index], dtype=np.float64)
        if box.shape == (4,):
            return box
    return keypoints_xyxy(key_points, detection_index)


def detection_confidence(key_points: sv.KeyPoints, detection_index: int) -> float:
    """Return model detection confidence for one row."""
    scores = key_points.detection_confidence
    if scores is None or len(scores) <= detection_index:
        return 0.0
    return float(scores[detection_index])


def nms_detection_indices(key_points: sv.KeyPoints, iou_threshold: float) -> list[int]:
    """Return detection indices to keep after confidence-sorted IoU-NMS."""
    num_detections = len(key_points)
    if num_detections <= 1:
        return list(range(num_detections))

    candidates: list[tuple[int, np.ndarray, float]] = []
    for detection_index in range(num_detections):
        box = detection_bbox(key_points, detection_index)
        if box is None:
            continue
        candidates.append((detection_index, box, detection_confidence(key_points, detection_index)))

    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0][0]]

    order = sorted(
        range(len(candidates)),
        key=lambda row: candidates[row][2],
        reverse=True,
    )
    keep_indices: list[int] = []
    suppressed: set[int] = set()
    for row in order:
        if row in suppressed:
            continue
        keep_indices.append(candidates[row][0])
        box_a = candidates[row][1]
        for other_row in order:
            if other_row == row or other_row in suppressed:
                continue
            if iou(box_a, candidates[other_row][1]) > iou_threshold:
                suppressed.add(other_row)
    keep_indices.sort()
    return keep_indices


# Backward-compatible private aliases
_keypoints_xyxy = keypoints_xyxy
_iou = iou
_hungarian_maximize = hungarian_maximize
