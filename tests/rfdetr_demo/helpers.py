# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for rfdetr_demo tests."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.inference.tune_cache import TunePreviewCache


def synthetic_key_points(
    *,
    num_persons: int,
    confidence: float,
    offset_x: float,
) -> sv.KeyPoints:
    xy = np.zeros((num_persons, 17, 2), dtype=np.float32)
    det_conf = np.full((num_persons,), confidence, dtype=np.float32)
    kp_conf = np.full((num_persons, 17), confidence, dtype=np.float32)
    visible = np.ones((num_persons, 17), dtype=bool)
    for det in range(num_persons):
        xy[det, 0] = (100.0 + offset_x + det * 40, 200.0)
    covariance = np.full((num_persons, 17, 2, 2), 25.0, dtype=np.float32)
    return sv.KeyPoints(
        xy=xy,
        visible=visible,
        keypoint_confidence=kp_conf,
        detection_confidence=det_conf,
        data={"covariance": covariance},
    )


def single_person_keypoints(*, x: float, y: float) -> sv.KeyPoints:
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    xy[0, 0] = (x, y)
    visible = np.ones((1, 17), dtype=bool)
    return sv.KeyPoints(
        xy=xy,
        visible=visible,
        keypoint_confidence=np.full((1, 17), 0.9, dtype=np.float32),
        detection_confidence=np.array([0.9], dtype=np.float32),
    )


def populate_tune_cache(
    cache: TunePreviewCache,
    *,
    person_counts: list[int],
    confidence: float = 0.8,
    frame_bgr: np.ndarray | None = None,
) -> None:
    bgr = frame_bgr if frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    for index, count in enumerate(person_counts):
        key_points = synthetic_key_points(
            num_persons=count,
            confidence=confidence,
            offset_x=float(index * 2),
        )
        cache.append_keypoint(
            frame_bgr=bgr,
            key_points=key_points,
            frame_index=index,
            processed_count=index + 1,
        )
