# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for tune pipeline integration."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS
from rfdetr_demo.tuning.pipeline import run_tune_pipeline


def _synthetic_key_points(*, num_persons: int, confidence: float) -> sv.KeyPoints:
    xy = np.zeros((num_persons, 17, 2), dtype=np.float32)
    det_conf = np.full((num_persons,), confidence, dtype=np.float32)
    kp_conf = np.full((num_persons, 17), confidence, dtype=np.float32)
    visible = np.ones((num_persons, 17), dtype=bool)
    for det in range(num_persons):
        xy[det, 0] = (100.0 + det * 40, 200.0)
    covariance = np.full((num_persons, 17, 2, 2), 25.0, dtype=np.float32)
    return sv.KeyPoints(
        xy=xy,
        visible=visible,
        keypoint_confidence=kp_conf,
        detection_confidence=det_conf,
        data={"covariance": covariance},
    )


def test_run_tune_pipeline_proposes_for_unstable_counts() -> None:
    cache = TunePreviewCache(task="keypoint", fps=30.0, frame_stride=1)
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    for index, count in enumerate([6, 5, 7, 6, 5]):
        cache.append_keypoint(
            frame_bgr=frame_bgr,
            key_points=_synthetic_key_points(num_persons=count, confidence=0.8),
            frame_index=index,
            processed_count=index + 1,
        )
    result = run_tune_pipeline(cache, current=DEFAULT_PARAMETERS)
    assert result.metrics.frames == 5
    assert result.proposed.threshold >= DEFAULT_PARAMETERS.threshold
    assert isinstance(result.effectiveness.recommended, bool)
