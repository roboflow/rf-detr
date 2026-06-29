# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for auto-tune parameter optimization."""

from __future__ import annotations

from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.tuning.auto_tune import (
    DEFAULT_PARAMETERS,
    analyze_tune_cache,
    propose_parameters,
    run_auto_tune,
)
from tests.rfdetr_demo.helpers import populate_tune_cache


def test_propose_raises_threshold_for_excess_persons() -> None:
    cache = TunePreviewCache(task="keypoint", fps=30.0, frame_stride=1)
    populate_tune_cache(cache, person_counts=[6, 5, 7, 6, 5])
    metrics = analyze_tune_cache(cache, current=DEFAULT_PARAMETERS)
    proposed = propose_parameters(metrics, current=DEFAULT_PARAMETERS)
    assert proposed.threshold > DEFAULT_PARAMETERS.threshold
    assert any("検出人数過多" in reason for reason in proposed.reasons)


def test_propose_lowers_threshold_for_under_count_instability() -> None:
    from rfdetr_demo.tuning.auto_tune_types import AnomalyFlags, CacheQualityMetrics

    metrics = CacheQualityMetrics(
        frames=10,
        avg_persons=4.5,
        person_count_std=0.9,
        person_count_min=4,
        person_count_max=5,
        low_confidence_ratio=0.05,
        mean_joint_confidence=0.8,
        motion_speed_rejections=0,
        motion_oscillation_corrections=0,
        rejection_rate_per_joint=0.0,
        centroid_jump_rate=0.0,
        covariance_spread_ratio=1.0,
        anomalies=AnomalyFlags(
            excess_person_detections=False,
            unstable_person_count=True,
        ),
    )
    proposed = propose_parameters(metrics, current=DEFAULT_PARAMETERS)
    assert proposed.threshold < DEFAULT_PARAMETERS.threshold
    assert any("不足側" in reason for reason in proposed.reasons)


def test_analyze_tune_cache_includes_stabilized_metrics() -> None:
    cache = TunePreviewCache(task="keypoint", fps=30.0, frame_stride=1)
    populate_tune_cache(cache, person_counts=[2, 2, 1, 2, 2])
    metrics = analyze_tune_cache(cache, current=DEFAULT_PARAMETERS)
    assert metrics.frames == 5
    assert metrics.stabilized_person_count_std >= 0.0
    assert 0.0 <= metrics.track_break_rate <= 1.0


def test_propose_lowers_threshold_for_high_track_break_rate() -> None:
    from rfdetr_demo.tuning.auto_tune_types import AnomalyFlags, CacheQualityMetrics

    metrics = CacheQualityMetrics(
        frames=10,
        avg_persons=4.5,
        person_count_std=0.5,
        person_count_min=4,
        person_count_max=5,
        low_confidence_ratio=0.05,
        mean_joint_confidence=0.8,
        motion_speed_rejections=0,
        motion_oscillation_corrections=0,
        rejection_rate_per_joint=0.0,
        centroid_jump_rate=0.0,
        covariance_spread_ratio=1.0,
        track_break_rate=0.45,
        anomalies=AnomalyFlags(high_track_break_rate=True),
    )
    proposed = propose_parameters(metrics, current=DEFAULT_PARAMETERS)
    assert proposed.threshold < DEFAULT_PARAMETERS.threshold
    assert any("トラック途切れ" in reason for reason in proposed.reasons)


def test_auto_tune_recommends_when_rejection_high() -> None:
    cache = TunePreviewCache(task="keypoint", fps=60.0, frame_stride=1)
    counts = [5, 7, 4, 6, 5, 7, 6, 5, 6, 4] * 3
    populate_tune_cache(cache, person_counts=counts, confidence=0.45)
    proposed, metrics, effectiveness = run_auto_tune(cache, current=DEFAULT_PARAMETERS)
    assert metrics.frames == len(counts)
    assert proposed.threshold >= DEFAULT_PARAMETERS.threshold
    assert isinstance(effectiveness.recommended, bool)
