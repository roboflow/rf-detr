# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Data types for automatic tune-preview parameter optimization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurrentParameters:
    """Snapshot of user-controlled inference parameters."""

    threshold: float = 0.6
    keypoint_threshold: float = 0.25
    motion_max_speed_fraction: float = 0.20
    motion_ema_alpha: float = 0.45
    motion_filter_enabled: bool = True
    motion_oscillation_enabled: bool = True
    ellipse_sigma: float = 1.5
    heatmap_opacity: float = 0.38


DEFAULT_PARAMETERS = CurrentParameters()


@dataclass
class AnomalyFlags:
    """Detected quality anomalies in a tune-preview cache."""

    excess_person_detections: bool = False
    unstable_person_count: bool = False
    high_low_confidence_ratio: bool = False
    low_mean_confidence: bool = False
    high_motion_rejection_rate: bool = False
    high_centroid_jump_rate: bool = False
    high_covariance_spread: bool = False


@dataclass
class CacheQualityMetrics:
    """Metrics derived from cached raw inference (no model re-run)."""

    frames: int
    avg_persons: float
    person_count_std: float
    person_count_min: int
    person_count_max: int
    low_confidence_ratio: float
    mean_joint_confidence: float
    motion_speed_rejections: int
    motion_oscillation_corrections: int
    rejection_rate_per_joint: float
    centroid_jump_rate: float
    covariance_spread_ratio: float
    anomalies: AnomalyFlags = field(default_factory=AnomalyFlags)


@dataclass
class ProposedParameters:
    """Optimized parameter set with human-readable rationale."""

    threshold: float
    keypoint_threshold: float
    motion_max_speed_fraction: float
    motion_ema_alpha: float
    motion_filter_enabled: bool
    motion_oscillation_enabled: bool
    ellipse_sigma: float
    heatmap_opacity: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class AutoTuneEffectiveness:
    """Before/after simulation comparing current vs proposed parameters."""

    recommended: bool
    confidence: float
    before_rejection_rate: float
    after_rejection_rate: float
    before_person_std: float
    after_person_std: float
    rejection_improvement_pct: float
    person_stability_improvement_pct: float
    summary: str
