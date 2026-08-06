# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared types for person tracking pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import supervision as sv

TRACK_IS_GHOST_KEY = "track_is_ghost"
TRACK_ID_KEY = "track_id"


@dataclass(frozen=True)
class PersonTrackSettings:
    """Parameters for NMS, association, hold, and optional center sticky."""

    enabled: bool = True
    nms_iou_threshold: float = 0.50
    match_iou_threshold: float = 0.15
    max_missed: int = 2
    max_tracks: int = 32
    hysteresis_enabled: bool = True
    new_track_min_confidence: float = 0.65
    sticky_center_track: bool = False
    sticky_max_missed: int = 4
    center_x_fraction: tuple[float, float] = (0.28, 0.48)
    expected_person_count: int = 0
    fill_below_expected: bool = True
    fill_extra_missed: int = 3
    motion_enabled: bool = True
    motion_smoothing: float = 0.5
    motion_max_speed: float = 120.0
    motion_gate_factor: float = 1.5
    reid_enabled: bool = False
    reid_weight: float = 0.3
    reid_similarity_threshold: float = 0.5
    reid_max_gallery_frames: int = 60
    reid_ema: float = 0.9
    reid_backend: str = "histogram"
    reid_model_path: str | None = None


@dataclass
class TrackPipelineStats:
    """Per-frame counters updated by the tracking pipeline."""

    raw_count: int = 0
    nms_count: int = 0
    active_track_count: int = 0
    ghost_count: int = 0


@dataclass
class TrackDiagnostic:
    """One stabilized person instance for audit / debugging."""

    track_id: int
    cx: float
    cy: float
    confidence: float
    is_ghost: bool
    missed: int
    matched_this_frame: bool


@dataclass
class TrackPipelineResult:
    """Output of one tracking pipeline pass."""

    key_points: sv.KeyPoints
    stats: TrackPipelineStats
    diagnostics: list[TrackDiagnostic] = field(default_factory=list)


@dataclass
class TrackedKeyPoints:
    """KeyPoints with pipeline metadata (alias-friendly wrapper)."""

    key_points: sv.KeyPoints
    stats: TrackPipelineStats
    diagnostics: list[TrackDiagnostic] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: TrackPipelineResult) -> TrackedKeyPoints:
        return cls(
            key_points=result.key_points,
            stats=result.stats,
            diagnostics=result.diagnostics,
        )


def is_person_track_enabled() -> bool:
    """Return False when ``RFDETR_DETECTION_STABILIZER=0``."""
    return os.environ.get("RFDETR_DETECTION_STABILIZER", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def person_track_settings_from_env(
    *,
    base: PersonTrackSettings | None = None,
) -> PersonTrackSettings:
    """Build settings from environment overrides."""
    settings = base or PersonTrackSettings()
    max_missed_raw = os.environ.get("RFDETR_MAX_MISSED")
    hysteresis_raw = os.environ.get("RFDETR_TRACK_HYSTERESIS", "").strip().lower()
    new_track_conf_raw = os.environ.get("RFDETR_NEW_TRACK_MIN_CONFIDENCE")
    sticky_raw = os.environ.get("RFDETR_STICKY_CENTER_TRACK", "").strip().lower()
    sticky_max_raw = os.environ.get("RFDETR_STICKY_MAX_MISSED")
    expected_raw = os.environ.get("RFDETR_EXPECTED_PERSON_COUNT")
    fill_extra_raw = os.environ.get("RFDETR_FILL_EXTRA_MISSED")
    fill_below_raw = os.environ.get("RFDETR_FILL_BELOW_EXPECTED", "").strip().lower()
    motion_raw = os.environ.get("RFDETR_TRACK_MOTION", "").strip().lower()
    motion_smoothing_raw = os.environ.get("RFDETR_MOTION_SMOOTHING")
    motion_max_speed_raw = os.environ.get("RFDETR_MOTION_MAX_SPEED")
    motion_gate_factor_raw = os.environ.get("RFDETR_MOTION_GATE_FACTOR")
    reid_raw = os.environ.get("RFDETR_TRACK_REID", "").strip().lower()
    reid_weight_raw = os.environ.get("RFDETR_REID_WEIGHT")
    reid_similarity_raw = os.environ.get("RFDETR_REID_SIMILARITY")
    reid_gallery_frames_raw = os.environ.get("RFDETR_REID_GALLERY_FRAMES")
    reid_ema_raw = os.environ.get("RFDETR_REID_EMA")
    reid_backend_raw = os.environ.get("RFDETR_REID_BACKEND", "").strip().lower()
    reid_model_raw = os.environ.get("RFDETR_REID_MODEL")
    kwargs: dict[str, object] = {}
    if max_missed_raw is not None:
        kwargs["max_missed"] = max(0, int(max_missed_raw))
    if hysteresis_raw in {"1", "true", "yes", "on"}:
        kwargs["hysteresis_enabled"] = True
    elif hysteresis_raw in {"0", "false", "no", "off"}:
        kwargs["hysteresis_enabled"] = False
    if new_track_conf_raw is not None:
        kwargs["new_track_min_confidence"] = max(0.0, min(1.0, float(new_track_conf_raw)))
    if sticky_raw in {"1", "true", "yes", "on"}:
        kwargs["sticky_center_track"] = True
    elif sticky_raw in {"0", "false", "no", "off"}:
        kwargs["sticky_center_track"] = False
    if sticky_max_raw is not None:
        kwargs["sticky_max_missed"] = max(1, int(sticky_max_raw))
    if expected_raw is not None:
        kwargs["expected_person_count"] = max(0, int(expected_raw))
    if fill_extra_raw is not None:
        kwargs["fill_extra_missed"] = max(0, int(fill_extra_raw))
    if fill_below_raw in {"1", "true", "yes", "on"}:
        kwargs["fill_below_expected"] = True
    elif fill_below_raw in {"0", "false", "no", "off"}:
        kwargs["fill_below_expected"] = False
    if motion_raw in {"1", "true", "yes", "on"}:
        kwargs["motion_enabled"] = True
    elif motion_raw in {"0", "false", "no", "off"}:
        kwargs["motion_enabled"] = False
    if motion_smoothing_raw is not None:
        kwargs["motion_smoothing"] = max(0.0, min(1.0, float(motion_smoothing_raw)))
    if motion_max_speed_raw is not None:
        kwargs["motion_max_speed"] = max(0.0, float(motion_max_speed_raw))
    if motion_gate_factor_raw is not None:
        kwargs["motion_gate_factor"] = max(0.0, float(motion_gate_factor_raw))
    if reid_raw in {"1", "true", "yes", "on"}:
        kwargs["reid_enabled"] = True
    elif reid_raw in {"0", "false", "no", "off"}:
        kwargs["reid_enabled"] = False
    if reid_weight_raw is not None:
        kwargs["reid_weight"] = max(0.0, min(1.0, float(reid_weight_raw)))
    if reid_similarity_raw is not None:
        kwargs["reid_similarity_threshold"] = max(0.0, min(1.0, float(reid_similarity_raw)))
    if reid_gallery_frames_raw is not None:
        kwargs["reid_max_gallery_frames"] = max(0, int(reid_gallery_frames_raw))
    if reid_ema_raw is not None:
        kwargs["reid_ema"] = max(0.0, min(1.0, float(reid_ema_raw)))
    if reid_backend_raw in {"histogram", "embedding"}:
        kwargs["reid_backend"] = reid_backend_raw
    if reid_model_raw is not None:
        kwargs["reid_model_path"] = reid_model_raw
    if not kwargs:
        return settings
    from dataclasses import replace

    return replace(settings, **kwargs)


# Backward-compatible aliases
StabilizationStats = TrackPipelineStats
StabilizationResult = TrackPipelineResult
