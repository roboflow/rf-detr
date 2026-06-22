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
    sticky_center_track: bool = False
    sticky_max_missed: int = 4
    center_x_fraction: tuple[float, float] = (0.28, 0.48)


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
    sticky_raw = os.environ.get("RFDETR_STICKY_CENTER_TRACK", "").strip().lower()
    sticky_max_raw = os.environ.get("RFDETR_STICKY_MAX_MISSED")
    kwargs: dict[str, object] = {}
    if max_missed_raw is not None:
        kwargs["max_missed"] = max(0, int(max_missed_raw))
    if sticky_raw in {"1", "true", "yes", "on"}:
        kwargs["sticky_center_track"] = True
    elif sticky_raw in {"0", "false", "no", "off"}:
        kwargs["sticky_center_track"] = False
    if sticky_max_raw is not None:
        kwargs["sticky_max_missed"] = max(1, int(sticky_max_raw))
    if not kwargs:
        return settings
    from dataclasses import replace

    return replace(settings, **kwargs)


# Backward-compatible aliases
StabilizationStats = TrackPipelineStats
StabilizationResult = TrackPipelineResult
