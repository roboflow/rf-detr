# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Stabilize per-frame keypoint detections with IoU-NMS and short track hold.

Deprecated facade — prefer :class:`PersonTrackPipeline` from ``tracking.pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import (
    detection_bbox,
    detection_confidence,
    nms_detection_indices,
)
from rfdetr_demo.tracking.keypoints_ops import (
    is_track_ghost,
    merge_key_points,
    partition_live_and_ghost,
    subset_key_points,
)
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.track_store import TrackStore
from rfdetr_demo.tracking.types import (
    TRACK_IS_GHOST_KEY,
    PersonTrackSettings,
    StabilizationResult,
    StabilizationStats,
    TrackDiagnostic,
    TrackPipelineResult,
    is_person_track_enabled,
    person_track_settings_from_env,
)

DetectionStabilizerSettings = PersonTrackSettings


def is_detection_stabilizer_enabled() -> bool:
    """Return False when ``RFDETR_DETECTION_STABILIZER=0``."""
    return is_person_track_enabled()


@dataclass
class DetectionStabilizer:
    """Apply IoU-NMS and short missed-frame hold to keypoint detections."""

    settings: PersonTrackSettings = field(default_factory=PersonTrackSettings)
    frame_width: int = 1280
    _store: TrackStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = TrackStore(settings=self.settings, frame_width=self.frame_width)

    def reset(self) -> None:
        """Clear track history."""
        self._store.reset()

    def apply(
        self,
        key_points: sv.KeyPoints,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> StabilizationResult:
        """Return stabilized detections for one frame (``frame`` enables ReID)."""
        result = self._store.apply(key_points, frame_index, frame)
        return StabilizationResult(
            key_points=result.key_points,
            stats=StabilizationStats(
                raw_count=result.stats.raw_count,
                nms_count=result.stats.nms_count,
                active_track_count=result.stats.active_track_count,
                ghost_count=result.stats.ghost_count,
            ),
            diagnostics=result.diagnostics,
        )


__all__ = [
    "DetectionStabilizer",
    "DetectionStabilizerSettings",
    "PersonTrackPipeline",
    "StabilizationResult",
    "StabilizationStats",
    "TRACK_IS_GHOST_KEY",
    "TrackDiagnostic",
    "TrackPipelineResult",
    "detection_bbox",
    "detection_confidence",
    "is_detection_stabilizer_enabled",
    "is_track_ghost",
    "merge_key_points",
    "nms_detection_indices",
    "partition_live_and_ghost",
    "person_track_settings_from_env",
    "subset_key_points",
]
