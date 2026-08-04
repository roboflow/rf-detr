# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unified person tracking pipeline (NMS, association, hold, optional motion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.track_store import TrackStore
from rfdetr_demo.tracking.types import (
    PersonTrackSettings,
    TrackPipelineResult,
    TrackPipelineStats,
    person_track_settings_from_env,
)

if TYPE_CHECKING:
    from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter


@dataclass
class PersonTrackPipeline:
    """Single entry for per-frame person track stabilization and optional motion filter."""

    settings: PersonTrackSettings = field(default_factory=PersonTrackSettings)
    frame_width: int = 1280
    frame_height: int = 480
    _store: TrackStore = field(init=False, repr=False)
    _temporal_filter: KeypointTemporalFilter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._store = TrackStore(settings=self.settings, frame_width=self.frame_width)

    @classmethod
    def from_env(
        cls,
        *,
        frame_width: int,
        frame_height: int,
        temporal_filter: KeypointTemporalFilter | None = None,
        base_settings: PersonTrackSettings | None = None,
    ) -> PersonTrackPipeline:
        """Build pipeline with environment overrides for ``max_missed`` / sticky."""
        settings = person_track_settings_from_env(base=base_settings)
        return cls(
            settings=settings,
            frame_width=frame_width,
            frame_height=frame_height,
            _temporal_filter=temporal_filter,
        )

    def reset(self) -> None:
        """Clear track and optional motion state."""
        self._store.reset()
        if self._temporal_filter is not None:
            self._temporal_filter.reset()

    def apply(
        self,
        key_points: sv.KeyPoints,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> TrackPipelineResult:
        """Run NMS, association, hold, then optional keypoint motion filtering.

        Args:
            key_points: Raw per-frame keypoint detections.
            frame_index: Frame counter forwarded to the track store.
            frame: Optional BGR frame used for appearance ReID re-association.
        """
        result = self._store.apply(key_points, frame_index, frame)
        if self._temporal_filter is not None:
            filtered = self._temporal_filter.apply(result.key_points, frame_index)
            return TrackPipelineResult(
                key_points=filtered,
                stats=result.stats,
                diagnostics=result.diagnostics,
            )
        return result

    @property
    def stats(self) -> TrackPipelineStats | None:
        """Most recent stats are returned from :meth:`apply`; property unused."""
        return None
