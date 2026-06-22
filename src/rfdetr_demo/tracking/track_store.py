# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Track state store: association, hold, and ghost flags."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import (
    detection_bbox,
    detection_confidence,
    hungarian_maximize,
    iou,
    nms_detection_indices,
)
from rfdetr_demo.tracking.keypoints_ops import merge_key_points, single_detection_key_points, subset_key_points
from rfdetr_demo.tracking.types import PersonTrackSettings, TrackDiagnostic, TrackPipelineResult, TrackPipelineStats


@dataclass
class TrackSnapshot:
    """One person track between frames."""

    track_id: int
    key_points: sv.KeyPoints
    box: np.ndarray
    missed: int = 0
    sticky: bool = False


def _center_x_range(frame_width: int, fraction: tuple[float, float]) -> tuple[float, float]:
    return fraction[0] * frame_width, fraction[1] * frame_width


def _box_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def _in_center_lane(cx: float, frame_width: int, fraction: tuple[float, float]) -> bool:
    x_min, x_max = _center_x_range(frame_width, fraction)
    return x_min <= cx <= x_max


def _track_diagnostic(
    track: TrackSnapshot,
    *,
    is_ghost: bool,
    matched_this_frame: bool,
) -> TrackDiagnostic:
    cx, cy = _box_center(track.box)
    return TrackDiagnostic(
        track_id=track.track_id,
        cx=cx,
        cy=cy,
        confidence=detection_confidence(track.key_points, 0),
        is_ghost=is_ghost,
        missed=track.missed,
        matched_this_frame=matched_this_frame,
    )


def _match_tracks_to_detections(
    tracks: list[TrackSnapshot],
    detection_boxes: list[np.ndarray],
    *,
    match_iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Return (track_idx, det_idx) pairs plus unmatched track/det indices."""
    if not tracks or not detection_boxes:
        return [], set(range(len(tracks))), set(range(len(detection_boxes)))

    cost = np.zeros((len(tracks), len(detection_boxes)), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for detection_index, det_box in enumerate(detection_boxes):
            cost[track_index, detection_index] = iou(track.box, det_box)

    pairs = hungarian_maximize(cost)
    matched: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for track_index, detection_index in pairs:
        if cost[track_index, detection_index] < match_iou_threshold:
            continue
        matched.append((track_index, detection_index))
        used_tracks.add(track_index)
        used_detections.add(detection_index)
    unmatched_tracks = set(range(len(tracks))) - used_tracks
    unmatched_detections = set(range(len(detection_boxes))) - used_detections
    return matched, unmatched_tracks, unmatched_detections


@dataclass
class TrackStore:
    """Own track snapshots, NMS, association, and short missed-frame hold."""

    settings: PersonTrackSettings = field(default_factory=PersonTrackSettings)
    frame_width: int = 1280
    _tracks: list[TrackSnapshot] = field(default_factory=list, init=False, repr=False)
    _next_track_id: int = field(default=0, init=False, repr=False)
    _sticky_track_id: int | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Clear track history."""
        self._tracks.clear()
        self._next_track_id = 0
        self._sticky_track_id = None

    def _max_missed_for(self, track: TrackSnapshot) -> int:
        if self.settings.sticky_center_track and track.sticky:
            return self.settings.sticky_max_missed
        return self.settings.max_missed

    def _maybe_mark_sticky(self, track: TrackSnapshot) -> None:
        if not self.settings.sticky_center_track:
            return
        cx, _ = _box_center(track.box)
        if _in_center_lane(cx, self.frame_width, self.settings.center_x_fraction):
            track.sticky = True
            self._sticky_track_id = track.track_id

    def apply(self, key_points: sv.KeyPoints, frame_index: int) -> TrackPipelineResult:
        """Return stabilized detections for one frame."""
        del frame_index
        raw_count = len(key_points)

        if not self.settings.enabled:
            return TrackPipelineResult(
                key_points=key_points,
                stats=TrackPipelineStats(
                    raw_count=raw_count,
                    nms_count=raw_count,
                    active_track_count=raw_count,
                    ghost_count=0,
                ),
                diagnostics=[],
            )

        if raw_count == 0:
            return self._apply_empty_frame(raw_count)

        nms_indices = nms_detection_indices(key_points, self.settings.nms_iou_threshold)
        nms_key_points = subset_key_points(key_points, nms_indices)
        nms_count = len(nms_key_points)

        detection_boxes: list[np.ndarray] = []
        for detection_index in range(nms_count):
            box = detection_bbox(nms_key_points, detection_index)
            detection_boxes.append(
                box.copy() if box is not None else np.zeros(4, dtype=np.float64),
            )

        matched, unmatched_tracks, unmatched_detections = _match_tracks_to_detections(
            self._tracks,
            detection_boxes,
            match_iou_threshold=self.settings.match_iou_threshold,
        )

        if self.settings.sticky_center_track and self._sticky_track_id is not None:
            sticky_index = next(
                (index for index, track in enumerate(self._tracks) if track.track_id == self._sticky_track_id),
                None,
            )
            if sticky_index is not None and sticky_index in unmatched_tracks and unmatched_detections:
                sticky_box = self._tracks[sticky_index].box
                best_det: int | None = None
                best_iou = self.settings.match_iou_threshold
                for detection_index in unmatched_detections:
                    score = iou(sticky_box, detection_boxes[detection_index])
                    cx, _ = _box_center(detection_boxes[detection_index])
                    if _in_center_lane(cx, self.frame_width, self.settings.center_x_fraction) and score >= best_iou:
                        best_iou = score
                        best_det = detection_index
                if best_det is not None:
                    matched.append((sticky_index, best_det))
                    unmatched_tracks.discard(sticky_index)
                    unmatched_detections.discard(best_det)

        matched_track_indices = {track_index for track_index, _ in matched}
        output_parts: list[sv.KeyPoints] = []
        ghost_flags: list[bool] = []
        track_ids: list[int] = []
        diagnostics: list[TrackDiagnostic] = []
        ghost_count = 0

        for track_index, detection_index in matched:
            track = self._tracks[track_index]
            snapshot = single_detection_key_points(nms_key_points, detection_index)
            track.key_points = snapshot
            track.box = detection_boxes[detection_index].copy()
            track.missed = 0
            self._maybe_mark_sticky(track)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(_track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        for detection_index in sorted(unmatched_detections):
            if len(self._tracks) >= self.settings.max_tracks:
                break
            snapshot = single_detection_key_points(nms_key_points, detection_index)
            track = TrackSnapshot(
                track_id=self._next_track_id,
                key_points=snapshot,
                box=detection_boxes[detection_index].copy(),
                missed=0,
            )
            self._next_track_id += 1
            self._maybe_mark_sticky(track)
            self._tracks.append(track)
            matched_track_indices.add(len(self._tracks) - 1)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(_track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        for track_index in sorted(unmatched_tracks):
            track = self._tracks[track_index]
            track.missed += 1
            if track.missed <= self._max_missed_for(track):
                output_parts.append(track.key_points)
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                ghost_count += 1
                diagnostics.append(_track_diagnostic(track, is_ghost=True, matched_this_frame=False))

        self._tracks = [
            track
            for track_index, track in enumerate(self._tracks)
            if track_index in matched_track_indices or track.missed <= self._max_missed_for(track)
        ]

        stabilized = merge_key_points(output_parts, ghost_flags=ghost_flags, track_ids=track_ids)
        return TrackPipelineResult(
            key_points=stabilized,
            stats=TrackPipelineStats(
                raw_count=raw_count,
                nms_count=nms_count,
                active_track_count=len(stabilized),
                ghost_count=ghost_count,
            ),
            diagnostics=diagnostics,
        )

    def _apply_empty_frame(self, raw_count: int) -> TrackPipelineResult:
        ghost_parts: list[sv.KeyPoints] = []
        ghost_flags: list[bool] = []
        track_ids: list[int] = []
        diagnostics: list[TrackDiagnostic] = []
        ghost_count = 0
        surviving_tracks: list[TrackSnapshot] = []
        for track in self._tracks:
            track.missed += 1
            if track.missed <= self._max_missed_for(track):
                ghost_parts.append(track.key_points)
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                ghost_count += 1
                surviving_tracks.append(track)
                diagnostics.append(_track_diagnostic(track, is_ghost=True, matched_this_frame=False))
        self._tracks = surviving_tracks
        stabilized = merge_key_points(ghost_parts, ghost_flags=ghost_flags, track_ids=track_ids)
        return TrackPipelineResult(
            key_points=stabilized,
            stats=TrackPipelineStats(
                raw_count=raw_count,
                nms_count=0,
                active_track_count=len(stabilized),
                ghost_count=ghost_count,
            ),
            diagnostics=diagnostics,
        )
