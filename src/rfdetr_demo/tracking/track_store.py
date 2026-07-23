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
from rfdetr_demo.tracking.keypoints_ops import (
    merge_key_points,
    shift_key_points,
    single_detection_key_points,
    subset_key_points,
)
from rfdetr_demo.tracking.types import PersonTrackSettings, TrackDiagnostic, TrackPipelineResult, TrackPipelineStats


@dataclass
class TrackSnapshot:
    """One person track between frames."""

    track_id: int
    key_points: sv.KeyPoints
    box: np.ndarray
    missed: int = 0
    sticky: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))


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
    track_boxes: list[np.ndarray],
    detection_boxes: list[np.ndarray],
    *,
    match_iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Return (track_idx, det_idx) pairs plus unmatched track/det indices.

    ``track_boxes`` are the per-track boxes to match against, which may be
    motion-predicted rather than the last observed position.
    """
    if not track_boxes or not detection_boxes:
        return [], set(range(len(track_boxes))), set(range(len(detection_boxes)))

    cost = np.zeros((len(track_boxes), len(detection_boxes)), dtype=np.float64)
    for track_index, track_box in enumerate(track_boxes):
        for detection_index, det_box in enumerate(detection_boxes):
            cost[track_index, detection_index] = iou(track_box, det_box)

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
    unmatched_tracks = set(range(len(track_boxes))) - used_tracks
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

    def _expected_count(self) -> int:
        return max(0, self.settings.expected_person_count)

    def _hold_limit_for(self, track: TrackSnapshot, current_output_count: int) -> int:
        """Extend hold when output is below the expected person count."""
        base = self._max_missed_for(track)
        expected = self._expected_count()
        if expected > 0 and self.settings.fill_below_expected and current_output_count < expected:
            return base + self.settings.fill_extra_missed
        return base

    def _cap_output(
        self,
        output_parts: list[sv.KeyPoints],
        ghost_flags: list[bool],
        track_ids: list[int],
        diagnostics: list[TrackDiagnostic],
    ) -> tuple[list[sv.KeyPoints], list[bool], list[int], list[TrackDiagnostic]]:
        """Drop lowest-priority tracks when count exceeds ``expected_person_count``."""
        expected = self._expected_count()
        if expected <= 0 or len(output_parts) <= expected:
            return output_parts, ghost_flags, track_ids, diagnostics

        scored: list[tuple[int, int, float]] = []
        for index, (key_points, is_ghost) in enumerate(zip(output_parts, ghost_flags, strict=True)):
            scored.append(
                (
                    index,
                    0 if is_ghost else 1,
                    detection_confidence(key_points, 0),
                ),
            )
        scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
        keep = {row[0] for row in scored[:expected]}
        dropped_ids = {track_ids[index] for index in range(len(track_ids)) if index not in keep}
        if dropped_ids:
            self._tracks = [track for track in self._tracks if track.track_id not in dropped_ids]

        trimmed_parts: list[sv.KeyPoints] = []
        trimmed_flags: list[bool] = []
        trimmed_ids: list[int] = []
        trimmed_diagnostics: list[TrackDiagnostic] = []
        for index in sorted(keep):
            trimmed_parts.append(output_parts[index])
            trimmed_flags.append(ghost_flags[index])
            trimmed_ids.append(track_ids[index])
            trimmed_diagnostics.append(diagnostics[index])
        return trimmed_parts, trimmed_flags, trimmed_ids, trimmed_diagnostics

    def _finalize_output(
        self,
        output_parts: list[sv.KeyPoints],
        ghost_flags: list[bool],
        track_ids: list[int],
        diagnostics: list[TrackDiagnostic],
        *,
        raw_count: int,
        nms_count: int,
    ) -> TrackPipelineResult:
        output_parts, ghost_flags, track_ids, diagnostics = self._cap_output(
            output_parts,
            ghost_flags,
            track_ids,
            diagnostics,
        )
        ghost_count = sum(1 for is_ghost in ghost_flags if is_ghost)
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

    def _predicted_box(self, track: TrackSnapshot) -> np.ndarray:
        """Return the track box advanced by one step of its velocity."""
        if not self.settings.motion_enabled:
            return track.box
        shift = np.array(
            [track.velocity[0], track.velocity[1], track.velocity[0], track.velocity[1]],
            dtype=np.float64,
        )
        return track.box + shift

    def _update_velocity(self, track: TrackSnapshot, new_box: np.ndarray) -> None:
        """Blend the observed center displacement into the track velocity (EMA)."""
        if not self.settings.motion_enabled:
            return
        old_cx, old_cy = _box_center(track.box)
        new_cx, new_cy = _box_center(new_box)
        measured = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float64)
        beta = self.settings.motion_smoothing
        track.velocity = beta * track.velocity + (1.0 - beta) * measured
        max_speed = self.settings.motion_max_speed
        if max_speed > 0:
            np.clip(track.velocity, -max_speed, max_speed, out=track.velocity)

    def _advance_ghost(self, track: TrackSnapshot) -> sv.KeyPoints:
        """Move a held track forward by its velocity and return shifted keypoints."""
        if not self.settings.motion_enabled or not np.any(track.velocity):
            return track.key_points
        dx = float(track.velocity[0])
        dy = float(track.velocity[1])
        track.box = self._predicted_box(track)
        track.key_points = shift_key_points(track.key_points, dx, dy)
        return track.key_points

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

        track_boxes = [self._predicted_box(track) for track in self._tracks]
        matched, unmatched_tracks, unmatched_detections = _match_tracks_to_detections(
            track_boxes,
            detection_boxes,
            match_iou_threshold=self.settings.match_iou_threshold,
        )

        if self.settings.sticky_center_track and self._sticky_track_id is not None:
            sticky_index = next(
                (index for index, track in enumerate(self._tracks) if track.track_id == self._sticky_track_id),
                None,
            )
            if sticky_index is not None and sticky_index in unmatched_tracks and unmatched_detections:
                sticky_box = track_boxes[sticky_index]
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

        for track_index, detection_index in matched:
            track = self._tracks[track_index]
            snapshot = single_detection_key_points(nms_key_points, detection_index)
            det_box = detection_boxes[detection_index].copy()
            self._update_velocity(track, det_box)
            track.key_points = snapshot
            track.box = det_box
            track.missed = 0
            self._maybe_mark_sticky(track)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(_track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        for detection_index in sorted(unmatched_detections):
            if len(self._tracks) >= self.settings.max_tracks:
                break
            if self.settings.hysteresis_enabled:
                confidence = detection_confidence(nms_key_points, detection_index)
                if confidence < self.settings.new_track_min_confidence:
                    continue
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
            if track.missed <= self._hold_limit_for(track, len(output_parts)):
                ghost_key_points = self._advance_ghost(track)
                output_parts.append(ghost_key_points)
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                diagnostics.append(_track_diagnostic(track, is_ghost=True, matched_this_frame=False))

        self._tracks = [
            track
            for track_index, track in enumerate(self._tracks)
            if track_index in matched_track_indices or track.missed <= self._hold_limit_for(track, len(output_parts))
        ]

        return self._finalize_output(
            output_parts,
            ghost_flags,
            track_ids,
            diagnostics,
            raw_count=raw_count,
            nms_count=nms_count,
        )

    def _apply_empty_frame(self, raw_count: int) -> TrackPipelineResult:
        ghost_parts: list[sv.KeyPoints] = []
        ghost_flags: list[bool] = []
        track_ids: list[int] = []
        diagnostics: list[TrackDiagnostic] = []
        surviving_tracks: list[TrackSnapshot] = []
        for track in self._tracks:
            track.missed += 1
            if track.missed <= self._hold_limit_for(track, len(ghost_parts)):
                ghost_parts.append(self._advance_ghost(track))
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                surviving_tracks.append(track)
                diagnostics.append(_track_diagnostic(track, is_ghost=True, matched_this_frame=False))
        self._tracks = surviving_tracks
        return self._finalize_output(
            ghost_parts,
            ghost_flags,
            track_ids,
            diagnostics,
            raw_count=raw_count,
            nms_count=0,
        )
