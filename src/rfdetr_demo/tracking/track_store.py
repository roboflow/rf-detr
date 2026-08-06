# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Track state store: association, hold, and ghost flags."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.appearance import (
    AppearanceEncoder,
    appearance_roi,
    build_appearance_encoder,
    histogram_similarity,
)
from rfdetr_demo.tracking.bbox import (
    detection_bbox,
    detection_confidence,
    hungarian_maximize,
    iou,
    nms_detection_indices,
)
from rfdetr_demo.tracking.keypoints_ops import (
    merge_key_points,
    single_detection_key_points,
    subset_key_points,
    transform_key_points,
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
    size_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    descriptor: np.ndarray | None = None


@dataclass
class GalleryEntry:
    """A recently retired track kept for appearance-based revival."""

    track_id: int
    descriptor: np.ndarray
    velocity: np.ndarray
    size_velocity: np.ndarray
    age: int = 0


def _center_x_range(frame_width: int, fraction: tuple[float, float]) -> tuple[float, float]:
    return fraction[0] * frame_width, fraction[1] * frame_width


def _box_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def _box_size(box: np.ndarray) -> tuple[float, float]:
    return float(box[2] - box[0]), float(box[3] - box[1])


def _center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    return float(np.hypot(ax - bx, ay - by))


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
    gate_distances: list[float] | None = None,
    track_descriptors: list[np.ndarray | None] | None = None,
    det_descriptors: list[np.ndarray | None] | None = None,
    reid_weight: float = 0.0,
    similarity_fn: Callable[[np.ndarray | None, np.ndarray | None], float] = histogram_similarity,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Return (track_idx, det_idx) pairs plus unmatched track/det indices.

    ``track_boxes`` are the per-track boxes to match against, which may be
    motion-predicted rather than the last observed position. When
    ``gate_distances`` is given, a track/detection pair whose center distance
    exceeds ``gate_distances[track_idx]`` is disqualified, which suppresses
    implausible long-range matches (ID swaps) between crossing people. When
    ``reid_weight > 0`` and both descriptor lists are supplied, the cost blends
    IoU with appearance similarity ``(1 - w) * iou + w * sim``.
    """
    if not track_boxes or not detection_boxes:
        return [], set(range(len(track_boxes))), set(range(len(detection_boxes)))

    blend_appearance = reid_weight > 0.0 and track_descriptors is not None and det_descriptors is not None
    cost = np.zeros((len(track_boxes), len(detection_boxes)), dtype=np.float64)
    for track_index, track_box in enumerate(track_boxes):
        gate = gate_distances[track_index] if gate_distances is not None else None
        for detection_index, det_box in enumerate(detection_boxes):
            if gate is not None and _center_distance(track_box, det_box) > gate:
                continue
            score = iou(track_box, det_box)
            if blend_appearance:
                similarity = similarity_fn(
                    track_descriptors[track_index],
                    det_descriptors[detection_index],
                )
                score = (1.0 - reid_weight) * score + reid_weight * similarity
            cost[track_index, detection_index] = score

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
    _gallery: list[GalleryEntry] = field(default_factory=list, init=False, repr=False)
    _encoder: AppearanceEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._encoder = build_appearance_encoder(
            backend=self.settings.reid_backend,
            model_path=self.settings.reid_model_path,
        )

    def reset(self) -> None:
        """Clear track history."""
        self._tracks.clear()
        self._gallery.clear()
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
        """Return the track box advanced by one step of center and size velocity."""
        if not self.settings.motion_enabled:
            return track.box
        cx, cy = _box_center(track.box)
        width, height = _box_size(track.box)
        new_cx = cx + float(track.velocity[0])
        new_cy = cy + float(track.velocity[1])
        new_width = max(1.0, width + float(track.size_velocity[0]))
        new_height = max(1.0, height + float(track.size_velocity[1]))
        return np.array(
            [
                new_cx - new_width / 2.0,
                new_cy - new_height / 2.0,
                new_cx + new_width / 2.0,
                new_cy + new_height / 2.0,
            ],
            dtype=np.float64,
        )

    def _gate_distance(self, predicted_box: np.ndarray) -> float | None:
        """Return the max plausible center jump for a track, or None if disabled."""
        factor = self.settings.motion_gate_factor
        if not self.settings.motion_enabled or factor <= 0:
            return None
        width, height = _box_size(predicted_box)
        return factor * 0.5 * (width + height) + self.settings.motion_max_speed

    def _update_velocity(self, track: TrackSnapshot, new_box: np.ndarray) -> None:
        """Blend observed center and size change into the track velocities (EMA)."""
        if not self.settings.motion_enabled:
            return
        old_cx, old_cy = _box_center(track.box)
        new_cx, new_cy = _box_center(new_box)
        old_width, old_height = _box_size(track.box)
        new_width, new_height = _box_size(new_box)
        measured_center = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float64)
        measured_size = np.array([new_width - old_width, new_height - old_height], dtype=np.float64)
        beta = self.settings.motion_smoothing
        track.velocity = beta * track.velocity + (1.0 - beta) * measured_center
        track.size_velocity = beta * track.size_velocity + (1.0 - beta) * measured_size
        max_speed = self.settings.motion_max_speed
        if max_speed > 0:
            np.clip(track.velocity, -max_speed, max_speed, out=track.velocity)
            np.clip(track.size_velocity, -max_speed, max_speed, out=track.size_velocity)

    def _advance_ghost(self, track: TrackSnapshot) -> sv.KeyPoints:
        """Move a held track forward by its motion model and return shifted keypoints."""
        if not self.settings.motion_enabled or not (np.any(track.velocity) or np.any(track.size_velocity)):
            return track.key_points
        old_cx, old_cy = _box_center(track.box)
        old_width, old_height = _box_size(track.box)
        predicted = self._predicted_box(track)
        new_width, new_height = _box_size(predicted)
        scale_x = new_width / old_width if old_width > 0 else 1.0
        scale_y = new_height / old_height if old_height > 0 else 1.0
        track.box = predicted
        track.key_points = transform_key_points(
            track.key_points,
            dx=float(track.velocity[0]),
            dy=float(track.velocity[1]),
            scale_x=scale_x,
            scale_y=scale_y,
            center_x=old_cx,
            center_y=old_cy,
        )
        return track.key_points

    def _reid_active(self, frame: np.ndarray | None) -> bool:
        return self.settings.reid_enabled and frame is not None

    def _detection_descriptor(
        self,
        frame: np.ndarray,
        nms_key_points: sv.KeyPoints,
        detection_index: int,
        box: np.ndarray,
    ) -> np.ndarray | None:
        """Compute the appearance descriptor for one detection, if possible."""
        roi = appearance_roi(nms_key_points, detection_index, box)
        if roi is None or self._encoder is None:
            return None
        return self._encoder.encode(frame, roi)

    def _appearance_similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        """Similarity between two descriptors under the active encoder."""
        if self._encoder is None:
            return 0.0
        return self._encoder.similarity(left, right)

    def _update_descriptor(self, track: TrackSnapshot, descriptor: np.ndarray | None) -> None:
        """Blend a fresh descriptor into the track descriptor with EMA."""
        if descriptor is None or self._encoder is None:
            return
        if track.descriptor is None:
            track.descriptor = descriptor
            return
        track.descriptor = self._encoder.combine(track.descriptor, descriptor, self.settings.reid_ema)

    def _age_gallery(self) -> None:
        """Advance gallery ages and drop entries older than the retention window."""
        if not self._gallery:
            return
        limit = self.settings.reid_max_gallery_frames
        survivors: list[GalleryEntry] = []
        for entry in self._gallery:
            entry.age += 1
            if entry.age <= limit:
                survivors.append(entry)
        self._gallery = survivors

    def _retire_to_gallery(self, track: TrackSnapshot) -> None:
        """Store a dropped track's appearance for later revival."""
        if not self.settings.reid_enabled or self.settings.reid_max_gallery_frames <= 0:
            return
        if track.descriptor is None:
            return
        self._gallery.append(
            GalleryEntry(
                track_id=track.track_id,
                descriptor=track.descriptor,
                velocity=track.velocity.copy(),
                size_velocity=track.size_velocity.copy(),
                age=0,
            ),
        )

    def _pop_gallery_match(self, descriptor: np.ndarray | None) -> GalleryEntry | None:
        """Return and remove the best gallery entry above the similarity threshold."""
        if descriptor is None or not self._gallery:
            return None
        best_entry: GalleryEntry | None = None
        best_similarity = self.settings.reid_similarity_threshold
        for entry in self._gallery:
            similarity = self._appearance_similarity(entry.descriptor, descriptor)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_entry = entry
        if best_entry is not None:
            self._gallery.remove(best_entry)
        return best_entry

    def _maybe_mark_sticky(self, track: TrackSnapshot) -> None:
        if not self.settings.sticky_center_track:
            return
        cx, _ = _box_center(track.box)
        if _in_center_lane(cx, self.frame_width, self.settings.center_x_fraction):
            track.sticky = True
            self._sticky_track_id = track.track_id

    def apply(
        self,
        key_points: sv.KeyPoints,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> TrackPipelineResult:
        """Return stabilized detections for one frame.

        Args:
            key_points: Raw per-frame keypoint detections.
            frame_index: Frame counter (unused directly; reserved for callers).
            frame: Optional BGR frame used for appearance ReID; required only
                when ``reid_enabled`` is set.
        """
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

        self._age_gallery()
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

        reid_active = self._reid_active(frame)
        det_descriptors: list[np.ndarray | None] = [None] * nms_count
        if reid_active:
            for detection_index in range(nms_count):
                det_descriptors[detection_index] = self._detection_descriptor(
                    frame,
                    nms_key_points,
                    detection_index,
                    detection_boxes[detection_index],
                )

        track_boxes = [self._predicted_box(track) for track in self._tracks]
        gate_distances = [self._gate_distance(box) for box in track_boxes]
        matched, unmatched_tracks, unmatched_detections = _match_tracks_to_detections(
            track_boxes,
            detection_boxes,
            match_iou_threshold=self.settings.match_iou_threshold,
            gate_distances=(gate_distances if self.settings.motion_gate_factor > 0 else None),
            track_descriptors=([track.descriptor for track in self._tracks] if reid_active else None),
            det_descriptors=(det_descriptors if reid_active else None),
            reid_weight=(self.settings.reid_weight if reid_active else 0.0),
            similarity_fn=self._appearance_similarity,
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
            self._update_descriptor(track, det_descriptors[detection_index])
            self._maybe_mark_sticky(track)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(_track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        # Appearance revival: reclaim a retired track id before minting a new one.
        if reid_active:
            for detection_index in sorted(unmatched_detections):
                entry = self._pop_gallery_match(det_descriptors[detection_index])
                if entry is None:
                    continue
                snapshot = single_detection_key_points(nms_key_points, detection_index)
                track = TrackSnapshot(
                    track_id=entry.track_id,
                    key_points=snapshot,
                    box=detection_boxes[detection_index].copy(),
                    missed=0,
                    velocity=entry.velocity,
                    size_velocity=entry.size_velocity,
                    descriptor=entry.descriptor,
                )
                self._update_descriptor(track, det_descriptors[detection_index])
                self._maybe_mark_sticky(track)
                self._tracks.append(track)
                matched_track_indices.add(len(self._tracks) - 1)
                unmatched_detections.discard(detection_index)
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
            self._update_descriptor(track, det_descriptors[detection_index])
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

        kept_tracks: list[TrackSnapshot] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_track_indices or track.missed <= self._hold_limit_for(track, len(output_parts)):
                kept_tracks.append(track)
            else:
                self._retire_to_gallery(track)
        self._tracks = kept_tracks

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
            else:
                self._retire_to_gallery(track)
        self._tracks = surviving_tracks
        return self._finalize_output(
            ghost_parts,
            ghost_flags,
            track_ids,
            diagnostics,
            raw_count=raw_count,
            nms_count=0,
        )
