# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Temporal plausibility filter for keypoint video tracks.

Rejects or smooths joint positions that imply implausible human motion speed
or rapid back-and-forth oscillation (hunting) between processed frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.keypoints_ops import is_track_ghost, track_ids_from_key_points

RejectMode = Literal["hold", "hide"]


@dataclass(frozen=True)
class MotionPlausibilitySettings:
    """Thresholds for temporal keypoint gating and smoothing."""

    enabled: bool = True
    max_speed_fraction_per_sec: float = 0.35
    ema_alpha: float = 0.55
    suppress_oscillation: bool = True
    oscillation_flip_threshold: int = 3
    oscillation_min_speed_fraction: float = 0.08
    covariance_sigma_multiplier: float = 4.0
    use_covariance_gate: bool = True
    reject_mode: RejectMode = "hold"
    max_consecutive_holds: int = 4
    track_match_fraction: float = 0.12


@dataclass
class MotionFilterStats:
    """Counters updated while filtering a sequence."""

    speed_rejections: int = 0
    covariance_rejections: int = 0
    oscillation_corrections: int = 0
    smoothed_joints: int = 0


@dataclass
class _JointState:
    last_xy: np.ndarray | None = None
    last_velocity: np.ndarray | None = None
    flip_count: int = 0
    consecutive_holds: int = 0
    last_frame_index: int | None = None


@dataclass
class _TrackState:
    joints: list[_JointState] = field(default_factory=list)
    last_centroid: np.ndarray | None = None
    last_frame_index: int = -1


def _frame_diagonal(frame_width: int, frame_height: int) -> float:
    return float(np.hypot(frame_width, frame_height))


def _detection_centroid(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    xy = key_points.xy[detection_index]
    visible = key_points.visible
    if visible is not None:
        mask = visible[detection_index]
        points = xy[mask]
    else:
        points = xy[~np.all(np.isclose(xy, 0), axis=1)]
    if len(points) == 0:
        return None
    hip_indices = (11, 12)
    hip_points = [xy[idx] for idx in hip_indices if idx < len(xy) and not np.allclose(xy[idx], 0)]
    if len(hip_points) >= 1:
        return np.mean(np.asarray(hip_points, dtype=np.float64), axis=0)
    return np.mean(points.astype(np.float64), axis=0)


def _covariance_trace(key_points: sv.KeyPoints, detection_index: int, joint_index: int) -> float | None:
    cov_raw = key_points.data.get("covariance")
    if cov_raw is None:
        return None
    cov = np.asarray(cov_raw, dtype=np.float64)
    if cov.shape[:2] != key_points.xy.shape[:2]:
        return None
    matrix = cov[detection_index, joint_index]
    if not np.isfinite(matrix).all():
        return None
    return float(matrix[0, 0] + matrix[1, 1])


class KeypointTemporalFilter:
    """Stateful filter applied sequentially across video frames."""

    def __init__(
        self,
        settings: MotionPlausibilitySettings,
        *,
        frame_width: int,
        frame_height: int,
        fps: float,
        frame_stride: int,
    ) -> None:
        self.settings = settings
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = max(fps, 1.0)
        self.frame_stride = max(frame_stride, 1)
        self.diagonal = _frame_diagonal(frame_width, frame_height)
        self.stats = MotionFilterStats()
        self._tracks: list[_TrackState] = []
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        """Clear track history and statistics."""
        self._tracks.clear()
        self._last_frame_index = None
        self.stats = MotionFilterStats()

    def _resolve_track_ids(self, key_points: sv.KeyPoints) -> list[int | None]:
        """Map each detection index to a stable track id from pipeline metadata."""
        track_ids = track_ids_from_key_points(key_points)
        if any(track_id is not None for track_id in track_ids):
            return track_ids
        if len(key_points) == 1:
            return [0]
        return [None] * len(key_points)

    def _ensure_track_states(self, track_ids: list[int | None]) -> None:
        max_id = max((track_id for track_id in track_ids if track_id is not None), default=-1)
        while len(self._tracks) <= max_id:
            self._tracks.append(_TrackState())

    def _effective_dt_sec(self, frame_index: int, previous_frame_index: int | None) -> float:
        if previous_frame_index is None:
            return self.frame_stride / self.fps
        delta_frames = max(1, frame_index - previous_frame_index)
        return delta_frames / self.fps

    def _ensure_joint_states(self, track: _TrackState, num_joints: int) -> None:
        while len(track.joints) < num_joints:
            track.joints.append(_JointState())

    def apply(self, key_points: sv.KeyPoints, frame_index: int) -> sv.KeyPoints:
        """Return a copy of ``key_points`` with implausible motion corrected."""
        if not self.settings.enabled or len(key_points) == 0:
            self._last_frame_index = frame_index
            return key_points

        filtered = sv.KeyPoints(
            xy=key_points.xy.copy(),
            keypoint_confidence=(
                key_points.keypoint_confidence.copy()
                if key_points.keypoint_confidence is not None
                else None
            ),
            detection_confidence=(
                key_points.detection_confidence.copy()
                if key_points.detection_confidence is not None
                else None
            ),
            visible=key_points.visible.copy() if key_points.visible is not None else None,
            class_id=key_points.class_id.copy() if key_points.class_id is not None else None,
            data=dict(key_points.data),
        )

        track_ids = self._resolve_track_ids(filtered)
        self._ensure_track_states(track_ids)
        for detection_index, track_id in enumerate(track_ids):
            if track_id is None:
                continue
            centroid = _detection_centroid(filtered, detection_index)
            if centroid is not None:
                self._tracks[track_id].last_centroid = centroid.copy()
            self._tracks[track_id].last_frame_index = self._last_frame_index or 0

        max_speed_px_per_sec = self.settings.max_speed_fraction_per_sec * self.diagonal
        oscillation_min_speed = self.settings.oscillation_min_speed_fraction * self.diagonal

        for detection_index, track_id in enumerate(track_ids):
            if track_id is None or is_track_ghost(filtered, detection_index):
                continue
            track = self._tracks[track_id]
            num_joints = filtered.xy.shape[1]
            self._ensure_joint_states(track, num_joints)

            for joint_index in range(num_joints):
                if filtered.visible is not None and not filtered.visible[detection_index, joint_index]:
                    continue
                current_xy = filtered.xy[detection_index, joint_index].astype(np.float64)
                if np.allclose(current_xy, 0):
                    continue

                joint_state = track.joints[joint_index]
                dt_sec = self._effective_dt_sec(frame_index, joint_state.last_frame_index)
                output_xy = current_xy.copy()
                rejected = False

                if joint_state.last_xy is not None and dt_sec > 0:
                    displacement = output_xy - joint_state.last_xy
                    distance = float(np.linalg.norm(displacement))
                    speed = distance / dt_sec
                    velocity = displacement / dt_sec

                    if speed > max_speed_px_per_sec:
                        rejected = True
                        self.stats.speed_rejections += 1

                    if (
                        not rejected
                        and self.settings.use_covariance_gate
                        and distance > 0
                    ):
                        trace_curr = _covariance_trace(filtered, detection_index, joint_index)
                        if trace_curr is not None:
                            sigma_scale = self.settings.covariance_sigma_multiplier * np.sqrt(
                                max(trace_curr, 1e-6),
                            )
                            if distance > sigma_scale:
                                rejected = True
                                self.stats.covariance_rejections += 1

                    if (
                        not rejected
                        and self.settings.suppress_oscillation
                        and joint_state.last_velocity is not None
                    ):
                        prev_speed = float(np.linalg.norm(joint_state.last_velocity))
                        curr_speed = float(np.linalg.norm(velocity))
                        if (
                            prev_speed >= oscillation_min_speed
                            and curr_speed >= oscillation_min_speed
                            and float(np.dot(velocity, joint_state.last_velocity)) < 0
                        ):
                            joint_state.flip_count += 1
                        else:
                            joint_state.flip_count = max(0, joint_state.flip_count - 1)

                        if joint_state.flip_count >= self.settings.oscillation_flip_threshold:
                            rejected = True
                            self.stats.oscillation_corrections += 1
                            joint_state.flip_count = 0

                    if rejected:
                        if self.settings.reject_mode == "hide":
                            if filtered.visible is not None:
                                filtered.visible[detection_index, joint_index] = False
                            output_xy = current_xy
                            joint_state.consecutive_holds = 0
                        else:
                            joint_state.consecutive_holds += 1
                            if joint_state.consecutive_holds >= self.settings.max_consecutive_holds:
                                output_xy = current_xy.copy()
                                joint_state.consecutive_holds = 0
                                rejected = False
                            else:
                                output_xy = joint_state.last_xy.copy()
                        velocity = (
                            (output_xy - joint_state.last_xy) / dt_sec
                            if joint_state.last_xy is not None
                            else velocity
                        )
                    elif self.settings.ema_alpha < 1.0 and joint_state.last_xy is not None:
                        alpha = self.settings.ema_alpha
                        output_xy = alpha * output_xy + (1.0 - alpha) * joint_state.last_xy
                        self.stats.smoothed_joints += 1
                        velocity = (output_xy - joint_state.last_xy) / dt_sec
                        joint_state.consecutive_holds = 0
                    else:
                        joint_state.consecutive_holds = 0

                    joint_state.last_velocity = velocity.copy()
                else:
                    joint_state.last_velocity = None
                    joint_state.flip_count = 0

                filtered.xy[detection_index, joint_index] = output_xy
                joint_state.last_xy = output_xy.copy()
                joint_state.last_frame_index = frame_index

        self._last_frame_index = frame_index
        return filtered


def apply_temporal_filter(
    key_points: sv.KeyPoints,
    temporal_filter: KeypointTemporalFilter,
    frame_index: int,
) -> sv.KeyPoints:
    """Apply ``temporal_filter`` and return filtered keypoints."""
    return temporal_filter.apply(key_points, frame_index)
