# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tune-preview cache quality metrics."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr.visualize.keypoints import key_points_for_display
from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter, MotionPlausibilitySettings
from rfdetr_demo.inference.tune_cache import TunePreviewCache, deserialize_key_points
from rfdetr_demo.tracking.keypoints_ops import attach_track_ids
from rfdetr_demo.tracking.person_associator import PersonAssociator
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings
from rfdetr_demo.tuning.auto_tune_types import (
    AnomalyFlags,
    CacheQualityMetrics,
    CurrentParameters,
)


def count_persons(key_points: sv.KeyPoints, threshold: float) -> int:
    if len(key_points) == 0:
        return 0
    if key_points.detection_confidence is not None:
        return int(np.sum(key_points.detection_confidence >= threshold))
    return len(key_points)


def joint_confidences(key_points: sv.KeyPoints, keypoint_threshold: float) -> list[float]:
    display = key_points_for_display(key_points, keypoint_threshold=keypoint_threshold)
    values: list[float] = []
    if display.keypoint_confidence is None:
        return values
    visible = display.visible
    for det_index in range(len(display)):
        for joint_index, confidence in enumerate(display.keypoint_confidence[det_index]):
            if visible is not None and not visible[det_index, joint_index]:
                continue
            if np.allclose(display.xy[det_index, joint_index], 0):
                continue
            values.append(float(confidence))
    return values


def detection_centroid(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    xy = key_points.xy[detection_index]
    visible = key_points.visible
    if visible is not None:
        points = xy[visible[detection_index]]
    else:
        points = xy[~np.all(np.isclose(xy, 0), axis=1)]
    if len(points) == 0:
        return None
    return np.mean(points.astype(np.float64), axis=0)


def largest_detection_centroid(key_points: sv.KeyPoints, threshold: float) -> np.ndarray | None:
    if len(key_points) == 0:
        return None
    best_index = 0
    best_score = -1.0
    for det_index in range(len(key_points)):
        if key_points.detection_confidence is not None:
            score = float(key_points.detection_confidence[det_index])
            if score < threshold:
                continue
        else:
            score = 1.0
        if score > best_score:
            best_score = score
            best_index = det_index
    return detection_centroid(key_points, best_index)


def covariance_spread_ratio(key_points: sv.KeyPoints) -> float:
    cov_raw = key_points.data.get("covariance")
    if cov_raw is None:
        return 1.0
    cov = np.asarray(cov_raw, dtype=np.float64)
    traces = cov[..., 0, 0] + cov[..., 1, 1]
    finite = traces[np.isfinite(traces) & (traces > 0)]
    if len(finite) < 4:
        return 1.0
    median = float(np.median(finite))
    if median <= 0:
        return 1.0
    return float(np.max(finite) / median)


def analyze_tune_cache(
    cache: TunePreviewCache,
    *,
    current: CurrentParameters,
) -> CacheQualityMetrics:
    """Compute quality metrics from cached tune-preview frames."""
    if not cache.has_entries:
        return CacheQualityMetrics(
            frames=0,
            avg_persons=0.0,
            person_count_std=0.0,
            person_count_min=0,
            person_count_max=0,
            low_confidence_ratio=0.0,
            mean_joint_confidence=0.0,
            motion_speed_rejections=0,
            motion_oscillation_corrections=0,
            rejection_rate_per_joint=0.0,
            centroid_jump_rate=0.0,
            covariance_spread_ratio=1.0,
        )

    sample = cache.latest
    frame_height, frame_width = sample.frame_bgr.shape[:2] if sample is not None else (480, 640)
    motion = MotionPlausibilitySettings(
        enabled=current.motion_filter_enabled,
        max_speed_fraction_per_sec=current.motion_max_speed_fraction,
        ema_alpha=current.motion_ema_alpha,
        suppress_oscillation=current.motion_oscillation_enabled,
    )
    temporal_filter = KeypointTemporalFilter(
        motion,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=cache.fps,
        frame_stride=cache.frame_stride,
    )
    associator = PersonAssociator()
    track_pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(),
        frame_width=frame_width,
        frame_height=frame_height,
    )

    person_counts: list[int] = []
    stabilized_person_counts: list[int] = []
    ghost_frames = 0
    confidences: list[float] = []
    low_conf = 0
    total_joints = 0
    centroid_jumps = 0
    centroid_checks = 0
    prev_centroid: np.ndarray | None = None
    cov_spreads: list[float] = []

    diagonal = float(np.hypot(frame_width, frame_height))
    jump_limit = current.motion_max_speed_fraction * diagonal * (cache.frame_stride / max(cache.fps, 1.0))

    for entry in cache.entries:
        if entry.key_points_payload is None:
            continue
        raw = deserialize_key_points(entry.key_points_payload)
        person_counts.append(count_persons(raw, current.threshold))
        track_result = track_pipeline.apply(raw, entry.frame_index)
        live_count = track_result.stats.active_track_count - track_result.stats.ghost_count
        stabilized_person_counts.append(live_count)
        if track_result.stats.ghost_count > 0:
            ghost_frames += 1
        cov_spreads.append(covariance_spread_ratio(raw))

        for confidence in joint_confidences(raw, current.keypoint_threshold):
            confidences.append(confidence)
            total_joints += 1
            if confidence < 0.5:
                low_conf += 1

        centroid = largest_detection_centroid(raw, current.threshold)
        if centroid is not None and prev_centroid is not None:
            centroid_checks += 1
            if float(np.linalg.norm(centroid - prev_centroid)) > jump_limit:
                centroid_jumps += 1
        if centroid is not None:
            prev_centroid = centroid.copy()

        temporal_filter.apply(
            attach_track_ids(raw, associator.assign(raw)),
            entry.frame_index,
        )

    person_std = float(np.std(person_counts)) if person_counts else 0.0
    stabilized_std = float(np.std(stabilized_person_counts)) if stabilized_person_counts else 0.0
    track_break_rate = ghost_frames / max(len(stabilized_person_counts), 1)
    avg_persons = float(np.mean(person_counts)) if person_counts else 0.0
    low_conf_ratio = low_conf / max(total_joints, 1)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    rejection_rate = temporal_filter.stats.speed_rejections / max(total_joints, 1)
    centroid_jump_rate = centroid_jumps / max(centroid_checks, 1)

    anomalies = AnomalyFlags(
        excess_person_detections=avg_persons > 2.5,
        unstable_person_count=stabilized_std > 0.8 if stabilized_person_counts else person_std > 0.8,
        high_low_confidence_ratio=low_conf_ratio > 0.12,
        low_mean_confidence=mean_conf < 0.55 and total_joints > 0,
        high_motion_rejection_rate=rejection_rate > 0.08,
        high_centroid_jump_rate=centroid_jump_rate > 0.15,
        high_covariance_spread=float(np.median(cov_spreads)) > 6.0 if cov_spreads else False,
        high_track_break_rate=track_break_rate > 0.30,
    )

    return CacheQualityMetrics(
        frames=len(person_counts),
        avg_persons=avg_persons,
        person_count_std=person_std,
        person_count_min=min(person_counts) if person_counts else 0,
        person_count_max=max(person_counts) if person_counts else 0,
        low_confidence_ratio=low_conf_ratio,
        mean_joint_confidence=mean_conf,
        motion_speed_rejections=temporal_filter.stats.speed_rejections,
        motion_oscillation_corrections=temporal_filter.stats.oscillation_corrections,
        rejection_rate_per_joint=rejection_rate,
        centroid_jump_rate=centroid_jump_rate,
        covariance_spread_ratio=float(np.median(cov_spreads)) if cov_spreads else 1.0,
        stabilized_person_count_std=stabilized_std,
        stabilized_person_count_min=min(stabilized_person_counts) if stabilized_person_counts else 0,
        stabilized_person_count_max=max(stabilized_person_counts) if stabilized_person_counts else 0,
        track_break_rate=track_break_rate,
        anomalies=anomalies,
    )
