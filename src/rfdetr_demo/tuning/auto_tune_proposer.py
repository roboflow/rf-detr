# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Parameter proposal and effectiveness evaluation for auto-tune."""

from __future__ import annotations

import numpy as np

from rfdetr.visualize.keypoints import key_points_for_display
from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter, MotionPlausibilitySettings
from rfdetr_demo.inference.tune_cache import TunePreviewCache, deserialize_key_points
from rfdetr_demo.tracking.keypoints_ops import attach_track_ids
from rfdetr_demo.tracking.person_associator import PersonAssociator
from rfdetr_demo.tuning.auto_tune_metrics import count_persons
from rfdetr_demo.tuning.auto_tune_types import (
    AutoTuneEffectiveness,
    CacheQualityMetrics,
    CurrentParameters,
    ProposedParameters,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def propose_parameters(
    metrics: CacheQualityMetrics,
    *,
    current: CurrentParameters,
) -> ProposedParameters:
    """Map detected anomalies to bounded parameter adjustments."""
    threshold = current.threshold
    keypoint_threshold = current.keypoint_threshold
    motion_speed = current.motion_max_speed_fraction
    ema_alpha = current.motion_ema_alpha
    ellipse_sigma = current.ellipse_sigma
    heatmap_opacity = current.heatmap_opacity
    motion_enabled = current.motion_filter_enabled
    oscillation_enabled = current.motion_oscillation_enabled
    reasons: list[str] = []

    if metrics.frames == 0:
        return ProposedParameters(
            threshold=threshold,
            keypoint_threshold=keypoint_threshold,
            motion_max_speed_fraction=motion_speed,
            motion_ema_alpha=ema_alpha,
            motion_filter_enabled=motion_enabled,
            motion_oscillation_enabled=oscillation_enabled,
            ellipse_sigma=ellipse_sigma,
            heatmap_opacity=heatmap_opacity,
            reasons=["試走キャッシュが空のため現状維持"],
        )

    anomalies = metrics.anomalies

    if anomalies.excess_person_detections:
        bump = clamp(0.05 * (metrics.avg_persons - 2.0), 0.05, 0.20)
        threshold = clamp(threshold + bump, 0.45, 0.85)
        reasons.append(
            f"検出人数過多 (平均 {metrics.avg_persons:.1f}) -> 閾値 +{bump:.2f}",
        )

    if anomalies.unstable_person_count:
        under_count = metrics.person_count_min < metrics.avg_persons - 0.3
        over_count = metrics.person_count_max > metrics.avg_persons + 0.8
        if under_count and not over_count:
            threshold = clamp(threshold - 0.05, 0.45, 0.85)
            reasons.append(
                f"人数変動（不足側 {metrics.person_count_min}-{metrics.person_count_max}）"
                " -> 閾値 -0.05",
            )
        elif over_count:
            threshold = clamp(threshold + 0.05, 0.45, 0.85)
            reasons.append(
                f"人数変動（過多側 {metrics.person_count_min}-{metrics.person_count_max}）"
                " -> 閾値 +0.05",
            )
        else:
            reasons.append(
                f"人数変動 ({metrics.person_count_min}-{metrics.person_count_max})"
                " -> 検出安定化レイヤを優先（閾値維持）",
            )

    if anomalies.high_low_confidence_ratio or anomalies.low_mean_confidence:
        target = clamp(0.15 + metrics.low_confidence_ratio * 0.5, 0.15, 0.45)
        if target > keypoint_threshold:
            keypoint_threshold = target
            reasons.append(
                f"低信頼関節 {metrics.low_confidence_ratio:.0%} -> 関節信頼度 {target:.2f}",
            )

    if anomalies.high_motion_rejection_rate or anomalies.high_centroid_jump_rate:
        motion_enabled = True
        oscillation_enabled = True
        reduction = 0.05 if metrics.rejection_rate_per_joint < 0.2 else 0.08
        motion_speed = clamp(motion_speed - reduction, 0.08, 0.50)
        ema_alpha = clamp(ema_alpha - 0.10, 0.30, 0.85)
        reasons.append(
            f"速度異常/重心ジャンプ -> 最大速度 {motion_speed:.2f}, 平滑化 alpha {ema_alpha:.2f}",
        )

    if anomalies.high_covariance_spread:
        ellipse_sigma = clamp(ellipse_sigma - 0.2, 1.0, 2.5)
        heatmap_opacity = clamp(heatmap_opacity - 0.05, 0.20, 0.55)
        reasons.append("不確実性の偏り -> sigma/不透明度を抑制")

    if not reasons:
        reasons.append("重大な異常なし - デフォルトを維持")

    return ProposedParameters(
        threshold=round(threshold, 2),
        keypoint_threshold=round(keypoint_threshold, 2),
        motion_max_speed_fraction=round(motion_speed, 2),
        motion_ema_alpha=round(ema_alpha, 2),
        motion_filter_enabled=motion_enabled,
        motion_oscillation_enabled=oscillation_enabled,
        ellipse_sigma=round(ellipse_sigma, 2),
        heatmap_opacity=round(heatmap_opacity, 2),
        reasons=reasons,
    )


def simulate_rejection_rate(
    cache: TunePreviewCache,
    *,
    threshold: float,
    keypoint_threshold: float,
    motion_max_speed_fraction: float,
    motion_ema_alpha: float,
    motion_filter_enabled: bool,
    motion_oscillation_enabled: bool,
) -> tuple[float, float]:
    """Return (rejection_rate_per_joint, person_count_std) for a parameter set."""
    if not cache.has_entries:
        return 0.0, 0.0

    sample = cache.latest
    frame_height, frame_width = sample.frame_bgr.shape[:2] if sample is not None else (480, 640)
    motion = MotionPlausibilitySettings(
        enabled=motion_filter_enabled,
        max_speed_fraction_per_sec=motion_max_speed_fraction,
        ema_alpha=motion_ema_alpha,
        suppress_oscillation=motion_oscillation_enabled,
    )
    temporal_filter = KeypointTemporalFilter(
        motion,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=cache.fps,
        frame_stride=cache.frame_stride,
    )
    associator = PersonAssociator()

    person_counts: list[int] = []
    total_joints = 0

    for entry in cache.entries:
        if entry.key_points_payload is None:
            continue
        raw = deserialize_key_points(entry.key_points_payload)
        person_counts.append(count_persons(raw, threshold))
        display = key_points_for_display(raw, keypoint_threshold=keypoint_threshold)
        for det_index in range(len(display)):
            visible = display.visible
            for joint_index in range(display.xy.shape[1]):
                if visible is not None and not visible[det_index, joint_index]:
                    continue
                if np.allclose(display.xy[det_index, joint_index], 0):
                    continue
                total_joints += 1
        temporal_filter.apply(
            attach_track_ids(raw, associator.assign(raw)),
            entry.frame_index,
        )

    person_std = float(np.std(person_counts)) if person_counts else 0.0
    rejection_rate = temporal_filter.stats.speed_rejections / max(total_joints, 1)
    return rejection_rate, person_std


def evaluate_auto_tune(
    cache: TunePreviewCache,
    *,
    current: CurrentParameters,
    proposed: ProposedParameters,
) -> AutoTuneEffectiveness:
    """Compare current vs proposed parameters on cached frames."""
    before_rej, before_std = simulate_rejection_rate(
        cache,
        threshold=current.threshold,
        keypoint_threshold=current.keypoint_threshold,
        motion_max_speed_fraction=current.motion_max_speed_fraction,
        motion_ema_alpha=current.motion_ema_alpha,
        motion_filter_enabled=current.motion_filter_enabled,
        motion_oscillation_enabled=current.motion_oscillation_enabled,
    )
    after_rej, after_std = simulate_rejection_rate(
        cache,
        threshold=proposed.threshold,
        keypoint_threshold=proposed.keypoint_threshold,
        motion_max_speed_fraction=proposed.motion_max_speed_fraction,
        motion_ema_alpha=proposed.motion_ema_alpha,
        motion_filter_enabled=proposed.motion_filter_enabled,
        motion_oscillation_enabled=proposed.motion_oscillation_enabled,
    )

    rej_improve = (
        (before_rej - after_rej) / max(before_rej, 1e-6) * 100.0
        if before_rej > 0
        else 0.0
    )
    std_improve = (
        (before_std - after_std) / max(before_std, 1e-6) * 100.0
        if before_std > 0
        else 0.0
    )

    confidence = clamp(
        0.35 * (rej_improve / 100.0)
        + 0.35 * (std_improve / 100.0)
        + 0.30 * (1.0 if len(cache.entries) >= 20 else len(cache.entries) / 20.0),
        0.0,
        1.0,
    )
    recommended = confidence >= 0.25 and (rej_improve > 5.0 or std_improve > 5.0 or before_rej > 0.1)

    if not recommended:
        summary = (
            f"自動調整の効果は限定的 (信頼度 {confidence:.0%})。"
            f"拒否率 {before_rej:.1%}->{after_rej:.1%}, 人数σ {before_std:.2f}->{after_std:.2f}"
        )
    else:
        summary = (
            f"自動調整を推奨 (信頼度 {confidence:.0%}): "
            f"拒否率 {rej_improve:+.0f}%, 人数安定 {std_improve:+.0f}%"
        )

    return AutoTuneEffectiveness(
        recommended=recommended,
        confidence=confidence,
        before_rejection_rate=before_rej,
        after_rejection_rate=after_rej,
        before_person_std=before_std,
        after_person_std=after_std,
        rejection_improvement_pct=rej_improve,
        person_stability_improvement_pct=std_improve,
        summary=summary,
    )
