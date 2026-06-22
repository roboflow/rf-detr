# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tune-preview cache replay, auto_tune, and post-preview orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfdetr_demo.gui.state.job_state import TuneJobState
from rfdetr_demo.gui.state.tune_parameters import TuneParameters, keypoint_style
from rfdetr_demo.inference.overlays.detection import render_tune_cache_sequence
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings
from rfdetr_demo.inference.temporal_filter import MotionPlausibilitySettings
from rfdetr_demo.inference.types import TaskName
from rfdetr_demo.tuning.auto_tune import CurrentParameters, run_auto_tune


@dataclass(frozen=True)
class TuneLogLine:
    """One GUI log entry produced by tune operations."""

    message: str
    level: str = "info"


@dataclass(frozen=True)
class AutoTuneOutcome:
    """Result of an auto_tune pass for GUI application."""

    log_lines: list[TuneLogLine]
    proposed: Any | None = None
    apply_recommended: bool = False


@dataclass(frozen=True)
class TunePreviewCompletePlan:
    """Actions for the GUI after a tune preview segment finishes."""

    log_lines: list[TuneLogLine]
    progress_text: str
    status_message: str
    status_metrics: str
    run_auto_tune: bool
    refresh_live_preview: bool


class TuneController:
    """Stateless tune-preview operations (testable without Tk)."""

    @staticmethod
    def start_button_label(
        *,
        tune_state: TuneJobState,
        tune_mode: bool,
        compute_backend: str,
    ) -> str:
        if tune_state == TuneJobState.TUNE_PAUSED:
            return "本番実行"
        if tune_mode and compute_backend == "local":
            return "試走開始"
        return "開始"

    @staticmethod
    def to_current_parameters(parameters: TuneParameters) -> CurrentParameters:
        return CurrentParameters(
            threshold=parameters.threshold,
            keypoint_threshold=parameters.keypoint_threshold,
            motion_max_speed_fraction=parameters.motion_max_speed_fraction,
            motion_ema_alpha=parameters.motion_ema_alpha,
            motion_filter_enabled=parameters.motion_filter_enabled,
            motion_oscillation_enabled=parameters.motion_oscillation_enabled,
            ellipse_sigma=parameters.ellipse_sigma,
            heatmap_opacity=parameters.heatmap_opacity,
        )

    @staticmethod
    def build_motion_settings(parameters: TuneParameters) -> MotionPlausibilitySettings:
        return MotionPlausibilitySettings(
            enabled=parameters.motion_filter_enabled,
            max_speed_fraction_per_sec=parameters.motion_max_speed_fraction,
            ema_alpha=parameters.motion_ema_alpha,
            suppress_oscillation=parameters.motion_oscillation_enabled,
        )

    @staticmethod
    def build_keypoint_overlay_settings(
        parameters: TuneParameters,
        tune_cache: Any,
    ) -> KeypointOverlaySettings:
        frame_width, frame_height = 1920, 1080
        if tune_cache.has_entries:
            sample = tune_cache.latest
            if sample is not None:
                frame_height, frame_width = sample.frame_bgr.shape[:2]
        return KeypointOverlaySettings(
            keypoint_threshold=parameters.keypoint_threshold,
            uncertainty_enabled=parameters.keypoint_uncertainty_enabled,
            uncertainty_style=keypoint_style(parameters),
            ellipse_sigma=parameters.ellipse_sigma,
            max_ellipse_axis=parameters.max_ellipse_axis,
            heatmap_opacity=parameters.heatmap_opacity,
            heatmap_decay=parameters.heatmap_decay,
            vertex_radius=parameters.vertex_radius,
            frame_width=frame_width,
            frame_height=frame_height,
            motion=TuneController.build_motion_settings(parameters),
        )

    @staticmethod
    def render_live_preview(
        tune_cache: Any,
        parameters: TuneParameters,
        *,
        video_fps: float,
        frame_stride: int,
    ) -> Any:
        keypoint_settings = (
            TuneController.build_keypoint_overlay_settings(parameters, tune_cache)
            if parameters.task == "keypoint"
            else None
        )
        return render_tune_cache_sequence(
            tune_cache,
            task=parameters.task,
            threshold=parameters.threshold,
            person_only=parameters.person_only,
            keypoint_settings=keypoint_settings,
            fps=video_fps,
            frame_stride=frame_stride,
        )

    @staticmethod
    def live_preview_status_metrics(parameters: TuneParameters) -> str:
        return (
            f"σ={parameters.ellipse_sigma:.1f} · "
            f"不透明度={parameters.heatmap_opacity:.2f} · "
            f"{parameters.uncertainty_style}"
        )

    @staticmethod
    def run_auto_tune(
        tune_cache: Any,
        parameters: TuneParameters,
        *,
        apply: bool,
    ) -> AutoTuneOutcome:
        if not tune_cache.has_entries:
            return AutoTuneOutcome(
                log_lines=[TuneLogLine("自動調整: 試走キャッシュがありません", level="warn")],
            )
        if parameters.task != "keypoint":
            return AutoTuneOutcome(
                log_lines=[TuneLogLine("自動調整: キーポイントタスク専用です", level="warn")],
            )

        proposed, metrics, effectiveness = run_auto_tune(
            tune_cache,
            current=TuneController.to_current_parameters(parameters),
        )
        log_lines = [
            TuneLogLine(
                f"異常検知: 人数 {metrics.avg_persons:.1f} (σ={metrics.person_count_std:.2f}), "
                f"低信頼 {metrics.low_confidence_ratio:.0%}, "
                f"速度拒否率 {metrics.rejection_rate_per_joint:.1%}",
            ),
            *[TuneLogLine(f"  -> {reason}") for reason in proposed.reasons],
            TuneLogLine(effectiveness.summary),
        ]
        if apply and effectiveness.recommended:
            log_lines.append(TuneLogLine("自動調整を適用しました。プレビューを更新します。"))
            return AutoTuneOutcome(
                log_lines=log_lines,
                proposed=proposed,
                apply_recommended=True,
            )
        if apply and not effectiveness.recommended:
            log_lines.append(
                TuneLogLine(
                    "自動調整は適用しませんでした（効果が限定的）。手動調整を推奨します。",
                    level="warn",
                ),
            )
        return AutoTuneOutcome(log_lines=log_lines, proposed=proposed, apply_recommended=False)

    @staticmethod
    def plan_tune_preview_complete(
        summary: dict[str, Any],
        *,
        cache_count: int,
        live_preview_enabled: bool,
        auto_tune_enabled: bool,
        task: TaskName,
        parameters: TuneParameters,
    ) -> TunePreviewCompletePlan:
        preview_sec = summary.get("max_source_seconds", "?")
        processed = summary.get("processed_frames", "?")
        progress_text = f"試走完了  ·  {preview_sec} 秒  ·  推論 {processed} フレーム  ·  キャッシュ {cache_count}"
        log_lines = [
            TuneLogLine(
                f"試走完了: {processed} 推論フレーム, "
                f"{summary.get('total_detections', '?')} インスタンス, "
                f"{summary.get('elapsed_sec', '?')} 秒",
            ),
            TuneLogLine(f"試走出力: {summary['target']}"),
        ]
        speed_rej = int(summary.get("motion_speed_rejections", 0))
        osc_corr = int(summary.get("motion_oscillation_corrections", 0))
        if speed_rej or osc_corr:
            log_lines.append(
                TuneLogLine(
                    f"時系列フィルタ: 速度超過 {speed_rej} 件, 振動抑制 {osc_corr} 件",
                ),
            )

        run_auto = False
        refresh = False
        if live_preview_enabled and cache_count > 0:
            log_lines.append(
                TuneLogLine(
                    f"リアルタイムプレビュー: {cache_count} フレームをキャッシュ — "
                    "不確実性パラメータ変更で即反映",
                ),
            )
            run_auto = auto_tune_enabled and task == "keypoint"
            refresh = True
        else:
            log_lines.append(
                TuneLogLine(
                    "閾値・フレーム間隔・不確実性などを調整し、「本番実行」で全編を処理してください。",
                ),
            )

        status_metrics = TuneController.live_preview_status_metrics(parameters) if refresh else ""
        return TunePreviewCompletePlan(
            log_lines=log_lines,
            progress_text=progress_text,
            status_message="試走完了 — リアルタイム調整中" if refresh else "試走完了 — パラメータを調整してください",
            status_metrics=status_metrics,
            run_auto_tune=run_auto,
            refresh_live_preview=refresh,
        )
