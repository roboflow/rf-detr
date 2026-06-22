# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run local / Vast video jobs from GUI configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

from rfdetr_demo.gui.state.job_state import RunConfig, StartJobError, StartJobPlan, TuneJobState
from rfdetr_demo.gui.state.ui_bindings import (
    parse_tune_preview_seconds,
    resolve_tune_preview_path,
)
from rfdetr_demo.inference.runner import run_demo
from rfdetr_demo.inference.temporal_filter import MotionPlausibilitySettings
from rfdetr_demo.inference.types import PreviewCallback, ProgressCallback
from rfdetr_demo.vast.runner import VastVideoJobConfig, run_video_demo_on_vast
from rfdetr_demo.vast.start_phases import VastProgressUpdate


@dataclass(frozen=True)
class LocalJobCallbacks:
    """Thread-safe hooks invoked by :class:`RunController` during local runs."""

    progress_callback: ProgressCallback | None = None
    preview_callback: PreviewCallback | None = None
    cancel_event: Event | None = None
    frame_audit_log_callback: Callable[[str], None] | None = None


@dataclass(frozen=True)
class VastJobCallbacks:
    """Thread-safe hooks invoked by :class:`RunController` during Vast runs."""

    cancel_event: Event | None = None
    log_callback: Callable[[str], None] | None = None
    phase_callback: Callable[[VastProgressUpdate], None] | None = None


class RunController:
    """Prepare and execute GUI inference jobs (testable without Tk)."""

    @staticmethod
    def prepare_start(
        config: RunConfig,
        *,
        tune_state: TuneJobState,
    ) -> StartJobPlan | StartJobError:
        if not config.source_path.is_file():
            return StartJobError(
                title="入力エラー",
                message=f"動画が見つかりません:\n{config.source_path}",
            )

        is_full_run_after_tune = (
            tune_state == TuneJobState.TUNE_PAUSED and config.compute_backend == "local"
        )
        is_tune_preview_run = False
        max_source_seconds: float | None = None
        output_path = config.output_path

        if is_full_run_after_tune:
            pass
        elif config.compute_backend == "local" and config.tune_mode:
            try:
                max_source_seconds = parse_tune_preview_seconds(config.tune_preview_seconds)
            except ValueError as error:
                return StartJobError(title="入力エラー", message=str(error))
            output_path = resolve_tune_preview_path(output_path)
            is_tune_preview_run = True
        elif config.tune_mode and config.compute_backend == "vast":
            return StartJobError(
                title="試走モード",
                message=(
                    "試走＋調整モードはローカル実行のみ利用できます。\n"
                    "実行先を「ローカル」に変更してください。"
                ),
            )

        if config.compute_backend == "vast" and config.vast_offer_id is None:
            return StartJobError(
                title="Vast.ai",
                message="GPU オファーが選択されていません。\n「GPU 検索」で一覧を取得してください。",
            )

        effective_max_frames = None if is_tune_preview_run else config.max_frames
        return StartJobPlan(
            config=config,
            output_path=output_path,
            is_tune_preview_run=is_tune_preview_run,
            is_full_run_after_tune=is_full_run_after_tune,
            max_source_seconds=max_source_seconds,
            effective_max_frames=effective_max_frames,
        )

    @staticmethod
    def build_motion_settings(config: RunConfig) -> MotionPlausibilitySettings:
        return MotionPlausibilitySettings(
            enabled=config.motion_filter_enabled,
            max_speed_fraction_per_sec=config.motion_max_speed_fraction,
            ema_alpha=config.motion_ema_alpha,
            suppress_oscillation=config.motion_oscillation_enabled,
        )

    @staticmethod
    def build_local_job_kwargs(
        plan: StartJobPlan,
        *,
        callbacks: LocalJobCallbacks,
        tune_cache: Any | None,
    ) -> dict[str, Any]:
        config = plan.config
        return {
            "source_path": config.source_path,
            "target_path": plan.output_path,
            "task": config.task,
            "model_size": config.model_size,
            "threshold": config.threshold,
            "frame_stride": config.frame_stride,
            "max_frames": plan.effective_max_frames,
            "person_only": config.person_only,
            "keypoint_threshold": config.keypoint_threshold,
            "keypoint_uncertainty_style": config.keypoint_uncertainty_style,
            "ellipse_sigma": config.ellipse_sigma,
            "max_ellipse_axis": config.max_ellipse_axis,
            "heatmap_opacity": config.heatmap_opacity,
            "heatmap_decay": config.heatmap_decay,
            "vertex_radius": config.vertex_radius,
            "keypoint_uncertainty_enabled": config.keypoint_uncertainty_enabled,
            "progress_callback": callbacks.progress_callback,
            "preview_callback": callbacks.preview_callback,
            "cancel_event": callbacks.cancel_event,
            "max_source_seconds": plan.max_source_seconds,
            "tune_cache": tune_cache if plan.is_tune_preview_run else None,
            "motion_settings": (
                RunController.build_motion_settings(config)
                if config.task == "keypoint"
                else None
            ),
            "frame_audit_log_callback": callbacks.frame_audit_log_callback,
        }

    @staticmethod
    def build_vast_config(plan: StartJobPlan) -> VastVideoJobConfig:
        config = plan.config
        assert config.vast_offer_id is not None
        return VastVideoJobConfig(
            source_path=config.source_path,
            target_path=plan.output_path,
            task=config.task,
            model_size=config.model_size,
            threshold=config.threshold,
            frame_stride=config.frame_stride,
            max_frames=config.max_frames,
            person_only=config.person_only,
            keypoint_threshold=0.0,
            keypoint_uncertainty_style=config.keypoint_uncertainty_style,
            ellipse_sigma=config.ellipse_sigma,
            max_ellipse_axis=config.max_ellipse_axis,
            offer_id=config.vast_offer_id,
            api_key=config.vast_api_key,
            destroy_on_finish=config.vast_destroy_on_finish,
            user_acknowledged=True,
        )

    @staticmethod
    def run_local(**job_kwargs: Any) -> dict[str, Any]:
        return run_demo(**job_kwargs)

    @staticmethod
    def run_vast(
        config: VastVideoJobConfig,
        *,
        callbacks: VastJobCallbacks,
    ) -> dict[str, Any]:
        return run_video_demo_on_vast(
            config,
            cancel_event=callbacks.cancel_event,
            log_callback=callbacks.log_callback,
            phase_callback=callbacks.phase_callback,
        )


def frame_audit_gui_lines(summary: dict[str, Any]) -> list[str]:
    """Return GUI log lines for a frame audit payload embedded in ``run_demo`` summary."""
    audit_payload = summary.get("frame_audit")
    if not isinstance(audit_payload, dict):
        return []
    from rfdetr_demo.media.frame_audit import FrameAuditSummary

    audit_summary = FrameAuditSummary(
        run_id=str(audit_payload.get("run_id", "")),
        source_relpath=str(summary.get("source", "")),
        task=str(summary.get("task", "")),
        frames_logged=int(audit_payload.get("frames_logged", 0)),
        evaluation=audit_payload.get("evaluation", {}),
        run_dir_relpath=str(audit_payload.get("run_dir_relpath", "")),
    )
    return audit_summary.gui_log_lines()
