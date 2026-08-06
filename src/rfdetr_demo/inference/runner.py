# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Core video inference runner (model load, callback wiring, I/O)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import supervision as sv

from rfdetr_demo.inference.callbacks import (
    make_detection_callback,
    make_detection_track_callback,
    make_keypoint_callback,
    make_segmentation_callback,
)
from rfdetr_demo.inference.models import (
    build_detection_model,
    build_keypoint_model,
    build_segmentation_model,
)
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, resolve_uncertainty_max_axis
from rfdetr_demo.inference.progress import compose_preview_callback, compose_progress_callback
from rfdetr_demo.inference.temporal_filter import MotionPlausibilitySettings
from rfdetr_demo.inference.types import (
    KeypointUncertaintyStyle,
    ModelSize,
    PreviewCallback,
    ProgressCallback,
    TaskName,
    VideoProcessingCancelledError,
)
from rfdetr_demo.inference.video_io import (
    cleanup_partial_video,
    finalize_video_path,
    partial_video_path,
    probe_video_size,
    process_video,
)
from rfdetr_demo.media.frame_audit import (
    ConfidentialFrameAuditLogger,
    is_frame_audit_enabled,
    wrap_callback_with_frame_audit,
)

logger = logging.getLogger(__name__)


def _default_motion_settings() -> MotionPlausibilitySettings:
    from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS

    defaults = DEFAULT_PARAMETERS
    return MotionPlausibilitySettings(
        enabled=defaults.motion_filter_enabled,
        max_speed_fraction_per_sec=defaults.motion_max_speed_fraction,
        ema_alpha=defaults.motion_ema_alpha,
        suppress_oscillation=defaults.motion_oscillation_enabled,
    )


def _build_task_callback(
    *,
    task: TaskName,
    model_size: ModelSize,
    threshold: float,
    person_only: bool,
    stats: dict[str, int],
    frame_width: int,
    frame_height: int,
    video_fps: float,
    frame_stride: int,
    keypoint_threshold: float,
    keypoint_uncertainty_style: KeypointUncertaintyStyle,
    keypoint_uncertainty_enabled: bool,
    ellipse_sigma: float,
    max_ellipse_axis: float | None,
    heatmap_opacity: float,
    heatmap_decay: float,
    vertex_radius: int,
    motion_settings: Any | None,
    tune_cache: Any | None,
    model_resolution: int | None = None,
    detect_track: bool = False,
) -> Callable[[Any, int], Any]:
    if task == "keypoint":
        model = build_keypoint_model(resolution=model_resolution)
        if motion_settings is None:
            motion_settings = _default_motion_settings()
        overlay_style: KeypointUncertaintyStyle = (
            keypoint_uncertainty_style
            if keypoint_uncertainty_enabled and keypoint_uncertainty_style != "none"
            else "none"
        )
        overlay_settings = KeypointOverlaySettings(
            keypoint_threshold=keypoint_threshold,
            uncertainty_enabled=keypoint_uncertainty_enabled,
            uncertainty_style=overlay_style,
            ellipse_sigma=ellipse_sigma,
            max_ellipse_axis=max_ellipse_axis,
            heatmap_opacity=heatmap_opacity,
            heatmap_decay=heatmap_decay,
            vertex_radius=vertex_radius,
            frame_width=frame_width,
            frame_height=frame_height,
            motion=motion_settings,
        )
        temporal_filter = None
        person_track_pipeline = None
        if motion_settings is not None and motion_settings.enabled:
            from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter

            temporal_filter = KeypointTemporalFilter(
                motion_settings,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=video_fps,
                frame_stride=frame_stride,
            )
        from rfdetr_demo.tracking.detection_stabilizer import is_detection_stabilizer_enabled
        from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
        from rfdetr_demo.tracking.types import PersonTrackSettings

        if is_detection_stabilizer_enabled():
            person_track_pipeline = PersonTrackPipeline.from_env(
                frame_width=frame_width,
                frame_height=frame_height,
                temporal_filter=temporal_filter,
                base_settings=PersonTrackSettings(enabled=True),
            )
            temporal_filter = None
        return make_keypoint_callback(
            model,
            threshold,
            overlay_settings,
            stats,
            tune_cache=tune_cache,
            temporal_filter=temporal_filter,
            person_track_pipeline=person_track_pipeline,
        )

    if task == "segment":
        model = build_segmentation_model(model_size)
        return make_segmentation_callback(
            model,
            threshold,
            person_only,
            sv.MaskAnnotator(),
            sv.LabelAnnotator(),
            stats,
            tune_cache=tune_cache,
        )

    model = build_detection_model(model_size, resolution=model_resolution)
    if detect_track:
        from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
        from rfdetr_demo.tracking.types import PersonTrackSettings

        track_pipeline = PersonTrackPipeline.from_env(
            frame_width=frame_width,
            frame_height=frame_height,
            base_settings=PersonTrackSettings(enabled=True),
        )
        return make_detection_track_callback(
            model,
            threshold,
            person_only,
            stats,
            track_pipeline,
            tune_cache=tune_cache,
        )
    return make_detection_callback(
        model,
        threshold,
        person_only,
        sv.BoxAnnotator(),
        sv.LabelAnnotator(),
        stats,
        tune_cache=tune_cache,
    )


def run_demo(
    source_path: Path,
    target_path: Path,
    task: TaskName,
    model_size: ModelSize,
    threshold: float,
    frame_stride: int,
    max_frames: int | None,
    person_only: bool,
    keypoint_threshold: float = 0.25,
    keypoint_uncertainty_style: KeypointUncertaintyStyle = "none",
    ellipse_sigma: float = 1.5,
    max_ellipse_axis: float | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    preview_stride: int | None = None,
    preview_min_interval_sec: float = 0.12,
    preview_max_width: int = 400,
    cancel_event: Event | None = None,
    progress_file: Path | None = None,
    max_source_seconds: float | None = None,
    tune_cache: Any | None = None,
    heatmap_opacity: float = 0.38,
    heatmap_decay: float = 3.0,
    vertex_radius: int = 4,
    keypoint_uncertainty_enabled: bool = True,
    motion_settings: Any | None = None,
    frame_audit_log_callback: Callable[[str], None] | None = None,
    model_resolution: int | None = None,
    detect_track: bool = False,
) -> dict[str, Any]:
    """Process ``source_path`` and write annotated video to ``target_path``.

    ``model_resolution`` overrides the detection/keypoint model input resolution
    (higher improves recall on small/distant people); ``None`` uses the default.
    ``detect_track`` (detect task only) tracks person detections through the
    box-IoU tracker, drawing stable ids and a live count.
    """
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source video not found: {source_path}. Place a video under confidential/media/input/ or pass --source."
        )
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_source_seconds is not None and max_source_seconds <= 0:
        raise ValueError(f"max_source_seconds must be > 0, got {max_source_seconds}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = partial_video_path(target_path)

    stats: dict[str, int] = {
        "processed_frames": 0,
        "total_detections": 0,
        "frame_raw_detections": 0,
        "frame_nms_detections": 0,
        "frame_active_tracks": 0,
        "frame_ghost_tracks": 0,
        "frame_live_tracks": 0,
        "motion_speed_rejections": 0,
        "motion_covariance_rejections": 0,
        "motion_oscillation_corrections": 0,
        "motion_smoothed_joints": 0,
    }
    frame_width, frame_height, video_fps = probe_video_size(source_path)
    resolved_max_axis = (
        resolve_uncertainty_max_axis(
            max_ellipse_axis,
            frame_width=frame_width,
            frame_height=frame_height,
            style=keypoint_uncertainty_style,
        )
        if task == "keypoint" and keypoint_uncertainty_style != "none"
        else None
    )

    callback = _build_task_callback(
        task=task,
        model_size=model_size,
        threshold=threshold,
        person_only=person_only,
        stats=stats,
        frame_width=frame_width,
        frame_height=frame_height,
        video_fps=video_fps,
        frame_stride=frame_stride,
        keypoint_threshold=keypoint_threshold,
        keypoint_uncertainty_style=keypoint_uncertainty_style,
        keypoint_uncertainty_enabled=keypoint_uncertainty_enabled,
        ellipse_sigma=ellipse_sigma,
        max_ellipse_axis=max_ellipse_axis,
        heatmap_opacity=heatmap_opacity,
        heatmap_decay=heatmap_decay,
        vertex_radius=vertex_radius,
        motion_settings=motion_settings,
        tune_cache=tune_cache,
        model_resolution=model_resolution,
        detect_track=detect_track,
    )

    frame_audit: ConfidentialFrameAuditLogger | None = None
    if is_frame_audit_enabled():
        frame_audit = ConfidentialFrameAuditLogger(source_path=source_path, task=task)

        def _on_frame_audited(record: Any) -> None:
            if frame_audit_log_callback is None:
                return
            line = (
                f"[機密] #{record.processed_count:02d} "
                f"src={record.frame_index} 検出={record.instance_count} "
                f"→ {record.image_relpath}"
            )
            frame_audit_log_callback(line)

        callback = wrap_callback_with_frame_audit(
            callback,
            audit=frame_audit,
            stats=stats,
            on_frame_logged=_on_frame_audited if frame_audit_log_callback else None,
        )

    started = time.perf_counter()
    if partial_path.exists():
        cleanup_partial_video(partial_path)

    combined_progress = compose_progress_callback(progress_callback, progress_file)
    preview_throttle = compose_preview_callback(
        preview_callback,
        preview_stride=preview_stride,
        preview_min_interval_sec=preview_min_interval_sec,
        preview_max_width=preview_max_width,
        frame_stride=frame_stride,
    )

    try:
        process_video(
            source_path=source_path,
            target_path=partial_path,
            callback=callback,
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_source_seconds=max_source_seconds,
            stats=stats,
            progress_callback=combined_progress,
            preview_throttle=preview_throttle,
            cancel_event=cancel_event,
        )
        finalize_video_path(partial_path, target_path)
    except VideoProcessingCancelledError:
        cleanup_partial_video(partial_path)
        raise
    except Exception:
        cleanup_partial_video(partial_path)
        raise
    elapsed_sec = time.perf_counter() - started

    summary: dict[str, Any] = {
        "source": str(source_path.resolve()),
        "target": str(target_path.resolve()),
        "task": task,
        "model_size": model_size if task in {"detect", "segment"} else "keypoint-preview",
        "threshold": threshold,
        "frame_stride": frame_stride,
        "person_only": person_only if task in {"detect", "segment"} else None,
        "keypoint_threshold": keypoint_threshold if task == "keypoint" else None,
        "keypoint_uncertainty_style": keypoint_uncertainty_style if task == "keypoint" else None,
        "ellipse_sigma": ellipse_sigma if task == "keypoint" else None,
        "max_ellipse_axis": resolved_max_axis if task == "keypoint" else None,
        "heatmap_opacity": heatmap_opacity if task == "keypoint" else None,
        "heatmap_decay": heatmap_decay if task == "keypoint" else None,
        "vertex_radius": vertex_radius if task == "keypoint" else None,
        "motion_speed_rejections": stats.get("motion_speed_rejections", 0),
        "motion_oscillation_corrections": stats.get("motion_oscillation_corrections", 0),
        "processed_frames": stats["processed_frames"],
        "total_detections": stats["total_detections"],
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if max_source_seconds is not None:
        summary["max_source_seconds"] = max_source_seconds
        summary["tune_preview"] = True
    if stats["processed_frames"] > 0:
        summary["avg_detections_per_frame"] = round(
            stats["total_detections"] / stats["processed_frames"],
            2,
        )
        summary["avg_fps"] = round(stats["processed_frames"] / elapsed_sec, 2)
    if frame_audit is not None:
        audit_summary = frame_audit.finalize()
        if audit_summary is not None:
            summary["frame_audit"] = {
                "run_id": audit_summary.run_id,
                "run_dir_relpath": audit_summary.run_dir_relpath,
                "frames_logged": audit_summary.frames_logged,
                "evaluation": audit_summary.evaluation,
            }
    return summary
