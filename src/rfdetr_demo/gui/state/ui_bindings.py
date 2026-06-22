# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build :class:`RunConfig` from Tk form variables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from rfdetr_demo.gui.state.job_state import ComputeBackend, RunConfig
from rfdetr_demo.inference.types import KeypointUncertaintyStyle, ModelSize, TaskName
from rfdetr_demo.paths import default_output_path


def parse_task(task_var_value: str) -> TaskName:
    if task_var_value == "keypoint":
        return "keypoint"
    if task_var_value == "segment":
        return "segment"
    return "detect"


def parse_max_frames(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("最大フレームは 1 以上の整数にしてください。")
    return parsed


def parse_tune_preview_seconds(value: float) -> float:
    if value <= 0:
        raise ValueError("試走時間は 0 より大きい秒数にしてください。")
    if value > 600:
        raise ValueError("試走時間は 600 秒以下にしてください。")
    return value


def parse_max_ellipse_axis(value: float) -> float | None:
    return None if value <= 0 else value


def resolve_output_path(
    *,
    source_path: Path,
    task: TaskName,
    explicit_output: str,
    keypoint_uncertainty_enabled: bool,
    keypoint_uncertainty_style: KeypointUncertaintyStyle,
) -> Path:
    if explicit_output.strip():
        return Path(explicit_output.strip())
    return default_output_path(
        source_path,
        task,
        keypoint_uncertainty=keypoint_uncertainty_enabled,
        keypoint_uncertainty_style=keypoint_uncertainty_style,
    )


def resolve_tune_preview_path(final_output: Path) -> Path:
    return final_output.with_name(f"{final_output.stem}_tune_preview{final_output.suffix}")


def keypoint_uncertainty_style_for_task(
    *,
    task: TaskName,
    enabled: bool,
    style: str,
) -> KeypointUncertaintyStyle:
    if task == "keypoint" and enabled:
        return cast(KeypointUncertaintyStyle, style)
    return "none"


def build_run_config(app: Any) -> RunConfig:
    """Build a :class:`RunConfig` from ``VideoDemoGuiApp`` tk variables."""
    task = parse_task(app.task_var.get())
    style = keypoint_uncertainty_style_for_task(
        task=task,
        enabled=bool(app.keypoint_uncertainty_var.get()),
        style=app.uncertainty_style_var.get(),
    )
    source_path = Path(app.source_var.get().strip())
    output_path = resolve_output_path(
        source_path=source_path,
        task=task,
        explicit_output=app.output_var.get(),
        keypoint_uncertainty_enabled=bool(app.keypoint_uncertainty_var.get()),
        keypoint_uncertainty_style=style,
    )
    offer = app._selected_vast_offer() if app.compute_var.get() == "vast" else None
    return RunConfig(
        source_path=source_path,
        output_path=output_path,
        task=task,
        model_size=cast(ModelSize, app.model_var.get()),
        threshold=float(app.threshold_var.get()),
        frame_stride=int(app.frame_stride_var.get()),
        max_frames=parse_max_frames(app.max_frames_var.get()),
        person_only=bool(app.person_only_var.get()),
        keypoint_threshold=float(app.keypoint_threshold_var.get()),
        keypoint_uncertainty_style=style,
        keypoint_uncertainty_enabled=bool(app.keypoint_uncertainty_var.get()),
        ellipse_sigma=float(app.ellipse_sigma_var.get()),
        max_ellipse_axis=parse_max_ellipse_axis(float(app.max_ellipse_axis_var.get())),
        heatmap_opacity=float(app.heatmap_opacity_var.get()),
        heatmap_decay=float(app.heatmap_decay_var.get()),
        vertex_radius=int(app.vertex_radius_var.get()),
        compute_backend=cast(ComputeBackend, app.compute_var.get()),
        tune_mode=bool(app.tune_mode_var.get()),
        tune_preview_seconds=float(app.tune_preview_seconds_var.get()),
        preview_enabled=bool(app.preview_enabled_var.get()),
        motion_filter_enabled=bool(app.motion_filter_var.get()),
        motion_max_speed_fraction=float(app.motion_max_speed_var.get()),
        motion_ema_alpha=float(app.motion_ema_alpha_var.get()),
        motion_oscillation_enabled=bool(app.motion_oscillation_var.get()),
        vast_offer_id=offer.offer_id if offer is not None else None,
        vast_api_key=app._resolve_vast_api_key_input(),
        vast_destroy_on_finish=bool(app.vast_destroy_var.get()),
    )
