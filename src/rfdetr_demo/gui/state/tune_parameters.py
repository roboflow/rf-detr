# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tune-preview parameter snapshot from GUI form values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rfdetr_demo.gui.state.ui_bindings import parse_max_ellipse_axis, parse_task
from rfdetr_demo.inference.types import KeypointUncertaintyStyle, TaskName


@dataclass(frozen=True)
class TuneParameters:
    """Parameters used for tune cache replay and auto_tune."""

    task: TaskName
    threshold: float
    keypoint_threshold: float
    person_only: bool
    motion_filter_enabled: bool
    motion_max_speed_fraction: float
    motion_ema_alpha: float
    motion_oscillation_enabled: bool
    ellipse_sigma: float
    heatmap_opacity: float
    heatmap_decay: float
    vertex_radius: int
    max_ellipse_axis: float | None
    keypoint_uncertainty_enabled: bool
    uncertainty_style: str


def build_tune_parameters(app: Any) -> TuneParameters:
    """Build :class:`TuneParameters` from ``VideoDemoGuiApp`` tk variables."""
    task = parse_task(app.task_var.get())
    style = (
        app.uncertainty_style_var.get()
        if task == "keypoint" and app.keypoint_uncertainty_var.get()
        else "none"
    )
    return TuneParameters(
        task=task,
        threshold=float(app.threshold_var.get()),
        keypoint_threshold=float(app.keypoint_threshold_var.get()),
        person_only=bool(app.person_only_var.get()),
        motion_filter_enabled=bool(app.motion_filter_var.get()),
        motion_max_speed_fraction=float(app.motion_max_speed_var.get()),
        motion_ema_alpha=float(app.motion_ema_alpha_var.get()),
        motion_oscillation_enabled=bool(app.motion_oscillation_var.get()),
        ellipse_sigma=float(app.ellipse_sigma_var.get()),
        heatmap_opacity=float(app.heatmap_opacity_var.get()),
        heatmap_decay=float(app.heatmap_decay_var.get()),
        vertex_radius=int(app.vertex_radius_var.get()),
        max_ellipse_axis=parse_max_ellipse_axis(float(app.max_ellipse_axis_var.get())),
        keypoint_uncertainty_enabled=bool(app.keypoint_uncertainty_var.get()),
        uncertainty_style=style,
    )


def keypoint_style(parameters: TuneParameters) -> KeypointUncertaintyStyle:
    if parameters.task == "keypoint" and parameters.keypoint_uncertainty_enabled:
        return cast(KeypointUncertaintyStyle, parameters.uncertainty_style)
    return "none"
