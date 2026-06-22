# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint overlay rendering and uncertainty annotators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import supervision as sv

from rfdetr.visualize.keypoints import key_points_for_display
from rfdetr_demo.inference.temporal_filter import MotionPlausibilitySettings
from rfdetr_demo.inference.types import KeypointUncertaintyStyle
from rfdetr_demo.tracking.detection_stabilizer import detection_bbox, partition_live_and_ghost


@dataclass(frozen=True)
class KeypointOverlaySettings:
    """Parameters that control keypoint skeleton and uncertainty rendering."""

    keypoint_threshold: float = 0.0
    uncertainty_enabled: bool = True
    uncertainty_style: KeypointUncertaintyStyle = "heatmap"
    ellipse_sigma: float = 1.5
    max_ellipse_axis: float | None = None
    heatmap_opacity: float = 0.38
    heatmap_decay: float = 3.0
    vertex_radius: int = 4
    frame_width: int = 1920
    frame_height: int = 1080
    motion: MotionPlausibilitySettings | None = None


def resolve_uncertainty_max_axis(
    user_value: float | None,
    *,
    frame_width: int,
    frame_height: int,
    style: KeypointUncertaintyStyle,
) -> float | None:
    """Resolve the pixel cap applied to uncertainty ellipse axes."""
    if user_value is not None and user_value <= 0:
        return None
    if user_value is not None and user_value > 0:
        return float(user_value)
    if style in {"heatmap", "magnitude", "outline", "cross", "filled"}:
        from rfdetr_demo.inference.uncertainty_viz import resolve_max_ellipse_axis

        return resolve_max_ellipse_axis(None, frame_width=frame_width, frame_height=frame_height)
    from rfdetr_demo.inference.uncertainty_viz import DEFAULT_UNCERTAINTY_MAX_AXIS_PX, resolve_max_ellipse_axis

    return resolve_max_ellipse_axis(
        DEFAULT_UNCERTAINTY_MAX_AXIS_PX,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def build_keypoint_uncertainty_annotator(
    style: KeypointUncertaintyStyle,
    *,
    ellipse_sigma: float,
    max_ellipse_axis: float | None,
    heatmap_opacity: float = 0.38,
    heatmap_decay: float = 3.0,
) -> Any:
    """Return a Supervision or custom annotator for per-joint uncertainty zones."""
    if style == "none":
        return None
    if style == "heatmap":
        from rfdetr_demo.inference.uncertainty_viz import KeypointJointHeatmapAnnotator

        heatmap_max_axis = max_ellipse_axis if max_ellipse_axis is not None else 36.0
        return KeypointJointHeatmapAnnotator(
            sigma=ellipse_sigma,
            max_axis=heatmap_max_axis,
            opacity=heatmap_opacity,
            decay=heatmap_decay,
        )
    if style == "magnitude":
        from rfdetr_demo.inference.uncertainty_viz import KeypointMagnitudeHeatmapAnnotator

        heatmap_max_axis = max_ellipse_axis if max_ellipse_axis is not None else 36.0
        return KeypointMagnitudeHeatmapAnnotator(
            sigma=ellipse_sigma,
            max_axis=heatmap_max_axis,
            opacity=heatmap_opacity,
            decay=heatmap_decay,
        )
    if style == "outline":
        from rfdetr_demo.inference.uncertainty_viz import KeypointOutlineAnnotator

        outline_max_axis = max_ellipse_axis if max_ellipse_axis is not None else 36.0
        return KeypointOutlineAnnotator(
            sigma=ellipse_sigma,
            max_axis=outline_max_axis,
        )
    if style == "cross":
        from rfdetr_demo.inference.uncertainty_viz import KeypointCrossAnnotator

        cross_max_axis = max_ellipse_axis if max_ellipse_axis is not None else 36.0
        return KeypointCrossAnnotator(
            sigma=ellipse_sigma,
            max_axis=cross_max_axis,
        )
    if style == "filled":
        from rfdetr_demo.inference.uncertainty_viz import KeypointFilledEllipseAnnotator

        filled_max_axis = max_ellipse_axis if max_ellipse_axis is not None else 36.0
        return KeypointFilledEllipseAnnotator(
            sigma=ellipse_sigma,
            max_axis=filled_max_axis,
            opacity=min(1.0, max(0.05, heatmap_opacity)),
        )
    kwargs: dict[str, Any] = {
        "sigma": ellipse_sigma,
        "color": sv.Color.ROBOFLOW,
        "opacity": min(1.0, max(0.05, heatmap_opacity)) if style == "halo" else 0.45,
    }
    if max_ellipse_axis is not None:
        kwargs["max_axis"] = max_ellipse_axis
    if style == "halo":
        return sv.VertexEllipseHaloAnnotator(**kwargs)
    return sv.VertexEllipseAnnotator(**kwargs)


def _render_ghost_hold_indicators(
    frame_bgr: np.ndarray,
    ghost_key_points: sv.KeyPoints,
) -> np.ndarray:
    """Draw a faint bbox for ghost holds instead of a frozen skeleton."""
    if len(ghost_key_points) == 0:
        return frame_bgr
    annotated = frame_bgr.copy()
    for detection_index in range(len(ghost_key_points)):
        box = detection_bbox(ghost_key_points, detection_index)
        if box is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "HOLD",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _render_live_keypoint_overlay(
    frame_bgr: np.ndarray,
    live_key_points: sv.KeyPoints,
    settings: KeypointOverlaySettings,
) -> np.ndarray:
    """Render skeleton / uncertainty overlays for non-ghost detections only."""
    if len(live_key_points) == 0:
        return frame_bgr

    key_points = key_points_for_display(
        live_key_points,
        keypoint_threshold=settings.keypoint_threshold,
    )
    resolved_max_axis = (
        resolve_uncertainty_max_axis(
            settings.max_ellipse_axis,
            frame_width=settings.frame_width,
            frame_height=settings.frame_height,
            style=settings.uncertainty_style if settings.uncertainty_enabled else "none",
        )
        if settings.uncertainty_enabled and settings.uncertainty_style != "none"
        else None
    )
    style = settings.uncertainty_style if settings.uncertainty_enabled else "none"
    uncertainty_annotator = build_keypoint_uncertainty_annotator(
        style,
        ellipse_sigma=settings.ellipse_sigma,
        max_ellipse_axis=resolved_max_axis,
        heatmap_opacity=settings.heatmap_opacity,
        heatmap_decay=settings.heatmap_decay,
    )
    edge_color = sv.Color.WHITE if style in {"heatmap", "magnitude"} else sv.Color.ROBOFLOW
    edge_thickness = 1 if style in {"heatmap", "magnitude"} else 2
    vertex_annotator = sv.VertexAnnotator(color=edge_color, radius=settings.vertex_radius)
    edge_annotator = sv.EdgeAnnotator(color=edge_color, thickness=edge_thickness)

    annotated = frame_bgr.copy()
    if uncertainty_annotator is not None and "covariance" in key_points.data:
        annotated = uncertainty_annotator.annotate(annotated, key_points)
    annotated = edge_annotator.annotate(annotated, key_points)
    if style in {"heatmap", "magnitude"}:
        from rfdetr_demo.inference.uncertainty_viz import annotate_joint_colored_vertices

        return annotate_joint_colored_vertices(
            annotated,
            key_points,
            radius=settings.vertex_radius,
        )
    return vertex_annotator.annotate(annotated, key_points)


def render_keypoint_overlay(
    frame_bgr: np.ndarray,
    key_points_raw: sv.KeyPoints,
    settings: KeypointOverlaySettings,
) -> np.ndarray:
    """Apply skeleton and uncertainty overlays without running inference."""
    live_key_points, ghost_key_points = partition_live_and_ghost(key_points_raw)
    annotated = _render_live_keypoint_overlay(frame_bgr, live_key_points, settings)
    return _render_ghost_hold_indicators(annotated, ghost_key_points)
