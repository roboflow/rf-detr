# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Heatmap-style keypoint uncertainty annotators."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from supervision.draw.base import ImageType
from supervision.key_points.core import KeyPoints
from supervision.utils.conversion import ensure_cv2_image_for_class_method

from rfdetr_demo.inference.uncertainty.constants import (
    DEFAULT_HEATMAP_DECAY,
    DEFAULT_HEATMAP_OPACITY,
    DEFAULT_UNCERTAINTY_MAX_AXIS_PX,
    DEFAULT_UNCERTAINTY_SIGMA,
)
from rfdetr_demo.inference.uncertainty.geometry import (
    clamp_ellipse_axes,
    covariance_trace,
    decompose_covariance,
    joint_index_to_bgr,
)


def trace_to_bgr(normalized: float) -> tuple[int, int, int]:
    """Map normalized uncertainty [0, 1] to a BGR color using the JET colormap."""
    value = int(np.clip(normalized, 0.0, 1.0) * 255)
    jet_pixel = cv2.applyColorMap(np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
    return int(jet_pixel[0]), int(jet_pixel[1]), int(jet_pixel[2])


@dataclass(frozen=True)
class KeypointJointHeatmapAnnotator:
    """Draw per-joint colored uncertainty halos from pixel-space covariances."""

    sigma: float = DEFAULT_UNCERTAINTY_SIGMA
    opacity: float = DEFAULT_HEATMAP_OPACITY
    max_axis: float = DEFAULT_UNCERTAINTY_MAX_AXIS_PX
    decay: float = DEFAULT_HEATMAP_DECAY

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """Overlay joint-index heatmap ellipses on ``scene``."""
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances_raw = key_points.data.get("covariance")
        if covariances_raw is None:
            raise ValueError("key_points.data must contain 'covariance' with shape (N, K, 2, 2).")

        covariances = np.asarray(covariances_raw, dtype=np.float32)
        expected_shape = (*key_points.xy.shape[:2], 2, 2)
        if covariances.shape != expected_shape:
            raise ValueError(
                f"Expected covariance shape {expected_shape}, got {covariances.shape}.",
            )

        traces: list[float] = []
        entries: list[tuple[int, int, float, float, npt.NDArray[np.float32]]] = []
        for detection_index, xy in enumerate(key_points.xy):
            for point_index, (x, y) in enumerate(xy):
                if np.allclose((x, y), 0):
                    continue
                if (
                    key_points.visible is not None
                    and not key_points.visible[detection_index, point_index]
                ):
                    continue
                covariance = covariances[detection_index, point_index]
                trace_value = covariance_trace(covariance)
                if not np.isfinite(trace_value):
                    continue
                entries.append((detection_index, point_index, float(x), float(y), covariance))
                traces.append(trace_value)

        if not entries:
            return scene

        min_trace = min(traces)
        max_trace = max(traces)
        trace_span = max(max_trace - min_trace, 1e-6)

        h, w = scene.shape[:2]
        composite: npt.NDArray[np.float32] = scene.astype(np.float32)

        for _det_index, point_index, x, y, covariance in entries:
            decomposition = decompose_covariance(covariance)
            if decomposition is None:
                continue
            eigenvalues, _eigenvectors, angle = decomposition
            trace_value = covariance_trace(covariance)
            uncertainty_strength = (trace_value - min_trace) / trace_span
            peak_opacity = self.opacity * (0.35 + 0.65 * uncertainty_strength)

            ax, ay = clamp_ellipse_axes(
                eigenvalues,
                sigma=self.sigma,
                max_axis=self.max_axis,
            )
            center = (round(x), round(y))

            pad = 2
            x_min = max(center[0] - ax - pad, 0)
            x_max = min(center[0] + ax + pad, w)
            y_min = max(center[1] - ay - pad, 0)
            y_max = min(center[1] + ay + pad, h)
            if x_min >= x_max or y_min >= y_max:
                continue

            ys = np.arange(y_min, y_max, dtype=np.float32) - center[1]
            xs = np.arange(x_min, x_max, dtype=np.float32) - center[0]
            grid_x, grid_y = np.meshgrid(xs, ys)

            angle_rad = np.radians(-angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rx = grid_x * cos_a - grid_y * sin_a
            ry = grid_x * sin_a + grid_y * cos_a

            dist_sq = (rx / ax) ** 2 + (ry / ay) ** 2
            inside = dist_sq <= 1.0

            falloff = np.zeros_like(dist_sq)
            falloff[inside] = (1.0 - dist_sq[inside]) ** self.decay
            scaled_alpha = falloff * peak_opacity

            bgr = np.array(joint_index_to_bgr(point_index), dtype=np.float32)
            roi = composite[y_min:y_max, x_min:x_max]
            alpha_3 = scaled_alpha[:, :, np.newaxis]
            roi[:] = roi * (1.0 - alpha_3) + bgr * alpha_3

        np.copyto(scene, composite.astype(np.uint8))
        return scene


@dataclass(frozen=True)
class KeypointMagnitudeHeatmapAnnotator:
    """Draw uncertainty halos colored by magnitude (JET), not joint index."""

    sigma: float = DEFAULT_UNCERTAINTY_SIGMA
    opacity: float = DEFAULT_HEATMAP_OPACITY
    max_axis: float = DEFAULT_UNCERTAINTY_MAX_AXIS_PX
    decay: float = DEFAULT_HEATMAP_DECAY

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances_raw = key_points.data.get("covariance")
        if covariances_raw is None:
            return scene

        covariances = np.asarray(covariances_raw, dtype=np.float32)
        traces: list[float] = []
        entries: list[tuple[float, float, float, npt.NDArray[np.float32]]] = []
        for detection_index, xy in enumerate(key_points.xy):
            for point_index, (x, y) in enumerate(xy):
                if np.allclose((x, y), 0):
                    continue
                if (
                    key_points.visible is not None
                    and not key_points.visible[detection_index, point_index]
                ):
                    continue
                covariance = covariances[detection_index, point_index]
                trace_value = covariance_trace(covariance)
                if not np.isfinite(trace_value):
                    continue
                entries.append((float(x), float(y), trace_value, covariance))
                traces.append(trace_value)

        if not entries:
            return scene

        min_trace = min(traces)
        max_trace = max(traces)
        trace_span = max(max_trace - min_trace, 1e-6)
        h, w = scene.shape[:2]
        composite: npt.NDArray[np.float32] = scene.astype(np.float32)

        for x, y, trace_value, covariance in entries:
            decomposition = decompose_covariance(covariance)
            if decomposition is None:
                continue
            eigenvalues, _eigenvectors, angle = decomposition
            uncertainty_strength = (trace_value - min_trace) / trace_span
            peak_opacity = self.opacity * (0.35 + 0.65 * uncertainty_strength)
            bgr = np.array(trace_to_bgr(uncertainty_strength), dtype=np.float32)

            ax, ay = clamp_ellipse_axes(
                eigenvalues,
                sigma=self.sigma,
                max_axis=self.max_axis,
            )
            center = (round(x), round(y))
            pad = 2
            x_min = max(center[0] - ax - pad, 0)
            x_max = min(center[0] + ax + pad, w)
            y_min = max(center[1] - ay - pad, 0)
            y_max = min(center[1] + ay + pad, h)
            if x_min >= x_max or y_min >= y_max:
                continue

            ys = np.arange(y_min, y_max, dtype=np.float32) - center[1]
            xs = np.arange(x_min, x_max, dtype=np.float32) - center[0]
            grid_x, grid_y = np.meshgrid(xs, ys)
            angle_rad = np.radians(-angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rx = grid_x * cos_a - grid_y * sin_a
            ry = grid_x * sin_a + grid_y * cos_a
            dist_sq = (rx / ax) ** 2 + (ry / ay) ** 2
            inside = dist_sq <= 1.0
            falloff = np.zeros_like(dist_sq)
            falloff[inside] = (1.0 - dist_sq[inside]) ** self.decay
            scaled_alpha = falloff * peak_opacity
            roi = composite[y_min:y_max, x_min:x_max]
            alpha_3 = scaled_alpha[:, :, np.newaxis]
            roi[:] = roi * (1.0 - alpha_3) + bgr * alpha_3

        np.copyto(scene, composite.astype(np.uint8))
        return scene
