# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Outline, cross, and filled ellipse uncertainty annotators."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from supervision.draw.base import ImageType
from supervision.key_points.core import KeyPoints
from supervision.utils.conversion import ensure_cv2_image_for_class_method

from rfdetr_demo.inference.uncertainty.constants import (
    DEFAULT_UNCERTAINTY_MAX_AXIS_PX,
    DEFAULT_UNCERTAINTY_SIGMA,
)
from rfdetr_demo.inference.uncertainty.geometry import (
    clamp_ellipse_axes,
    decompose_covariance,
    joint_index_to_bgr,
)


@dataclass(frozen=True)
class KeypointOutlineAnnotator:
    """Draw uncertainty ellipse outlines only (no fill)."""

    sigma: float = DEFAULT_UNCERTAINTY_SIGMA
    max_axis: float = DEFAULT_UNCERTAINTY_MAX_AXIS_PX
    thickness: int = 2
    use_joint_colors: bool = True

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances_raw = key_points.data.get("covariance")
        if covariances_raw is None:
            return scene

        covariances = np.asarray(covariances_raw, dtype=np.float32)
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
                decomposition = decompose_covariance(covariance)
                if decomposition is None:
                    continue
                eigenvalues, _eigenvectors, angle = decomposition
                ax, ay = clamp_ellipse_axes(
                    eigenvalues,
                    sigma=self.sigma,
                    max_axis=self.max_axis,
                )
                color = (
                    joint_index_to_bgr(point_index)
                    if self.use_joint_colors
                    else (0, 165, 255)
                )
                cv2.ellipse(
                    scene,
                    center=(round(x), round(y)),
                    axes=(ax, ay),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=self.thickness,
                    lineType=cv2.LINE_AA,
                )
        return scene


@dataclass(frozen=True)
class KeypointCrossAnnotator:
    """Draw crosshairs whose arm length reflects uncertainty."""

    sigma: float = DEFAULT_UNCERTAINTY_SIGMA
    max_axis: float = DEFAULT_UNCERTAINTY_MAX_AXIS_PX
    thickness: int = 2

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances_raw = key_points.data.get("covariance")
        if covariances_raw is None:
            return scene

        covariances = np.asarray(covariances_raw, dtype=np.float32)
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
                decomposition = decompose_covariance(covariance)
                if decomposition is None:
                    continue
                eigenvalues, _eigenvectors, _angle = decomposition
                ax, ay = clamp_ellipse_axes(
                    eigenvalues,
                    sigma=self.sigma,
                    max_axis=self.max_axis,
                )
                arm = max(ax, ay)
                color = joint_index_to_bgr(point_index)
                cx, cy = round(x), round(y)
                cv2.line(scene, (cx - arm, cy), (cx + arm, cy), color, self.thickness, cv2.LINE_AA)
                cv2.line(scene, (cx, cy - arm), (cx, cy + arm), color, self.thickness, cv2.LINE_AA)
        return scene


@dataclass(frozen=True)
class KeypointFilledEllipseAnnotator:
    """Draw solid semi-transparent uncertainty ellipses with joint-index colors."""

    sigma: float = DEFAULT_UNCERTAINTY_SIGMA
    opacity: float = 0.35
    max_axis: float = DEFAULT_UNCERTAINTY_MAX_AXIS_PX

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances_raw = key_points.data.get("covariance")
        if covariances_raw is None:
            return scene

        covariances = np.asarray(covariances_raw, dtype=np.float32)
        overlay = scene.copy()
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
                decomposition = decompose_covariance(covariance)
                if decomposition is None:
                    continue
                eigenvalues, _eigenvectors, angle = decomposition
                ax, ay = clamp_ellipse_axes(
                    eigenvalues,
                    sigma=self.sigma,
                    max_axis=self.max_axis,
                )
                color = joint_index_to_bgr(point_index)
                cv2.ellipse(
                    overlay,
                    center=(round(x), round(y)),
                    axes=(ax, ay),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        return cv2.addWeighted(overlay, self.opacity, scene, 1.0 - self.opacity, 0)


def annotate_joint_colored_vertices(
    scene: npt.NDArray[np.uint8],
    key_points: KeyPoints,
    *,
    radius: int = 4,
) -> npt.NDArray[np.uint8]:
    """Draw keypoint dots using the same joint-index palette as the heatmap."""
    if len(key_points) == 0:
        return scene

    for detection_index, xy in enumerate(key_points.xy):
        for point_index, (x, y) in enumerate(xy):
            if np.allclose((x, y), 0):
                continue
            if (
                key_points.visible is not None
                and not key_points.visible[detection_index, point_index]
            ):
                continue
            cv2.circle(
                img=scene,
                center=(int(x), int(y)),
                radius=radius,
                color=joint_index_to_bgr(point_index),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                img=scene,
                center=(int(x), int(y)),
                radius=radius,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    return scene
