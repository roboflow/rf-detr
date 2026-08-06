# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lightweight CPU appearance descriptors (color histograms) for ReID.

The descriptor is an HSV hue-saturation histogram of a person's torso region,
usable to re-associate tracks after long occlusions without an embedding model.
OpenCV is imported lazily so the tracking package stays importable without it;
only the ReID code path requires ``cv2``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import supervision as sv

# COCO-17 torso joints: left/right shoulder, left/right hip.
_TORSO_JOINTS: tuple[int, ...] = (5, 6, 11, 12)


def _visible_torso_points(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    """Return visible torso joint coordinates for one detection, or None."""
    xy = key_points.xy[detection_index]
    if key_points.visible is not None:
        visible = key_points.visible[detection_index]
    else:
        visible = ~np.all(np.isclose(xy, 0), axis=1)
    joints = [index for index in _TORSO_JOINTS if index < len(xy) and visible[index]]
    if len(joints) < 2:
        return None
    return xy[joints]


def appearance_roi(
    key_points: sv.KeyPoints,
    detection_index: int,
    person_box: np.ndarray,
) -> np.ndarray | None:
    """Return an ``[x1, y1, x2, y2]`` torso ROI, preferring keypoints over the box.

    Falls back to the central band of ``person_box`` (horizontally centered,
    vertical 20-65%) when torso joints are not visible, which keeps most of the
    background out of the color histogram.

    Args:
        key_points: Detections for the current frame.
        detection_index: Row to build an ROI for.
        person_box: Full-person ``[x1, y1, x2, y2]`` fallback box.

    Returns:
        A float ROI box, or None when it would be degenerate.
    """
    points = _visible_torso_points(key_points, detection_index)
    if points is not None:
        x1 = float(points[:, 0].min())
        y1 = float(points[:, 1].min())
        x2 = float(points[:, 0].max())
        y2 = float(points[:, 1].max())
        if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
            return np.array([x1, y1, x2, y2], dtype=np.float64)

    bx1, by1, bx2, by2 = (float(value) for value in person_box)
    width = bx2 - bx1
    height = by2 - by1
    if width < 2.0 or height < 2.0:
        return None
    return np.array(
        [
            bx1 + 0.25 * width,
            by1 + 0.20 * height,
            bx2 - 0.25 * width,
            by1 + 0.65 * height,
        ],
        dtype=np.float64,
    )


def crop_roi(frame_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    """Return the ROI sub-image clipped to frame bounds, or None if degenerate."""
    height, width = frame_bgr.shape[:2]
    x1 = max(0, min(width, int(round(float(roi[0])))))
    y1 = max(0, min(height, int(round(float(roi[1])))))
    x2 = max(0, min(width, int(round(float(roi[2])))))
    y2 = max(0, min(height, int(round(float(roi[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def appearance_histogram(
    frame_bgr: np.ndarray,
    roi: np.ndarray,
    *,
    hue_bins: int = 8,
    saturation_bins: int = 8,
) -> np.ndarray | None:
    """Return an L1-normalized HSV hue-saturation histogram for an ROI crop.

    Args:
        frame_bgr: Full frame in BGR order (as delivered by OpenCV capture).
        roi: ``[x1, y1, x2, y2]`` region to describe.
        hue_bins: Number of hue bins.
        saturation_bins: Number of saturation bins.

    Returns:
        A flattened ``hue_bins * saturation_bins`` descriptor summing to 1, or
        None when the crop is empty or fully unsaturated.
    """
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "Appearance ReID requires OpenCV. Install it via the demo extras "
            "(e.g. `uv sync --all-groups`) or set RFDETR_TRACK_REID=0."
        ) from error

    crop = crop_roi(frame_bgr, roi)
    if crop is None:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [hue_bins, saturation_bins],
        [0, 180, 0, 256],
    )
    total = float(histogram.sum())
    if total <= 0.0:
        return None
    return (histogram.flatten() / total).astype(np.float64)


def histogram_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Return histogram-intersection similarity in ``[0, 1]``.

    Both inputs are assumed L1-normalized; missing descriptors score 0.
    """
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(np.minimum(left, right).sum())


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Return cosine similarity clamped to ``[0, 1]`` for L2-normalized vectors."""
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(left, right)))))


class AppearanceEncoder(ABC):
    """Turn an ROI crop into a comparable descriptor.

    Implementations own their own descriptor space (histogram vs embedding),
    the similarity metric, and how to blend descriptors over time, so the
    tracker stays agnostic to the ReID backend.
    """

    @abstractmethod
    def encode(self, frame_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
        """Return a descriptor for the ROI, or None when it cannot be computed."""

    @abstractmethod
    def similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        """Return a similarity in ``[0, 1]`` between two descriptors."""

    @abstractmethod
    def combine(self, previous: np.ndarray, fresh: np.ndarray, ema: float) -> np.ndarray:
        """Blend a fresh descriptor into a running one and renormalize."""


class HistogramEncoder(AppearanceEncoder):
    """HSV color-histogram descriptor (no extra dependencies, CPU only)."""

    def encode(self, frame_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
        return appearance_histogram(frame_bgr, roi)

    def similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        return histogram_similarity(left, right)

    def combine(self, previous: np.ndarray, fresh: np.ndarray, ema: float) -> np.ndarray:
        blended = ema * previous + (1.0 - ema) * fresh
        total = float(blended.sum())
        return blended / total if total > 0 else fresh


# ImageNet normalization used by common person-ReID backbones (e.g. OSNet).
_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmbeddingEncoder(AppearanceEncoder):
    """ONNX person-ReID embedding descriptor (grayscale- and color-robust).

    Requires the ``[reid]`` extra (``onnxruntime``) and an ONNX model path. The
    session is created lazily on first use so importing this module stays cheap
    and the model is only loaded when ReID embedding is actually enabled.
    """

    def __init__(self, model_path: str, *, input_width: int = 128, input_height: int = 256) -> None:
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self._session: object | None = None
        self._input_name: str | None = None

    def _ensure_session(self) -> object:
        if self._session is not None:
            return self._session
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ReID embedding model not found: {self.model_path!r}. Set RFDETR_REID_MODEL "
                "to an ONNX person-ReID model, or use RFDETR_REID_BACKEND=histogram.",
            )
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - exercised only without onnxruntime
            raise ImportError(
                "ReID embedding requires onnxruntime. Install the extra: pip install -e '.[reid]'.",
            ) from error
        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name
        return session

    def encode(self, frame_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
        crop = crop_roi(frame_bgr, roi)
        if crop is None:
            return None
        import cv2

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_width, self.input_height))
        normalized = (rgb.astype(np.float32) / 255.0 - _REID_MEAN) / _REID_STD
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        session = self._ensure_session()
        outputs = session.run(None, {self._input_name: tensor})
        feature = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else None

    def similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        return cosine_similarity(left, right)

    def combine(self, previous: np.ndarray, fresh: np.ndarray, ema: float) -> np.ndarray:
        blended = ema * previous + (1.0 - ema) * fresh
        norm = float(np.linalg.norm(blended))
        return blended / norm if norm > 0 else fresh


def build_appearance_encoder(*, backend: str, model_path: str | None) -> AppearanceEncoder:
    """Return the appearance encoder for a backend name.

    Args:
        backend: ``"histogram"`` (default, dependency-free) or ``"embedding"``.
        model_path: ONNX model path, required when ``backend == "embedding"``.

    Returns:
        A concrete :class:`AppearanceEncoder`.
    """
    if backend == "embedding":
        return EmbeddingEncoder(model_path or "")
    return HistogramEncoder()
