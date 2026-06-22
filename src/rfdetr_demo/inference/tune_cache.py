# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Cache inference results during tune preview and re-render overlays without re-inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import supervision as sv

TuneTaskName = Literal["detect", "segment", "keypoint"]


def _copy_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def serialize_key_points(key_points: sv.KeyPoints) -> dict[str, Any]:
    """Return a deep copy of keypoint arrays suitable for later re-rendering."""
    data: dict[str, Any] = {}
    if key_points.data:
        for key, value in key_points.data.items():
            data[key] = _copy_array(value)
    return {
        "xy": key_points.xy.copy(),
        "visible": key_points.visible.copy() if key_points.visible is not None else None,
        "confidence": key_points.confidence.copy() if key_points.confidence is not None else None,
        "data": data,
    }


def deserialize_key_points(payload: dict[str, Any]) -> sv.KeyPoints:
    """Rebuild a ``KeyPoints`` instance from ``serialize_key_points`` output."""
    return sv.KeyPoints(
        xy=payload["xy"],
        visible=payload.get("visible"),
        confidence=payload.get("confidence"),
        data=payload.get("data") or {},
    )


def serialize_detections(detections: sv.Detections) -> dict[str, Any]:
    """Return a deep copy of detection arrays suitable for later re-rendering."""
    data: dict[str, Any] = {}
    if detections.data:
        for key, value in detections.data.items():
            data[key] = _copy_array(value)
    return {
        "xyxy": detections.xyxy.copy(),
        "confidence": detections.confidence.copy(),
        "class_id": detections.class_id.copy(),
        "mask": detections.mask.copy() if detections.mask is not None else None,
        "data": data,
    }


def deserialize_detections(payload: dict[str, Any]) -> sv.Detections:
    """Rebuild a ``Detections`` instance from ``serialize_detections`` output."""
    return sv.Detections(
        xyxy=payload["xyxy"],
        confidence=payload["confidence"],
        class_id=payload["class_id"],
        mask=payload.get("mask"),
        data=payload.get("data") or {},
    )


@dataclass
class TuneCacheEntry:
    """One inferred frame stored for live overlay re-rendering."""

    frame_bgr: npt.NDArray[np.uint8]
    frame_index: int
    processed_count: int
    task: TuneTaskName
    key_points_payload: dict[str, Any] | None = None
    detections_payload: dict[str, Any] | None = None


@dataclass
class TunePreviewCache:
    """In-memory store of tune-preview inference results."""

    task: TuneTaskName
    person_only: bool = False
    fps: float = 30.0
    frame_stride: int = 1
    entries: list[TuneCacheEntry] = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()

    def append_keypoint(
        self,
        *,
        frame_bgr: npt.NDArray[np.uint8],
        key_points: sv.KeyPoints,
        frame_index: int,
        processed_count: int,
    ) -> None:
        self.entries.append(
            TuneCacheEntry(
                frame_bgr=frame_bgr.copy(),
                frame_index=frame_index,
                processed_count=processed_count,
                task="keypoint",
                key_points_payload=serialize_key_points(key_points),
            ),
        )

    def append_detection(
        self,
        *,
        frame_bgr: npt.NDArray[np.uint8],
        detections: sv.Detections,
        frame_index: int,
        processed_count: int,
        task: TuneTaskName,
    ) -> None:
        self.entries.append(
            TuneCacheEntry(
                frame_bgr=frame_bgr.copy(),
                frame_index=frame_index,
                processed_count=processed_count,
                task=task,
                detections_payload=serialize_detections(detections),
            ),
        )

    @property
    def has_entries(self) -> bool:
        return len(self.entries) > 0

    @property
    def latest(self) -> TuneCacheEntry | None:
        if not self.entries:
            return None
        return self.entries[-1]
