# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-frame inference callbacks for video demo tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import supervision as sv

from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.detr import RFDETR
from rfdetr.visualize.keypoints import key_points_for_display
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.types import COCO_PERSON_CLASS_ID
from rfdetr_demo.tracking.keypoints_ops import (
    detections_to_key_points,
    is_track_ghost,
    track_ids_from_key_points,
)


def _color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track id."""
    hue = (int(track_id) * 47) % 180
    pixel = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_tracked_boxes(frame_bgr: np.ndarray, key_points: sv.KeyPoints) -> np.ndarray:
    """Draw id-colored boxes (with a `*` ghost mark) and a live count banner."""
    annotated = frame_bgr.copy()
    track_ids = track_ids_from_key_points(key_points)
    boxes = key_points.data.get("xyxy") if key_points.data else None
    live = 0
    for index, track_id in enumerate(track_ids):
        if track_id is None or boxes is None or index >= len(boxes):
            continue
        ghost = is_track_ghost(key_points, index)
        if not ghost:
            live += 1
        color = _color_for_track_id(track_id)
        x1, y1, x2, y2 = (int(round(float(value))) for value in boxes[index])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1 if ghost else 2)
        label = f"{track_id}*" if ghost else str(track_id)
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    total = sum(1 for track_id in track_ids if track_id is not None)
    cv2.putText(annotated, f"count: {total} (live {live})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated


def make_detection_callback(
    model: RFDETR,
    threshold: float,
    person_only: bool,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    stats: dict[str, int],
    tune_cache: Any | None = None,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a callback that runs COCO object detection per frame."""

    def callback(frame_bgr: np.ndarray, index: int) -> np.ndarray:
        stats["processed_frames"] += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(
            frame_rgb,
            threshold=threshold,
            include_source_image=False,
        )
        if tune_cache is not None:
            tune_cache.append_detection(
                frame_bgr=frame_bgr,
                detections=detections,
                frame_index=index,
                processed_count=stats["processed_frames"],
                task="detect",
            )
        if person_only and len(detections) > 0:
            person_mask = detections.class_id == COCO_PERSON_CLASS_ID
            detections = detections[person_mask]
        stats["total_detections"] += len(detections)
        labels = [
            f"{COCO_CLASSES[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
        ]
        annotated = frame_bgr.copy()
        annotated = box_annotator.annotate(annotated, detections)
        return label_annotator.annotate(annotated, detections, labels)

    return callback


def make_detection_track_callback(
    model: RFDETR,
    threshold: float,
    person_only: bool,
    stats: dict[str, int],
    track_pipeline: Any,
    tune_cache: Any | None = None,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a callback that detects persons and tracks them (ids + stable count).

    Person detections flow through the box-IoU tracker (``track_pipeline``), so
    motion prediction, the motion gate, and appearance ReID all apply. Boxes are
    drawn colored by track id with a live-count banner.
    """

    def callback(frame_bgr: np.ndarray, index: int) -> np.ndarray:
        stats["processed_frames"] += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=threshold, include_source_image=False)
        if person_only and len(detections) > 0:
            detections = detections[detections.class_id == COCO_PERSON_CLASS_ID]
        if tune_cache is not None:
            tune_cache.append_detection(
                frame_bgr=frame_bgr,
                detections=detections,
                frame_index=index,
                processed_count=stats["processed_frames"],
                task="detect",
            )
        key_points = detections_to_key_points(detections)
        result = track_pipeline.apply(key_points, index, frame=frame_bgr)
        frame_stats = result.stats
        stats["frame_raw_detections"] = frame_stats.raw_count
        stats["frame_nms_detections"] = frame_stats.nms_count
        stats["frame_active_tracks"] = frame_stats.active_track_count
        stats["frame_ghost_tracks"] = frame_stats.ghost_count
        stats["frame_live_tracks"] = frame_stats.active_track_count - frame_stats.ghost_count
        stats["total_detections"] += frame_stats.active_track_count
        return _draw_tracked_boxes(frame_bgr, result.key_points)

    return callback


def make_segmentation_callback(
    model: RFDETR,
    threshold: float,
    person_only: bool,
    mask_annotator: sv.MaskAnnotator,
    label_annotator: sv.LabelAnnotator,
    stats: dict[str, int],
    tune_cache: Any | None = None,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a callback that runs COCO instance segmentation per frame."""

    def callback(frame_bgr: np.ndarray, index: int) -> np.ndarray:
        stats["processed_frames"] += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(
            frame_rgb,
            threshold=threshold,
            include_source_image=False,
        )
        if tune_cache is not None:
            tune_cache.append_detection(
                frame_bgr=frame_bgr,
                detections=detections,
                frame_index=index,
                processed_count=stats["processed_frames"],
                task="segment",
            )
        if person_only and len(detections) > 0:
            person_mask = detections.class_id == COCO_PERSON_CLASS_ID
            detections = detections[person_mask]
        stats["total_detections"] += len(detections)
        labels = [
            f"{COCO_CLASSES[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
        ]
        annotated = frame_bgr.copy()
        annotated = mask_annotator.annotate(annotated, detections)
        return label_annotator.annotate(annotated, detections, labels)

    return callback


def make_keypoint_callback(
    model: RFDETR,
    threshold: float,
    overlay_settings: KeypointOverlaySettings,
    stats: dict[str, int],
    tune_cache: Any | None = None,
    temporal_filter: Any | None = None,
    detection_stabilizer: Any | None = None,
    person_track_pipeline: Any | None = None,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a callback that runs COCO person keypoint inference per frame."""
    track_pipeline = person_track_pipeline or detection_stabilizer

    def callback(frame_bgr: np.ndarray, index: int) -> np.ndarray:
        stats["processed_frames"] += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(
            frame_rgb,
            threshold=threshold,
            include_source_image=False,
        )
        if tune_cache is not None:
            tune_cache.append_keypoint(
                frame_bgr=frame_bgr,
                key_points=key_points,
                frame_index=index,
                processed_count=stats["processed_frames"],
            )
        if track_pipeline is not None:
            stabilized = track_pipeline.apply(key_points, index, frame=frame_bgr)
            key_points = stabilized.key_points
            frame_stats = stabilized.stats
            stats["frame_raw_detections"] = frame_stats.raw_count
            stats["frame_nms_detections"] = frame_stats.nms_count
            stats["frame_active_tracks"] = frame_stats.active_track_count
            stats["frame_ghost_tracks"] = frame_stats.ghost_count
            stats["frame_live_tracks"] = frame_stats.active_track_count - frame_stats.ghost_count
        if temporal_filter is not None:
            key_points = temporal_filter.apply(key_points, index)
            motion_stats = temporal_filter.stats
            stats["motion_speed_rejections"] = motion_stats.speed_rejections
            stats["motion_covariance_rejections"] = motion_stats.covariance_rejections
            stats["motion_oscillation_corrections"] = motion_stats.oscillation_corrections
            stats["motion_smoothed_joints"] = motion_stats.smoothed_joints
        display_points = key_points_for_display(
            key_points,
            keypoint_threshold=overlay_settings.keypoint_threshold,
        )
        active_count = stats.get("frame_active_tracks", len(display_points))
        stats["total_detections"] += active_count
        return render_keypoint_overlay(frame_bgr, key_points, overlay_settings)

    return callback
