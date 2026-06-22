# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Detection, segmentation, and tune-cache overlay rendering."""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.types import COCO_PERSON_CLASS_ID, TaskName


def render_detection_overlay(
    frame_bgr: np.ndarray,
    detections_raw: sv.Detections,
    *,
    threshold: float,
    person_only: bool,
) -> np.ndarray:
    """Re-annotate detection boxes after threshold or class filtering."""
    detections = detections_raw
    if len(detections) > 0:
        confidence_mask = detections.confidence >= threshold
        detections = detections[confidence_mask]
    if person_only and len(detections) > 0:
        person_mask = detections.class_id == COCO_PERSON_CLASS_ID
        detections = detections[person_mask]

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    labels = [
        f"{COCO_CLASSES[int(class_id)]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
    ]
    annotated = frame_bgr.copy()
    annotated = box_annotator.annotate(annotated, detections)
    return label_annotator.annotate(annotated, detections, labels)


def render_segment_overlay(
    frame_bgr: np.ndarray,
    detections_raw: sv.Detections,
    *,
    threshold: float,
    person_only: bool,
) -> np.ndarray:
    """Re-annotate segmentation masks after threshold or class filtering."""
    detections = detections_raw
    if len(detections) > 0:
        confidence_mask = detections.confidence >= threshold
        detections = detections[confidence_mask]
    if person_only and len(detections) > 0:
        person_mask = detections.class_id == COCO_PERSON_CLASS_ID
        detections = detections[person_mask]

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    labels = [
        f"{COCO_CLASSES[int(class_id)]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
    ]
    annotated = frame_bgr.copy()
    annotated = mask_annotator.annotate(annotated, detections)
    return label_annotator.annotate(annotated, detections, labels)


def render_tune_cache_entry(
    entry: Any,
    *,
    task: TaskName,
    threshold: float,
    person_only: bool,
    keypoint_settings: KeypointOverlaySettings | None = None,
) -> np.ndarray:
    """Re-render one cached tune-preview frame with updated overlay settings."""
    from rfdetr_demo.inference.tune_cache import TuneCacheEntry, deserialize_detections, deserialize_key_points

    assert isinstance(entry, TuneCacheEntry)
    if task == "keypoint" and entry.key_points_payload is not None and keypoint_settings is not None:
        key_points = deserialize_key_points(entry.key_points_payload)
        return render_keypoint_overlay(entry.frame_bgr, key_points, keypoint_settings)
    if entry.detections_payload is not None:
        detections = deserialize_detections(entry.detections_payload)
        if task == "segment":
            return render_segment_overlay(
                entry.frame_bgr,
                detections,
                threshold=threshold,
                person_only=person_only,
            )
        return render_detection_overlay(
            entry.frame_bgr,
            detections,
            threshold=threshold,
            person_only=person_only,
        )
    return entry.frame_bgr.copy()


def render_tune_cache_sequence(
    cache: Any,
    *,
    task: TaskName,
    threshold: float,
    person_only: bool,
    keypoint_settings: KeypointOverlaySettings | None,
    fps: float,
    frame_stride: int,
) -> list[tuple[np.ndarray, int, int]]:
    """Re-render all cached tune frames with optional temporal filtering replay."""
    from rfdetr_demo.inference.tune_cache import TunePreviewCache

    assert isinstance(cache, TunePreviewCache)
    rendered: list[tuple[np.ndarray, int, int]] = []
    temporal_filter = None
    if (
        task == "keypoint"
        and keypoint_settings is not None
        and keypoint_settings.motion is not None
        and keypoint_settings.motion.enabled
    ):
        from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter

        temporal_filter = KeypointTemporalFilter(
            keypoint_settings.motion,
            frame_width=keypoint_settings.frame_width,
            frame_height=keypoint_settings.frame_height,
            fps=fps,
            frame_stride=frame_stride,
        )

    for entry in cache.entries:
        if task == "keypoint" and entry.key_points_payload is not None and keypoint_settings is not None:
            from rfdetr_demo.inference.tune_cache import deserialize_key_points

            key_points = deserialize_key_points(entry.key_points_payload)
            if temporal_filter is not None:
                key_points = temporal_filter.apply(key_points, entry.frame_index)
            frame = render_keypoint_overlay(entry.frame_bgr, key_points, keypoint_settings)
        else:
            frame = render_tune_cache_entry(
                entry,
                task=task,
                threshold=threshold,
                person_only=person_only,
                keypoint_settings=keypoint_settings,
            )
        rendered.append((frame, entry.frame_index, entry.processed_count))
    return rendered
