# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Person detection association across frames using IoU and Hungarian matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import hungarian_maximize, iou, keypoints_xyxy

_keypoints_xyxy = keypoints_xyxy
_iou = iou
_hungarian_maximize = hungarian_maximize


@dataclass
class PersonAssociator:
    """Map detections to stable track ids using IoU-based Hungarian assignment."""

    iou_threshold: float = 0.15
    max_tracks: int = 32

    def __post_init__(self) -> None:
        self._track_boxes: list[np.ndarray | None] = []

    def reset(self) -> None:
        """Clear track history."""
        self._track_boxes.clear()

    def assign(self, key_points: sv.KeyPoints) -> list[int | None]:
        """Return track id per detection index."""
        num_detections = len(key_points)
        assignments: list[int | None] = [None] * num_detections
        if num_detections == 0:
            return assignments

        boxes: list[np.ndarray | None] = [
            keypoints_xyxy(key_points, detection_index) for detection_index in range(num_detections)
        ]

        if num_detections == 1 and boxes[0] is not None:
            if not self._track_boxes:
                self._track_boxes.append(boxes[0].copy())
            assignments[0] = 0
            self._track_boxes[0] = boxes[0].copy()
            return assignments

        valid_detections = [index for index, box in enumerate(boxes) if box is not None]
        if not valid_detections:
            return assignments

        num_tracks = len(self._track_boxes)
        if num_tracks == 0:
            for detection_index in valid_detections:
                box = boxes[detection_index]
                assert box is not None
                track_id = len(self._track_boxes)
                self._track_boxes.append(box.copy())
                assignments[detection_index] = track_id
            return assignments

        cost = np.zeros((len(valid_detections), num_tracks), dtype=np.float64)
        for row, detection_index in enumerate(valid_detections):
            det_box = boxes[detection_index]
            assert det_box is not None
            for track_id in range(num_tracks):
                track_box = self._track_boxes[track_id]
                if track_box is None:
                    continue
                cost[row, track_id] = iou(det_box, track_box)

        pairs = hungarian_maximize(cost)
        used_tracks: set[int] = set()
        for row, track_id in pairs:
            if cost[row, track_id] < self.iou_threshold:
                continue
            detection_index = valid_detections[row]
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
            box = boxes[detection_index]
            assert box is not None
            self._track_boxes[track_id] = box.copy()

        for detection_index in valid_detections:
            if assignments[detection_index] is not None:
                continue
            if len(self._track_boxes) >= self.max_tracks:
                continue
            box = boxes[detection_index]
            assert box is not None
            track_id = len(self._track_boxes)
            self._track_boxes.append(box.copy())
            assignments[detection_index] = track_id

        return assignments
