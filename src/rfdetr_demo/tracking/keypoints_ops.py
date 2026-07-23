# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""KeyPoints row operations for the tracking pipeline."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.types import TRACK_ID_KEY, TRACK_IS_GHOST_KEY


def _subset_key_points_data(
    data: dict[str, object] | None,
    indices: list[int],
    num_detections: int,
) -> dict[str, object]:
    if not data:
        return {}
    idx = np.asarray(indices, dtype=int)
    subset: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == num_detections:
            subset[key] = value[idx]
        else:
            subset[key] = value
    return subset


def empty_key_points_like(template: sv.KeyPoints) -> sv.KeyPoints:
    """Return a zero-length ``KeyPoints`` with the same joint count as ``template``."""
    num_joints = template.xy.shape[1]
    return sv.KeyPoints(
        xy=np.zeros((0, num_joints, 2), dtype=template.xy.dtype),
        visible=(np.zeros((0, num_joints), dtype=bool) if template.visible is not None else None),
        keypoint_confidence=(
            np.zeros((0, num_joints), dtype=np.float32) if template.keypoint_confidence is not None else None
        ),
        detection_confidence=(np.zeros((0,), dtype=np.float32) if template.detection_confidence is not None else None),
        class_id=(np.zeros((0,), dtype=np.int64) if template.class_id is not None else None),
        data={},
    )


def subset_key_points(key_points: sv.KeyPoints, indices: list[int]) -> sv.KeyPoints:
    """Return a new ``KeyPoints`` containing only ``indices``."""
    num_detections = len(key_points)
    if not indices:
        return empty_key_points_like(key_points)

    idx = np.asarray(indices, dtype=int)
    return sv.KeyPoints(
        xy=key_points.xy[idx],
        visible=key_points.visible[idx] if key_points.visible is not None else None,
        keypoint_confidence=(
            key_points.keypoint_confidence[idx] if key_points.keypoint_confidence is not None else None
        ),
        detection_confidence=(
            key_points.detection_confidence[idx] if key_points.detection_confidence is not None else None
        ),
        class_id=key_points.class_id[idx] if key_points.class_id is not None else None,
        data=_subset_key_points_data(key_points.data, indices, num_detections),
    )


def single_detection_key_points(key_points: sv.KeyPoints, detection_index: int) -> sv.KeyPoints:
    """Extract one detection row as a length-1 ``KeyPoints``."""
    return subset_key_points(key_points, [detection_index])


def shift_key_points(key_points: sv.KeyPoints, dx: float, dy: float) -> sv.KeyPoints:
    """Return a copy with every visible joint translated by ``(dx, dy)``.

    Zero-padded (invisible) joints are left untouched so they keep acting as
    absent points. Any ``data['xyxy']`` box is translated by the same offset.

    Args:
        key_points: Source detections to translate.
        dx: Horizontal shift in pixels applied to visible joints.
        dy: Vertical shift in pixels applied to visible joints.

    Returns:
        A new ``KeyPoints`` with shifted coordinates.
    """
    if len(key_points) == 0 or (dx == 0.0 and dy == 0.0):
        return key_points

    xy = key_points.xy.copy()
    offset = np.asarray([dx, dy], dtype=xy.dtype)
    if key_points.visible is not None:
        mask = key_points.visible
    else:
        mask = ~np.all(np.isclose(xy, 0), axis=2)
    xy[mask] += offset

    data = dict(key_points.data) if key_points.data else {}
    xyxy = data.get("xyxy")
    if isinstance(xyxy, np.ndarray) and xyxy.shape[-1] == 4:
        shifted = xyxy.astype(np.float64, copy=True)
        shifted[..., [0, 2]] += dx
        shifted[..., [1, 3]] += dy
        data["xyxy"] = shifted.astype(xyxy.dtype, copy=False)

    return sv.KeyPoints(
        xy=xy,
        visible=key_points.visible,
        keypoint_confidence=key_points.keypoint_confidence,
        detection_confidence=key_points.detection_confidence,
        class_id=key_points.class_id,
        data=data,
    )


def is_track_ghost(key_points: sv.KeyPoints, detection_index: int) -> bool:
    """Return True when a detection row is a stabilizer ghost hold."""
    flags = key_points.data.get(TRACK_IS_GHOST_KEY) if key_points.data else None
    if flags is None or len(flags) <= detection_index:
        return False
    return bool(flags[detection_index])


def track_id_at(key_points: sv.KeyPoints, detection_index: int) -> int | None:
    """Return pipeline track id for one detection row, if present."""
    ids = key_points.data.get(TRACK_ID_KEY) if key_points.data else None
    if ids is None or len(ids) <= detection_index:
        return None
    return int(ids[detection_index])


def partition_live_and_ghost(key_points: sv.KeyPoints) -> tuple[sv.KeyPoints, sv.KeyPoints]:
    """Split detections into live (matched this frame) and ghost hold rows."""
    if len(key_points) == 0:
        empty = empty_key_points_like(key_points)
        return empty, empty
    flags = key_points.data.get(TRACK_IS_GHOST_KEY) if key_points.data else None
    if flags is None:
        return key_points, empty_key_points_like(key_points)
    live_indices = [index for index, is_ghost in enumerate(flags) if not is_ghost]
    ghost_indices = [index for index, is_ghost in enumerate(flags) if is_ghost]
    return (
        subset_key_points(key_points, live_indices),
        subset_key_points(key_points, ghost_indices),
    )


def _attach_row_metadata(
    key_points: sv.KeyPoints,
    *,
    ghost_flags: list[bool] | None = None,
    track_ids: list[int] | None = None,
) -> sv.KeyPoints:
    data = dict(key_points.data) if key_points.data else {}
    if ghost_flags is not None:
        data[TRACK_IS_GHOST_KEY] = np.asarray(ghost_flags, dtype=bool)
    if track_ids is not None:
        data[TRACK_ID_KEY] = np.asarray(track_ids, dtype=np.int64)
    return sv.KeyPoints(
        xy=key_points.xy,
        visible=key_points.visible,
        keypoint_confidence=key_points.keypoint_confidence,
        detection_confidence=key_points.detection_confidence,
        class_id=key_points.class_id,
        data=data,
    )


def merge_key_points(
    parts: list[sv.KeyPoints],
    *,
    ghost_flags: list[bool] | None = None,
    track_ids: list[int] | None = None,
) -> sv.KeyPoints:
    """Concatenate multiple ``KeyPoints`` along the detection axis."""
    non_empty = [part for part in parts if len(part) > 0]
    if not non_empty:
        if parts:
            return empty_key_points_like(parts[0])
        return sv.KeyPoints(xy=np.zeros((0, 17, 2), dtype=np.float32))
    if len(non_empty) == 1:
        merged = non_empty[0]
        if ghost_flags is not None or track_ids is not None:
            return _attach_row_metadata(merged, ghost_flags=ghost_flags, track_ids=track_ids)
        return merged

    xy = np.concatenate([part.xy for part in non_empty], axis=0)
    visible = None
    if non_empty[0].visible is not None:
        visible = np.concatenate([part.visible for part in non_empty], axis=0)
    keypoint_confidence = None
    if non_empty[0].keypoint_confidence is not None:
        keypoint_confidence = np.concatenate(
            [part.keypoint_confidence for part in non_empty],
            axis=0,
        )
    detection_confidence = None
    if non_empty[0].detection_confidence is not None:
        detection_confidence = np.concatenate(
            [part.detection_confidence for part in non_empty],
            axis=0,
        )
    class_id = None
    if non_empty[0].class_id is not None:
        class_id = np.concatenate([part.class_id for part in non_empty], axis=0)

    merged_data: dict[str, object] = {}
    if non_empty[0].data:
        for key in non_empty[0].data:
            if key in {TRACK_IS_GHOST_KEY, TRACK_ID_KEY}:
                continue
            values = [part.data.get(key) for part in non_empty]
            if all(isinstance(value, np.ndarray) for value in values):
                merged_data[key] = np.concatenate(values, axis=0)
            else:
                merged_data[key] = non_empty[0].data[key]
    merged = sv.KeyPoints(
        xy=xy,
        visible=visible,
        keypoint_confidence=keypoint_confidence,
        detection_confidence=detection_confidence,
        class_id=class_id,
        data=merged_data,
    )
    return _attach_row_metadata(merged, ghost_flags=ghost_flags, track_ids=track_ids)


def attach_track_ids(key_points: sv.KeyPoints, track_ids: list[int | None]) -> sv.KeyPoints:
    """Attach explicit track ids to each detection row (``None`` → -1)."""
    if len(track_ids) != len(key_points):
        raise ValueError("track_ids length must match key_points detections")
    ids = np.asarray([track_id if track_id is not None else -1 for track_id in track_ids], dtype=np.int64)
    data = dict(key_points.data) if key_points.data else {}
    data[TRACK_ID_KEY] = ids
    return sv.KeyPoints(
        xy=key_points.xy,
        visible=key_points.visible,
        keypoint_confidence=key_points.keypoint_confidence,
        detection_confidence=key_points.detection_confidence,
        class_id=key_points.class_id,
        data=data,
    )


def track_ids_from_key_points(key_points: sv.KeyPoints) -> list[int | None]:
    """Read pipeline track ids; ``-1`` and missing rows map to ``None``."""
    raw = key_points.data.get(TRACK_ID_KEY) if key_points.data else None
    if raw is None:
        return [None] * len(key_points)
    ids = np.asarray(raw, dtype=np.int64)
    return [None if int(value) < 0 else int(value) for value in ids]
