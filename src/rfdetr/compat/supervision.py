# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Supervision API compatibility shims.

These helpers bridge the RF-DETR predict() pipeline across supported Supervision versions.
The module-level ``_KEYPOINTS_ACCEPTS_NEW_API`` flag avoids repeated signature inspection
on every call.

Supported Supervision versions: 0.21+ (new API); 0.20.x (legacy fallback).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from supervision import KeyPoints

_KEYPOINTS_ACCEPTS_NEW_API: bool = "keypoint_confidence" in inspect.signature(KeyPoints.__init__).parameters
"""True when the installed Supervision exposes ``keypoint_confidence`` in the KeyPoints constructor (≥0.21)."""


def _attach_detection_metadata(detections: Any, key: str, value: Any) -> None:
    """Attach a metadata entry to a Supervision Detections object.

    Works across Supervision versions that lack a ``.metadata`` attribute (older
    releases) by creating the dict on the object when absent.

    Args:
        detections: A Supervision ``Detections`` object (any supported version).
        key: Metadata key to set.
        value: Metadata value to attach. ``None`` is not accepted.

    Raises:
        ValueError: If ``value`` is ``None``.

    Examples:
        >>> from types import SimpleNamespace
        >>> det = SimpleNamespace()
        >>> _attach_detection_metadata(det, "source_image", "img.jpg")
        >>> det.metadata
        {'source_image': 'img.jpg'}

        >>> det2 = SimpleNamespace(metadata={"existing": 1})
        >>> _attach_detection_metadata(det2, "source_image", "img.jpg")
        >>> det2.metadata == {"existing": 1, "source_image": "img.jpg"}
        True
    """
    if value is None:
        raise ValueError(f"metadata value for key {key!r} must not be None")
    metadata = getattr(detections, "metadata", None)
    if metadata is None:
        metadata = {}
        detections.metadata = metadata
    metadata[key] = value


def _make_keypoints(
    keypoints_cls: Any,
    xy: np.ndarray,
    keypoint_confidence: np.ndarray,
    detection_confidence: np.ndarray | None,
    class_id: np.ndarray | None,
    visible: np.ndarray,
    data: dict[str, Any],
) -> Any:
    """Create a Supervision KeyPoints object with version-compatibility fallback.

    Uses the module-level ``_KEYPOINTS_ACCEPTS_NEW_API`` flag (computed once at import)
    to avoid repeated signature inspection per call.  On Supervision ≥ 0.21, the new
    constructor API is used directly.  On Supervision 0.20.x the legacy positional
    ``confidence`` parameter is used and the additional attributes expected by the
    RF-DETR pipeline are patched onto the returned instance after construction.
    Supervision 0.20.x ``KeyPoints`` is a non-frozen dataclass without ``__slots__``,
    so post-construction attribute assignment is safe for that target version.

    Args:
        keypoints_cls: Supervision ``KeyPoints`` class or a compatible subclass.
        xy: Keypoint coordinates array with shape ``(N, K, 2)``, where ``N`` is the
            number of detections and ``K`` is the number of keypoints per detection.
            Must be a 3-D ndarray.
        keypoint_confidence: Per-keypoint confidence values, shape ``(N, K)``.
        detection_confidence: Per-detection confidence values, shape ``(N,)``, or ``None``.
        class_id: Per-detection class ids, shape ``(N,)``, or ``None``.
        visible: Per-keypoint visibility mask, shape ``(N, K)``.
        data: Per-detection extra data dict forwarded to the constructor.

    Returns:
        A Supervision ``KeyPoints`` instance.  On Supervision < 0.21, the attributes
        ``xy``, ``confidence``, ``keypoint_confidence``, ``detection_confidence``, and
        ``visible`` are patched directly onto the instance after construction.

    Raises:
        ValueError: If ``xy`` is not a 3-D ndarray.
        TypeError: If the constructor raises ``TypeError`` for a reason unrelated to
            the missing ``keypoint_confidence`` parameter (re-raised unchanged).

    Examples:
        >>> import numpy as np
        >>> from supervision import KeyPoints
        >>> xy = np.zeros((1, 17, 2), dtype=np.float32)
        >>> kp_conf = np.ones((1, 17), dtype=np.float32)
        >>> kp = _make_keypoints(KeyPoints, xy, kp_conf, None, None, kp_conf > 0, {})
        >>> kp.xy.shape
        (1, 17, 2)
    """
    if xy.ndim != 3:
        raise ValueError(f"xy must be a 3-D ndarray with shape (N, K, 2), got shape {xy.shape}")
    if _KEYPOINTS_ACCEPTS_NEW_API:
        return keypoints_cls(
            xy=xy,
            keypoint_confidence=keypoint_confidence,
            detection_confidence=detection_confidence,
            class_id=class_id,
            visible=visible,
            data=data,
        )
    # Fallback for Supervision 0.20.x: the keypoint_confidence constructor
    # parameter did not exist; build via the legacy positional confidence arg.
    constructor_xy = xy
    constructor_confidence = keypoint_confidence
    if xy.shape[0] == 0:
        constructor_xy = np.empty((0, 0, 2), dtype=xy.dtype)
        constructor_confidence = np.empty((0, 0), dtype=keypoint_confidence.dtype)
    key_points = keypoints_cls(
        xy=constructor_xy,
        class_id=class_id,
        confidence=constructor_confidence,
        data=data,
    )
    key_points.xy = xy
    key_points.confidence = keypoint_confidence
    key_points.keypoint_confidence = keypoint_confidence
    key_points.detection_confidence = detection_confidence
    key_points.visible = visible
    if xy.shape[0] == 0:
        # instance attribute — DO NOT promote to class; prevents cross-instance state leak
        key_points.is_empty = _empty_keypoints_is_empty
    return key_points


def _empty_keypoints_is_empty() -> bool:
    """Return ``True``; used as an instance-level ``is_empty`` override for legacy KeyPoints.

    Assigned to empty ``KeyPoints`` instances constructed via the 0.20.x fallback
    path in ``_make_keypoints``.

    Examples:
        >>> _empty_keypoints_is_empty()
        True
    """
    return True
