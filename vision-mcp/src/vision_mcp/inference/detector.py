"""Turning model output into contract detections.

This module owns the only place where `supervision` result objects are read, so the rest of the
engine deals in `Detection` payloads and never in raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import supervision as sv

from vision_mcp.api_contract import BoundingBox, Detection
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.inference.images import ImageLoader, LoadedImage
from vision_mcp.inference.models import LoadedModel, ModelManager


@dataclass(slots=True)
class InferenceOutput:
    """One inference: the image it ran on, what it found and how long it took."""

    image: LoadedImage
    detections: list[Detection]
    raw: sv.Detections
    inference_ms: float


class Detector:
    """Runs models over images and normalises the results."""

    def __init__(self, manager: ModelManager, loader: ImageLoader) -> None:
        self._manager = manager
        self._loader = loader

    async def detect_source(
        self,
        model: str,
        source: str,
        confidence: float | None = None,
        classes: list[str] | None = None,
    ) -> InferenceOutput:
        """Load *source*, run *model* over it and return normalised detections."""
        image = await self._loader.load(source)
        return await self.detect_image(model, image, confidence=confidence, classes=classes)

    async def detect_image(
        self,
        model: str,
        image: LoadedImage,
        confidence: float | None = None,
        classes: list[str] | None = None,
    ) -> InferenceOutput:
        """Run *model* over an already-decoded image."""
        entry = self._manager.entry(model)
        threshold = confidence if confidence is not None else entry.confidence
        loaded = await self._manager.acquire(model)
        validate_classes(classes, loaded.class_names)

        def call(target: LoadedModel) -> Any:
            return target.model.predict(image.array, threshold=threshold)

        result, elapsed_ms = await self._manager.run(model, call)
        # Filter before converting so `raw[i]` and `detections[i]` describe the same object;
        # tracking and annotation both rely on that alignment.
        raw = filter_by_class(to_detections(result, image), loaded.class_names, classes)
        detections = convert(raw, loaded.class_names)
        return InferenceOutput(image=image, detections=detections, raw=raw, inference_ms=elapsed_ms)


def validate_classes(classes: list[str] | None, class_names: list[str]) -> None:
    """Reject filters naming labels this model cannot produce.

    Raises:
        VisionError: INVALID_ARGUMENT listing the unknown class names.
    """
    if not classes:
        return
    unknown = sorted(set(classes) - set(class_names))
    if unknown:
        raise VisionError(
            ErrorCode.INVALID_ARGUMENT,
            "Model does not predict the requested classes.",
            {"unknown_classes": unknown, "class_count": len(class_names)},
        )


def to_detections(result: Any, image: LoadedImage) -> sv.Detections:
    """Normalise `predict` output to `sv.Detections`, deriving boxes for keypoint models."""
    if isinstance(result, list):
        result = result[0] if result else sv.Detections.empty()
    if isinstance(result, sv.Detections):
        return result
    if isinstance(result, sv.KeyPoints):
        return _keypoints_to_detections(result)
    raise VisionError(
        ErrorCode.INFERENCE_FAILED,
        "Model returned an unsupported result type.",
        {"type": type(result).__name__, "source": image.label},
    )


def _keypoints_to_detections(keypoints: sv.KeyPoints) -> sv.Detections:
    """Wrap keypoint output in `sv.Detections` using the keypoint extent as the box."""
    if len(keypoints) == 0:
        return sv.Detections.empty()
    xy = np.asarray(keypoints.xy, dtype=np.float32)
    boxes = np.stack(
        [xy[:, :, 0].min(axis=1), xy[:, :, 1].min(axis=1), xy[:, :, 0].max(axis=1), xy[:, :, 1].max(axis=1)],
        axis=1,
    )
    scores = (
        np.asarray(keypoints.confidence, dtype=np.float32).mean(axis=1)
        if keypoints.confidence is not None
        else np.ones(len(keypoints), dtype=np.float32)
    )
    class_id = (
        np.asarray(keypoints.class_id, dtype=int)
        if keypoints.class_id is not None
        else np.zeros(len(keypoints), dtype=int)
    )
    detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_id)
    detections.data["keypoints"] = xy
    if keypoints.confidence is not None:
        detections.data["keypoint_confidence"] = np.asarray(keypoints.confidence, dtype=np.float32)
    return detections


def convert(
    detections: sv.Detections,
    class_names: list[str],
    classes: list[str] | None = None,
) -> list[Detection]:
    """Convert `sv.Detections` to contract detections, applying an optional class-name filter."""
    wanted = set(classes) if classes else None
    masks = detections.mask
    keypoints = detections.data.get("keypoints")
    keypoint_confidence = detections.data.get("keypoint_confidence")
    attached_names = detections.data.get("class_name")
    tracker_ids = detections.tracker_id
    out: list[Detection] = []
    for index in range(len(detections)):
        class_id = int(detections.class_id[index]) if detections.class_id is not None else -1
        name = (
            str(attached_names[index])
            if attached_names is not None and str(attached_names[index])
            else class_names[class_id]
            if 0 <= class_id < len(class_names)
            else str(class_id)
        )
        if wanted is not None and name not in wanted:
            continue
        x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])
        instance_keypoints = None
        if keypoints is not None:
            scores = None if keypoint_confidence is None else keypoint_confidence[index]
            instance_keypoints = _keypoint_rows(keypoints[index], scores)
        out.append(
            Detection(
                class_id=class_id,
                class_name=name,
                confidence=round(
                    float(detections.confidence[index]) if detections.confidence is not None else 1.0, 4
                ),
                box=BoundingBox(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1)),
                track_id=None if tracker_ids is None else int(tracker_ids[index]),
                mask_area_px=None if masks is None else int(np.count_nonzero(masks[index])),
                keypoints=instance_keypoints,
            )
        )
    return out


def _keypoint_rows(xy: np.ndarray[Any, Any], confidence: np.ndarray[Any, Any] | None) -> list[list[float]]:
    """Flatten one instance's keypoints to `[x, y, score]` rows."""
    rows: list[list[float]] = []
    for index, point in enumerate(xy):
        score = 1.0 if confidence is None else float(confidence[index])
        rows.append([round(float(point[0]), 1), round(float(point[1]), 1), round(score, 3)])
    return rows


def filter_by_class(
    detections: sv.Detections, class_names: list[str], classes: list[str] | None
) -> sv.Detections:
    """Subset raw detections by class name, for paths that keep working with `sv.Detections`."""
    if not classes or len(detections) == 0 or detections.class_id is None:
        return detections
    attached_names = detections.data.get("class_name")
    if attached_names is not None:
        keep = np.array([str(name) in set(classes) for name in attached_names], dtype=bool)
    else:
        wanted = {index for index, name in enumerate(class_names) if name in set(classes)}
        keep = np.array([int(value) in wanted for value in detections.class_id], dtype=bool)
    return cast("sv.Detections", detections[keep])


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """Detections grouped by class name, ordered by descending count then name."""
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
