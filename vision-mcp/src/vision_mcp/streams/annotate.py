"""Drawing for the debug preview.

Annotation is not part of the inference path. It runs only while a preview client is
connected, on a copy of the last frame, at the preview's own frame rate. Nothing here is
allowed to mutate pipeline state or write anything to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from vision_mcp.api_contract import Detection


@dataclass(slots=True)
class Overlay:
    """The zones, lines and counters drawn on top of a frame."""

    zones: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    lines: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=dict)
    zone_counts: dict[str, int] = field(default_factory=dict)
    line_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    hud: list[str] = field(default_factory=list)


_ZONE_COLOR = (56, 189, 248)
_LINE_COLOR = (250, 204, 21)
_HUD_BACKGROUND = (17, 24, 39)
_HUD_TEXT = (243, 244, 246)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    """Draws detections, tracks, zones, lines and a heads-up display onto RGB frames."""

    def __init__(self) -> None:
        palette = sv.ColorPalette.DEFAULT
        self._boxes = sv.BoxAnnotator(color=palette, thickness=2)
        self._labels = sv.LabelAnnotator(color=palette, text_scale=0.45, text_thickness=1, text_padding=4)
        self._traces = sv.TraceAnnotator(color=palette, thickness=2, trace_length=30)

    def render(
        self,
        frame: np.ndarray[Any, Any],
        raw: sv.Detections,
        detections: list[Detection],
        overlay: Overlay | None = None,
    ) -> np.ndarray[Any, Any]:
        """Return an annotated copy of *frame*; the input is never modified."""
        canvas: npt.NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8).copy()
        overlay = overlay or Overlay()
        self._draw_zones(canvas, overlay)
        self._draw_lines(canvas, overlay)
        if len(raw) > 0:
            canvas = self._boxes.annotate(scene=canvas, detections=raw)
            if raw.tracker_id is not None:
                canvas = self._traces.annotate(scene=canvas, detections=raw)
            # supervision types LabelAnnotator.annotate for PIL images only; its
            # decorator accepts and returns an ndarray at runtime.
            canvas = cast(
                "npt.NDArray[np.uint8]",
                self._labels.annotate(
                    scene=canvas,  # type: ignore[arg-type]
                    detections=raw,
                    labels=[_label(item) for item in detections],
                ),
            )
        if overlay.hud:
            _draw_hud(canvas, overlay.hud)
        return canvas

    def _draw_zones(self, canvas: np.ndarray[Any, Any], overlay: Overlay) -> None:
        """Outline each polygon zone and print its current occupancy."""
        for name, polygon in overlay.zones.items():
            if len(polygon) < 3:
                continue
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], isClosed=True, color=_ZONE_COLOR, thickness=2)
            count = overlay.zone_counts.get(name)
            caption = name if count is None else f"{name}: {count}"
            anchor = min(polygon, key=lambda point: (point[1], point[0]))
            _draw_caption(canvas, caption, (anchor[0], max(14, anchor[1] - 6)), _ZONE_COLOR)

    def _draw_lines(self, canvas: np.ndarray[Any, Any], overlay: Overlay) -> None:
        """Draw each counting line with its in/out totals."""
        for name, (start, end) in overlay.lines.items():
            cv2.line(canvas, start, end, _LINE_COLOR, thickness=2)
            counts = overlay.line_counts.get(name)
            caption = name if counts is None else f"{name}  in {counts[0]} / out {counts[1]}"
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            _draw_caption(canvas, caption, (midpoint[0], max(14, midpoint[1] - 8)), _LINE_COLOR)


def _label(detection: Detection) -> str:
    """`#track class 0.87` when tracked, `class 0.87` otherwise."""
    prefix = "" if detection.track_id is None else f"#{detection.track_id} "
    return f"{prefix}{detection.class_name} {detection.confidence:.2f}"


def _draw_caption(
    canvas: np.ndarray[Any, Any], text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    """Small filled caption box so text stays readable over any background."""
    (width, height), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
    left, top = origin
    cv2.rectangle(canvas, (left, top - height - 4), (left + width + 6, top + 3), _HUD_BACKGROUND, -1)
    cv2.putText(canvas, text, (left + 3, top), _FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray[Any, Any], lines: list[str]) -> None:
    """Stats panel in the top-left corner."""
    padding = 6
    sizes = [cv2.getTextSize(line, _FONT, 0.45, 1)[0] for line in lines]
    width = max(size[0] for size in sizes) + padding * 2
    height = sum(size[1] + 6 for size in sizes) + padding
    panel = canvas[0:height, 0:width]
    if panel.size:
        canvas[0:height, 0:width] = cv2.addWeighted(
            panel, 0.35, np.full_like(panel, _HUD_BACKGROUND, dtype=np.uint8), 0.65, 0
        )
    offset = padding + sizes[0][1]
    for line, size in zip(lines, sizes, strict=True):
        cv2.putText(canvas, line, (padding, offset), _FONT, 0.45, _HUD_TEXT, 1, cv2.LINE_AA)
        offset += size[1] + 6
