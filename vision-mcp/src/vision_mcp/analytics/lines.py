"""Directional counting lines.

`supervision` ships `LineZone`, but its 0.29 implementation calls `np.cross` on
two-dimensional vectors, which NumPy removed in 2.x — it raises on every frame. The maths
is four lines long, so this module owns it rather than pinning NumPy backwards.

A crossing is edge-triggered per track: the anchor's side of the line is remembered between
frames and a crossing is reported only when that side flips. Tracks are anchored at the
bottom centre of their box, which is where an object meets the ground plane and is far
steadier than the centroid when a box grows or shrinks near the camera.

Direction is defined by the line's orientation: walking from `start` to `end`, a track
crossing right to left counts as `in`, and left to right as `out`. Swapping the endpoints
in configuration therefore swaps the two counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import supervision as sv

Direction = Literal["in", "out"]

_FORGET_AFTER = 512
"""Updates a track may go unseen before its remembered side is dropped."""


class CountingLine:
    """One configured line, plus the per-track state needed to count crossings."""

    def __init__(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self._start = (float(start[0]), float(start[1]))
        self._end = (float(end[0]), float(end[1]))
        self._sides: dict[int, int] = {}
        self._seen: dict[int, int] = {}
        self._tick = 0
        self.in_count = 0
        self.out_count = 0

    def update(self, tracked: sv.Detections) -> list[tuple[int, Direction]]:
        """Fold one frame in, returning `(track_id, direction)` for tracks that crossed."""
        self._tick += 1
        crossings: list[tuple[int, Direction]] = []
        if len(tracked) and tracked.tracker_id is not None:
            for track_id, anchor in zip(tracked.tracker_id, _anchors(tracked.xyxy), strict=False):
                crossing = self._observe(int(track_id), anchor)
                if crossing is not None:
                    crossings.append(crossing)
        self._forget()
        return crossings

    @property
    def counts(self) -> tuple[int, int]:
        """Cumulative `(in, out)` totals since the stream started."""
        return self.in_count, self.out_count

    def _observe(self, track_id: int, anchor: tuple[float, float]) -> tuple[int, Direction] | None:
        """Record one track's position, reporting a crossing when its side flips."""
        self._seen[track_id] = self._tick
        side = _side(self._start, self._end, anchor)
        previous = self._sides.get(track_id)
        if side != 0:
            self._sides[track_id] = side
        if previous is None or side == 0 or side == previous:
            return None
        direction: Direction = "in" if side > 0 else "out"
        if direction == "in":
            self.in_count += 1
        else:
            self.out_count += 1
        return track_id, direction

    def _forget(self) -> None:
        """Drop state for tracks that have been gone long enough to never return."""
        stale = [track_id for track_id, tick in self._seen.items() if self._tick - tick > _FORGET_AFTER]
        for track_id in stale:
            self._seen.pop(track_id, None)
            self._sides.pop(track_id, None)


def build_lines(points: Mapping[str, tuple[tuple[int, int], tuple[int, int]]]) -> dict[str, CountingLine]:
    """Build a counter per configured line, from already-resolved pixel endpoints."""
    return {name: CountingLine(start, end) for name, (start, end) in points.items()}


def _anchors(boxes: np.ndarray[Any, Any]) -> list[tuple[float, float]]:
    """Bottom-centre anchor for each `xyxy` box."""
    return [(float(box[0] + box[2]) / 2.0, float(box[3])) for box in boxes]


def _side(start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]) -> int:
    """Which side of the directed line the point falls on: 1, -1, or 0 when on it."""
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0
