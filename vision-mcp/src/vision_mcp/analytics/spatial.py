# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Zone occupancy and line crossings.

Zones and lines are configured in normalised coordinates and resolved to pixels the first time a frame
arrives, so a stream whose resolution changes mid-run re-resolves cleanly.

Everything here is edge-triggered: an entry is emitted when a track first appears inside a zone, an exit when
it has been absent for a short grace period (boxes flicker across a boundary far more often than objects do),
and an occupancy violation only on the frame the limit is first exceeded — never once per frame while it stays
exceeded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import supervision as sv

from vision_mcp.analytics.lines import CountingLine, build_lines
from vision_mcp.config import StreamEntry
from vision_mcp.streams.preview import resolve_lines, resolve_zones

#: Consecutive frames a track must be absent from a zone before it counts as an exit.
_EXIT_GRACE_FRAMES = 2


@dataclass(slots=True)
class ZoneVisit:
    """One track's stay in one zone."""

    zone: str
    track_id: int
    class_name: str
    entered_at: float
    exited_at: float | None = None

    @property
    def dwell_seconds(self) -> float | None:
        """Seconds spent in the zone, or None while the visit is still open."""
        return None if self.exited_at is None else max(0.0, self.exited_at - self.entered_at)


@dataclass(slots=True)
class Crossing:
    """One track crossing one counting line."""

    line: str
    track_id: int
    class_name: str
    at: float
    direction: Literal["in", "out"]


@dataclass(slots=True)
class Violation:
    """A zone whose occupancy has just risen above its configured limit."""

    zone: str
    occupancy: int
    limit: int


@dataclass(slots=True)
class SpatialUpdate:
    """What one frame changed spatially."""

    zone_counts: dict[str, int] = field(default_factory=dict)
    zone_by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    line_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    entries: list[ZoneVisit] = field(default_factory=list)
    exits: list[ZoneVisit] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)


class SpatialIndex:
    """Zone membership and line counters for one stream."""

    def __init__(self, entry: StreamEntry) -> None:
        self._entry = entry
        self._size: tuple[int, int] | None = None
        self._zones: dict[str, sv.PolygonZone] = {}
        self._lines: dict[str, CountingLine] = {}
        self._polygons: dict[str, list[tuple[int, int]]] = {}
        self._points: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self._open: dict[str, dict[int, ZoneVisit]] = {name: {} for name in entry.zones}
        self._absent: dict[str, dict[int, int]] = {name: {} for name in entry.zones}
        self._over_limit: set[str] = set()

    @property
    def polygons(self) -> dict[str, list[tuple[int, int]]]:
        """Resolved zone polygons in pixels, empty until the first frame."""
        return self._polygons

    @property
    def line_points(self) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
        """Resolved line endpoints in pixels, empty until the first frame."""
        return self._points

    @property
    def open_visits(self) -> list[ZoneVisit]:
        """Every visit currently in progress, across all zones."""
        return [visit for visits in self._open.values() for visit in visits.values()]

    def configure(self, width: int, height: int) -> None:
        """Resolve zones and lines for a frame size, rebuilding only when it changes."""
        if self._size == (width, height):
            return
        self._size = (width, height)
        self._polygons = resolve_zones(self._entry, width, height)
        self._points = resolve_lines(self._entry, width, height)
        self._zones = {
            name: sv.PolygonZone(polygon=np.array(polygon, dtype=int))
            for name, polygon in self._polygons.items()
        }
        self._lines = build_lines(self._points)

    def update(self, tracked: sv.Detections, classes: Mapping[int, str], at: float) -> SpatialUpdate:
        """Fold one frame's tracked detections into zone membership and line counters.

        Args:
            tracked: Detections carrying tracker IDs, from `Tracker.update`.
            classes: Track ID to class name, used to label events.
            at: Wall-clock capture time of the frame, in epoch seconds.
        """
        update = SpatialUpdate()
        for name, zone in self._zones.items():
            self._update_zone(name, _inside(zone, tracked), classes, at, update)
        for name, line in self._lines.items():
            update.crossings.extend(_cross(name, line, tracked, classes, at))
            update.line_counts[name] = line.counts
        return update

    def close_tracks(self, track_ids: Iterable[int], at: float) -> list[ZoneVisit]:
        """Close open visits for tracks that have been retired."""
        closed: list[ZoneVisit] = []
        for track_id in track_ids:
            for name, visits in self._open.items():
                visit = visits.pop(int(track_id), None)
                self._absent[name].pop(int(track_id), None)
                if visit is not None:
                    visit.exited_at = at
                    closed.append(visit)
        return closed

    def _update_zone(
        self,
        name: str,
        present: set[int],
        classes: Mapping[int, str],
        at: float,
        update: SpatialUpdate,
    ) -> None:
        """Apply one zone's membership changes to *update*."""
        visits = self._open[name]
        for track_id in sorted(present - set(visits)):
            visit = ZoneVisit(name, track_id, classes.get(track_id, "unknown"), at)
            visits[track_id] = visit
            update.entries.append(visit)
        update.exits.extend(self._departures(name, present, at))
        occupancy = len(visits)
        update.zone_counts[name] = occupancy
        update.zone_by_class[name] = _by_class(visits.values())
        violation = self._violation(name, occupancy)
        if violation is not None:
            update.violations.append(violation)

    def _departures(self, name: str, present: set[int], at: float) -> list[ZoneVisit]:
        """Close visits absent for the grace period; reset the counter for those still in."""
        visits, absent = self._open[name], self._absent[name]
        left: list[ZoneVisit] = []
        for track_id in sorted(set(visits) - present):
            absent[track_id] = absent.get(track_id, 0) + 1
            if absent[track_id] > _EXIT_GRACE_FRAMES:
                visit = visits.pop(track_id)
                del absent[track_id]
                visit.exited_at = at
                left.append(visit)
        for track_id in present:
            absent.pop(track_id, None)
        return left

    def _violation(self, name: str, occupancy: int) -> Violation | None:
        """Report a limit breach once, on the frame it starts."""
        limit = self._entry.occupancy_limits.get(name)
        if limit is None or occupancy <= limit:
            self._over_limit.discard(name)
            return None
        if name in self._over_limit:
            return None
        self._over_limit.add(name)
        return Violation(zone=name, occupancy=occupancy, limit=limit)


def _inside(zone: sv.PolygonZone, tracked: sv.Detections) -> set[int]:
    """Track IDs whose anchor point falls inside *zone*."""
    if len(tracked) == 0 or tracked.tracker_id is None:
        return set()
    mask = cast("np.ndarray[tuple[int], np.dtype[np.bool_]]", zone.trigger(detections=tracked))
    return {int(track_id) for track_id, hit in zip(tracked.tracker_id, mask, strict=False) if bool(hit)}


def _cross(
    name: str,
    line: CountingLine,
    tracked: sv.Detections,
    classes: Mapping[int, str],
    at: float,
) -> list[Crossing]:
    """Crossings of one line in this frame."""
    return [
        Crossing(
            line=name,
            track_id=track_id,
            class_name=classes.get(track_id, "unknown"),
            at=at,
            direction=direction,
        )
        for track_id, direction in line.update(tracked)
    ]


def _by_class(visits: Iterable[ZoneVisit]) -> dict[str, int]:
    """Occupancy split by class name, ordered by descending count then name."""
    counts: dict[str, int] = {}
    for visit in visits:
        counts[visit.class_name] = counts.get(visit.class_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
