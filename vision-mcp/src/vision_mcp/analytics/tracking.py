"""ByteTrack wrapper producing stable per-stream track records.

Track IDs are temporary motion-consistency handles. They are not identities: nothing here
recognises faces, matches a person across streams, or survives a stream restart. A track
that disappears for longer than the configured lost-track window is retired for good.

`supervision`'s ByteTrack does the association work. This module owns the bookkeeping the
tools need on top of it: first/last seen wall-clock times, frame counts, mean confidence
and lifecycle records worth persisting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

from vision_mcp.api_contract import BoundingBox, Detection
from vision_mcp.config import TrackingConfig

_INDEX_KEY = "vision_index"
"""Data column used to map tracked rows back to their source detection index."""


@dataclass(slots=True)
class TrackRecord:
    """The life so far of one tracked object."""

    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    frames: int
    confidence_sum: float
    confidence: float
    box: BoundingBox

    @property
    def mean_confidence(self) -> float:
        """Average confidence over every frame this track appeared in."""
        return self.confidence_sum / max(self.frames, 1)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock seconds between the first and most recent sighting."""
        return max(0.0, self.last_seen - self.first_seen)


@dataclass(slots=True)
class TrackedFrame:
    """One frame after association."""

    detections: list[Detection]
    """Every detection in the frame, with `track_id` filled in where one was assigned."""

    raw: sv.Detections
    """The same detections as `sv.Detections`, carrying tracker IDs for annotation."""

    tracked: sv.Detections
    """Only the rows ByteTrack assigned an ID to; zone and line logic uses this subset."""

    active: list[TrackRecord]
    """Records seen in this frame."""

    expired: list[TrackRecord]
    """Records retired in this frame because they exceeded the lost-track window."""


class Tracker:
    """Per-stream ByteTrack instance plus track lifecycle bookkeeping."""

    def __init__(self, config: TrackingConfig, processing_fps: float) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=max(1, round(config.lost_track_seconds * processing_fps)),
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=max(1, round(processing_fps)),
        )
        self._lost_seconds = config.lost_track_seconds
        self._records: dict[int, TrackRecord] = {}
        self._live: set[int] = set()

    @property
    def records(self) -> dict[int, TrackRecord]:
        """Every track still within the lost-track window, keyed by track ID."""
        return self._records

    @property
    def active_count(self) -> int:
        """Tracks seen in the most recent processed frame."""
        return len(self._live)

    @property
    def active_records(self) -> list[TrackRecord]:
        """Track records seen in the latest processed frame only."""
        return [self._records[track_id] for track_id in self._live if track_id in self._records]

    def update(self, raw: sv.Detections, detections: list[Detection], at: float) -> TrackedFrame:
        """Associate one frame's detections with existing tracks.

        Args:
            raw: Detections for the frame, index-aligned with *detections*.
            detections: Contract detections for the same frame.
            at: Wall-clock capture time of the frame, in epoch seconds.
        """
        raw.data[_INDEX_KEY] = np.arange(len(raw), dtype=int)
        tracked = self._tracker.update_with_detections(raw)
        by_index = _ids_by_index(tracked)
        labelled = [
            item.model_copy(update={"track_id": by_index.get(index)}) for index, item in enumerate(detections)
        ]
        raw.tracker_id = _tracker_ids(len(labelled), by_index)
        self._live = set(by_index.values())
        active = [self._touch(item, at) for item in labelled if item.track_id is not None]
        return TrackedFrame(
            detections=labelled,
            raw=raw,
            tracked=tracked,
            active=active,
            expired=self._expire(at),
        )

    def flush(self) -> list[TrackRecord]:
        """Retire and return every remaining record, for persistence at shutdown."""
        remaining = list(self._records.values())
        self._records.clear()
        self._live.clear()
        return remaining

    def _touch(self, detection: Detection, at: float) -> TrackRecord:
        """Create or extend the record for *detection*'s track."""
        track_id = int(detection.track_id or 0)
        record = self._records.get(track_id)
        if record is None:
            record = TrackRecord(
                track_id=track_id,
                class_name=detection.class_name,
                first_seen=at,
                last_seen=at,
                frames=0,
                confidence_sum=0.0,
                confidence=detection.confidence,
                box=detection.box,
            )
            self._records[track_id] = record
        record.last_seen = max(record.last_seen, at)
        record.frames += 1
        record.confidence_sum += detection.confidence
        record.confidence = detection.confidence
        record.class_name = detection.class_name
        record.box = detection.box
        return record

    def _expire(self, at: float) -> list[TrackRecord]:
        """Drop records unseen for longer than the lost-track window."""
        stale = [
            record
            for track_id, record in self._records.items()
            if track_id not in self._live and at - record.last_seen > self._lost_seconds
        ]
        for record in stale:
            del self._records[record.track_id]
        return stale


def _ids_by_index(tracked: sv.Detections) -> dict[int, int]:
    """Map source detection index to assigned track ID."""
    indices = tracked.data.get(_INDEX_KEY) if len(tracked) else None
    if indices is None or tracked.tracker_id is None:
        return {}
    return {int(index): int(track_id) for index, track_id in zip(indices, tracked.tracker_id, strict=False)}


def _tracker_ids(count: int, by_index: dict[int, int]) -> np.ndarray[Any, Any] | None:
    """Tracker ID column for annotation; untracked rows get distinct negative IDs.

    Distinct negatives keep `supervision`'s trace annotator from stitching unrelated
    untracked boxes into one path.
    """
    if not by_index:
        return None
    return np.array([by_index.get(index, -index - 1) for index in range(count)], dtype=int)
