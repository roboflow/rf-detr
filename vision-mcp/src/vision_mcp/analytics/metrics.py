"""Per-stream metric accumulation, flushed to SQLite in fixed buckets.

The inference loop only ever increments counters here — no I/O, no locks beyond the GIL.
A separate aggregator task turns the accumulated numbers into rows every
`metrics.aggregation_interval_seconds`, which is what keeps the database small enough to
answer `get_detection_rate` over a week without storing a row per frame.

Counting rules, kept deliberately explicit because they are easy to conflate:

- `frame_detections` sums detections over frames. Ten frames with three people each is 30.
- `unique_objects` counts distinct track IDs and lives in the `tracks` table.
- Confidence histograms bucket every detection, not every track.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from vision_mcp.api_contract import Detection
from vision_mcp.storage.database import Statement

CONFIDENCE_BINS = 10
"""Histogram resolution: ten 0.1-wide bins spanning [0, 1]."""


def safe_rate(numerator: float, denominator: float) -> float:
    """Return zero for an empty denominator instead of raising or returning NaN."""
    return 0.0 if denominator <= 0 else numerator / denominator


@dataclass(slots=True)
class _ClassTotals:
    """Running totals for one class name within the current bucket."""

    detections: int = 0
    confidence_sum: float = 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean detection confidence, or zero when the class was never seen."""
        return self.confidence_sum / max(self.detections, 1)


@dataclass(slots=True)
class BucketCounts:
    """Frame counters supplied by the runtime at flush time."""

    captured: int = 0
    processed: int = 0
    dropped: int = 0


class MetricsCollector:
    """Accumulates one stream's metrics between flushes."""

    def __init__(self, stream_id: str, latency_samples: int) -> None:
        self._stream_id = stream_id
        self._latency: deque[float] = deque(maxlen=latency_samples)
        self._classes: dict[str, _ClassTotals] = {}
        self._histogram: list[int] = [0] * CONFIDENCE_BINS
        self._frame_detections = 0
        self._processed = 0

    @property
    def processed_frames(self) -> int:
        """Frames folded in since the last flush."""
        return self._processed

    def observe(self, detections: list[Detection], inference_ms: float) -> None:
        """Fold one processed frame into the current bucket."""
        self._processed += 1
        self._frame_detections += len(detections)
        self._latency.append(inference_ms)
        for detection in detections:
            totals = self._classes.setdefault(detection.class_name, _ClassTotals())
            totals.detections += 1
            totals.confidence_sum += detection.confidence
            self._histogram[_bin_index(detection.confidence)] += 1

    def flush(self, bucket_start: float, bucket_seconds: float, counts: BucketCounts) -> list[Statement]:
        """Turn the current bucket into rows and reset the accumulators.

        Args:
            bucket_start: Epoch seconds at which this bucket opened.
            bucket_seconds: Bucket width, used to derive rates on read.
            counts: Capture-side frame counters measured over the same bucket.
        """
        statements = [self._metrics_row(bucket_start, bucket_seconds, counts)]
        statements.extend(_summary_rows(self._stream_id, bucket_start, self._classes))
        statements.extend(_histogram_rows(self._stream_id, bucket_start, self._histogram))
        self._reset()
        return statements

    def _metrics_row(self, bucket_start: float, bucket_seconds: float, counts: BucketCounts) -> Statement:
        """The processing_metrics row for the current bucket."""
        ordered = sorted(self._latency)
        mean = round(sum(ordered) / len(ordered), 2) if ordered else None
        return (
            "INSERT INTO processing_metrics (stream_id, bucket_start, bucket_seconds, captured_frames,"
            " processed_frames, dropped_frames, frame_detections, latency_samples, latency_mean_ms,"
            " latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_max_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._stream_id,
                bucket_start,
                bucket_seconds,
                counts.captured,
                counts.processed,
                counts.dropped,
                self._frame_detections,
                len(ordered),
                mean,
                percentile(ordered, 50),
                percentile(ordered, 95),
                percentile(ordered, 99),
                round(ordered[-1], 2) if ordered else None,
            ),
        )

    def _reset(self) -> None:
        """Clear the accumulators for the next bucket."""
        self._latency.clear()
        self._classes = {}
        self._histogram = [0] * CONFIDENCE_BINS
        self._frame_detections = 0
        self._processed = 0


def percentile(ordered: list[float], rank: float) -> float | None:
    """Nearest-rank percentile of an already sorted list."""
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, round(percentile_index(len(ordered), rank))))
    return round(ordered[index], 2)


def percentile_index(count: int, rank: float) -> float:
    """Zero-based index for *rank* over *count* samples."""
    return (rank / 100.0) * count - 1


def _bin_index(confidence: float) -> int:
    """Histogram bin for one confidence value, clamped to the last bin at 1.0."""
    return min(CONFIDENCE_BINS - 1, max(0, int(confidence * CONFIDENCE_BINS)))


def _summary_rows(stream_id: str, bucket_start: float, classes: dict[str, _ClassTotals]) -> list[Statement]:
    """One detection_summaries row per class seen in the bucket."""
    return [
        (
            "INSERT INTO detection_summaries (stream_id, bucket_start, class_name, detections,"
            " mean_confidence) VALUES (?, ?, ?, ?, ?)",
            (stream_id, bucket_start, name, totals.detections, round(totals.mean_confidence, 4)),
        )
        for name, totals in sorted(classes.items())
        if totals.detections
    ]


def _histogram_rows(stream_id: str, bucket_start: float, histogram: list[int]) -> list[Statement]:
    """One confidence_histogram row per non-empty bin."""
    return [
        (
            "INSERT INTO confidence_histogram (stream_id, bucket_start, bin_index, count)"
            " VALUES (?, ?, ?, ?)",
            (stream_id, bucket_start, index, count),
        )
        for index, count in enumerate(histogram)
        if count
    ]


@dataclass(slots=True)
class HealthSample:
    """One stream_health row, written on the same cadence as the metric buckets."""

    stream_id: str
    at: float
    state: str
    health: str
    processed_fps: float
    queue_depth: int
    dropped_frames: int
    last_error: str | None = None

    def statement(self) -> Statement:
        """The INSERT for this sample."""
        return (
            "INSERT INTO stream_health (stream_id, at, state, health, processed_fps, queue_depth,"
            " dropped_frames, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.stream_id,
                self.at,
                self.state,
                self.health,
                self.processed_fps,
                self.queue_depth,
                self.dropped_frames,
                self.last_error,
            ),
        )
