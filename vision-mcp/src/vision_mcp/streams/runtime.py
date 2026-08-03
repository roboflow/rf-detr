"""The per-stream inference worker.

One async task per stream pulls frames off the bounded queue, runs the model on the
model's own thread, and publishes live state. Because the queue drops the oldest frame
under pressure, the worker always sees recent frames rather than a growing backlog. It
holds the latest frame in memory for the debug preview and writes nothing to disk.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import supervision as sv

from vision_mcp.api_contract import (
    Detection,
    Health,
    ImageSize,
    LiveSnapshot,
    StreamState,
    StreamStatus,
    StreamSummary,
)
from vision_mcp.clock import monotonic, utc_iso, utc_iso_or_none
from vision_mcp.config import EngineConfig, StreamEntry
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.inference.detector import Detector, InferenceOutput
from vision_mcp.inference.images import LoadedImage
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import redact
from vision_mcp.streams.capture import CaptureThread, source_label
from vision_mcp.streams.frames import Frame, LatestFrameQueue

logger = get_logger("vision-mcp.stream")

_STALL_SECONDS = 10.0
"""Connected but no processed frame for this long counts as unhealthy."""

_FPS_WINDOW = 30
"""Processed-frame timestamps kept for the rolling frame-rate estimate."""

_DEGRADED_FPS_RATIO = 0.5
"""Sustained throughput below this fraction of the target counts as degraded."""

_DEGRADED_DROP_RATIO = 0.5
"""Dropping more than this fraction of captured frames counts as degraded."""


class StreamObserver(Protocol):
    """Analytics attached to a stream. Called once per processed frame, in order."""

    async def observe(self, stream_id: str, frame: Frame, output: InferenceOutput) -> ObservedState:
        """Consume one inference result and return what should be reported live."""

    async def close(self) -> None:
        """Release anything the observer holds."""


@dataclass(slots=True)
class ObservedState:
    """What an observer wants reflected in live status and on the preview."""

    detections: list[Detection]
    raw: sv.Detections
    active_tracks: int = 0
    zone_counts: dict[str, int] | None = None
    line_counts: dict[str, tuple[int, int]] | None = None


@dataclass(slots=True)
class PreviewFrame:
    """The most recent processed frame, kept in memory for the preview only."""

    array: np.ndarray[Any, Any]
    raw: sv.Detections
    detections: list[Detection]
    index: int
    captured_at: float
    inference_ms: float
    zone_counts: dict[str, int]
    line_counts: dict[str, tuple[int, int]]


class StreamRuntime:
    """Owns one stream: its capture thread, its queue and its inference loop."""

    def __init__(
        self,
        stream_id: str,
        entry: StreamEntry,
        config: EngineConfig,
        detector: Detector,
        observer: StreamObserver | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.entry = entry
        self._config = config
        self._detector = detector
        self._observer = observer
        self._queue: LatestFrameQueue | None = None
        self._capture: CaptureThread | None = None
        self._task: asyncio.Task[None] | None = None
        self._started_at: str | None = None
        self._stopping = False

        self._processed_frames = 0
        self._processed_at: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._latency_ms: deque[float] = deque(maxlen=config.metrics.latency_samples)
        self._last_processed_at: float | None = None
        self._last_error: str | None = None
        self._counts_by_class: dict[str, int] = {}
        self._current_objects = 0
        self._active_tracks = 0
        self._preview: PreviewFrame | None = None

    @property
    def running(self) -> bool:
        """Whether the worker task is still alive."""
        return self._task is not None and not self._task.done()

    @property
    def queue_capacity(self) -> int:
        """Configured queue bound, reported even before the stream starts."""
        return self.entry.queue_size

    async def start(self) -> None:
        """Open the source and begin processing. Idempotent."""
        if self.running:
            return
        loop = asyncio.get_running_loop()
        self._reset_live_state()
        self._stopping = False
        self._queue = LatestFrameQueue(self.entry.queue_size, loop)
        resolution = self._config.models[self.entry.model].effective_resolution
        self._capture = CaptureThread(
            stream_id=self.stream_id,
            entry=self.entry,
            queue=self._queue,
            security=self._config.security,
            target_long_edge=resolution,
        )
        self._capture.start()
        self._started_at = utc_iso()
        self._task = asyncio.create_task(self._run(), name=f"stream:{self.stream_id}")
        logger.info(
            "stream started",
            extra={
                "stream_id": self.stream_id,
                "model": self.entry.model,
                "source": self._capture.label,
                "target_fps": self.entry.processing_fps,
            },
        )

    def _reset_live_state(self) -> None:
        """Clear counters and frame state so a restart begins a fresh live session."""
        self._processed_frames = 0
        self._processed_at.clear()
        self._latency_ms.clear()
        self._last_processed_at = None
        self._last_error = None
        self._counts_by_class = {}
        self._current_objects = 0
        self._active_tracks = 0
        self._preview = None

    async def stop(self) -> None:
        """Stop capture, drain the worker and release the source."""
        self._stopping = True
        if self._queue is not None:
            self._queue.close()
        if self._capture is not None:
            await asyncio.to_thread(self._capture.stop)
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        if self._observer is not None:
            await self._observer.close()
        self._preview = None
        logger.info("stream stopped", extra={"stream_id": self.stream_id})

    async def _run(self) -> None:
        """Pull frames until the queue closes, processing the newest available one."""
        queue = self._queue
        if queue is None:  # pragma: no cover - start() always assigns
            return
        while True:
            frame = await queue.get()
            if frame is None:
                break
            try:
                await self._process(frame)
            except asyncio.CancelledError:
                raise
            except VisionError as exc:
                self._last_error = redact(exc.message)
                logger.warning(
                    "stream inference failed",
                    extra={"stream_id": self.stream_id, "code": exc.code.value, "error": exc.message},
                )
                await asyncio.sleep(0.5)
            except Exception as exc:  # a worker must outlive one bad frame
                self._last_error = redact(exc)
                logger.exception("stream frame failed", extra={"stream_id": self.stream_id})
                await asyncio.sleep(0.5)

    async def _process(self, frame: Frame) -> None:
        """Run the model over one frame and publish the results."""
        image = LoadedImage(array=frame.array, size=frame.size, label=self.stream_id)
        output = await self._detector.detect_image(
            self.entry.model,
            image,
            confidence=self.entry.confidence,
            classes=self.entry.classes,
        )
        observed = ObservedState(detections=output.detections, raw=output.raw)
        if self._observer is not None:
            observed = await self._observer.observe(self.stream_id, frame, output)

        now = monotonic()
        self._processed_frames += 1
        self._processed_at.append(now)
        self._latency_ms.append(output.inference_ms)
        self._last_processed_at = now
        self._last_error = None
        self._counts_by_class = dict(Counter(item.class_name for item in observed.detections))
        self._current_objects = len(observed.detections)
        self._active_tracks = observed.active_tracks
        self._preview = PreviewFrame(
            array=frame.array,
            raw=observed.raw,
            detections=observed.detections,
            index=frame.index,
            captured_at=frame.captured_at,
            inference_ms=output.inference_ms,
            zone_counts=observed.zone_counts or {},
            line_counts=observed.line_counts or {},
        )

    def preview_frame(self) -> PreviewFrame | None:
        """The latest processed frame, or None if nothing has been processed yet."""
        return self._preview

    def latency_samples(self) -> list[float]:
        """Recent inference latencies in milliseconds, oldest first."""
        return list(self._latency_ms)

    @property
    def processed_fps(self) -> float:
        """Rolling processed frame rate over the recent window."""
        if len(self._processed_at) < 2:
            return 0.0
        span = self._processed_at[-1] - self._processed_at[0]
        if span <= 0:
            return 0.0
        return round((len(self._processed_at) - 1) / span, 2)

    @property
    def state(self) -> StreamState:
        """Capture state, or `stopped` when the stream is not running."""
        if self._capture is None or self._stopping:
            return "stopped"
        return self._capture.state.snapshot().state

    def health(self) -> tuple[Health, list[str]]:
        """Classify the stream and explain every reason it is not healthy."""
        if self._capture is None:
            return "unknown", ["stream not started"]
        capture = self._capture.state.snapshot()
        reasons: list[str] = []
        if self._stopping or capture.state == "stopped":
            return "unknown", ["stream stopped"]
        if capture.state == "disconnected":
            return "unhealthy", [capture.last_error or "capture source disconnected"]
        if not self._capture.alive:
            return "unhealthy", ["capture thread is not running"]
        if capture.state == "starting" and self._last_processed_at is None:
            return "unknown", ["waiting for the first frame"]

        unhealthy = False
        if capture.state == "reconnecting":
            reasons.append("reconnecting to capture source")
        if self._last_processed_at is not None:
            idle = monotonic() - self._last_processed_at
            if idle > _STALL_SECONDS:
                unhealthy = True
                reasons.append(f"no frame processed for {idle:.0f}s")
        fps = self.processed_fps
        if len(self._processed_at) >= _FPS_WINDOW and fps < self.entry.processing_fps * _DEGRADED_FPS_RATIO:
            reasons.append(f"processing {fps:.1f} fps against a target of {self.entry.processing_fps:.1f}")
        dropped = self.dropped_frames
        if capture.captured_frames > _FPS_WINDOW:
            ratio = dropped / max(capture.captured_frames, 1)
            if ratio > _DEGRADED_DROP_RATIO:
                reasons.append(f"dropping {ratio:.0%} of captured frames")
        if self._last_error is not None:
            reasons.append(self._last_error)
        if unhealthy:
            return "unhealthy", reasons
        if reasons:
            return "degraded", reasons
        return "healthy", []

    @property
    def dropped_frames(self) -> int:
        """Frames the queue discarded because inference could not keep up."""
        return 0 if self._queue is None else self._queue.dropped

    @property
    def queue_depth(self) -> int:
        """Frames waiting for inference right now."""
        return 0 if self._queue is None else self._queue.depth

    @property
    def queue_high_water(self) -> int:
        """Deepest the current queue has reached."""
        return 0 if self._queue is None else self._queue.high_water

    @property
    def capture_alive(self) -> bool:
        """Whether the capture thread is running."""
        return self._capture is not None and self._capture.alive

    @property
    def frame_size(self) -> ImageSize | None:
        """Size inference actually runs on, once the first frame has arrived."""
        return None if self._capture is None else self._capture.state.snapshot().frame_size

    @property
    def source(self) -> str:
        """Log-safe source description, with credentials redacted."""
        return self._capture.label if self._capture is not None else source_label(self.entry.source)

    def summary(self) -> StreamSummary:
        """Row for `list_streams`."""
        health, _ = self.health()
        return StreamSummary(
            stream_id=self.stream_id,
            state=self.state,
            health=health,
            model=self.entry.model,
            source=self.source,
            processed_frames=self._processed_frames,
            current_objects=self._current_objects,
            active_tracks=self._active_tracks,
        )

    def status(self) -> StreamStatus:
        """Full answer for `get_stream_status`."""
        capture = self._capture.state.snapshot() if self._capture is not None else None
        health, reasons = self.health()
        return StreamStatus(
            stream_id=self.stream_id,
            state=self.state,
            health=health,
            health_reasons=reasons,
            model=self.entry.model,
            source=self.source,
            frame_size=None if capture is None else capture.frame_size,
            target_fps=self.entry.processing_fps,
            processed_fps=self.processed_fps,
            captured_frames=0 if capture is None else capture.captured_frames,
            processed_frames=self._processed_frames,
            dropped_frames=self.dropped_frames,
            queue_depth=self.queue_depth,
            queue_capacity=self.queue_capacity,
            active_tracks=self._active_tracks,
            current_objects=self._current_objects,
            last_frame_at=utc_iso_or_none(self._preview.captured_at if self._preview else None),
            started_at=self._started_at,
            reconnect_attempts=0 if capture is None else capture.reconnect_attempts,
            last_error=self._last_error or (None if capture is None else capture.last_error),
        )

    def snapshot(self) -> LiveSnapshot:
        """Answer for `get_live_snapshot`."""
        health, _ = self.health()
        return LiveSnapshot(
            stream_id=self.stream_id,
            state=self.state,
            health=health,
            counts_by_class=dict(self._counts_by_class),
            current_objects=self._current_objects,
            active_tracks=self._active_tracks,
            processed_fps=self.processed_fps,
            queue_depth=self.queue_depth,
            queue_capacity=self.queue_capacity,
            dropped_frames=self.dropped_frames,
            inference_ms_p50=_percentile(self._latency_ms, 50),
            last_frame_at=utc_iso_or_none(self._preview.captured_at if self._preview else None),
        )

    def require_running(self) -> None:
        """Guard for tools that need live frames.

        Raises:
            VisionError: STREAM_DISCONNECTED when the stream is not currently connected.
        """
        if not self.running or self.state in ("disconnected", "stopped"):
            raise VisionError(
                ErrorCode.STREAM_DISCONNECTED,
                "Stream is not currently connected.",
                {"stream_id": self.stream_id, "state": self.state},
            )


def _percentile(samples: deque[float], percentile: float) -> float | None:
    """Nearest-rank percentile over the latency window."""
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, round(percentile / 100 * len(ordered)) - 1))
    return round(ordered[rank], 2)
