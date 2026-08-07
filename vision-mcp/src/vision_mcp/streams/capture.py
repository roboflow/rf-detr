# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""One blocking capture thread per stream.

`cv2.VideoCapture.read()` blocks, so it cannot live on the event loop. Each stream owns exactly one thread which opens
its source, paces itself to `processing_fps`, downscales once, and pushes into the bounded queue. Everything the rest of
the engine needs to know about connection state is published through `CaptureState`, read under a lock.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_mcp.api_contract import ImageSize, StreamState
from vision_mcp.clock import monotonic
from vision_mcp.config import SecurityConfig, StreamEntry
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import redact_source, validate_local_path, validate_stream_url
from vision_mcp.streams.frames import Frame, LatestFrameQueue

logger = get_logger("vision-mcp.capture")

_URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")
_READ_FAILURE_LIMIT = 5
"""Consecutive failed reads tolerated before the source counts as dropped."""

_LIVE_GRAB_INTERVAL = 0.01
"""How often a live source is drained between processed frames, in seconds."""

_MACOS_CAMERA_FPS = 30.0
"""Stable AVFoundation webcam mode requested before the first read."""


@dataclass
class CaptureState:
    """Everything the status tools report about the capture side of a stream."""

    state: StreamState = "starting"
    frame_size: ImageSize | None = None
    captured_frames: int = 0
    reconnect_attempts: int = 0
    last_error: str | None = None
    last_frame_at: float | None = None
    started_at: float | None = None
    source_fps: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> CaptureState:
        """A consistent copy for a status response."""
        with self._lock:
            return CaptureState(
                state=self.state,
                frame_size=self.frame_size,
                captured_frames=self.captured_frames,
                reconnect_attempts=self.reconnect_attempts,
                last_error=self.last_error,
                last_frame_at=self.last_frame_at,
                started_at=self.started_at,
                source_fps=self.source_fps,
            )

    def update(self, **fields: Any) -> None:
        """Publish new values atomically."""
        with self._lock:
            for name, value in fields.items():
                setattr(self, name, value)


def resolve_source(source: int | str, security: SecurityConfig) -> tuple[int | str, str]:
    """Validate a configured source and return `(opencv target, log-safe label)`.

    Integers are camera indices. Strings are either a stream URL checked against the
    outbound policy, or a local file resolved inside `security.filesystem_roots`.

    Raises:
        VisionError: UNSUPPORTED_SOURCE when the source is neither a camera, an allowed
            URL, nor a readable file under a configured root.
    """
    if isinstance(source, int):
        if source < 0:
            raise VisionError(ErrorCode.UNSUPPORTED_SOURCE, "Camera index must not be negative.")
        return source, f"camera:{source}"
    lowered = source.lower()
    if lowered.startswith(_URL_SCHEMES):
        url = validate_stream_url(source, security.allowed_url_hosts, security.allow_private_network)
        return url, redact_source(url)
    path = validate_local_path(security.filesystem_roots, source)
    if not path.is_file():
        raise VisionError(
            ErrorCode.UNSUPPORTED_SOURCE, "Stream source file does not exist.", {"source": path.name}
        )
    return str(path), path.name


class CaptureThread:
    """Opens a source, paces it, downscales it and feeds the stream queue."""

    def __init__(
        self,
        stream_id: str,
        entry: StreamEntry,
        queue: LatestFrameQueue,
        security: SecurityConfig,
        target_long_edge: int | None,
    ) -> None:
        self._stream_id = stream_id
        self._entry = entry
        self._queue = queue
        self._target, self._label = resolve_source(entry.source, security)
        self._long_edge = target_long_edge
        self._interval = 1.0 / entry.processing_fps
        self._live = isinstance(entry.source, int) or str(entry.source).lower().startswith(_URL_SCHEMES)
        # AVFoundation webcams can stop delivering frames after VideoCapture.grab().
        # CAP_PROP_BUFFERSIZE keeps their single unread frame recent, so pace cameras
        # with a timed wait and reserve grab-based draining for URLs and files.
        self._grab_while_waiting = not isinstance(entry.source, int)
        self._grab_interval = _LIVE_GRAB_INTERVAL
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"capture:{stream_id}", daemon=True)
        self.state = CaptureState()

    @property
    def label(self) -> str:
        """Log-safe description of the source; RTSP credentials never appear."""
        return self._label

    @property
    def alive(self) -> bool:
        """Whether the capture thread is still running."""
        return self._thread.is_alive()

    def start(self) -> None:
        """Begin capturing."""
        self.state.update(started_at=monotonic(), state="starting")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to finish and wait briefly for it."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.state.update(state="stopped")

    def _run(self) -> None:
        """Connect, capture until told to stop, reconnect with backoff in between."""
        backoff = self._entry.reconnect.initial_seconds
        index = 0
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self._sleep_backoff(backoff)
                backoff = min(backoff * self._entry.reconnect.multiplier, self._entry.reconnect.max_seconds)
                continue
            backoff = self._entry.reconnect.initial_seconds
            try:
                index = self._pump(capture, index)
            finally:
                capture.release()
            if self._stop.is_set():
                break
            self.state.update(state="reconnecting")
            self._sleep_backoff(backoff)
            backoff = min(backoff * self._entry.reconnect.multiplier, self._entry.reconnect.max_seconds)
        self.state.update(state="stopped")
        self._queue.close()
        logger.info("capture stopped", extra={"stream_id": self._stream_id})

    def _open(self) -> cv2.VideoCapture | None:
        """Open the source, or record why it failed and return None."""
        attempts = self.state.snapshot().reconnect_attempts
        try:
            capture = _video_capture(self._target)
        except cv2.error as exc:  # pragma: no cover - backend specific
            self.state.update(
                state="disconnected",
                last_error=redact_source(str(exc)),
                reconnect_attempts=attempts + 1,
            )
            return None
        if not capture.isOpened():
            capture.release()
            self.state.update(
                state="disconnected",
                last_error="Could not open capture source.",
                reconnect_attempts=attempts + 1,
            )
            logger.warning(
                "capture source unavailable",
                extra={"stream_id": self._stream_id, "source": self._label, "attempt": attempts + 1},
            )
            return None
        # Keep the driver's own buffer at one frame so reads return live video, not history.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(self._target, int) and sys.platform == "darwin":
            capture.set(cv2.CAP_PROP_FPS, max(_MACOS_CAMERA_FPS, self._entry.processing_fps))
        reported = capture.get(cv2.CAP_PROP_FPS)
        source_fps = float(reported) if reported and reported > 0 else None
        # A live camera is drained as fast as it produces; a file is stepped at its own
        # frame rate so playback tracks wall clock instead of racing to the end.
        if self._live or source_fps is None:
            self._grab_interval = _LIVE_GRAB_INTERVAL
        else:
            self._grab_interval = 1.0 / source_fps
        self.state.update(state="connected", last_error=None, source_fps=source_fps)
        logger.info(
            "capture connected",
            extra={"stream_id": self._stream_id, "source": self._label, "source_fps": source_fps},
        )
        return capture

    def _pump(self, capture: cv2.VideoCapture, index: int) -> int:
        """Read until the source dies or we are stopped.

        Returns the next frame index.
        """
        failures = 0
        black_frames = 0
        black_warning_logged = False
        next_due = monotonic()
        while not self._stop.is_set():
            now = monotonic()
            if now < next_due:
                if not self._grab_while_waiting:
                    self._stop.wait(min(next_due - now, self._grab_interval))
                    continue
                # Grab without decoding to keep the socket drained while we wait our turn.
                if not capture.grab():
                    failures += 1
                    if failures >= _READ_FAILURE_LIMIT:
                        break
                    self._stop.wait(self._grab_interval)
                    continue
                self._stop.wait(min(next_due - now, self._grab_interval))
                continue
            ok, raw = capture.read()
            if not ok or raw is None:
                failures += 1
                if failures >= _READ_FAILURE_LIMIT:
                    self.state.update(last_error="Capture source stopped returning frames.")
                    break
                self._stop.wait(0.05)
                continue
            failures = 0
            if index < _BLACK_FRAME_LIMIT + 1 and not np.any(raw):
                black_frames += 1
                if black_frames >= _BLACK_FRAME_LIMIT and not black_warning_logged:
                    logger.warning(
                        "webcam returned several entirely black frames; "
                        "macOS camera permission may be denied. "
                        "Enable access in System Settings → Privacy & Security → Camera",
                        extra={"stream_id": self._stream_id, "source": self._label},
                    )
                    black_warning_logged = True
            else:
                black_frames = 0
            next_due = now + self._interval
            index += 1
            self._publish(raw, index)
        return index

    def _publish(self, raw: np.ndarray[Any, Any], index: int) -> None:
        """Downscale once, convert to RGB and offer the frame to the worker."""
        array = _downscale(raw, self._long_edge)
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        size = ImageSize(width=width, height=height)
        captured = monotonic()
        state = self.state.snapshot()
        self.state.update(captured_frames=state.captured_frames + 1, frame_size=size, last_frame_at=captured)
        frame = Frame(
            array=rgb,
            size=size,
            index=index,
            captured_at=_wall_clock(),
            captured_monotonic=captured,
        )
        self._queue.push(frame)

    def _sleep_backoff(self, seconds: float) -> None:
        """Wait out the backoff, but wake immediately on stop."""
        attempts = self.state.snapshot().reconnect_attempts
        logger.info(
            "reconnecting to capture source",
            extra={
                "stream_id": self._stream_id,
                "source": self._label,
                "backoff_seconds": round(seconds, 2),
                "attempt": attempts,
            },
        )
        self._stop.wait(seconds)


def _downscale(array: np.ndarray[Any, Any], long_edge: int | None) -> np.ndarray[Any, Any]:
    """Shrink to the model's input size before queueing.

    Never upscales.
    """
    if long_edge is None:
        return array
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= long_edge:
        return array
    scale = long_edge / longest
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(array, target, interpolation=cv2.INTER_AREA)


def _video_capture(target: int | str) -> cv2.VideoCapture:
    """Open macOS webcams explicitly with AVFoundation; use auto-selection elsewhere."""
    if isinstance(target, int) and sys.platform == "darwin":
        return cv2.VideoCapture(target, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(target)


def _wall_clock() -> float:
    """Wall-clock capture time, used only for timestamps that leave the process."""
    return time.time()


def source_label(source: int | str) -> str:
    """Log-safe label for a configured source without opening it."""
    if isinstance(source, int):
        return f"camera:{source}"
    if source.lower().startswith(_URL_SCHEMES):
        return redact_source(source)
    return Path(source).name


_BLACK_FRAME_LIMIT = 3
"""Consecutive all-black startup frames required before warning about permissions."""
