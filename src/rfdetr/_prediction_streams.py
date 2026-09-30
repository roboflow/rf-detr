# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bounded background capture for live prediction sources."""

from collections import deque
from collections.abc import Generator
from math import isfinite
from threading import Condition, Event, Thread
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import numpy as np

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class _LiveCapture:
    """Own one capture and its bounded frame queue."""

    def __init__(self, source: str | int, vid_stride: int, stream_buffer: bool, *, finite: bool = False) -> None:
        """Read an initial frame before starting background capture."""
        try:
            import cv2
        except ImportError as error:
            raise ImportError(
                'Live prediction requires OpenCV. Install it with `uv pip install "rfdetr[stream]"`.'
            ) from error

        self.source = source
        self.network = isinstance(source, str) and urlparse(source).scheme.lower() in {
            "http",
            "https",
            "rtsp",
            "rtsps",
            "rtmp",
            "tcp",
        }
        self.vid_stride = vid_stride
        self.stream_buffer = stream_buffer
        self.frames: deque[np.ndarray[Any, Any]] = deque()
        self.condition = Condition()
        self.stop = Event()
        self.error: BaseException | None = None
        self.finished = False
        self.capture = (
            cv2.VideoCapture(source)
            if not self.network
            else cv2.VideoCapture(
                source,
                cv2.CAP_ANY,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
        )
        try:
            if not self.capture.isOpened():
                raise ValueError("Could not open the video source.")
            success, frame = self.capture.read()
            if not success:
                raise ValueError("Could not read the first frame from the live video source.")
            frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.finite = finite or (
                isinstance(frame_count, (int, float)) and isfinite(frame_count) and frame_count > 0
            )
            self.first: np.ndarray[Any, Any] | None = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except BaseException:
            self.capture.release()
            raise
        self.thread = Thread(target=self._run, name="rfdetr-live-capture", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.capture.release()
            raise

    def _run(self) -> None:
        """Capture frames until closed, reconnecting after transient read failures."""
        import cv2  # Loaded at the optional dependency boundary in __init__.

        failures = 0
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stop.is_set() or len(self.frames) < 30)
                if self.stop.is_set():
                    break
                success = False
                frame = None
                for _ in range(self.vid_stride):
                    success, frame = self.capture.read()
                    if not success or self.stop.is_set():
                        break
                if self.stop.is_set():
                    break
                if not success or frame is None:
                    if self.finite:
                        if self.network:
                            logger.warning(
                                "Finite network stream stopped at end of file or a read failure. "
                                "OpenCV cannot distinguish EOF from a timeout; results may be incomplete."
                            )
                        return
                    failures += 1
                    if failures > 3:
                        raise RuntimeError("Live video source failed after three reconnect attempts.")
                    if self.stop.wait(0.1):
                        break
                    if not self.network:
                        self.capture.open(self.source)
                    else:
                        self.capture.open(
                            self.source,
                            cv2.CAP_ANY,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
                        )
                    continue
                failures = 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.condition:
                    if not self.stream_buffer:
                        self.frames.clear()
                    self.frames.append(rgb)
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self) -> np.ndarray[Any, Any] | None:
        """Return the initial frame, then the next available queued frame."""
        if self.first is not None:
            first, self.first = self.first, None
            return first
        with self.condition:
            self.condition.wait_for(
                lambda: bool(self.frames) or self.error is not None or self.finished or self.stop.is_set()
            )
            if self.frames:
                frame = self.frames.popleft()
                self.condition.notify_all()
                return frame
            if self.error is not None:
                raise self.error
            return None


def iter_live_frames(
    sources: list[str | int],
    *,
    vid_stride: int = 1,
    stream_buffer: bool = False,
    finite_sources: frozenset[str | int] = frozenset(),
) -> Generator[list[np.ndarray[Any, Any]], None, None]:
    """Yield RGB batches in source order with bounded background capture.

    Each source contributes one frame per batch. Buffered mode retains up to 30
    pending frames per source; the default keeps only the latest pending frame.
    Closing the iterator stops every worker. A backend blocked in a native read
    releases its capture when that read returns. Network backends must support
    OpenCV open/read timeouts. Known finite sources or sources with a known frame count stop at EOF;
    the group stops when its shortest source ends. Unknown-length sources
    treat failed reads as disconnects and retry up to three times.

    Args:
        sources: Camera indexes or live stream URLs.
        vid_stride: Number of captured frames between predictions.
        stream_buffer: Retain pending frames instead of replacing them.
        finite_sources: Sources known to be finite even without frame-count metadata.

    Yields:
        One RGB image per source.
    """
    if not sources:
        raise ValueError("At least one live video source is required.")
    captures: list[_LiveCapture] = []
    try:
        for source in sources:
            captures.append(_LiveCapture(source, vid_stride, stream_buffer, finite=source in finite_sources))
        while True:
            batch = []
            for capture in captures:
                frame = capture.read()
                if frame is None:
                    return
                batch.append(frame)
            yield batch
    finally:
        deadline = monotonic() + 5
        for capture in captures:
            capture.stop.set()
            with capture.condition:
                capture.condition.notify_all()
        for capture in captures:
            capture.thread.join(timeout=max(0, deadline - monotonic()))
