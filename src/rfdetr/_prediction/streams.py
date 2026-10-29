# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bounded background capture for live prediction sources."""

from collections import deque
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from math import isfinite
from threading import Condition, Event, Thread
from time import monotonic
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

import numpy as np

from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Pending frames retained per source in buffered mode.
_BUFFER_FRAMES = 30
#: OpenCV open and read timeout, in milliseconds, applied to every network stream.
_NETWORK_TIMEOUT_MSEC = 5000
#: URL schemes that always name a live stream. Source classification and capture timeouts both read this set.
LIVE_STREAM_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "tcp", "udp", "srt"})
#: URL schemes opened with OpenCV open and read timeouts: live streams plus HTTP videos.
_NETWORK_SCHEMES = LIVE_STREAM_SCHEMES | {"http", "https"}
#: Failed reads of an unknown-length source retried before giving up.
#: The failure message and the ``iter_live_frames`` docstring spell this number out.
_MAX_RECONNECTS = 3
#: Seconds to wait before reopening a source after a failed read.
_RECONNECT_DELAY = 0.1
#: Seconds allowed for all worker threads to exit once the frame iterator closes.
_CLOSE_TIMEOUT = 5


def _import_cv2(task: str) -> ModuleType:
    """Import OpenCV, or explain how to install it.

    Args:
        task: Prediction kind that needs OpenCV, used to start the error message.

    Returns:
        The imported ``cv2`` module.

    Raises:
        ImportError: If OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            f'{task} prediction requires OpenCV. Install it with `uv pip install "rfdetr[stream]"`.'
        ) from error
    return cv2


def _is_network_source(source: str | int) -> bool:
    """Tell whether OpenCV opens the source as a network stream.

    Args:
        source: Camera index, file path or URL. Only a string with a live-stream or HTTP(S) scheme counts.

    Returns:
        True for a network source, False for a camera index or a local path.

    Examples:
        >>> _is_network_source("rtsp://camera.local/live")
        True
        >>> _is_network_source("https://example.com/video.mp4")
        True
        >>> _is_network_source("video.mp4")
        False
        >>> _is_network_source(0)
        False
    """
    return isinstance(source, str) and urlparse(source).scheme.lower() in _NETWORK_SCHEMES


def _capture_args(source: str | int) -> tuple[Any, ...]:
    """Build the positional arguments for ``cv2.VideoCapture`` and ``VideoCapture.open``.

    Network streams receive open and read timeouts so a dead host cannot block forever;
    cameras and files take the source alone. The caller must have imported OpenCV through
    ``_import_cv2`` first.

    Args:
        source: Camera index, file path or URL to open.

    Returns:
        ``(source,)`` for cameras and files, or ``(source, cv2.CAP_ANY, timeouts)`` for a network stream, where
        ``timeouts`` is the OpenCV open and read timeout parameter list.

    Examples:
        >>> _capture_args(0)
        (0,)
        >>> _capture_args("clip.mp4")
        ('clip.mp4',)
    """
    if not _is_network_source(source):
        return (source,)
    import cv2  # Imported by the caller through _import_cv2; this reuses the loaded module.

    timeouts = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        _NETWORK_TIMEOUT_MSEC,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        _NETWORK_TIMEOUT_MSEC,
    ]
    return (source, cv2.CAP_ANY, timeouts)


class _LiveCapture:
    """Own one OpenCV capture, its worker thread and its bounded frame queue.

    The initial frame is read on the constructing thread so open and read failures surface immediately. A daemon worker
    then keeps reading, converting frames to RGB and queueing them. Setting ``stop`` ends the worker, which releases
    the capture on exit.

    Args:
        source: Camera index, file path or stream URL.
        vid_stride: Number of captured frames between kept frames.
        stream_buffer: Queue up to ``_BUFFER_FRAMES`` pending frames instead of keeping only the latest.
        finite: Treat the source as ending, even without frame-count metadata.

    Attributes:
        finite (bool): The source ends at end of file instead of reconnecting after a failed read.
        stream_buffer (bool): Pending frames are queued; always True for a finite source.
        error (BaseException | None): Exception that stopped the worker, re-raised by :meth:`read`.
        finished (bool): The worker has exited.
    """

    def __init__(self, source: str | int, vid_stride: int, stream_buffer: bool, *, finite: bool = False) -> None:
        """Open the capture, read an initial frame, and start the background worker.

        Args:
            source: Camera index, file path or stream URL.
            vid_stride: Number of captured frames between kept frames.
            stream_buffer: Queue pending frames instead of keeping only the latest.
            finite: Treat the source as ending, even without frame-count metadata.

        Raises:
            ImportError: If OpenCV is not installed.
            ValueError: If the source cannot be opened or its first frame cannot be read.
        """
        cv2 = _import_cv2("Live")

        self.source = source
        self.network = _is_network_source(source)
        self.vid_stride = vid_stride
        self.frames: deque[np.ndarray[Any, Any]] = deque()
        self.condition = Condition()
        self.stop = Event()
        self.error: BaseException | None = None
        self.finished = False
        self.capture = cv2.VideoCapture(*_capture_args(source))
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
            self.stream_buffer = stream_buffer or self.finite
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
        """Capture frames on the worker thread until closed, reconnecting after transient read failures.

        The loop waits while the queue is full, skips ``vid_stride - 1`` frames, reads one, and queues its RGB copy
        (replacing older frames unless buffered). A finite source stops at the first failed read. An unknown-length
        source reopens the capture after a short delay and raises ``RuntimeError`` after more than ``_MAX_RECONNECTS``
        consecutive failures. Any exception is stored in ``error`` for :meth:`read`; the capture is always released and
        ``finished`` set on exit.
        """
        import cv2  # Loaded at the optional dependency boundary in __init__.

        failures = 0
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stop.is_set() or len(self.frames) < _BUFFER_FRAMES)
                if self.stop.is_set():
                    break
                # grab() advances past skipped frames without decoding them; only the kept frame is read.
                success = True
                frame = None
                for _ in range(self.vid_stride - 1):
                    success = self.capture.grab()
                    if not success or self.stop.is_set():
                        break
                if success and not self.stop.is_set():
                    success, frame = self.capture.read()
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
                    if failures > _MAX_RECONNECTS:
                        raise RuntimeError("Live video source failed after three reconnect attempts.")
                    if self.stop.wait(_RECONNECT_DELAY):
                        break
                    self.capture.open(*_capture_args(self.source))
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
        """Return the initial frame, then the next available queued frame, blocking until one exists.

        Returns:
            The next RGB frame in HWC layout, or None once the worker has finished (or been stopped) and the queue is
            empty.

        Raises:
            BaseException: The error that stopped the worker, once every queued frame has been read.
        """
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


def _open_captures(
    sources: list[str | int], *, vid_stride: int, stream_buffer: bool, finite_sources: frozenset[str | int]
) -> list[_LiveCapture]:
    """Open one capture per source, overlapping the slow network opens.

    Network streams can block for the full open timeout plus the first read, so with several
    sources they open on worker threads while cameras and files open one by one on the calling
    thread (camera backends may require it). The result is all or nothing: when any source
    fails, every open already in flight finishes, every capture that did open is closed, and the
    first error is raised.

    Args:
        sources: Camera indexes, files or stream URLs, in batch order.
        vid_stride: Number of captured frames between predictions.
        stream_buffer: Retain pending frames instead of replacing them.
        finite_sources: Sources known to be finite even without frame-count metadata.

    Returns:
        One running capture per source, in source order.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If a source cannot be opened or its first frame cannot be read.
    """

    def open_one(source: str | int) -> _LiveCapture:
        """Open one capture and read its first frame.

        Args:
            source: Camera index, file path or stream URL.

        Returns:
            A running capture, finite when the source is in ``finite_sources``.
        """
        return _LiveCapture(source, vid_stride, stream_buffer, finite=source in finite_sources)

    network = [index for index, source in enumerate(sources) if len(sources) > 1 and _is_network_source(source)]
    opened: dict[int, _LiveCapture] = {}
    futures: dict[int, Future[_LiveCapture]] = {}
    error: BaseException | None = None
    try:
        # Leaving the pool context waits for every in-flight open, success or failure.
        with ThreadPoolExecutor(max_workers=max(1, len(network)), thread_name_prefix="rfdetr-live-open") as pool:
            for index in network:
                futures[index] = pool.submit(open_one, sources[index])
            for index, source in enumerate(sources):
                if index not in futures:
                    opened[index] = open_one(source)
    except BaseException as caught:
        error = caught
    for index, future in futures.items():
        failure = future.exception()
        if failure is None:
            opened[index] = future.result()
        elif error is None:
            error = failure
    if error is not None:
        _close_captures(list(opened.values()))
        raise error
    return [opened[index] for index in range(len(sources))]


def _close_captures(captures: list[_LiveCapture]) -> None:
    """Stop every capture worker and wait a bounded time for each to release its device.

    All workers are signalled first, then joined against one shared deadline of ``_CLOSE_TIMEOUT`` seconds. A worker
    blocked in a native read is not forced to exit; it releases its capture when that read returns.

    Args:
        captures: Captures to stop. An empty list does nothing.
    """
    deadline = monotonic() + _CLOSE_TIMEOUT
    for capture in captures:
        capture.stop.set()
        with capture.condition:
            capture.condition.notify_all()
    for capture in captures:
        capture.thread.join(timeout=max(0, deadline - monotonic()))


def iter_live_frames(
    sources: list[str | int],
    *,
    vid_stride: int = 1,
    stream_buffer: bool = False,
    finite_sources: frozenset[str | int] = frozenset(),
) -> Generator[list[np.ndarray[Any, Any]], None, None]:
    """Yield RGB batches in source order with bounded background capture.

    Each source contributes one frame per batch. Buffered mode retains up to 30
    pending frames per source; live sources default to the latest pending frame.
    Finite sources always use buffered mode to preserve selected frames.
    Network sources open concurrently; cameras and files open in order on the
    calling thread, and any failure releases every capture that did open.
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
        captures = _open_captures(
            sources, vid_stride=vid_stride, stream_buffer=stream_buffer, finite_sources=finite_sources
        )
        while True:
            batch = []
            for capture in captures:
                frame = capture.read()
                if frame is None:
                    return
                batch.append(frame)
            yield batch
    finally:
        _close_captures(captures)
