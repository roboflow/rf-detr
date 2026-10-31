# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Live capture behavior through prediction with mocked device boundaries."""

import logging
import threading
from collections.abc import Callable
from contextlib import closing
from itertools import islice
from pathlib import Path
from threading import Event
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from rfdetr.inference import RFDETRInference

from ..helpers import _DummyRFDETR, _requires_opencv, _scripted_capture

#: Name every background capture thread carries, so tests can tell leaked workers apart.
_WORKER_NAME = "rfdetr-live-capture"


def _capture_workers() -> set[threading.Thread]:
    """Return the background capture threads that are alive right now.

    Examples:
        >>> isinstance(_capture_workers(), set)
        True
    """
    return {thread for thread in threading.enumerate() if thread.name == _WORKER_NAME}


def _capture_by_source(captures: dict[str, MagicMock]) -> Callable[..., MagicMock]:
    """Return a ``cv2.VideoCapture`` stand-in that hands out each capture by its source.

    Network sources open concurrently, so a fake cannot rely on the order of the calls.

    Examples:
        >>> first = MagicMock()
        >>> _capture_by_source({"rtsp://a": first})("rtsp://a", 0, []) is first
        True
    """
    return lambda source, *args: captures[source]


def _opened_on(threads: dict[str, int], name: str) -> Callable[[], bool]:
    """Return an ``isOpened`` stand-in that records the thread which checked it.

    The capture constructor calls ``isOpened`` on the thread that opens the source, so the
    recorded thread identifies where the open ran.

    Examples:
        >>> threads: dict[str, int] = {}
        >>> _opened_on(threads, "camera")()
        True
        >>> threads["camera"] == threading.get_ident()
        True
    """

    def is_opened() -> bool:
        """Record the calling thread and report the capture as open."""
        threads[name] = threading.get_ident()
        return True

    return is_opened


class TestLivePredictions:
    """Exercise live sources through the public prediction API."""

    @_requires_opencv
    @pytest.mark.parametrize("vid_stride", [1, 2])
    def test_finite_streams_preserve_frames_for_slow_consumer(self, tmp_path: Path, vid_stride: int) -> None:
        """Finite videos retain selected frames even when latest-frame mode is requested."""
        source = tmp_path / "videos.streams"
        source.write_text("https://example.com/a.mp4\nhttps://example.com/b.mp4\n")
        urls = ("https://example.com/a.mp4", "https://example.com/b.mp4")
        captures = {
            url: _scripted_capture(
                [(True, np.full((24, 32, 3), value + offset, dtype=np.uint8)) for value in range(1, 6)]
            )
            for url, offset in zip(urls, (0, 10))
        }
        finished = [Event(), Event()]
        for capture, event in zip(captures.values(), finished):
            capture.get.return_value = 0
            capture.release.side_effect = event.set

        with patch("cv2.VideoCapture", side_effect=_capture_by_source(captures)):
            results = RFDETRInference(_DummyRFDETR()).predict(
                source, stream=True, stream_buffer=False, vid_stride=vid_stride
            )
            try:
                first = next(results)
                assert all(event.wait(30) for event in finished)
                predictions = [first, *results]
            finally:
                results.close()

        expected = [1, 11, 2, 12, 3, 13, 4, 14, 5, 15] if vid_stride == 1 else [1, 11, 3, 13, 5, 15]
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in predictions] == expected

    @_requires_opencv
    @pytest.mark.parametrize("entry", ["video.mp4", "https://example.com/video.mp4"])
    def test_manifest_video_without_frame_count_stops_at_eof(self, tmp_path: Path, entry: str) -> None:
        """Known video files end without reconnecting when frame-count metadata is absent."""
        (tmp_path / "video.mp4").touch()
        source = tmp_path / "video.streams"
        source.write_text(entry)
        capture = MagicMock()
        capture.get.return_value = 0
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))] + [(False, None)] * 4
        with patch("cv2.VideoCapture", return_value=capture):
            assert len(list(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True))) == 1
        capture.open.assert_not_called()
        capture.release.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://camera.example/live",
            "rtmp://camera.example/live",
            "tcp://camera.example/live",
            "udp://@239.0.0.1:1234",
            "srt://camera.example:9000?mode=caller",
        ],
    )
    def test_live_stream_urls_open_with_capture_timeouts(self, url: str) -> None:
        """Every live stream scheme opens with bounded OpenCV open and read timeouts.

        A UDP or SRT source is a network stream like RTSP, so it must not fall through to the unbounded local-file open
        that a camera index or video path uses; a silent sender would otherwise block the first ``next()`` forever.
        """
        import cv2

        capture = MagicMock()
        capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = RFDETRInference(_DummyRFDETR()).predict(url, stream=True)
            try:
                next(results)
            finally:
                results.close()
        open_capture.assert_called_once_with(
            url,
            cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )

    @_requires_opencv
    def test_finite_network_read_failure_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remote file ending on a failed read must not silently hide a timeout."""
        # The package logger does not propagate by default, so caplog only sees it once propagation is enabled.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        source = tmp_path / "video.streams"
        source.write_text("https://example.com/video.mp4")
        capture = MagicMock()
        capture.get.return_value = 100
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch("cv2.VideoCapture", return_value=capture):
            assert len(list(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True))) == 1
        assert "end of file or a read failure" in caplog.text
        capture.open.assert_not_called()

    @_requires_opencv
    def test_lazy_rgb_batches_and_close(self, tmp_path: Path) -> None:
        """Open lazily, keep source order, convert colors, and release all captures."""
        source = tmp_path / "cameras.streams"
        source.write_text("0\n1\n")
        first = MagicMock()
        second = MagicMock()
        first.read.return_value = (True, np.full((24, 32, 3), [1, 2, 3], dtype=np.uint8))
        second.read.return_value = (True, np.full((24, 32, 3), [4, 5, 6], dtype=np.uint8))
        model = _DummyRFDETR()
        batch_sizes = []
        assert model.model.model is not None
        handle = model.model.model.register_forward_pre_hook(lambda module, args: batch_sizes.append(len(args[0])))
        with patch("cv2.VideoCapture", side_effect=[first, second]) as open_capture:
            results = RFDETRInference(model).predict(source, stream=True, stream_buffer=True)
            open_capture.assert_not_called()
            try:
                pixels = [cast(sv.Detections, next(results)).metadata["source_image"][0, 0].tolist() for _ in range(2)]
                assert pixels == [[3, 2, 1], [6, 5, 4]]
                assert batch_sizes == [2]
            finally:
                results.close()
        handle.remove()
        first.release.assert_called_once()
        second.release.assert_called_once()

    @_requires_opencv
    def test_buffered_frames_preserve_order_before_error(self) -> None:
        """Buffered mode delivers all captured frames before propagating failure."""
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3)] + [
            RuntimeError("decoder failure")
        ]
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True, stream_buffer=True)
            assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(3)] == [1, 2, 3]
            with pytest.raises(RuntimeError, match="decoder failure"):
                next(results)
        capture.release.assert_called_once()

    @_requires_opencv
    def test_latest_frame_drops_pending_frames(self) -> None:
        """A slow consumer receives the latest pending frame after the first frame."""
        finished = Event()
        reads = iter([(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3)] + [(False, None)])
        capture = MagicMock()
        capture.read.side_effect = lambda: next(reads)
        capture.open.side_effect = lambda *args: finished.set()
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True)
            try:
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 1
                assert finished.wait(30)
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 3
            finally:
                results.close()
        capture.release.assert_called_once()

    @_requires_opencv
    def test_stride_skips_live_frames(self) -> None:
        """Stride applies after the initial frame, skips frames without decoding them, and keeps source colors.

        Skipped frames advance the capture through ``grab()``, so only the three kept frames are decoded.
        """
        capture = _scripted_capture(
            [*((True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3, 4, 5)), RuntimeError("end")]
        )
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True, stream_buffer=True, vid_stride=2)
            try:
                assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(3)] == [
                    1,
                    3,
                    5,
                ]
            finally:
                results.close()
            assert capture.read.call_count == 3

    @_requires_opencv
    def test_reconnect_after_failed_read(self) -> None:
        """A transient failed read reconnects and returns the next valid frame."""
        capture = MagicMock()
        capture.read.side_effect = [
            (True, np.full((24, 32, 3), 1, dtype=np.uint8)),
            (False, None),
            (True, np.full((24, 32, 3), 2, dtype=np.uint8)),
            RuntimeError("end"),
        ]
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True, stream_buffer=True)
            try:
                assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(2)] == [
                    1,
                    2,
                ]
                capture.open.assert_called_once_with(0)
            finally:
                results.close()

    @_requires_opencv
    @pytest.mark.parametrize("url", ["rtsp://camera.example/live", "http://camera.example/feed.m3u8"])
    def test_network_reconnect_reopens_with_timeouts(self, url: str) -> None:
        """A network stream reconnects with the same open and read timeouts it was first opened with.

        A transient failed read on a camera feed must reopen through ``VideoCapture.open`` with the timeout arguments; a
        bare ``open(url)`` would let a dead host block the worker indefinitely.
        """
        import cv2

        timeouts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        capture = MagicMock()
        capture.read.side_effect = [
            (True, np.full((24, 32, 3), 1, dtype=np.uint8)),
            (False, None),
            (True, np.full((24, 32, 3), 2, dtype=np.uint8)),
            RuntimeError("end"),
        ]
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            with closing(RFDETRInference(_DummyRFDETR()).predict(url, stream=True, stream_buffer=True)) as results:
                pixels = [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(2)]
        assert pixels == [1, 2]
        open_capture.assert_called_once_with(url, cv2.CAP_ANY, timeouts)
        capture.open.assert_called_once_with(url, cv2.CAP_ANY, timeouts)

    @_requires_opencv
    def test_later_source_failure_releases_every_capture_and_thread(self, tmp_path: Path) -> None:
        """A later source failing mid-stream raises its error and leaves no capture or worker alive.

        Two cameras start and deliver a first batch; the second decoder then fails while the first keeps producing. The
        error must reach the caller, both captures must be released, and neither background thread may outlive the
        iterator.
        """
        source = tmp_path / "cameras.streams"
        source.write_text("0\n1\n")
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        first = MagicMock()
        second = MagicMock()
        first.read.return_value = (True, frame)
        second.read.side_effect = [(True, frame), RuntimeError("decoder failure")]
        workers_before = _capture_workers()
        with patch("cv2.VideoCapture", side_effect=[first, second]):
            results = RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True)
            assert len(list(islice(results, 2))) == 2  # one result per source in the first batch
            with pytest.raises(RuntimeError, match="decoder failure"):
                next(results)
        first.release.assert_called_once()
        second.release.assert_called_once()
        assert _capture_workers() == workers_before

    @_requires_opencv
    def test_network_sources_open_concurrently(self, tmp_path: Path) -> None:
        """Slow network opens overlap instead of adding up before the first batch.

        Each fake open blocks until the other source is opening too; with serial opens the first would wait out the
        barrier timeout and fail, as a dead host does in the real world.
        """
        urls = ["rtsp://camera.example/a", "rtsp://camera.example/b"]
        source = tmp_path / "cameras.streams"
        source.write_text("\n".join(urls))
        barrier = threading.Barrier(2, timeout=2)
        captures = {url: MagicMock() for url in urls}
        for value, capture in enumerate(captures.values(), start=1):
            capture.read.return_value = (True, np.full((24, 32, 3), value, dtype=np.uint8))
            capture.isOpened.side_effect = lambda: barrier.wait() is not None
        with patch("cv2.VideoCapture", side_effect=_capture_by_source(captures)):
            with closing(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True)) as results:
                pixels = [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(2)]
        assert pixels == [1, 2]

    @_requires_opencv
    def test_cameras_open_on_the_caller_thread(self, tmp_path: Path) -> None:
        """Cameras open in order on the caller's thread while network streams open elsewhere.

        Camera backends can require the thread that started them, so only network sources move to worker threads.
        """
        url = "rtsp://camera.example/a"
        source = tmp_path / "mixed.streams"
        source.write_text(f"0\n{url}\n")
        opened_on: dict[str, int] = {}
        captures = {url: MagicMock()}
        camera = MagicMock()
        for name, capture in (("camera", camera), ("network", captures[url])):
            capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
            capture.isOpened.side_effect = _opened_on(opened_on, name)
        with patch("cv2.VideoCapture", side_effect=lambda source, *args: camera if source == 0 else captures[source]):
            with closing(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True)) as results:
                assert len(list(islice(results, 2))) == 2
        assert opened_on["camera"] == threading.get_ident()
        assert opened_on["network"] != threading.get_ident()

    @_requires_opencv
    def test_failed_open_waits_for_in_flight_opens_and_releases_every_capture(self, tmp_path: Path) -> None:
        """A source that cannot open releases the sibling capture that was still opening.

        The healthy stream finishes opening only after its sibling has failed, so it is already in flight when the error
        arises. It must still be closed instead of leaking its device and worker thread, and the sibling's error must
        reach the caller.
        """
        slow_url, bad_url = "rtsp://camera.example/slow", "rtsp://camera.example/bad"
        source = tmp_path / "cameras.streams"
        source.write_text(f"{slow_url}\n{bad_url}\n")
        bad_opened = Event()
        slow = MagicMock()
        bad = MagicMock()
        slow.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        slow.isOpened.side_effect = lambda: bad_opened.wait(2)
        bad.isOpened.side_effect = lambda: bad_opened.set() or False
        workers_before = _capture_workers()
        with patch("cv2.VideoCapture", side_effect=_capture_by_source({slow_url: slow, bad_url: bad})):
            with pytest.raises(ValueError, match="Could not open the video source"):
                next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))
        slow.release.assert_called_once()
        bad.release.assert_called_once()
        assert _capture_workers() == workers_before

    @_requires_opencv
    def test_repeated_failed_reads_stop(self) -> None:
        """Permanent disconnection raises after a bounded number of reconnects."""
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))] + [(False, None)] * 4
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True)
            next(results)
            with pytest.raises(RuntimeError, match="three reconnect attempts"):
                next(results)
        assert capture.open.call_count == 3
        capture.release.assert_called_once()

    @_requires_opencv
    def test_unreadable_first_frame_fails_and_releases_capture(self) -> None:
        """A source that opens but yields no first frame raises a clear error and closes the device."""
        capture = MagicMock()
        capture.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True)
            with pytest.raises(ValueError, match="Could not read the first frame"):
                next(results)
        capture.release.assert_called_once()

    @_requires_opencv
    def test_partial_initialization_releases_open_capture(self, tmp_path: Path) -> None:
        """A failed second camera closes the already running first camera."""
        source = tmp_path / "cameras.streams"
        source.write_text("0\n1\n")
        first = MagicMock()
        second = MagicMock()
        first.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        second.isOpened.return_value = False
        with patch("cv2.VideoCapture", side_effect=[first, second]):
            with pytest.raises(ValueError, match="Could not open the video source"):
                next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))
        first.release.assert_called_once()
        second.release.assert_called_once()

    @_requires_opencv
    def test_buffered_capture_applies_backpressure(self) -> None:
        """A full queue pauses capture and resumes without discarding frames."""
        full = Event()
        resumed = Event()
        frames = iter((True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in range(100))
        capture = MagicMock()

        def read_frame() -> tuple[bool, np.ndarray[Any, Any]]:
            """Signal the queue boundary for this deterministic fake camera.

            Examples:
                >>> read_frame()  # doctest: +SKIP
                # Requires the enclosing test's events and capture mock.
            """
            if capture.read.call_count == 31:
                full.set()
            if capture.read.call_count == 32:
                resumed.set()
            return next(frames)

        capture.read.side_effect = read_frame
        with patch("cv2.VideoCapture", return_value=capture):
            results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True, stream_buffer=True)
            try:
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 0
                assert full.wait(30)
                assert not resumed.wait(0.05)
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 1
                assert resumed.wait(30)
                assert [
                    cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(30)
                ] == list(range(2, 32))
            finally:
                results.close()

    @_requires_opencv
    def test_finite_streams_stop_at_shortest_source(self, tmp_path: Path) -> None:
        """Known finite streams stop together without reconnecting or replaying."""
        source = tmp_path / "videos.streams"
        source.write_text("https://example.com/a.mp4\nhttps://example.com/b.mp4\n")
        first = MagicMock()
        second = MagicMock()
        first.get.return_value = 2
        second.get.return_value = 3
        first.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2)] + [
            (False, None)
        ]
        second.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (3, 4, 5)] + [
            (False, None)
        ]
        captures = {"https://example.com/a.mp4": first, "https://example.com/b.mp4": second}
        with patch("cv2.VideoCapture", side_effect=_capture_by_source(captures)):
            results = list(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True))
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 3, 2, 4]
        first.open.assert_not_called()
        second.open.assert_not_called()
        first.release.assert_called_once()
        second.release.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize("frame_count", [1, 10])
    def test_finite_stream_uses_eof_instead_of_frame_count(self, tmp_path: Path, frame_count: int) -> None:
        """An inaccurate frame estimate neither truncates nor replays a finite stream."""
        source = tmp_path / "video.streams"
        source.write_text("https://example.com/video.mp4\n")
        capture = MagicMock()
        capture.get.return_value = frame_count
        capture.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3)] + [
            (False, None)
        ]
        with patch("cv2.VideoCapture", return_value=capture):
            results = list(RFDETRInference(_DummyRFDETR()).predict(source, stream=True, stream_buffer=True))
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 2, 3]
        capture.open.assert_not_called()
        capture.release.assert_called_once()

    def test_live_capture_reports_missing_opencv(self) -> None:
        """Image-only installations receive an actionable optional-dependency error."""
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(
                ImportError,
                match=r'requires OpenCV\. Install it with `uv pip install "rfdetr\[stream\]"` '
                r'or `pip install "rfdetr\[headless\]"`\.',
            ):
                next(RFDETRInference(_DummyRFDETR()).predict(0, stream=True))
