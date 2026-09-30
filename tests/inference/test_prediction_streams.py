# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Live capture behavior through prediction with mocked device boundaries."""

from pathlib import Path
from threading import Event
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from .helpers import _DummyRFDETR


class TestLivePredictions:
    """Exercise live sources through the public prediction API."""

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
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", side_effect=[first, second]) as open_capture:
            results = model.predict(source, stream=True, stream_buffer=True)
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

    def test_buffered_frames_preserve_order_before_error(self) -> None:
        """Buffered mode delivers all captured frames before propagating failure."""
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3)] + [
            RuntimeError("decoder failure")
        ]
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True, stream_buffer=True)
            assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(3)] == [1, 2, 3]
            with pytest.raises(RuntimeError, match="decoder failure"):
                next(results)
        capture.release.assert_called_once()

    def test_latest_frame_drops_pending_frames(self) -> None:
        """A slow consumer receives the latest pending frame after the first frame."""
        finished = Event()
        reads = iter([(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3)] + [(False, None)])
        capture = MagicMock()
        capture.read.side_effect = lambda: next(reads)
        capture.open.side_effect = lambda *args: finished.set()
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True)
            try:
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 1
                assert finished.wait(2)
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 3
            finally:
                results.close()
        capture.release.assert_called_once()

    def test_stride_skips_live_frames(self) -> None:
        """Stride applies after the initial frame without changing source colors."""
        capture = MagicMock()
        capture.read.side_effect = [
            (True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2, 3, 4, 5)
        ] + [RuntimeError("end")]
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True, stream_buffer=True, vid_stride=2)
            try:
                assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(3)] == [
                    1,
                    3,
                    5,
                ]
            finally:
                results.close()

    def test_reconnect_after_failed_read(self) -> None:
        """A transient failed read reconnects and returns the next valid frame."""
        capture = MagicMock()
        capture.read.side_effect = [
            (True, np.full((24, 32, 3), 1, dtype=np.uint8)),
            (False, None),
            (True, np.full((24, 32, 3), 2, dtype=np.uint8)),
            RuntimeError("end"),
        ]
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True, stream_buffer=True)
            try:
                assert [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(2)] == [
                    1,
                    2,
                ]
                capture.open.assert_called_once_with(0)
            finally:
                results.close()

    def test_repeated_failed_reads_stop(self) -> None:
        """Permanent disconnection raises after a bounded number of reconnects."""
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))] + [(False, None)] * 4
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True)
            next(results)
            with pytest.raises(RuntimeError, match="three reconnect attempts"):
                next(results)
        assert capture.open.call_count == 3
        capture.release.assert_called_once()

    def test_partial_initialization_releases_open_capture(self, tmp_path: Path) -> None:
        """A failed second camera closes the already running first camera."""
        source = tmp_path / "cameras.streams"
        source.write_text("0\n1\n")
        first = MagicMock()
        second = MagicMock()
        first.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        second.isOpened.return_value = False
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", side_effect=[first, second]):
            with pytest.raises(ValueError, match="Could not open the video source"):
                next(_DummyRFDETR().predict(source, stream=True))
        first.release.assert_called_once()
        second.release.assert_called_once()

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
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True, stream_buffer=True)
            try:
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 0
                assert full.wait(2)
                assert not resumed.wait(0.05)
                assert cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] == 1
                assert resumed.wait(2)
                assert [
                    cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(30)
                ] == list(range(2, 32))
            finally:
                results.close()

    def test_finite_streams_stop_at_shortest_source(self, tmp_path: Path) -> None:
        """Known finite streams stop together without reconnecting or replaying."""
        source = tmp_path / "videos.streams"
        source.write_text("https://example.com/a.mp4\nhttps://example.com/b.mp4\n")
        first = MagicMock()
        second = MagicMock()
        first.get.return_value = 2
        second.get.return_value = 3
        first.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (1, 2)]
        second.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)) for value in (3, 4, 5)]
        with patch("rfdetr._prediction_streams.cv2.VideoCapture", side_effect=[first, second]):
            results = list(_DummyRFDETR().predict(source, stream=True, stream_buffer=True))
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 3, 2, 4]
        first.open.assert_not_called()
        second.open.assert_not_called()
        first.release.assert_called_once()
        second.release.assert_called_once()
