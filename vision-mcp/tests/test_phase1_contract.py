"""Fast checks for Phase 1 failures that are not reliable to spot in a live demo."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_mcp.analytics.metrics import percentile, safe_rate
from vision_mcp.api_contract import ImageSize, TimeWindow
from vision_mcp.config import EngineConfig, ModelEntry, SecurityConfig, StreamEntry
from vision_mcp.engine.queries import _dwell
from vision_mcp.engine.queries import HistoricalQueries
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.mcp_server.server import _artifact_id
from vision_mcp.query import TimeQuery, build_time_query
from vision_mcp.security import redact_data, resolve_within, validate_stream_url, validate_url
from vision_mcp.storage.database import Database
from vision_mcp.streams.capture import CaptureThread
from vision_mcp.streams.frames import Frame, LatestFrameQueue


def test_zero_denominator_rate_is_zero() -> None:
    assert safe_rate(0, 0) == 0.0
    assert safe_rate(7, 0) == 0.0


def test_latency_and_dwell_percentiles() -> None:
    ordered = [10.0, 20.0, 30.0, 40.0]
    assert percentile(ordered, 50) == 20.0
    assert percentile(ordered, 95) == 40.0
    dwell = _dwell("lobby", ordered)
    assert dwell.mean_seconds == 25.0
    assert dwell.p50_seconds == 20.0
    assert dwell.p95_seconds == 40.0
    assert dwell.max_seconds == 40.0


def test_window_interval_bucket_limit_rejected() -> None:
    with pytest.raises(VisionError) as caught:
        build_time_query("7d", "1s", now=1_000_000)
    assert caught.value.code == ErrorCode.INVALID_TIME_WINDOW
    assert caught.value.details["bucket_count"] == 604_800


def test_artifact_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(VisionError) as caught:
        resolve_within(tmp_path, "../outside.jpg")
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


def test_url_allowlist_rejection_and_rtsp_stream_acceptance() -> None:
    with pytest.raises(VisionError) as caught:
        validate_url("https://evil.example/image.jpg", ["images.example"], True)
    assert caught.value.code == ErrorCode.URL_NOT_ALLOWED
    assert (
        validate_stream_url("rtsp://camera.example/live", ["camera.example"], True)
        == "rtsp://camera.example/live"
    )
    with pytest.raises(VisionError):
        validate_url("rtsp://camera.example/live", ["camera.example"], True)


def test_credentials_and_rtsp_password_are_recursively_redacted() -> None:
    value = {
        "nested": [
            "rtsp://alice:hunter2@camera.example/live",
            {"password": "password=secret-value", "token": "token: abc123"},
        ]
    }
    rendered = str(redact_data(value))
    assert "hunter2" not in rendered
    assert "secret-value" not in rendered
    assert "abc123" not in rendered
    assert rendered.count("***REDACTED***") == 3


@pytest.mark.asyncio
async def test_bounded_queue_drops_oldest_and_increments_counter() -> None:
    queue = LatestFrameQueue(2, asyncio.get_running_loop())
    for index in (1, 2, 3):
        queue.push(
            Frame(
                array=np.zeros((1, 1, 3), dtype=np.uint8),
                size=ImageSize(width=1, height=1),
                index=index,
                captured_at=float(index),
                captured_monotonic=float(index),
            )
        )
    assert queue.depth == 2
    assert queue.high_water == 2
    assert queue.dropped == 1
    first = await queue.get()
    second = await queue.get()
    assert first is not None and first.index == 2
    assert second is not None and second.index == 3


def test_webcam_pacing_does_not_use_backend_grab(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.frames: list[Frame] = []

        def push(self, frame: Frame) -> None:
            self.frames.append(frame)

    class FakeStop:
        def __init__(self) -> None:
            self.stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def set(self) -> None:
            self.stopped = True

        def wait(self, timeout: float) -> bool:
            return self.stopped

    class FakeCapture:
        def __init__(self, stop: FakeStop) -> None:
            self.stop = stop
            self.reads = 0
            self.grabs = 0

        def read(self) -> tuple[bool, np.ndarray]:
            self.reads += 1
            if self.reads == 2:
                self.stop.set()
            return True, np.ones((2, 2, 3), dtype=np.uint8)

        def grab(self) -> bool:
            self.grabs += 1
            raise AssertionError("webcam pacing must not call backend grab")

    queue = FakeQueue()
    stop = FakeStop()
    capture = FakeCapture(stop)
    worker = CaptureThread(
        stream_id="webcam",
        entry=StreamEntry(source=0, model="demo", processing_fps=1.0),
        queue=queue,  # type: ignore[arg-type]
        security=SecurityConfig(),
        target_long_edge=None,
    )
    worker._stop = stop  # type: ignore[assignment]
    times = iter((0.0, 0.0, 0.0, 0.5, 1.0, 1.0))
    monkeypatch.setattr("vision_mcp.streams.capture.monotonic", lambda: next(times))

    assert worker._pump(capture, 0) == 2  # type: ignore[arg-type]
    assert capture.reads == 2
    assert capture.grabs == 0
    assert [frame.index for frame in queue.frames] == [1, 2]


def test_macos_webcam_open_selects_avfoundation_and_recovers_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.fps = 1.0
            self.settings: list[tuple[int, float]] = []

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

        def set(self, prop: int, value: float) -> bool:
            self.settings.append((prop, value))
            if prop == cv2.CAP_PROP_FPS:
                self.fps = value
            return True

        def get(self, prop: int) -> float:
            return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    calls: list[tuple[object, ...]] = []
    capture = FakeCapture()

    def fake_video_capture(*args: object) -> FakeCapture:
        calls.append(args)
        return capture

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    worker = CaptureThread(
        stream_id="webcam",
        entry=StreamEntry(source=0, model="demo", processing_fps=3.0),
        queue=None,  # type: ignore[arg-type]
        security=SecurityConfig(),
        target_long_edge=None,
    )

    opened = worker._open()

    assert opened is capture
    assert calls == [(0, cv2.CAP_AVFOUNDATION)]
    assert (cv2.CAP_PROP_FPS, 30.0) in capture.settings
    assert worker.state.snapshot().source_fps == 30.0


@pytest.mark.asyncio
async def test_throughput_uses_observed_bucket_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDatabase:
        async def fetch_all(self, sql: str, params: object = ()) -> list[dict[str, float]]:
            assert "bucket_seconds" in sql
            return [
                {"bucket_start": 700.0, "bucket_seconds": 30.0, "processed_frames": 90.0},
                {"bucket_start": 730.0, "bucket_seconds": 20.0, "processed_frames": 60.0},
            ]

    query = TimeQuery(
        start=100.0,
        end=1_000.0,
        window_seconds=900,
        interval_seconds=60,
        bucket_count=15,
    )
    view = TimeWindow(
        window="15m",
        interval="1m",
        start="1970-01-01T00:01:40.000Z",
        end="1970-01-01T00:16:40.000Z",
        bucket_count=15,
    )
    monkeypatch.setattr(HistoricalQueries, "resolve", lambda self, window, interval: (query, view))
    config = EngineConfig(
        models={"demo": ModelEntry(architecture="RFDETRNano")},
        streams={"webcam": StreamEntry(source=0, model="demo", processing_fps=3.0)},
    )
    historical = HistoricalQueries(FakeDatabase(), config)  # type: ignore[arg-type]

    result = await historical.throughput("webcam", "15m", "1m")

    assert result.processed_frames == 150
    assert result.processed_fps == 3.0
    assert result.buckets[10].value == 3.0


def test_artifact_resource_uri_is_routed_and_validated() -> None:
    artifact_id = "a" * 32

    assert _artifact_id(f"vision://artifacts/{artifact_id}") == artifact_id
    assert _artifact_id("vision://streams/webcam/status") is None
    with pytest.raises(VisionError) as caught:
        _artifact_id("vision://artifacts/../../secret")
    assert caught.value.code == ErrorCode.ARTIFACT_NOT_FOUND


@pytest.mark.asyncio
async def test_unique_objects_exclude_single_frame_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "vision.db")
    await database.start()
    try:
        await database.write(
            [
                (
                    "INSERT INTO tracks (stream_id, track_id, class_name, first_seen, last_seen,"
                    " frames, mean_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("webcam", 1, "person", 900.0, 990.0, 270, 0.95),
                ),
                (
                    "INSERT INTO tracks (stream_id, track_id, class_name, first_seen, last_seen,"
                    " frames, mean_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("webcam", 2, "person", 950.0, 950.0, 1, 0.93),
                ),
            ]
        )
        query = TimeQuery(
            start=100.0,
            end=1_000.0,
            window_seconds=900,
            interval_seconds=60,
            bucket_count=15,
        )
        view = TimeWindow(
            window="15m",
            interval="1m",
            start="1970-01-01T00:01:40.000Z",
            end="1970-01-01T00:16:40.000Z",
            bucket_count=15,
        )
        monkeypatch.setattr(HistoricalQueries, "resolve", lambda self, window, interval: (query, view))
        config = EngineConfig(
            models={"demo": ModelEntry(architecture="RFDETRNano")},
            streams={"webcam": StreamEntry(source=0, model="demo")},
        )

        result = await HistoricalQueries(database, config).unique_objects("webcam", "15m", "1m")

        assert result.unique_objects == 1
        assert result.by_class == {"person": 1}
    finally:
        await database.stop()
