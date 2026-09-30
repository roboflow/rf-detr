# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Prediction from manifests and optional source adapters."""

import io
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from .helpers import _DummyRFDETR


class TestManifestPrediction:
    """Manifest paths resolve relative to the manifest file."""

    @pytest.mark.parametrize("suffix,content", [(".txt", "b.png\na.png\n"), (".csv", 'source\n"b.png",a.png\n')])
    def test_manifest_preserves_order(self, tmp_path: Path, suffix: str, content: str) -> None:
        """Manifests preserve the explicit order of their image entries."""
        Image.new("RGB", (32, 24), (10, 20, 30)).save(tmp_path / "a.png")
        Image.new("RGB", (32, 24), (40, 50, 60)).save(tmp_path / "b.png")
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content)
        results = _DummyRFDETR().predict(manifest)
        assert isinstance(results, list)
        pixels = []
        for result in results:
            assert isinstance(result, sv.Detections)
            pixels.append(result.metadata["source_image"][0, 0].tolist())
        assert pixels == [[40, 50, 60], [10, 20, 30]]

    def test_nested_manifest_cycle_has_clear_error(self, tmp_path: Path) -> None:
        """Recursive manifests fail with a readable error."""
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("second.txt\n")
        second.write_text("first.txt\n")
        with pytest.raises(ValueError, match="manifest cycle"):
            _DummyRFDETR().predict(first)

    def test_nested_manifest_can_repeat_without_a_cycle(self, tmp_path: Path) -> None:
        """Repeated manifests produce repeated predictions."""
        Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
        (tmp_path / "nested.txt").write_text("image.png\n")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("nested.txt\nnested.txt\n")
        assert len(_DummyRFDETR().predict(manifest)) == 2

    def test_empty_manifest_returns_no_predictions(self, tmp_path: Path) -> None:
        """Empty manifests produce an empty result list."""
        manifest = tmp_path / "sources.txt"
        manifest.write_text("# comment\n\n")
        assert _DummyRFDETR().predict(manifest) == []


class TestOptionalSources:
    """Optional adapters preserve RGB frames and close their resources."""

    def test_screen_region_uses_monitor_offset_and_closes(self) -> None:
        """Screen coordinates are relative to the selected monitor."""
        module = MagicMock()
        capture = module.mss.return_value.__enter__.return_value
        capture.monitors = [
            {"left": 0, "top": 0, "width": 100, "height": 100},
            {"left": 100, "top": 200, "width": 100, "height": 100},
        ]
        capture.grab.return_value = np.full((24, 32, 4), (10, 20, 30, 255), dtype=np.uint8)
        with patch.dict("sys.modules", {"mss": module}):
            results = _DummyRFDETR().predict("screen 1 5 6 32 24", stream=True)
            result = next(results)
            results.close()
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        capture.grab.assert_called_once_with({"left": 105, "top": 206, "width": 32, "height": 24})
        module.mss.return_value.__exit__.assert_called_once()

    @pytest.mark.parametrize("source", ["screen -1", "screen 0 0 0 24", "screen 1 2", "screen invalid"])
    def test_invalid_screen_source_fails_before_capture(self, source: str) -> None:
        """Invalid screen arguments fail before the desktop is read."""
        with patch.dict("sys.modules", {"mss": None}):
            with pytest.raises(ValueError, match="Screen|screen"):
                next(_DummyRFDETR().predict(source, stream=True))

    @pytest.mark.parametrize("module,source", [("mss", "screen"), ("yt_dlp", "https://youtu.be/video")])
    def test_missing_optional_package_has_install_guidance(self, module: str, source: str) -> None:
        """Missing optional dependencies provide an actionable install hint."""
        with patch.dict("sys.modules", {module: None}):
            with pytest.raises(ImportError, match=r"rfdetr\[stream\]"):
                next(_DummyRFDETR().predict(source, stream=True))

    def test_youtube_resolves_page_without_downloading(self) -> None:
        """YouTube inference opens the resolved stream and converts RGB."""
        module = MagicMock()
        downloader = module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.return_value = {"url": "https://video.example/frames.mp4"}
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8)), (False, None)]
        with (
            patch.dict("sys.modules", {"yt_dlp": module}),
            patch("cv2.VideoCapture", return_value=capture) as open_capture,
        ):
            results = _DummyRFDETR().predict("https://youtu.be/video", stream=True)
            result = next(results)
            assert list(results) == []
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        downloader.extract_info.assert_called_once_with("https://youtu.be/video", download=False)
        assert open_capture.call_args.args[0] == "https://video.example/frames.mp4"
        capture.release.assert_called_once()


class TestSourceRouting:
    """Source routing preserves image URLs and validates stream manifests."""

    def test_suffixless_http_url_remains_an_image(self) -> None:
        """An image endpoint without a suffix keeps the existing scalar result."""
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="PNG")
        response = MagicMock()
        response.content = buffer.getvalue()
        with (
            patch("requests.get", return_value=response),
            patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture")) as capture,
        ):
            result = _DummyRFDETR().predict("https://images.example/render?id=42")
        assert isinstance(result, sv.Detections)
        capture.assert_not_called()

    @pytest.mark.parametrize("entry", ["screen", "missing.avi", "image.png", "https://images.example/image.png"])
    def test_invalid_stream_manifest_entries_fail_before_capture(self, tmp_path: Path, entry: str) -> None:
        """Stream manifests accept video captures rather than arbitrary images or screens."""
        manifest = tmp_path / "sources.streams"
        manifest.write_text(entry)
        with patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture")) as capture:
            with pytest.raises((ValueError, FileNotFoundError), match="[Ss]tream|[Vv]ideo"):
                next(_DummyRFDETR().predict(manifest, stream=True))
            capture.assert_not_called()

    def test_remote_video_uses_capture_timeouts(self) -> None:
        """Remote finite video reads use bounded OpenCV open and read calls."""
        import cv2

        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = _DummyRFDETR().predict("https://video.example/movie.mp4")
        assert isinstance(results, list)
        assert len(results) == 1
        open_capture.assert_called_once_with(
            "https://video.example/movie.mp4",
            cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )

    def test_eager_nested_live_manifest_warns_before_capture(self, tmp_path: Path) -> None:
        """Live sources inside nested manifests warn before capture starts."""
        (tmp_path / "nested.txt").write_text("0")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("nested.txt")
        with patch("cv2.VideoCapture", side_effect=RuntimeError("capture attempted")):
            with pytest.warns(UserWarning, match="stream=True"):
                with pytest.raises(RuntimeError, match="capture attempted"):
                    _DummyRFDETR().predict(manifest)

    def test_eager_youtube_video_does_not_warn_about_live_memory(self) -> None:
        """A resolved finite YouTube video produces results without a live warning."""
        module = MagicMock()
        downloader = module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.return_value = {"url": "https://video.example/frames.mp4", "is_live": False}
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch.dict("sys.modules", {"yt_dlp": module}), patch("cv2.VideoCapture", return_value=capture):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = _DummyRFDETR().predict("https://youtu.be/video")
        assert isinstance(results, list)
        assert len(results) == 1
        assert not any("stream=True" in str(warning.message) for warning in caught)

    def test_missing_opencv_has_install_guidance(self, tmp_path: Path) -> None:
        """Video decoding explains which optional dependency is missing."""
        source = tmp_path / "video.avi"
        source.touch()
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match=r"rfdetr\[stream\]"):
                next(_DummyRFDETR().predict(source, stream=True))

    def test_local_video_stream_manifest_requires_existing_file(self, tmp_path: Path) -> None:
        """A stream manifest opens its local video relative to the manifest."""
        video = tmp_path / "video.avi"
        video.touch()
        manifest = tmp_path / "sources.streams"
        manifest.write_text("video.avi")
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        capture.get.return_value = 1.0
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = _DummyRFDETR().predict(manifest, stream=True, stream_buffer=True)
            predictions = list(results)
        assert len(predictions) == 1
        open_capture.assert_called_once_with(str(video))
