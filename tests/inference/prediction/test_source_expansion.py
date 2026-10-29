# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Prediction from manifests and optional source adapters."""

import io
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from rfdetr.inference import RFDETRInference

from ..helpers import _DummyRFDETR, _requires_opencv


class TestManifestPrediction:
    """Manifest paths resolve relative to the manifest file."""

    @pytest.mark.parametrize(
        "suffix,content",
        [
            pytest.param(".txt", "b.png\na.png\n", id="txt"),
            pytest.param(".csv", "source\nb.png\na.png\n", id="csv-header-column"),
            pytest.param(".csv", '"b.png",a.png\n', id="csv-every-cell"),
            pytest.param(".csv", "label,path\ncat,b.png\ndog\ndog,a.png\n", id="csv-multi-column"),
        ],
    )
    def test_manifest_preserves_order(self, tmp_path: Path, suffix: str, content: str) -> None:
        """Manifests preserve the explicit order of their image entries."""
        Image.new("RGB", (32, 24), (10, 20, 30)).save(tmp_path / "a.png")
        Image.new("RGB", (32, 24), (40, 50, 60)).save(tmp_path / "b.png")
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content)
        results = RFDETRInference(_DummyRFDETR()).predict(manifest)
        assert isinstance(results, list)
        pixels = []
        for result in results:
            assert isinstance(result, sv.Detections)
            pixels.append(result.metadata["source_image"][0, 0].tolist())
        assert pixels == [[40, 50, 60], [10, 20, 30]]

    @pytest.mark.parametrize("name", ["source", "path"])
    def test_txt_first_entry_named_like_a_header_is_a_source(self, tmp_path: Path, name: str) -> None:
        """A text manifest has no header, so a first line naming a file called ``source`` or ``path`` is predicted.

        Only CSV manifests have a header row; skipping that line in a text manifest would silently drop a real image.
        """
        Image.new("RGB", (32, 24), (10, 20, 30)).save(tmp_path / name, format="PNG")
        manifest = tmp_path / "sources.txt"
        manifest.write_text(f"{name}\n")

        results = RFDETRInference(_DummyRFDETR()).predict(manifest)

        assert isinstance(results, list)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0].metadata["source_image"][0, 0], [10, 20, 30])

    def test_manifest_is_closed_before_its_entries_are_predicted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest is read in full and closed before its first entry runs.

        An entry can be a live stream that runs for hours; the manifest file must not stay open for that long.
        """
        Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("image.png\nimage.png\n")
        handles: list[io.IOBase] = []
        real_open = Path.open

        def recording_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            """Open the path normally and keep the handle so the test can check that it was closed."""
            handle = real_open(path, *args, **kwargs)
            handles.append(handle)
            return handle

        monkeypatch.setattr(Path, "open", recording_open)
        results = RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True)
        next(results)

        assert handles
        assert all(handle.closed for handle in handles)
        results.close()

    def test_nested_manifest_cycle_has_clear_error(self, tmp_path: Path) -> None:
        """Recursive manifests fail with a readable error."""
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("second.txt\n")
        second.write_text("first.txt\n")
        with pytest.raises(ValueError, match="manifest cycle"):
            RFDETRInference(_DummyRFDETR()).predict(first)

    def test_nested_manifest_can_repeat_without_a_cycle(self, tmp_path: Path) -> None:
        """Repeated manifests produce repeated predictions."""
        Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
        (tmp_path / "nested.txt").write_text("image.png\n")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("nested.txt\nnested.txt\n")
        assert len(RFDETRInference(_DummyRFDETR()).predict(manifest)) == 2

    @pytest.mark.parametrize(
        "suffix,content",
        [
            pytest.param(".txt", "# comment\n\n", id="txt-comments-and-blanks"),
            pytest.param(".csv", "source\n", id="csv-header-only"),
            pytest.param(".csv", "", id="csv-empty-file"),
        ],
    )
    def test_empty_manifest_returns_no_predictions(self, tmp_path: Path, suffix: str, content: str) -> None:
        """An eager call over a manifest that lists no sources returns an empty result list.

        An empty collection holds no inputs, so it is not an error; this differs from an empty list literal, which keeps
        raising because a batch with nothing to predict is a caller bug.
        """
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content)
        assert RFDETRInference(_DummyRFDETR()).predict(manifest) == []

    @pytest.mark.parametrize(
        "suffix,content",
        [
            pytest.param(".txt", "# comment\n\n", id="txt-comments-and-blanks"),
            pytest.param(".csv", "source\n", id="csv-header-only"),
        ],
    )
    def test_empty_manifest_streams_nothing(self, tmp_path: Path, suffix: str, content: str) -> None:
        """A streamed call over a manifest that lists no sources yields nothing instead of raising.

        A pipeline can feed a manifest that happens to be empty; ending the stream is the quiet half of the empty-source
        policy, while an empty directory or ``.streams`` file is the loud half.
        """
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content)
        assert list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True)) == []


def _symlinks_supported() -> bool:
    """Report whether this host can create a symbolic link, so symlink tests skip only where they cannot run.

    Windows without developer mode or elevation refuses ``symlink_to`` with ``OSError``. Probing the capability keeps
    the skip narrow instead of excluding a whole platform.

    Returns:
        ``True`` when a link to an existing file can be created in a temporary directory.

    Examples:
        >>> isinstance(_symlinks_supported(), bool)
        True
    """
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "target"
        target.touch()
        try:
            (Path(directory) / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


_skip_without_symlinks = pytest.mark.skipif(
    not _symlinks_supported(), reason="symbolic links cannot be created on this host"
)


def _source_pixels(results: object) -> list[list[int]]:
    """Return the first source-image pixel of every prediction, in result order.

    Args:
        results: The list a non-streamed ``predict`` call returns for an expanded source.

    Returns:
        One ``[r, g, b]`` triple per prediction.

    Examples:
        >>> detections = sv.Detections.empty()
        >>> detections.metadata["source_image"] = np.full((2, 2, 3), (1, 2, 3), dtype=np.uint8)
        >>> _source_pixels([detections])
        [[1, 2, 3]]
    """
    assert isinstance(results, list)
    return [result.metadata["source_image"][0, 0].tolist() for result in results]


class TestManifestEdgeCases:
    """Manifest entries at the edge of encoding, nesting, path and glob rules."""

    @pytest.fixture
    def image(self, tmp_path: Path) -> Path:
        """Write a 32x24 image whose pixels are all ``(10, 20, 30)`` and return its path."""
        path = tmp_path / "a.png"
        Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
        return path

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("", id="empty-file"),
            pytest.param("# comment\n\n", id="comments-and-blanks"),
        ],
    )
    def test_streams_manifest_without_sources_raises(self, tmp_path: Path, content: str, stream: bool) -> None:
        """A ``.streams`` file that lists no sources raises before any capture opens, eager or streamed.

        A concurrent-capture manifest exists to open cameras, so an empty one is a wrong path or a generation bug. The
        empty-source policy makes it loud in both modes, unlike an empty text manifest.
        """
        manifest = tmp_path / "sources.streams"
        manifest.write_text(content)
        with patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture")) as capture:
            with pytest.raises(ValueError, match="no sources"):
                list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=stream))
            capture.assert_not_called()

    @pytest.mark.parametrize(
        "suffix,content",
        [
            pytest.param(".txt", "﻿a.png\n", id="txt"),
            pytest.param(".csv", "﻿source\na.png\n", id="csv-header"),
        ],
    )
    def test_utf8_bom_is_not_part_of_the_first_entry(
        self, tmp_path: Path, image: Path, suffix: str, content: str
    ) -> None:
        """A manifest saved with a UTF-8 byte-order mark still resolves its first entry.

        Windows editors add the mark. Left in place it would glue itself to the first path or hide a CSV ``source``
        header, and the first image would be reported missing.
        """
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content, encoding="utf-8")

        assert _source_pixels(RFDETRInference(_DummyRFDETR()).predict(manifest)) == [[10, 20, 30]]

    def test_manifest_listing_itself_is_a_cycle(self, tmp_path: Path) -> None:
        """A manifest that names itself fails with the cycle error instead of recursing."""
        manifest = tmp_path / "sources.txt"
        manifest.write_text("sources.txt\n")

        with pytest.raises(ValueError, match="manifest cycle"):
            RFDETRInference(_DummyRFDETR()).predict(manifest)

    @_skip_without_symlinks
    def test_symlink_alias_of_a_manifest_is_a_cycle_on_the_real_file(self, tmp_path: Path) -> None:
        """A symbolic link back to the manifest is caught on its first visit, and the error names the real file.

        Cycle detection compares resolved paths. A comparison of spellings would admit the alias once as a new manifest
        and report the alias, not the file the user edits.
        """
        manifest = tmp_path / "sources.txt"
        manifest.write_text("alias.txt\n")
        (tmp_path / "alias.txt").symlink_to(manifest)

        with pytest.raises(ValueError, match=r"manifest cycle: .*sources\.txt\."):
            RFDETRInference(_DummyRFDETR()).predict(manifest)

    @pytest.mark.parametrize("stream", [False, True])
    def test_streams_manifest_nested_in_text_manifest_opens_its_captures(self, tmp_path: Path, stream: bool) -> None:
        """A ``.streams`` file listed by a ``.txt`` manifest is read relative to the outer manifest.

        The nested file keeps its concurrent-capture meaning: its video opens once, relative to the nested file, and
        yields its frames.
        """
        video = tmp_path / "video.avi"
        video.touch()
        (tmp_path / "cameras.streams").write_text("video.avi\n")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("cameras.streams\n")
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        capture.get.return_value = 1.0

        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            predictions = list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=stream, stream_buffer=True))

        assert len(predictions) == 1
        open_capture.assert_called_once_with(str(video))

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("../a.png", id="parent-relative"),
            pytest.param("{root}/a.png", id="absolute"),
        ],
    )
    def test_parent_relative_and_absolute_entries_resolve(self, tmp_path: Path, image: Path, entry: str) -> None:
        """Entries outside the manifest directory are followed as written; nothing confines them.

        A manifest is trusted user input, like a path passed to ``predict``. ``..`` climbs from the manifest directory
        and an absolute path ignores it.
        """
        directory = tmp_path / "lists"
        directory.mkdir()
        manifest = directory / "sources.txt"
        manifest.write_text(entry.format(root=tmp_path.as_posix()) + "\n")

        assert _source_pixels(RFDETRInference(_DummyRFDETR()).predict(manifest)) == [[10, 20, 30]]

    def test_glob_entry_expands_relative_to_manifest(self, tmp_path: Path) -> None:
        """A glob entry matches inside the manifest directory and predicts its matches in sorted order."""
        images = tmp_path / "images"
        images.mkdir()
        Image.new("RGB", (32, 24), (40, 50, 60)).save(images / "b.png")
        Image.new("RGB", (32, 24), (10, 20, 30)).save(images / "a.png")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("images/*.png\n")

        assert _source_pixels(RFDETRInference(_DummyRFDETR()).predict(manifest)) == [[10, 20, 30], [40, 50, 60]]

    def test_glob_entry_without_matches_raises(self, tmp_path: Path) -> None:
        """A glob entry that matches nothing raises, because an empty location is a likely wrong path."""
        manifest = tmp_path / "sources.txt"
        manifest.write_text("images/*.png\n")

        with pytest.raises(FileNotFoundError, match="No supported images or videos"):
            RFDETRInference(_DummyRFDETR()).predict(manifest)


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
            results = RFDETRInference(_DummyRFDETR()).predict("screen 1 5 6 32 24", stream=True)
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
                next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

    @pytest.mark.parametrize("module,source", [("mss", "screen"), ("yt_dlp", "https://youtu.be/video")])
    def test_missing_optional_package_has_install_guidance(self, module: str, source: str) -> None:
        """Missing optional dependencies provide an actionable install hint."""
        with patch.dict("sys.modules", {module: None}):
            with pytest.raises(ImportError, match=r"rfdetr\[stream\]"):
                next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

    @_requires_opencv
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
            results = RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True)
            result = next(results)
            assert list(results) == []
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        downloader.extract_info.assert_called_once_with("https://youtu.be/video", download=False)
        assert open_capture.call_args.args[0] == "https://video.example/frames.mp4"
        capture.release.assert_called_once()

    def test_stream_manifest_resolves_youtube_pages_concurrently(self, tmp_path: Path) -> None:
        """YouTube entries of a stream manifest resolve together instead of one after another.

        Each lookup blocks until the other is running, so a serial resolver would wait out the barrier timeout; results
        must still follow the manifest order.
        """
        pages = {
            "https://youtu.be/one": "https://video.example/one.mp4",
            "https://youtu.be/two": "https://video.example/two.mp4",
        }
        manifest = tmp_path / "videos.streams"
        manifest.write_text("\n".join(pages))
        barrier = threading.Barrier(2, timeout=2)
        module = MagicMock()
        downloader = module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.side_effect = lambda page, download: {"url": pages[page], "arrived": barrier.wait()}
        captures = {url: MagicMock() for url in pages.values()}
        for value, capture in enumerate(captures.values(), start=1):
            capture.get.return_value = 0
            capture.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)), (False, None)]
        with (
            patch.dict("sys.modules", {"yt_dlp": module}),
            patch("cv2.VideoCapture", side_effect=lambda source, *args: captures[source]),
        ):
            results = list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True, stream_buffer=True))
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 2]

    def test_youtube_resolution_caps_height_and_bounds_the_wait(self) -> None:
        """Resolution passes yt-dlp a height-capped format chain and an explicit socket timeout.

        Frames are resized to the model resolution, so selecting a 4K or 8K stream only costs decode memory; an
        unbounded wait on an unresponsive host would block the first ``next()`` of the prediction generator.
        """
        module = MagicMock()
        downloader = module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.return_value = {"url": "https://video.example/frames.mp4"}
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch.dict("sys.modules", {"yt_dlp": module}), patch("cv2.VideoCapture", return_value=capture):
            list(RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True))
        options = module.YoutubeDL.call_args.args[0]
        assert all("[height<=?" in selector for selector in options["format"].split("/"))
        assert 0 < options["socket_timeout"] <= 30

    def test_youtube_download_error_is_a_value_error_naming_the_source(self) -> None:
        """A yt-dlp resolution failure surfaces as ``ValueError`` that names the page and chains the original error.

        A private, region-locked or removed video makes yt-dlp raise its own ``DownloadError``. Callers handle the
        ``ValueError`` family that every other source problem uses, and the message must say which source failed.
        """
        module = MagicMock()
        module.utils.DownloadError = type("DownloadError", (Exception,), {})
        downloader = module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.side_effect = module.utils.DownloadError("ERROR: [youtube] video: Video unavailable")
        with patch.dict("sys.modules", {"yt_dlp": module}):
            with pytest.raises(ValueError, match=r"https://youtu\.be/video.*Video unavailable") as raised:
                next(RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True))
        assert isinstance(raised.value.__cause__, module.utils.DownloadError)


class TestSourceRouting:
    """Source routing preserves image URLs and validates stream manifests."""

    @_requires_opencv
    def test_suffixless_http_stream_manifest_opens_capture(self, tmp_path: Path) -> None:
        """A stream manifest explicitly opts an extensionless HTTP endpoint into video capture."""
        url = "https://camera.example/feed?id=42"
        manifest = tmp_path / "camera.streams"
        manifest.write_text(url)
        capture = MagicMock()
        capture.read.return_value = (True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8))
        with (
            patch("cv2.VideoCapture", return_value=capture) as open_capture,
            patch("requests.get", side_effect=AssertionError("unexpected image download")),
        ):
            results = RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True)
            open_capture.assert_not_called()
            try:
                result = next(results)
                assert isinstance(result, sv.Detections)
                np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
                assert open_capture.call_args.args[0] == url
            finally:
                results.close()
        capture.release.assert_called_once()

    @_requires_opencv
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
            result = RFDETRInference(_DummyRFDETR()).predict("https://images.example/render?id=42")
        assert isinstance(result, sv.Detections)
        capture.assert_not_called()

    @_requires_opencv
    @pytest.mark.parametrize("entry", ["screen", "missing.avi", "image.png", "https://images.example/image.png"])
    def test_invalid_stream_manifest_entries_fail_before_capture(self, tmp_path: Path, entry: str) -> None:
        """Stream manifests accept video captures rather than arbitrary images or screens."""
        manifest = tmp_path / "sources.streams"
        manifest.write_text(entry)
        with patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture")) as capture:
            with pytest.raises((ValueError, FileNotFoundError), match="[Ss]tream|[Vv]ideo"):
                next(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True))
            capture.assert_not_called()

    @_requires_opencv
    def test_remote_video_uses_capture_timeouts(self) -> None:
        """Remote finite video reads use bounded OpenCV open and read calls."""
        import cv2

        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = RFDETRInference(_DummyRFDETR()).predict("https://video.example/movie.mp4")
        assert isinstance(results, list)
        assert len(results) == 1
        open_capture.assert_called_once_with(
            "https://video.example/movie.mp4",
            cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )

    @_requires_opencv
    def test_eager_nested_live_manifest_warns_before_capture(self, tmp_path: Path) -> None:
        """Live sources inside nested manifests warn before capture starts."""
        (tmp_path / "nested.txt").write_text("0")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("nested.txt")
        with patch("cv2.VideoCapture", side_effect=RuntimeError("capture attempted")):
            with pytest.warns(UserWarning, match="stream=True"):
                with pytest.raises(RuntimeError, match="capture attempted"):
                    RFDETRInference(_DummyRFDETR()).predict(manifest)

    @_requires_opencv
    @pytest.mark.parametrize("source", [0, "flat.txt", "nested.txt"])
    def test_live_memory_warning_names_the_calling_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: int | str
    ) -> None:
        """The eager live-source warning points at the user's call, however deeply the source is nested.

        A camera index, a manifest of cameras and a manifest of manifests reach the warning through different numbers of
        generator frames. The warning must name the module that called ``predict``, not an ``rfdetr`` internal.
        """
        (tmp_path / "flat.txt").write_text("0")
        (tmp_path / "nested.txt").write_text("flat.txt")
        monkeypatch.chdir(tmp_path)
        with patch("cv2.VideoCapture", side_effect=RuntimeError("capture attempted")):
            with pytest.warns(UserWarning, match="stream=True") as record:
                with pytest.raises(RuntimeError, match="capture attempted"):
                    RFDETRInference(_DummyRFDETR()).predict(source)
        live = [warning for warning in record if "stream=True" in str(warning.message)]
        assert [Path(warning.filename).resolve() for warning in live] == [Path(__file__).resolve()]

    @_requires_opencv
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
                results = RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video")
        assert isinstance(results, list)
        assert len(results) == 1
        assert not any("Live results accumulate" in str(warning.message) for warning in caught)

    def test_large_eager_video_warns_about_retained_frames(self, tmp_path: Path) -> None:
        """An eager video whose frame count and size exceed the threshold warns before frames are decoded.

        Each eager result keeps its full source frame, so a long video can exhaust memory without ``stream=True``.
        """
        import cv2

        source = tmp_path / "video.mp4"
        source.touch()
        sizes = {cv2.CAP_PROP_FRAME_COUNT: 2.0, cv2.CAP_PROP_FRAME_HEIGHT: 24.0, cv2.CAP_PROP_FRAME_WIDTH: 32.0}
        capture = MagicMock()
        capture.get.side_effect = sizes.__getitem__
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))] * 2 + [(False, None)]
        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", 2 * 24 * 32 * 3),
        ):
            with pytest.warns(UserWarning, match="include_source_image=False"):
                results = RFDETRInference(_DummyRFDETR()).predict(source)
        assert len(results) == 2

    def test_large_eager_image_directory_warns_about_retained_frames(self, tmp_path: Path) -> None:
        """An eager directory estimates retained pixels from its image count and first image size."""
        for index in range(2):
            Image.new("RGB", (32, 24)).save(tmp_path / f"{index}.png")
        with patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", 2 * 24 * 32 * 3):
            with pytest.warns(UserWarning, match="include_source_image=False"):
                results = RFDETRInference(_DummyRFDETR()).predict(tmp_path)
        assert len(results) == 2

    @pytest.mark.parametrize(
        ("options", "threshold"),
        [
            pytest.param({"stream": True}, 1, id="streamed"),
            pytest.param({"include_source_image": False}, 1, id="no-source-image"),
            pytest.param({}, 2 * 24 * 32 * 3 + 1, id="below-threshold"),
        ],
    )
    def test_retained_frame_warning_is_silent(self, tmp_path: Path, options: dict[str, bool], threshold: int) -> None:
        """Streamed runs, runs without source images, and sources below the threshold do not warn."""
        for index in range(2):
            Image.new("RGB", (32, 24)).save(tmp_path / f"{index}.png")
        with patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", threshold):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = list(RFDETRInference(_DummyRFDETR()).predict(tmp_path, **options))
        assert len(results) == 2
        assert not any("include_source_image=False" in str(warning.message) for warning in caught)

    def test_missing_opencv_has_install_guidance(self, tmp_path: Path) -> None:
        """Video decoding explains which optional dependency is missing."""
        source = tmp_path / "video.avi"
        source.touch()
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(
                ImportError, match=r'requires OpenCV\. Install it with `uv pip install "rfdetr\[stream\]"`'
            ):
                next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

    @_requires_opencv
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
            results = RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True, stream_buffer=True)
            predictions = list(results)
        assert len(predictions) == 1
        open_capture.assert_called_once_with(str(video))
