# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Prediction from manifests and optional source adapters."""

import dataclasses
import io
import tempfile
import threading
import warnings
from contextlib import closing
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr._prediction import core, pipelines
from rfdetr.inference import RFDETRInference

from ..helpers import _DummyRFDETR, _image_dir, _requires_opencv, _source_pixels


@pytest.fixture
def youtube_module(mocker: MockerFixture) -> MagicMock:
    """A patched ``yt_dlp`` whose downloader resolves every page to ``https://video.example/frames.mp4``."""
    module = mocker.MagicMock()
    downloader = module.YoutubeDL.return_value.__enter__.return_value
    downloader.extract_info.return_value = {"url": "https://video.example/frames.mp4"}
    mocker.patch.dict("sys.modules", {"yt_dlp": module})
    return module


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

    def test_nested_manifest_can_repeat_without_a_cycle(self, tmp_path: Path) -> None:
        """Repeated manifests produce repeated predictions."""
        Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
        (tmp_path / "nested.txt").write_text("image.png\n")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("nested.txt\nnested.txt\n")
        assert len(RFDETRInference(_DummyRFDETR()).predict(manifest)) == 2

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "suffix,content",
        [
            pytest.param(".txt", "# comment\n\n", id="txt-comments-and-blanks"),
            pytest.param(".csv", "source\n", id="csv-header-only"),
            pytest.param(".csv", "", id="csv-empty-file"),
        ],
    )
    def test_empty_manifest_yields_no_predictions(
        self, tmp_path: Path, suffix: str, content: str, stream: bool
    ) -> None:
        """A manifest that lists no sources yields no predictions, as an empty list or an ended stream, without raising.

        An empty collection holds no inputs, so it is not an error; this differs from an empty list literal, which keeps
        raising because a batch with nothing to predict is a caller bug. Ending the stream is the quiet half of the
        empty-source policy, while an empty directory or ``.streams`` file is the loud half.
        """
        manifest = tmp_path / f"sources{suffix}"
        manifest.write_text(content)

        results = RFDETRInference(_DummyRFDETR()).predict(manifest, stream=stream)

        assert isinstance(results, list) is not stream
        assert list(results) == []


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


_requires_symlinks = pytest.mark.skipif(
    not _symlinks_supported(), reason="symbolic links cannot be created on this host"
)


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
    def test_streams_manifest_without_sources_raises(
        self, tmp_path: Path, content: str, stream: bool, mocker: MockerFixture
    ) -> None:
        """A ``.streams`` file that lists no sources raises before any capture opens, eager or streamed.

        A concurrent-capture manifest exists to open cameras, so an empty one is a wrong path or a generation bug. The
        empty-source policy makes it loud in both modes, unlike an empty text manifest.
        """
        manifest = tmp_path / "sources.streams"
        manifest.write_text(content)
        # Guard the package capture boundary so this validation also runs without OpenCV.
        video_import = mocker.patch("rfdetr._prediction.sources._import_cv2")
        live_import = mocker.patch("rfdetr._prediction.streams._import_cv2")

        with pytest.raises(ValueError, match="no sources"):
            list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=stream))

        video_import.assert_not_called()
        live_import.assert_not_called()

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

    @pytest.mark.parametrize(
        ("files", "start"),
        [
            pytest.param({"sources.txt": "sources.txt\n"}, "sources.txt", id="self-listing"),
            pytest.param({"first.txt": "second.txt\n", "second.txt": "first.txt\n"}, "first.txt", id="mutual"),
        ],
    )
    def test_manifest_cycle_raises_instead_of_recursing(
        self, tmp_path: Path, files: dict[str, str], start: str
    ) -> None:
        """A manifest that lists itself, directly or through another manifest, fails with a readable cycle error."""
        for name, content in files.items():
            (tmp_path / name).write_text(content)

        with pytest.raises(ValueError, match="manifest cycle"):
            RFDETRInference(_DummyRFDETR()).predict(tmp_path / start)

    @_requires_symlinks
    def test_symlink_alias_of_a_manifest_is_a_cycle_on_the_real_file(self, tmp_path: Path) -> None:
        """A symbolic link back to the manifest is caught on its first visit, and the error names the real file.

        Cycle detection compares resolved paths. A comparison of spellings would admit the alias once as a new manifest
        and report the alias, not the file the user edits.
        """
        manifest = tmp_path / "sources.txt"
        manifest.write_text("alias.txt\n")
        (tmp_path / "alias.txt").symlink_to(manifest)

        with pytest.raises(ValueError) as exc_info:
            RFDETRInference(_DummyRFDETR()).predict(manifest)

        assert "manifest cycle: " in str(exc_info.value)
        assert "sources.txt." in str(exc_info.value).split("manifest cycle: ", 1)[1]

    @_requires_opencv
    @pytest.mark.parametrize("stream", [False, True])
    def test_streams_manifest_nested_in_text_manifest_opens_its_captures(
        self, tmp_path: Path, stream: bool, mocker: MockerFixture
    ) -> None:
        """A ``.streams`` file listed by a ``.txt`` manifest is read relative to the outer manifest.

        The nested file keeps its concurrent-capture meaning: its video opens once, relative to the nested file, and
        yields its frames.
        """
        video = tmp_path / "video.avi"
        video.touch()
        (tmp_path / "cameras.streams").write_text("video.avi\n")
        manifest = tmp_path / "sources.txt"
        manifest.write_text("cameras.streams\n")
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        capture.get.return_value = 1.0

        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        predictions = list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=stream, stream_buffer=True))

        assert len(predictions) == 1
        open_capture.assert_called_once_with(str(video))

    @pytest.mark.parametrize("entry", ["../a.png", "{root}/a.png"])
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

    def test_screen_region_uses_monitor_offset_and_closes(self, mocker: MockerFixture) -> None:
        """Screen coordinates are relative to the selected monitor."""
        module = mocker.MagicMock()
        capture = module.mss.return_value.__enter__.return_value
        capture.monitors = [
            {"left": 0, "top": 0, "width": 100, "height": 100},
            {"left": 100, "top": 200, "width": 100, "height": 100},
        ]
        capture.grab.return_value = np.full((24, 32, 4), (10, 20, 30, 255), dtype=np.uint8)
        mocker.patch.dict("sys.modules", {"mss": module})

        results = RFDETRInference(_DummyRFDETR()).predict("screen 1 5 6 32 24", stream=True)
        result = next(results)
        results.close()

        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        capture.grab.assert_called_once_with({"left": 105, "top": 206, "width": 32, "height": 24})
        module.mss.return_value.__exit__.assert_called_once()

    @pytest.mark.parametrize("source", ["screen -1", "screen 0 0 0 24", "screen 1 2", "screen invalid"])
    def test_invalid_screen_source_fails_before_capture(self, source: str, mocker: MockerFixture) -> None:
        """Invalid screen arguments fail before the desktop is read."""
        mocker.patch.dict("sys.modules", {"mss": None})

        with pytest.raises(ValueError) as exc_info:
            next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

        assert "screen" in str(exc_info.value).lower()

    @pytest.mark.parametrize("module,source", [("mss", "screen"), ("yt_dlp", "https://youtu.be/video")])
    def test_missing_optional_package_has_install_guidance(
        self, module: str, source: str, mocker: MockerFixture
    ) -> None:
        """Missing optional dependencies provide an actionable install hint."""
        mocker.patch.dict("sys.modules", {module: None})

        with pytest.raises(ImportError) as exc_info:
            next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

        assert "rfdetr[stream]" in str(exc_info.value)

    @_requires_opencv
    def test_youtube_resolves_page_without_downloading(self, youtube_module: MagicMock, mocker: MockerFixture) -> None:
        """YouTube inference opens the resolved stream and converts RGB."""
        downloader = youtube_module.YoutubeDL.return_value.__enter__.return_value
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8)), (False, None)]
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)

        results = RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True)
        result = next(results)
        assert list(results) == []
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        downloader.extract_info.assert_called_once_with("https://youtu.be/video", download=False)
        assert open_capture.call_args.args[0] == "https://video.example/frames.mp4"
        capture.release.assert_called_once()

    @_requires_opencv
    def test_stream_manifest_resolves_youtube_pages_concurrently(
        self, tmp_path: Path, youtube_module: MagicMock, mocker: MockerFixture
    ) -> None:
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
        downloader = youtube_module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.side_effect = lambda page, download: {"url": pages[page], "arrived": barrier.wait()}
        captures = {url: mocker.MagicMock() for url in pages.values()}
        for value, capture in enumerate(captures.values(), start=1):
            capture.get.return_value = 0
            capture.read.side_effect = [(True, np.full((24, 32, 3), value, dtype=np.uint8)), (False, None)]
        mocker.patch("cv2.VideoCapture", side_effect=lambda source, *args: captures[source])

        results = list(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True, stream_buffer=True))

        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 2]

    @_requires_opencv
    def test_youtube_resolution_caps_height_and_bounds_the_wait(
        self, youtube_module: MagicMock, mocker: MockerFixture
    ) -> None:
        """Resolution passes yt-dlp a height-capped format chain and an explicit socket timeout.

        Frames are resized to the model resolution, so selecting a 4K or 8K stream only costs decode memory; an
        unbounded wait on an unresponsive host would block the first ``next()`` of the prediction generator.
        """
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        mocker.patch("cv2.VideoCapture", return_value=capture)

        list(RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True))

        options = youtube_module.YoutubeDL.call_args.args[0]
        assert all("[height<=?" in selector for selector in options["format"].split("/"))
        assert 0 < options["socket_timeout"] <= 30

    def test_youtube_download_error_is_a_value_error_naming_the_source(self, youtube_module: MagicMock) -> None:
        """A yt-dlp resolution failure surfaces as ``ValueError`` that names the page and chains the original error.

        A private, region-locked or removed video makes yt-dlp raise its own ``DownloadError``. Callers handle the
        ``ValueError`` family that every other source problem uses, and the message must say which source failed.
        """
        youtube_module.utils.DownloadError = type("DownloadError", (Exception,), {})
        downloader = youtube_module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.side_effect = youtube_module.utils.DownloadError(
            "ERROR: [youtube] video: Video unavailable"
        )

        with pytest.raises(ValueError) as raised:
            next(RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video", stream=True))

        assert "https://youtu.be/video" in str(raised.value)
        assert "Video unavailable" in str(raised.value)
        assert isinstance(raised.value.__cause__, youtube_module.utils.DownloadError)


class TestSourceRouting:
    """Source routing preserves image URLs and validates stream manifests."""

    @_requires_opencv
    def test_suffixless_http_stream_manifest_opens_capture(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A stream manifest explicitly opts an extensionless HTTP endpoint into video capture."""
        url = "https://camera.example/feed?id=42"
        manifest = tmp_path / "camera.streams"
        manifest.write_text(url)
        capture = mocker.MagicMock()
        capture.read.return_value = (True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8))
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        mocker.patch("requests.get", side_effect=AssertionError("unexpected image download"))

        with closing(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True)) as results:
            open_capture.assert_not_called()
            result = next(results)
            assert isinstance(result, sv.Detections)
            np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
            assert open_capture.call_args.args[0] == url
        capture.release.assert_called_once()

    @_requires_opencv
    def test_suffixless_http_url_remains_an_image(self, mocker: MockerFixture) -> None:
        """An image endpoint without a suffix keeps the existing scalar result."""
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="PNG")
        response = mocker.MagicMock()
        response.content = buffer.getvalue()
        mocker.patch("requests.get", return_value=response)
        capture = mocker.patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture"))

        result = RFDETRInference(_DummyRFDETR()).predict("https://images.example/render?id=42")

        assert isinstance(result, sv.Detections)
        capture.assert_not_called()

    @_requires_opencv
    @pytest.mark.parametrize("entry", ["screen", "missing.avi", "image.png", "https://images.example/image.png"])
    def test_invalid_stream_manifest_entries_fail_before_capture(
        self, tmp_path: Path, entry: str, mocker: MockerFixture
    ) -> None:
        """Stream manifests accept video captures rather than arbitrary images or screens."""
        manifest = tmp_path / "sources.streams"
        manifest.write_text(entry)
        capture = mocker.patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture"))
        with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
            next(RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True))
        assert any(word in str(exc_info.value).lower() for word in ("stream", "video"))
        capture.assert_not_called()

    @_requires_opencv
    def test_remote_video_uses_capture_timeouts(self, mocker: MockerFixture) -> None:
        """Remote finite video reads use bounded OpenCV open and read calls."""
        import cv2

        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict("https://video.example/movie.mp4")
        assert isinstance(results, list)
        assert len(results) == 1
        open_capture.assert_called_once_with(
            "https://video.example/movie.mp4",
            cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )

    @_requires_opencv
    @pytest.mark.parametrize("source", [0, "flat.txt", "nested.txt"])
    def test_live_memory_warning_names_the_calling_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: int | str, mocker: MockerFixture
    ) -> None:
        """The eager live-source warning fires before capture and points at the user's call, however deeply nested.

        A camera index, a manifest of cameras and a manifest of manifests reach the warning through different numbers of
        generator frames. The warning must name the module that called ``predict``, not an ``rfdetr`` internal.
        """
        (tmp_path / "flat.txt").write_text("0")
        (tmp_path / "nested.txt").write_text("flat.txt")
        monkeypatch.chdir(tmp_path)
        mocker.patch("cv2.VideoCapture", side_effect=RuntimeError("capture attempted"))
        with pytest.warns(UserWarning, match="stream=True") as record:
            with pytest.raises(RuntimeError, match="capture attempted"):
                RFDETRInference(_DummyRFDETR()).predict(source)
        live = [warning for warning in record if "stream=True" in str(warning.message)]
        assert [Path(warning.filename).resolve() for warning in live] == [Path(__file__).resolve()]

    @_requires_opencv
    def test_eager_youtube_video_does_not_warn_about_live_memory(
        self, youtube_module: MagicMock, mocker: MockerFixture
    ) -> None:
        """A resolved finite YouTube video produces results without a live warning."""
        downloader = youtube_module.YoutubeDL.return_value.__enter__.return_value
        downloader.extract_info.return_value = {"url": "https://video.example/frames.mp4", "is_live": False}
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        mocker.patch("cv2.VideoCapture", return_value=capture)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = RFDETRInference(_DummyRFDETR()).predict("https://youtu.be/video")

        assert isinstance(results, list)
        assert len(results) == 1
        assert not any("Live results accumulate" in str(warning.message) for warning in caught)

    @_requires_opencv
    def test_large_eager_video_warns_about_retained_frames(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """An eager video whose frame count and size exceed the threshold warns before frames are decoded.

        Each eager result keeps its full source frame, so a long video can exhaust memory without ``stream=True``.
        """
        import cv2

        source = tmp_path / "video.mp4"
        source.touch()
        sizes = {cv2.CAP_PROP_FRAME_COUNT: 2.0, cv2.CAP_PROP_FRAME_HEIGHT: 24.0, cv2.CAP_PROP_FRAME_WIDTH: 32.0}
        capture = mocker.MagicMock()
        capture.get.side_effect = sizes.__getitem__
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))] * 2 + [(False, None)]
        mocker.patch("cv2.VideoCapture", return_value=capture)
        mocker.patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", 2 * 24 * 32 * 3)

        with pytest.warns(UserWarning, match="include_source_image=False"):
            results = RFDETRInference(_DummyRFDETR()).predict(source)
        assert len(results) == 2

    def test_large_eager_image_directory_warns_about_retained_frames(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """An eager directory estimates retained pixels from its image count and first image size."""
        for index in range(2):
            Image.new("RGB", (32, 24)).save(tmp_path / f"{index}.png")
        mocker.patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", 2 * 24 * 32 * 3)
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
    def test_retained_frame_warning_is_silent(
        self, tmp_path: Path, options: dict[str, bool], threshold: int, mocker: MockerFixture
    ) -> None:
        """Streamed runs, runs without source images, and sources below the threshold do not warn."""
        for index in range(2):
            Image.new("RGB", (32, 24)).save(tmp_path / f"{index}.png")
        mocker.patch("rfdetr._prediction.sources._RETAINED_FRAME_WARNING_BYTES", threshold)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = list(RFDETRInference(_DummyRFDETR()).predict(tmp_path, **options))
        assert len(results) == 2
        assert not any("include_source_image=False" in str(warning.message) for warning in caught)

    def test_missing_opencv_has_install_guidance(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Video decoding explains which optional dependency is missing."""
        source = tmp_path / "video.avi"
        source.touch()
        mocker.patch.dict("sys.modules", {"cv2": None})

        with pytest.raises(ImportError) as exc_info:
            next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

        assert 'requires OpenCV. Install it with `uv pip install "rfdetr[stream]"`' in str(exc_info.value)

    @_requires_opencv
    def test_stream_manifest_opens_local_video_relative_to_manifest(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """A stream manifest opens its local video relative to the manifest."""
        video = tmp_path / "video.avi"
        video.touch()
        manifest = tmp_path / "sources.streams"
        manifest.write_text("video.avi")
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        capture.get.return_value = 1.0
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(manifest, stream=True, stream_buffer=True)
        predictions = list(results)
        assert len(predictions) == 1
        open_capture.assert_called_once_with(str(video))


class TestDefaultEagerBatch:
    """``batch=None`` predicts an eager directory or glob in batches of eight on a model without a fixed batch size."""

    @staticmethod
    def _batch_sizes(
        mocker: MockerFixture, source: Any, *, stream: bool = False, context_overrides: dict[str, Any] | None = None
    ) -> list[int]:
        """Predict ``source`` with ``batch=None`` and return the size of every batch that reached the model.

        Examples:
            Needs the pytest-mock ``mocker`` fixture, which a standalone doctest cannot create.

            >>> TestDefaultEagerBatch._batch_sizes(mocker, np.zeros((4, 4, 3), dtype=np.uint8))  # doctest: +SKIP
            [1]
        """
        model = _DummyRFDETR()
        sizes: list[int] = []
        original = core._predict_batch

        def recording_predict_batch(context_factory: Any, images: Any, *args: Any) -> Any:
            """Record the batch size, then predict the batch as the real function does."""
            sizes.append(len(images) if isinstance(images, list) else 1)
            return original(context_factory, images, *args)

        def context_factory() -> core.PredictionContext:
            """Return the dummy model's context with ``context_overrides`` applied."""
            return dataclasses.replace(model._prediction_context(), **(context_overrides or {}))

        mocker.patch.object(pipelines, "_predict_batch", side_effect=recording_predict_batch)
        results = core.predict(context_factory, source, stream=stream)
        if stream:
            list(results)
        return sizes

    def test_eager_directory_runs_in_batches_of_eight(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Ten files run as one batch of eight and a final partial batch of two."""
        assert self._batch_sizes(mocker, _image_dir(tmp_path)) == [8, 2]

    def test_eager_glob_runs_in_batches_of_eight(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A glob is a finite file source like a directory."""
        _image_dir(tmp_path)
        assert self._batch_sizes(mocker, str(tmp_path / "images" / "*.png")) == [8, 2]

    def test_streamed_directory_keeps_one_image_per_batch(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """``stream=True`` yields the first result after one image, not after a full batch."""
        assert self._batch_sizes(mocker, _image_dir(tmp_path), stream=True) == [1] * 10

    def test_fixed_batch_context_keeps_one_image_per_batch(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A compiled snapshot keeps the one-image default instead of failing its batch-size check."""
        overrides = {"batch_size": 1}
        assert self._batch_sizes(mocker, _image_dir(tmp_path), context_overrides=overrides) == [1] * 10

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            pytest.param({"fixed_shape": True}, [8, 2], id="unbounded"),
            pytest.param({"fixed_shape": True, "max_batch_size": 4}, [4, 4, 2], id="max-batch-below-eight"),
            pytest.param({"fixed_shape": True, "max_batch_size": 16}, [8, 2], id="max-batch-above-eight"),
            pytest.param({"max_batch_size": 1}, [1] * 10, id="max-batch-one"),
            pytest.param({"max_batch_size": 8}, [8, 2], id="max-batch-equals-eight"),
            pytest.param({"max_batch_size": 9}, [8, 2], id="max-batch-just-above-eight"),
        ],
    )
    def test_dynamic_batch_export_takes_the_eager_default(
        self, tmp_path: Path, mocker: MockerFixture, overrides: dict[str, Any], expected: list[int]
    ) -> None:
        """An export's fixed input resolution does not fix its batch axis, so it batches up to its own maximum."""
        assert self._batch_sizes(mocker, _image_dir(tmp_path), context_overrides=overrides) == expected

    def test_streamed_dynamic_batch_export_keeps_one_image_per_batch(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """A streamed directory stays at one image per batch even when the export allows larger batches."""
        overrides = {"max_batch_size": 4}
        batch_sizes = self._batch_sizes(mocker, _image_dir(tmp_path), stream=True, context_overrides=overrides)
        assert batch_sizes == [1] * 10

    def test_manifest_keeps_one_image_per_batch(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A manifest may name live sources, so it keeps the one-image default."""
        _image_dir(tmp_path, count=3)
        manifest = tmp_path / "sources.txt"
        manifest.write_text("images/00.png\nimages/01.png\nimages/02.png\n")
        assert self._batch_sizes(mocker, manifest) == [1, 1, 1]

    def test_compiled_snapshot_context_reports_its_batch_size(self) -> None:
        """A snapshot compiled for a batch size reports it, so the default batch never exceeds it."""
        model = _DummyRFDETR()
        model._optimized_has_been_compiled = True
        model._optimized_batch_size = 4

        assert model._prediction_context().batch_size == 4
