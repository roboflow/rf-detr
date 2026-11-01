# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Prediction contracts for file collections and streams."""

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import requests
import supervision as sv
import torch
from PIL import Image, UnidentifiedImageError
from pytest_mock import MockerFixture

from rfdetr._prediction.sources import has_only_finite_images
from rfdetr.inference import RFDETRInference
from tests._online import is_online

from ..helpers import _DummyModel, _DummyRFDETR, _FakeResponse, _png_bytes, _requires_opencv, _scripted_capture

if TYPE_CHECKING:
    from rfdetr.inference import ModelContext

_HTTP_IMAGE_URL = "http://images.cocodataset.org/val2017/000000397133.jpg"
_HTTP_HOST = "images.cocodataset.org"
_HTTP_PORT = 80


class _ExportableIdentity(torch.nn.Identity):
    """Identity module exposing the ``export()`` switch that native compilation calls before tracing.

    Examples:
        >>> _ExportableIdentity().export() is None
        True
    """

    def export(self) -> None:
        """Accept the export switch; an identity has no export-specific layers."""


@pytest.fixture
def forbid_capture(mocker: MockerFixture) -> None:
    """Fail on any attempt to start a screen or video capture, so image-only cases also run without OpenCV."""
    mocker.patch.dict("sys.modules", {"mss": None})
    mocker.patch("rfdetr._prediction.sources._import_cv2", side_effect=AssertionError("unexpected capture"))
    mocker.patch("rfdetr._prediction.streams._import_cv2", side_effect=AssertionError("unexpected capture"))


@pytest.fixture
def recorded_model() -> Generator[tuple[_DummyRFDETR, list[int]], None, None]:
    """A weight-free model, and the list of batch sizes its network receives."""
    model = _DummyRFDETR()
    sizes: list[int] = []
    assert model.model.model is not None
    handle = model.model.model.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
    yield model, sizes
    handle.remove()


class TestPredictSources:
    """Exercise source inputs through the public prediction API."""

    @pytest.mark.parametrize("filename", ["screen shot.png", "screen", "screen 1", "0"])
    @pytest.mark.parametrize("source_kind", ["file", "stream", "manifest"])
    @pytest.mark.usefixtures("forbid_capture")
    def test_existing_image_precedes_capture_syntax(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, source_kind: str
    ) -> None:
        """Existing image paths take precedence over screen and camera capture syntax."""
        monkeypatch.chdir(tmp_path)
        image_dir = tmp_path / "images" if source_kind == "manifest" else tmp_path
        image_dir.mkdir(exist_ok=True)
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(image_dir / filename, format="PNG")
        source = filename
        if source_kind == "manifest":
            manifest = image_dir / "sources.txt"
            manifest.write_text(filename)
            source = str(manifest)

        result = RFDETRInference(_DummyRFDETR()).predict(source, stream=source_kind == "stream")
        if source_kind != "file":
            results = list(result)
            assert len(results) == 1
            result = results[0]

        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [10, 20, 30])

    @pytest.mark.parametrize("dirname", ["screen", "screen 1", "0", "frames.txt", "frames.csv", "cameras.streams"])
    @pytest.mark.usefixtures("forbid_capture")
    def test_existing_directory_precedes_capture_and_manifest_syntax(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirname: str
    ) -> None:
        """An existing directory expands its images even when its name spells a capture or a manifest.

        Directory names such as ``screen`` or ``frames.txt`` are ordinary on disk; classifying them by spelling would
        start a desktop or webcam capture, or try to read the directory as a manifest file.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / dirname).mkdir()
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(tmp_path / dirname / "image.png")

        results = RFDETRInference(_DummyRFDETR()).predict(dirname)

        assert isinstance(results, list)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0].metadata["source_image"][0, 0], [10, 20, 30])

    def test_missing_image_named_like_screen_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        """A missing image whose name starts with ``screen`` reports the missing file, not screen syntax.

        ``screen shot.png`` ends in an image suffix, so it names a file; a screen-argument error would hide the typo.
        """
        monkeypatch.chdir(tmp_path)
        mocker.patch.dict("sys.modules", {"mss": None})
        with pytest.raises(FileNotFoundError):
            RFDETRInference(_DummyRFDETR()).predict("screen shot.png")

    @pytest.mark.parametrize("source", ["", "   "])
    def test_empty_source_raises_before_expanding_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
    ) -> None:
        """An empty or whitespace source string raises instead of predicting the working directory.

        ``Path("")`` normalizes to ``.``, so without a guard an empty string from a missing config value would expand
        every image in the current directory and hold all results in memory.
        """
        monkeypatch.chdir(tmp_path)
        Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
        with pytest.raises(ValueError, match="source is empty"):
            RFDETRInference(_DummyRFDETR()).predict(source)

    @pytest.mark.parametrize("animated", [False, True])
    @pytest.mark.parametrize("source_kind", ["file", "directory"])
    def test_gif_predicts_first_frame(self, tmp_path: Path, animated: bool, source_kind: str) -> None:
        """GIF files retain image return types and use only the first frame."""
        path = tmp_path / "image.gif"
        first = Image.new("RGB", (32, 24), color=(10, 20, 30))
        first.save(
            path,
            save_all=animated,
            append_images=[Image.new("RGB", (32, 24), color=(40, 50, 60))] if animated else [],
        )

        result = RFDETRInference(_DummyRFDETR()).predict(path if source_kind == "file" else tmp_path)

        if source_kind == "directory":
            assert isinstance(result, list)
            assert len(result) == 1
            result = result[0]
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [10, 20, 30])

    def test_directory_batches_flush_partial_batch(
        self, tmp_path: Path, recorded_model: tuple[_DummyRFDETR, list[int]]
    ) -> None:
        """Batch inference preserves source order and emits the final partial batch."""
        for index in range(5):
            Image.new("RGB", (32, 24), color=(index, 0, 0)).save(tmp_path / f"{index}.png")
        model, sizes = recorded_model

        results = list(RFDETRInference(model).predict(tmp_path, stream=True, batch=2))

        assert sizes == [2, 2, 1]
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == list(range(5))

    def test_compiled_native_batch_rejects_partial_final_batch(self, tmp_path: Path) -> None:
        """A native predictor compiled for batch two predicts full batches, then rejects the partial final batch.

        A traced native model runs only its compiled batch size, so a directory whose length is not a multiple of
        ``batch`` must yield the full batches first and then fail with the batch-size hint, as fixed-batch exports do.
        """
        for index in range(3):
            Image.new("RGB", (32, 24), color=(index, 0, 0)).save(tmp_path / f"{index}.png")
        model = _DummyRFDETR()
        model.model.model = _ExportableIdentity()
        predictor = model.inference(batch_size=2)
        results = predictor.predict(tmp_path, stream=True, batch=2)

        pixels = [cast(sv.Detections, next(results)).metadata["source_image"][0, 0, 0] for _ in range(2)]
        with pytest.raises(ValueError, match="Batch size mismatch"):
            next(results)
        assert pixels == [0, 1]

    def test_corrupt_directory_image_fails_after_earlier_results(self, tmp_path: Path) -> None:
        """An unreadable file inside a directory raises when reached, after earlier files have produced results.

        Directory expansion filters by extension only, so a corrupt file is decoded in source order; the stream must
        surface the decoder error instead of skipping the file or failing before the readable files are predicted.
        """
        Image.new("RGB", (32, 24), color=(10, 0, 0)).save(tmp_path / "a.png")
        (tmp_path / "b.png").write_bytes(b"not an image")
        Image.new("RGB", (32, 24), color=(30, 0, 0)).save(tmp_path / "c.png")
        results = RFDETRInference(_DummyRFDETR()).predict(tmp_path, stream=True)

        first = cast(sv.Detections, next(results))
        with pytest.raises(UnidentifiedImageError):
            next(results)
        assert first.metadata["source_image"][0, 0, 0] == 10

    @pytest.mark.parametrize("stream", [False, True])
    def test_bchw_tensor_preserves_order_and_rgb(self, stream: bool) -> None:
        """BCHW tensors produce one result per RGB image."""
        images = torch.zeros((2, 3, 24, 32))
        images[0, 0] = 1
        images[1, 2] = 1
        predictions = RFDETRInference(_DummyRFDETR()).predict(images, stream=stream, batch=2)
        assert isinstance(predictions, (list, Generator))
        results = list(predictions)
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0].tolist() for result in results] == [
            [255, 0, 0],
            [0, 0, 255],
        ]

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("batch", "expected_sizes"),
        [pytest.param(None, [3], id="default-one-pass"), pytest.param(2, [2, 1], id="explicit-bound")],
    )
    def test_bchw_tensor_batch_sizes(
        self,
        recorded_model: tuple[_DummyRFDETR, list[int]],
        stream: bool,
        batch: int | None,
        expected_sizes: list[int],
    ) -> None:
        """A BCHW tensor runs as one forward pass by default and splits only at an explicit batch bound.

        The tensor already is a batch, so the default keeps it whole; an explicit ``batch`` caps memory per pass.
        """
        model, sizes = recorded_model

        results = list(RFDETRInference(model).predict(torch.zeros((3, 3, 24, 32)), stream=stream, batch=batch))

        assert sizes == expected_sizes
        assert len(results) == 3

    @pytest.mark.parametrize(
        ("batch", "expected_sizes"),
        [pytest.param(None, [5], id="default-one-pass"), pytest.param(2, [2, 2, 1], id="explicit-bound")],
    )
    def test_eager_image_list_batch_sizes(
        self, recorded_model: tuple[_DummyRFDETR, list[int]], batch: int | None, expected_sizes: list[int]
    ) -> None:
        """An eager image list runs as one pass by default and honours an explicit batch bound in source order.

        A caller passing ``batch`` to bound memory on a long list must not silently get one pass over every image.
        """
        images = [np.full((24, 32, 3), index, dtype=np.uint8) for index in range(5)]
        model, sizes = recorded_model

        results = RFDETRInference(model).predict(images, batch=batch)

        assert sizes == expected_sizes
        assert isinstance(results, list)
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == list(range(5))

    @_requires_opencv
    def test_video_stride_and_partial_batch(
        self, tmp_path: Path, recorded_model: tuple[_DummyRFDETR, list[int]], mocker: MockerFixture
    ) -> None:
        """Frame stride keeps every Nth frame from frame 0, skips the rest undecoded, and flushes a partial batch.

        Nine frames with stride 2 keep five, so batches of two end with a single frame. The kept frames match the live-
        capture reader, so one video yields the same frames on both paths.
        """
        path = tmp_path / "frames.avi"
        path.touch()
        capture = _scripted_capture([(True, np.full((24, 32, 3), index, dtype=np.uint8)) for index in range(1, 10)])
        model, sizes = recorded_model
        mocker.patch("cv2.VideoCapture", return_value=capture)

        results = list(RFDETRInference(model).predict(path, stream=True, vid_stride=2, batch=2))

        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [1, 3, 5, 7, 9]
        assert sizes == [2, 2, 1]
        assert capture.read.call_count == 5
        capture.release.assert_called_once()

    @_requires_opencv
    def test_unreadable_first_video_frame_raises(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A video that opens but cannot decode its first frame raises instead of yielding no results.

        Treating that failed read as the end of the video would make a corrupt or unsupported file look like an empty
        one. Live capture already raises in the same situation, and the capture must still be released.
        """
        path = tmp_path / "broken.avi"
        path.touch()
        capture = _scripted_capture([])
        mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(path, stream=True)
        with pytest.raises(ValueError, match="Could not read the first frame"):
            next(results)
        capture.release.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize(("source", "warnings_logged"), [("https://example.com/clip.mp4", 1), ("clip.avi", 0)])
    def test_network_video_stop_is_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, warnings_logged: int, mocker: MockerFixture
    ) -> None:
        """A network video that stops after its first frame logs a warning; a local video ends silently.

        OpenCV reports a network read timeout and the end of the file the same way, so a truncated remote video would
        otherwise look complete. A local file has no timeout, so its end needs no warning.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "clip.avi").touch()
        capture = _scripted_capture([(True, np.zeros((24, 32, 3), dtype=np.uint8))])
        mocker.patch("cv2.VideoCapture", return_value=capture)
        logger = mocker.patch("rfdetr._prediction.sources.logger")

        results = list(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))

        assert len(results) == 1
        assert logger.warning.call_count == warnings_logged

    @_requires_opencv
    @pytest.mark.parametrize("source", [0, "clip.avi"])
    def test_one_channel_model_receives_grayscale_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: int | str, mocker: MockerFixture
    ) -> None:
        """Camera and video frames reach a one-channel model as grayscale with the luminance used for image files.

        Frame sources decode RGB arrays, which preprocessing passes through unchanged, so a one-channel model used to
        reject every frame. Using the Pillow conversion keeps frame and image-file predictions comparable.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "clip.avi").touch()
        model = _DummyRFDETR()
        model.model_config.num_channels = 1
        model.means = model.means[:1]
        model.stds = model.stds[:1]
        bgr = np.full((24, 32, 3), (231, 127, 17), dtype=np.uint8)
        expected = np.array(Image.new("RGB", (32, 24), (17, 127, 231)).convert("L"))
        capture = _scripted_capture([(True, bgr), RuntimeError("end")])
        mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(model).predict(source, stream=True)
        result = next(results)
        results.close()
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"], expected)

    def test_path_image_returns_single_prediction(self, tmp_path: Path) -> None:
        """A Path image has the same return type and pixels as a string path."""
        path = tmp_path / "image.png"
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path)

        result = RFDETRInference(_DummyRFDETR()).predict(path)

        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [10, 20, 30])

    def test_stream_yields_first_image_before_later_invalid_one_is_read(self) -> None:
        """A later invalid image does not prevent the first result; it raises only when its turn comes."""
        image = Image.new("RGB", (32, 24))
        results = RFDETRInference(_DummyRFDETR()).predict([image, np.zeros((24, 32, 4), dtype=np.uint8)], stream=True)

        assert isinstance(results, Generator)
        result = next(results)
        assert isinstance(result, sv.Detections)
        assert result.metadata["source_image"].shape == (24, 32, 3)
        with pytest.raises(ValueError, match="Invalid tensor image shape"):
            next(results)

    @pytest.mark.parametrize(
        "registered,openers,expected",
        [
            pytest.param({".heic": "HEIF"}, {"HEIF": (object(), None)}, 2, id="plugin-registered"),
            pytest.param({".heic": "HEIF"}, {}, 1, id="extension-without-opener"),
            pytest.param({}, {}, 1, id="no-plugin"),
        ],
    )
    def test_directory_includes_heif_only_when_pillow_can_open_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        registered: dict[str, str],
        openers: dict[str, object],
        expected: int,
    ) -> None:
        """Directory expansion picks up ``.heic`` only while a Pillow plugin has registered a HEIF opener.

        Without pillow-heif, Pillow cannot open HEIF, and one such file would stop an eager directory run after its
        earlier results. ``.dng`` is never listed because Pillow has no DNG opener. Pillow identifies files by their
        content, so PNG bytes saved under a ``.heic`` name decode without the plugin.
        """
        Image.new("RGB", (32, 24)).save(tmp_path / "a.png")
        Image.new("RGB", (32, 24)).save(tmp_path / "b.heic", format="PNG")
        Image.new("RGB", (32, 24)).save(tmp_path / "c.dng", format="PNG")
        monkeypatch.setattr(Image, "registered_extensions", lambda: registered)
        # Another test in the session may have registered a real HEIF opener; start every case without one.
        monkeypatch.delitem(Image.OPEN, "HEIF", raising=False)
        monkeypatch.setattr(Image, "OPEN", {**Image.OPEN, **openers})

        results = RFDETRInference(_DummyRFDETR()).predict(tmp_path)

        assert isinstance(results, list)
        assert len(results) == expected

    @pytest.mark.parametrize("source_kind", ["directory", "glob"])
    def test_collections_are_sorted_and_skip_nonmedia_files(self, tmp_path: Path, source_kind: str) -> None:
        """Directories and globs produce an ordered list of image predictions."""
        Image.new("RGB", (32, 24), color=(0, 0, 255)).save(tmp_path / "b.PNG")
        Image.new("RGB", (32, 24), color=(255, 0, 0)).save(tmp_path / "a.png")
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "nested").mkdir()
        Image.new("RGB", (32, 24)).save(tmp_path / "nested" / "ignored.png")
        source = tmp_path if source_kind == "directory" else str(tmp_path / "*.*")

        results = RFDETRInference(_DummyRFDETR()).predict(source)

        assert isinstance(results, list)
        pixels = []
        for result in results:
            assert isinstance(result, sv.Detections)
            pixels.append(result.metadata["source_image"][0, 0].tolist())
        assert pixels == [[255, 0, 0], [0, 0, 255]]

    @_requires_opencv
    @pytest.mark.parametrize("stream", [False, True])
    def test_video_predicts_all_frames_in_rgb(self, tmp_path: Path, stream: bool) -> None:
        """A real video yields RGB predictions in frame order."""
        import cv2  # optional dependency, guarded by the test's skipif

        path = tmp_path / "frames.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 5, (32, 24))
        if not writer.isOpened():
            writer.release()
            pytest.skip("The OpenCV build has no MJPG encoder.")
        try:
            writer.write(np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8))
            writer.write(np.full((24, 32, 3), (255, 0, 0), dtype=np.uint8))
        finally:
            writer.release()

        predictions = RFDETRInference(_DummyRFDETR()).predict(path, stream=stream)
        assert isinstance(predictions, (list, Generator))
        results = list(predictions)

        assert len(results) == 2
        assert isinstance(results[0], sv.Detections)
        assert isinstance(results[1], sv.Detections)
        np.testing.assert_allclose(results[0].metadata["source_image"][0, 0], [255, 0, 0], atol=5)
        np.testing.assert_allclose(results[1].metadata["source_image"][0, 0], [0, 0, 255], atol=5)

    @_requires_opencv
    @pytest.mark.parametrize(
        "source",
        [
            0,
            "0",
            "rtsp://camera.example/live",
            "rtsps://camera.example/live",
            "rtmp://camera.example/live",
            "tcp://camera.example/live",
            "udp://@239.0.0.1:1234",
            "srt://camera.example:9000?mode=caller",
            "http://camera.example/live.mjpg",
        ],
    )
    def test_live_capture_is_lazy_and_closes_on_early_exit(self, source: int | str, mocker: MockerFixture) -> None:
        """Live sources open on demand, convert RGB, and close explicitly."""
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8))]
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(source, stream=True)
        assert isinstance(results, Generator)
        open_capture.assert_not_called()

        result = next(results)
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [30, 20, 10])
        open_capture.assert_called_once()
        assert open_capture.call_args.args[0] == (0 if source == "0" else source)
        assert not torch.is_inference_mode_enabled()
        results.close()

        capture.release.assert_called_once()

    def test_source_image_mutation_does_not_modify_input(self) -> None:
        """Annotating retained source pixels must not modify the caller's image."""
        image = np.full((24, 32, 3), 127, dtype=np.uint8)

        result = RFDETRInference(_DummyRFDETR()).predict(image)

        assert isinstance(result, sv.Detections)
        result.metadata["source_image"][:] = 0
        np.testing.assert_array_equal(image, np.full((24, 32, 3), 127, dtype=np.uint8))

    @_requires_opencv
    @pytest.mark.parametrize("source", [0, "rtsp://camera.example/live"])
    def test_eager_live_capture_warns_before_opening(self, source: int | str, mocker: MockerFixture) -> None:
        """Eager live inference is allowed with a memory warning."""
        capture = mocker.MagicMock()
        capture.isOpened.return_value = False
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        with pytest.warns(UserWarning, match="accumulate in memory"):
            with pytest.raises(ValueError, match="Could not open"):
                RFDETRInference(_DummyRFDETR()).predict(source)
        open_capture.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize("option", ["batch", "vid_stride"])
    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_invalid_source_options_fail_before_capture(
        self, option: str, value: object, mocker: MockerFixture
    ) -> None:
        """Invalid batching and stride options fail without opening a device."""
        open_capture = mocker.patch("cv2.VideoCapture")
        with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
            RFDETRInference(_DummyRFDETR()).predict(0, stream=True, **cast(Any, {option: value}))
        open_capture.assert_not_called()

    def test_stream_buffer_requires_boolean(self) -> None:
        """A misspelled buffer value must not enable buffering implicitly."""
        with pytest.raises(ValueError, match="stream_buffer must be a boolean"):
            RFDETRInference(_DummyRFDETR()).predict([], stream_buffer=cast(Any, "false"))

    @_requires_opencv
    def test_video_glob_expands_before_capture(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A video glob opens each matching file, rather than the pattern."""
        path = tmp_path / "clip.avi"
        path.touch()
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(str(tmp_path / "*.avi"))

        assert isinstance(results, list)
        assert len(results) == 1
        open_capture.assert_called_once_with(str(path))
        capture.release.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize(
        ("capture_attrs", "options", "error", "message"),
        [
            pytest.param(
                {"read.return_value": (True, np.zeros((24, 32, 3), dtype=np.uint8))},
                {"shape": (13, 14)},
                ValueError,
                "divisible",
                id="invalid-prediction-shape",
            ),
            pytest.param(
                {"read.side_effect": RuntimeError("decoder failed")},
                {},
                RuntimeError,
                "decoder failed",
                id="decoder-error",
            ),
            pytest.param(
                {"isOpened.return_value": False}, {}, ValueError, "Could not open the video source", id="open-failure"
            ),
        ],
    )
    def test_capture_released_after_error(
        self,
        capture_attrs: dict[str, Any],
        options: dict[str, Any],
        error: type[Exception],
        message: str,
        mocker: MockerFixture,
    ) -> None:
        """A failed prediction, decoder read or device open releases the capture and surfaces a readable error."""
        capture = mocker.MagicMock()
        capture.configure_mock(**capture_attrs)
        mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(0, stream=True, **options)
        with pytest.raises(error, match=message):
            next(results)

        capture.release.assert_called_once()

    @_requires_opencv
    @pytest.mark.parametrize(
        "source,message",
        [
            pytest.param(-1, "non-negative integer", id="negative"),
            pytest.param(2**31, "no greater than 2147483647", id="above-int32"),
            pytest.param(True, "not a boolean", id="bool"),
            pytest.param(np.int64(-1), "non-negative integer", id="numpy-negative"),
            pytest.param(np.uint64(2**32), "no greater than 2147483647", id="numpy-above-int32"),
        ],
    )
    def test_invalid_camera_index_does_not_open_capture(self, source: int, message: str, mocker: MockerFixture) -> None:
        """Boolean, negative, and out-of-range indexes cannot select a camera."""
        open_capture = mocker.patch("cv2.VideoCapture")
        with pytest.raises(ValueError, match=message):
            next(RFDETRInference(_DummyRFDETR()).predict(source, stream=True))
        open_capture.assert_not_called()

    @_requires_opencv
    @pytest.mark.parametrize(
        "source,index",
        [
            pytest.param("2147483647", 2**31 - 1, id="string-int32-max"),
            pytest.param("007", 7, id="string-leading-zeros"),
            pytest.param(np.uint32(2**31 - 1), 2**31 - 1, id="numpy-int32-max"),
            pytest.param(np.int64(1), 1, id="numpy-int64"),
        ],
    )
    def test_camera_index_within_bounds_opens_capture(self, source: object, index: int, mocker: MockerFixture) -> None:
        """ASCII digit strings and NumPy integers up to the 32-bit maximum open that camera index as a Python ``int``.

        OpenCV encodes backend selection as offsets such as ``700 + n``, so the bound stays at the int32 range rather
        than at a count of attached cameras. Indexes often come from NumPy arrays; treating ``np.int64`` as image data
        would fail far from the call site.
        """
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8))]
        open_capture = mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict(cast(Any, source), stream=True)
        next(results)
        results.close()

        assert type(open_capture.call_args.args[0]) is int
        assert open_capture.call_args.args[0] == index

    @_requires_opencv
    @pytest.mark.parametrize("source", ["2147483648", pytest.param("9" * 30, id="thirty-digits")])
    def test_out_of_range_camera_string_raises_before_capture(self, source: str, mocker: MockerFixture) -> None:
        """A digit string beyond the 32-bit index range raises instead of opening a capture."""
        open_capture = mocker.patch("cv2.VideoCapture")
        with pytest.raises(ValueError, match="no greater than 2147483647"):
            RFDETRInference(_DummyRFDETR()).predict(source)
        open_capture.assert_not_called()

    @_requires_opencv
    @pytest.mark.parametrize(
        "source",
        [
            "1.0",
            "-1",
            "+1",
            pytest.param(" 0", id="leading-space"),
            pytest.param("0 ", id="trailing-space"),
            pytest.param("٠", id="arabic-indic-zero"),
            pytest.param("１", id="fullwidth-one"),
            pytest.param("²", id="superscript-two"),
        ],
    )
    def test_non_ascii_digit_spelling_is_looked_up_as_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, mocker: MockerFixture
    ) -> None:
        """Only plain ASCII digits name a camera; other numeric spellings are image paths.

        Unicode digits, signs, decimals and padded strings are plausible file names. Treating them as camera indexes
        would open a device the caller never asked for, so a missing file reports ``FileNotFoundError`` instead.
        """
        monkeypatch.chdir(tmp_path)
        mocker.patch("cv2.VideoCapture", side_effect=AssertionError("unexpected capture"))
        with pytest.raises(FileNotFoundError):
            RFDETRInference(_DummyRFDETR()).predict(source)

    @_requires_opencv
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("source_kind", ["directory", "glob", "video"])
    def test_missing_media_raises_file_not_found(
        self, tmp_path: Path, source_kind: str, stream: bool, mocker: MockerFixture
    ) -> None:
        """An empty directory or glob and a missing video file raise before capture, eager or streamed.

        An empty match almost always means a wrong path, so the empty-source policy makes it loud in both modes instead
        of returning no results.
        """
        source = {"directory": tmp_path, "glob": tmp_path / "*.png", "video": tmp_path / "missing.mp4"}[source_kind]
        open_capture = mocker.patch("cv2.VideoCapture")
        with pytest.raises(FileNotFoundError):
            list(RFDETRInference(_DummyRFDETR()).predict(source, stream=stream))
        open_capture.assert_not_called()

    @_requires_opencv
    def test_mixed_collection_preserves_input_and_frame_order(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A list containing images and video expands into one ordered result list."""
        path = tmp_path / "clip.avi"
        path.touch()
        capture = mocker.MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), 20, dtype=np.uint8)), (False, None)]
        first = np.full((24, 32, 3), 10, dtype=np.uint8)
        last = np.full((24, 32, 3), 30, dtype=np.uint8)
        mocker.patch("cv2.VideoCapture", return_value=capture)
        results = RFDETRInference(_DummyRFDETR()).predict([first, path, last])

        assert isinstance(results, list)
        pixels = []
        for result in results:
            assert isinstance(result, sv.Detections)
            pixels.append(result.metadata["source_image"][0, 0, 0])
        assert pixels == [10, 20, 30]

    @pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
    def test_stream_preserves_prediction_type_and_options(self, task: str) -> None:
        """Streaming uses the same threshold, output fields, and source-image option."""
        model = _DummyRFDETR()
        model.model = cast(
            "ModelContext", _DummyModel(include_masks=task == "segmentation", include_keypoints=task == "keypoints")
        )
        results = RFDETRInference(model).predict(
            Image.new("RGB", (32, 24)),
            threshold=0.95,
            include_source_image=False,
            shape=(28, 56),
            patch_size=14,
            stream=True,
        )
        result = next(results)
        results.close()

        assert len(result) == 0
        if task == "keypoints":
            assert isinstance(result, sv.KeyPoints)
            assert result.xy.shape == (0, 17, 2)
            assert "source_image" not in result.data
        else:
            assert isinstance(result, sv.Detections)
            assert "source_image" not in result.metadata
            if task == "segmentation":
                assert result.mask is not None
                assert result.mask.shape == (0, 4, 4)

    def test_empty_stream_has_no_results(self) -> None:
        """An empty image sequence produces an empty stream."""
        assert list(RFDETRInference(_DummyRFDETR()).predict([], stream=True)) == []


def test_predict_accepts_image_url() -> None:
    """A remote image URL is fetched and predicted as one image; skipped when the host is unreachable."""
    if not is_online(_HTTP_HOST, _HTTP_PORT):
        pytest.skip("Offline environment, skipping HTTP predict URL test.")
    model = _DummyRFDETR()
    detections = model.predict(_HTTP_IMAGE_URL)
    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)


class TestPredictURLFetch:
    """``predict()`` URL handling: robust HTTP fetch and correct local-path classification."""

    def test_http_error_status_raises_httperror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 404 response surfaces as ``requests.HTTPError``, not an opaque PIL error."""
        model = _DummyRFDETR()

        def _fake_get(url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse(content=b"not found", status_code=404)

        monkeypatch.setattr(requests, "get", _fake_get)
        with pytest.raises(requests.HTTPError):
            model.predict("http://example.com/missing.jpg")

    def test_timeout_is_passed_to_requests_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The HTTP fetch must pass an explicit ``timeout`` so it cannot hang forever."""
        model = _DummyRFDETR()
        captured: dict[str, object] = {}

        def _fake_get(url: str, **kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse(content=_png_bytes(), status_code=200)

        monkeypatch.setattr(requests, "get", _fake_get)
        detections = model.predict("http://example.com/image.png")
        assert isinstance(detections, sv.Detections)
        assert captured.get("timeout") == 30, "predict() must pass timeout=30 to requests.get"

    def test_local_file_named_like_http_is_not_fetched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A local file whose name starts with ``http`` must be opened, not fetched over HTTP."""
        img_path = tmp_path / "httpcam_frame.png"
        Image.new("RGB", (28, 28), (128, 128, 128)).save(str(img_path))
        model = _DummyRFDETR()

        def _fail_get(url: str, **kwargs: object) -> _FakeResponse:
            raise AssertionError(f"requests.get must not be called for local path {url!r}")

        monkeypatch.setattr(requests, "get", _fail_get)
        detections = model.predict(str(img_path))
        assert isinstance(detections, sv.Detections)


class TestHasOnlyFiniteImages:
    """Sources holding only image files, directories and globs are finite; any camera, video or manifest is not."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            pytest.param("images", True, id="directory"),
            pytest.param("images/*.png", True, id="glob"),
            pytest.param("images/a.png", True, id="file"),
            pytest.param(["images/a.png", "images"], True, id="list-of-files"),
            pytest.param(0, False, id="camera"),
            pytest.param(["images/a.png", 0], False, id="list-with-camera"),
            pytest.param("sources.txt", False, id="manifest"),
            pytest.param("clip.mp4", False, id="video"),
            pytest.param("rtsp://host/stream", False, id="live"),
        ],
    )
    def test_has_only_finite_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: Any, expected: bool
    ) -> None:
        """A source is finite only when it holds no camera, video, live stream or manifest."""
        (tmp_path / "images").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / "a.png")
        (tmp_path / "sources.txt").write_text("images/a.png\n")
        (tmp_path / "clip.mp4").write_bytes(b"")
        monkeypatch.chdir(tmp_path)

        assert has_only_finite_images(source) is expected
