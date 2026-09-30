# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Prediction contracts for file collections and streams."""

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from .helpers import _DummyModel, _DummyRFDETR

if TYPE_CHECKING:
    from rfdetr.inference import ModelContext


class TestPredictSources:
    """Exercise source inputs through the public prediction API."""

    def test_prediction_input_has_public_import(self) -> None:
        """Wrappers can annotate prediction inputs without private imports."""
        from rfdetr import PredictionInput
        from rfdetr.prediction import PredictionInput as PublicPredictionInput

        assert PredictionInput is PublicPredictionInput

    def test_directory_batches_flush_partial_batch(self, tmp_path: Path) -> None:
        """Batch inference preserves source order and emits the final partial batch."""
        for index in range(5):
            Image.new("RGB", (32, 24), color=(index, 0, 0)).save(tmp_path / f"{index}.png")
        model = _DummyRFDETR()
        sizes = []
        assert model.model.model is not None
        handle = model.model.model.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
        try:
            results = list(model.predict(tmp_path, stream=True, batch=2))
        finally:
            handle.remove()
        assert sizes == [2, 2, 1]
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == list(range(5))

    @pytest.mark.parametrize("stream", [False, True])
    def test_bchw_tensor_preserves_order_and_rgb(self, stream: bool) -> None:
        """BCHW tensors produce one result per RGB image."""
        images = torch.zeros((2, 3, 24, 32))
        images[0, 0] = 1
        images[1, 2] = 1
        predictions = _DummyRFDETR().predict(images, stream=stream, batch=2)
        assert isinstance(predictions, (list, Generator))
        results = list(predictions)
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0].tolist() for result in results] == [
            [255, 0, 0],
            [0, 0, 255],
        ]

    def test_video_stride_and_partial_batch(self, tmp_path: Path) -> None:
        """Frame stride selects every Nth frame and retains a partial final batch."""
        path = tmp_path / "frames.avi"
        path.touch()
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), index, dtype=np.uint8)) for index in range(1, 8)] + [
            (False, None)
        ]
        with patch("cv2.VideoCapture", return_value=capture):
            results = list(_DummyRFDETR().predict(path, stream=True, vid_stride=2, batch=2))
        assert [cast(sv.Detections, result).metadata["source_image"][0, 0, 0] for result in results] == [2, 4, 6]
        capture.release.assert_called_once()

    def test_path_image_returns_single_prediction(self, tmp_path: Path) -> None:
        """A Path image has the same return type and pixels as a string path."""
        path = tmp_path / "image.png"
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path)

        result = _DummyRFDETR().predict(path)

        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.metadata["source_image"][0, 0], [10, 20, 30])

    def test_stream_predicts_each_image_only_when_requested(self) -> None:
        """A later invalid image does not prevent the first result."""
        image = Image.new("RGB", (32, 24))
        results = _DummyRFDETR().predict([image, np.zeros((24, 32, 4), dtype=np.uint8)], stream=True)

        assert isinstance(results, Generator)
        result = next(results)
        assert isinstance(result, sv.Detections)
        assert result.metadata["source_image"].shape == (24, 32, 3)
        with pytest.raises(ValueError, match="Invalid tensor image shape"):
            next(results)

    @pytest.mark.parametrize("source_kind", ["directory", "glob"])
    def test_collections_are_sorted_and_skip_nonmedia_files(self, tmp_path: Path, source_kind: str) -> None:
        """Directories and globs produce an ordered list of image predictions."""
        Image.new("RGB", (32, 24), color=(0, 0, 255)).save(tmp_path / "b.PNG")
        Image.new("RGB", (32, 24), color=(255, 0, 0)).save(tmp_path / "a.png")
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "nested").mkdir()
        Image.new("RGB", (32, 24)).save(tmp_path / "nested" / "ignored.png")
        source = tmp_path if source_kind == "directory" else str(tmp_path / "*.*")

        results = _DummyRFDETR().predict(source)

        assert isinstance(results, list)
        pixels = []
        for result in results:
            assert isinstance(result, sv.Detections)
            pixels.append(result.metadata["source_image"][0, 0].tolist())
        assert pixels == [[255, 0, 0], [0, 0, 255]]

    @pytest.mark.parametrize("stream", [False, True])
    def test_video_predicts_all_frames_in_rgb(self, tmp_path: Path, stream: bool) -> None:
        """A real video yields RGB predictions in frame order."""
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

        predictions = _DummyRFDETR().predict(path, stream=stream)
        assert isinstance(predictions, (list, Generator))
        results = list(predictions)

        assert len(results) == 2
        assert isinstance(results[0], sv.Detections)
        assert isinstance(results[1], sv.Detections)
        np.testing.assert_allclose(results[0].metadata["source_image"][0, 0], [255, 0, 0], atol=5)
        np.testing.assert_allclose(results[1].metadata["source_image"][0, 0], [0, 0, 255], atol=5)

    @pytest.mark.parametrize(
        "source",
        [
            0,
            "0",
            "rtsp://camera.example/live",
            "rtsps://camera.example/live",
            "rtmp://camera.example/live",
            "tcp://camera.example/live",
            "http://camera.example/live.mjpg",
        ],
    )
    def test_live_capture_is_lazy_and_closes_on_early_exit(self, source: int | str) -> None:
        """Live sources open on demand, convert RGB, and close explicitly."""
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8))]
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = _DummyRFDETR().predict(source, stream=True)
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

        result = _DummyRFDETR().predict(image)

        assert isinstance(result, sv.Detections)
        result.metadata["source_image"][:] = 0
        np.testing.assert_array_equal(image, np.full((24, 32, 3), 127, dtype=np.uint8))

    @pytest.mark.parametrize("source", [0, "rtsp://camera.example/live"])
    def test_eager_live_capture_warns_before_opening(self, source: int | str) -> None:
        """Eager live inference is allowed with a memory warning."""
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            with pytest.warns(UserWarning, match="accumulate in memory"):
                with pytest.raises(ValueError, match="Could not open"):
                    _DummyRFDETR().predict(source)
            open_capture.assert_called_once()

    @pytest.mark.parametrize("option", ["batch", "vid_stride"])
    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_invalid_source_options_fail_before_capture(self, option: str, value: object) -> None:
        """Invalid batching and stride options fail without opening a device."""
        with patch("cv2.VideoCapture") as open_capture:
            with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
                _DummyRFDETR().predict(0, stream=True, **cast(Any, {option: value}))
            open_capture.assert_not_called()

    def test_stream_buffer_requires_boolean(self) -> None:
        """A misspelled buffer value must not enable buffering implicitly."""
        with pytest.raises(ValueError, match="stream_buffer must be a boolean"):
            _DummyRFDETR().predict([], stream_buffer=cast(Any, "false"))

    def test_video_glob_expands_before_capture(self, tmp_path: Path) -> None:
        """A video glob opens each matching file, rather than the pattern."""
        path = tmp_path / "clip.avi"
        path.touch()
        capture = MagicMock()
        capture.read.side_effect = [(True, np.zeros((24, 32, 3), dtype=np.uint8)), (False, None)]
        with patch("cv2.VideoCapture", return_value=capture) as open_capture:
            results = _DummyRFDETR().predict(str(tmp_path / "*.avi"))

        assert isinstance(results, list)
        assert len(results) == 1
        open_capture.assert_called_once_with(str(path))
        capture.release.assert_called_once()

    def test_capture_closes_after_prediction_error(self) -> None:
        """An invalid prediction shape releases the active video capture."""
        capture = MagicMock()
        capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, shape=(13, 14), stream=True)
            with pytest.raises(ValueError, match="divisible"):
                next(results)

        capture.release.assert_called_once()

    def test_capture_closes_after_read_error(self) -> None:
        """A decoder error releases the capture and reaches the caller."""
        capture = MagicMock()
        capture.read.side_effect = RuntimeError("decoder failed")
        with patch("cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True)
            with pytest.raises(RuntimeError, match="decoder failed"):
                next(results)

        capture.release.assert_called_once()

    def test_capture_open_failure_has_clear_error_and_releases_capture(self) -> None:
        """An unavailable source raises a readable error and closes its capture."""
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict(0, stream=True)
            with pytest.raises(ValueError, match="Could not open the video source"):
                next(results)

        capture.release.assert_called_once()

    @pytest.mark.parametrize("source", [-1, True])
    def test_invalid_camera_index_does_not_open_capture(self, source: int) -> None:
        """Boolean and negative indexes cannot select a camera."""
        with patch("cv2.VideoCapture") as open_capture:
            with pytest.raises(ValueError, match="non-negative integer"):
                next(_DummyRFDETR().predict(source, stream=True))
            open_capture.assert_not_called()

    @pytest.mark.parametrize("source_kind", ["directory", "glob", "video"])
    def test_missing_media_has_clear_error(self, tmp_path: Path, source_kind: str) -> None:
        """Empty collections and missing video files raise before capture."""
        source = {"directory": tmp_path, "glob": tmp_path / "*.png", "video": tmp_path / "missing.mp4"}[source_kind]
        with patch("cv2.VideoCapture") as open_capture:
            with pytest.raises(FileNotFoundError):
                next(_DummyRFDETR().predict(source, stream=True))
            open_capture.assert_not_called()

    def test_mixed_collection_preserves_input_and_frame_order(self, tmp_path: Path) -> None:
        """A list containing images and video expands into one ordered result list."""
        path = tmp_path / "clip.avi"
        path.touch()
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((24, 32, 3), 20, dtype=np.uint8)), (False, None)]
        first = np.full((24, 32, 3), 10, dtype=np.uint8)
        last = np.full((24, 32, 3), 30, dtype=np.uint8)
        with patch("cv2.VideoCapture", return_value=capture):
            results = _DummyRFDETR().predict([first, path, last])

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
        results = model.predict(
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
        assert list(_DummyRFDETR().predict([], stream=True)) == []
