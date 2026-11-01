# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the batch scheduling in :mod:`rfdetr._prediction.pipelines`."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr._prediction import core, pipelines
from rfdetr._prediction.sources import has_only_finite_images
from rfdetr.inference import RFDETRInference
from tests._markers import _requires_cuda, _requires_mps

from ..helpers import _DummyRFDETR, _image_dir, _source_pixels


class TestDecodeAhead:
    """Finite file sources decode the next batch on worker threads while the current batch runs."""

    @staticmethod
    def _recording_batches(batches: list[list[Any]], pulls: list[int], error: Exception | None = None):
        """Yield ``batches`` and record how many have been pulled, optionally raising after the last one.

        Examples:
            >>> pulls = []
            >>> list(TestDecodeAhead._recording_batches([["a"], ["b"]], pulls)), pulls
            ([['a'], ['b']], [1, 2])
        """
        for batch in batches:
            pulls.append(len(pulls) + 1)
            yield batch
        if error is not None:
            raise error

    def test_next_batch_is_pulled_and_submitted_before_the_current_one_is_used(self, tmp_path: Path) -> None:
        """When batch k reaches the caller, batch k+1 has already left the source and its decode was submitted."""
        paths = []
        for name in ("a.jpg", "b.png", "c.jpg"):
            Image.new("RGB", (8, 6), (1, 2, 3)).save(tmp_path / name)
            paths.append(str(tmp_path / name))
        pulls: list[int] = []
        with ThreadPoolExecutor(2) as pool:
            batches = pipelines._decode_ahead(self._recording_batches([paths[:2], paths[2:]], pulls), pool)
            first = next(batches)
            assert pulls == [1, 2]
            second = next(batches)
            assert next(batches, None) is None
        assert [item.path for item in first + second] == paths
        assert all(isinstance(item, core._PrefetchedImage) for item in first + second)
        assert all(item.decoded.result().shape == (3, 6, 8) for item in first + second)

    def test_non_file_items_pass_through(self) -> None:
        """In-memory images and URLs are not decoded ahead; they keep their own path through ``_predict_batch``."""
        image = Image.new("RGB", (4, 4))
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        url = "https://example.com/image.jpg"
        with ThreadPoolExecutor(1) as pool:
            (batch,) = pipelines._decode_ahead(iter([[image, array, url]]), pool)
        assert batch[0] is image and batch[1] is array and batch[2] == url

    def test_source_error_is_raised_after_the_current_batch(self) -> None:
        """A source that fails while the next batch is read still hands over the batch before it first."""
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        with ThreadPoolExecutor(1) as pool:
            batches = pipelines._decode_ahead(self._recording_batches([[array]], [], FileNotFoundError("gone")), pool)
            assert next(batches)[0] is array
            with pytest.raises(FileNotFoundError, match="gone"):
                next(batches)

    def test_single_image_batches_keep_every_decode_worker_busy(self) -> None:
        """At batch size one, the first batch reaches the caller with one decode per worker already read behind it.

        Reading one batch ahead left all but one decode worker idle while the model ran a single image.
        """
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        pulls: list[int] = []
        with ThreadPoolExecutor(1) as pool:
            batches = pipelines._decode_ahead(self._recording_batches([[array]] * 8, pulls), pool)
            next(batches)
            assert len(pulls) == 1 + pipelines._DECODE_WORKERS

    def test_source_error_is_raised_after_every_batch_read_before_it(self) -> None:
        """A source failing while several batches are read ahead still hands over each of them before raising."""
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        with ThreadPoolExecutor(1) as pool:
            batches = pipelines._decode_ahead(
                self._recording_batches([[array]] * 3, [], FileNotFoundError("gone")), pool
            )
            assert len([next(batches), next(batches), next(batches)]) == 3
            with pytest.raises(FileNotFoundError, match="gone"):
                next(batches)

    @staticmethod
    def _decode_threads(
        mocker: MockerFixture, root: Path, device: str, backend: str = "pytorch"
    ) -> tuple[list[list[int]], list[str]]:
        """Predict five PNG files from ``root`` in batches of two and return their source pixels and decode threads.

        ``backend`` is the runtime the prediction context reports; any value but ``"pytorch"`` stands for an exported
        runtime.

        Examples:
            Needs the pytest-mock ``mocker`` fixture, which a standalone doctest cannot create.

            >>> TestDecodeAhead._decode_threads(mocker, tmp_path, "cpu")  # doctest: +SKIP
            ([[0, 20, 30], [30, 20, 30], [60, 20, 30], [90, 20, 30], [120, 20, 30]], ['MainThread', ...])
        """
        for index in range(5):
            # PNG keeps the pixel exact; JPEG would round it.
            Image.new("RGB", (32, 24), (index * 30, 20, 30)).save(root / f"{index}.png")
        threads: list[str] = []
        original = core._decode_local_image

        def recording_decode(path: str) -> torch.Tensor:
            """Record the decoding thread, then decode as the real function does."""
            threads.append(threading.current_thread().name)
            return original(path)

        model = _DummyRFDETR()
        model.model.device = torch.device(device)
        native_context = model._prediction_context

        def reported_context() -> core.PredictionContext:
            """Return the native context, reporting ``backend`` as its runtime."""
            return dataclasses.replace(native_context(), runtime_info={"backend": backend, "device": device})

        # Decoded ahead by this module, or inline by the batch itself when the model runs on the CPU.
        mocker.patch.object(pipelines, "_decode_local_image", side_effect=recording_decode)
        mocker.patch.object(core, "_decode_local_image", side_effect=recording_decode)
        mocker.patch.object(model, "_prediction_context", side_effect=reported_context)
        results = RFDETRInference(model).predict(root, batch=2)
        return _source_pixels(results), threads

    @_requires_mps
    def test_accelerator_directory_predictions_match_and_decode_off_the_main_thread(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """On an accelerator, a directory returns the same images in the same order, decoded by worker threads."""
        pixels, threads = self._decode_threads(mocker, tmp_path, "mps")

        assert pixels == [[index * 30, 20, 30] for index in range(5)]
        assert len(threads) == 5
        assert threading.main_thread().name not in threads

    def test_cpu_model_decodes_on_the_main_thread(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A CPU forward pass already uses every core, so decode threads would compete with it."""
        pixels, threads = self._decode_threads(mocker, tmp_path, "cpu")

        assert pixels == [[index * 30, 20, 30] for index in range(5)]
        assert threads == [threading.main_thread().name] * 5

    def test_exported_runtime_decodes_off_the_main_thread(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """An exported runtime reports a CPU context whatever hardware it runs on, and still decodes ahead."""
        pixels, threads = self._decode_threads(mocker, tmp_path, "cpu", backend="coreml")

        assert pixels == [[index * 30, 20, 30] for index in range(5)]
        assert len(threads) == 5
        assert threading.main_thread().name not in threads

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
    def test_only_finite_image_sources_decode_ahead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: Any, expected: bool
    ) -> None:
        """Live, video and manifest sources keep reading one batch at a time, so a live frame is never read early."""
        (tmp_path / "images").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / "a.png")
        (tmp_path / "sources.txt").write_text("images/a.png\n")
        (tmp_path / "clip.mp4").write_bytes(b"")
        monkeypatch.chdir(tmp_path)

        assert has_only_finite_images(source) is expected


class TestPreparedAhead:
    """On CUDA, a finite source builds each batch's model input while the batch before it runs."""

    @staticmethod
    def _recording_prepare(events: list[str], failing: str | None = None) -> Callable[[list[str]], SimpleNamespace]:
        """Return a ``prepare`` stand-in recording each batch it builds, whose ``run`` records the forward pass.

        Each batch holds one label, which stands in for the batch tensor; the label ``failing`` raises ``ValueError``.

        Examples:
            >>> events = []
            >>> prepared = TestPreparedAhead._recording_prepare(events)(["a"])
            >>> prepared.context.run(prepared.batch_tensor), events
            ({}, ['prepare a', 'run a'])
        """

        def run(batch: str) -> dict[str, torch.Tensor]:
            """Record the forward pass of ``batch``."""
            events.append(f"run {batch}")
            return {}

        def prepare(images: list[str]) -> SimpleNamespace:
            """Record building ``images``, raising for the ``failing`` label."""
            (label,) = images
            events.append(f"prepare {label}")
            if label == failing:
                raise ValueError(f"bad image {label}")
            return SimpleNamespace(context=SimpleNamespace(run=run), batch_tensor=label)

        return prepare

    def test_native_model_prepares_next_batch_between_forward_and_decode(self, mocker: MockerFixture) -> None:
        """A native forward pass is queued, the next batch is prepared, and only then do the results wait for it.

        Post-processing is the first step that waits for the device, so preparing the next batch before it keeps the
        host busy while the device runs the forward pass.
        """
        events: list[str] = []
        mocker.patch.object(
            pipelines,
            "_decode_batch",
            side_effect=lambda prepared, _: events.append(f"decode {prepared.batch_tensor}") or [prepared.batch_tensor],
        )
        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"]]), self._recording_prepare(events), torch.device("cuda"), in_worker=False
        )

        assert list(results) == ["a", "b"]
        assert events == ["prepare a", "run a", "prepare b", "decode a", "run b", "decode b"]

    def test_error_preparing_next_batch_is_raised_after_current_results(self, mocker: MockerFixture) -> None:
        """An invalid image in the next batch still lets the current batch's results reach the caller first.

        The serial loop raised it when it reached that batch, after yielding every result before it.
        """
        mocker.patch.object(pipelines, "_decode_batch", side_effect=lambda prepared, _: [prepared.batch_tensor])
        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"]]), self._recording_prepare([], failing="b"), torch.device("cuda"), in_worker=False
        )

        assert next(results) == "a"
        with pytest.raises(ValueError, match="bad image b"):
            next(results)

    @pytest.mark.parametrize(
        "device,backend,expected",
        [
            pytest.param("cuda", "pytorch", [False], id="cuda-native"),
            pytest.param("cuda", "tensorrt", [True], id="cuda-export"),
            pytest.param("cpu", "pytorch", [], id="cpu-native"),
            pytest.param("cpu", "onnxruntime", [], id="cpu-export"),
            pytest.param("mps", "pytorch", [], id="mps-native"),
        ],
    )
    def test_only_cuda_contexts_prepare_ahead(
        self, tmp_path: Path, mocker: MockerFixture, device: str, backend: str, expected: list[bool]
    ) -> None:
        """Only a CUDA context prepares ahead, on a worker for an exported runtime; every other device runs serially.

        The prediction functions are replaced, so no device is touched: this checks the routing alone.
        """
        Image.new("RGB", (8, 6)).save(tmp_path / "a.png")
        context = dataclasses.replace(
            _DummyRFDETR()._prediction_context(),
            device=torch.device(device),
            runtime_info={"backend": backend, "device": device},
        )
        in_worker_flags: list[bool] = []

        def predict_prepared_ahead(*args: Any, in_worker: bool) -> Any:
            """Record the worker choice and predict nothing."""
            in_worker_flags.append(in_worker)
            return iter([])

        mocker.patch.object(pipelines, "_predict_prepared_ahead", side_effect=predict_prepared_ahead)
        mocker.patch.object(pipelines, "_predict_batch", return_value=[])
        results = pipelines._predict_stream(
            lambda: context,
            str(tmp_path),
            0.5,
            None,
            None,
            True,
            antialias=None,
            batch=1,
            vid_stride=1,
            stream_buffer=False,
            warn_on_live=False,
        )

        assert list(results) == []
        assert in_worker_flags == expected

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("backend", ["pytorch", "tensorrt"])
    def test_cuda_directory_matches_one_serial_batch(self, tmp_path: Path, mocker: MockerFixture, backend: str) -> None:
        """Preparing ahead, on this thread or on a worker's stream, returns exactly what one serial batch returns.

        ``tensorrt`` makes the native dummy model report an exported runtime, which routes it to the worker path.
        """
        images = _image_dir(tmp_path, 5)
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda")
        native_context = model._prediction_context
        mocker.patch.object(
            model,
            "_prediction_context",
            side_effect=lambda: dataclasses.replace(native_context(), runtime_info={"backend": backend}),
        )
        predictor = RFDETRInference(model)
        paths = sorted(str(path) for path in images.iterdir())
        expected = core._predict_batch(predictor._prediction_context, paths, 0.5, None, None, True, None)

        actual = predictor.predict(images, batch=2)

        assert _source_pixels(actual) == _source_pixels(expected)
        assert [result.xyxy.tolist() for result in actual] == [result.xyxy.tolist() for result in expected]
        assert [result.confidence.tolist() for result in actual] == [result.confidence.tolist() for result in expected]
