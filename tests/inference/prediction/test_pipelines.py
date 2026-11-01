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
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr._prediction import core, pipelines
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

        def record_decode(prepared: Any, _: object) -> list[Any]:
            """Record decoding ``prepared`` and return its input as the batch's only result."""
            events.append(f"decode {prepared.batch_tensor}")
            return [prepared.batch_tensor]

        mocker.patch.object(pipelines, "_decode_batch", side_effect=record_decode)
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

    def test_error_preparing_a_batch_never_read_is_logged_at_debug_level(self, mocker: MockerFixture) -> None:
        """Closing the stream before a batch that failed to prepare logs that error instead of dropping it silently.

        The serial loop would never have prepared that batch, so the error is not raised; a debug record keeps it
        findable. The logger call is spied on directly: pytest 9.1 also captures on the non-propagating ``rf-detr``
        logger, so with propagation forced on ``caplog`` holds each record twice.
        """
        debug = mocker.spy(pipelines.logger, "debug")
        mocker.patch.object(pipelines, "_decode_batch", side_effect=lambda prepared, _: [prepared.batch_tensor])
        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"]]), self._recording_prepare([], failing="b"), torch.device("cuda"), in_worker=False
        )

        next(results)
        results.close()

        debug.assert_called_once()
        assert str(debug.call_args.kwargs["exc_info"]) == "bad image b"

    def test_batch_holding_cuda_tensor_is_prepared_on_the_caller_thread(self, mocker: MockerFixture) -> None:
        """An exported runtime's worker never prepares a batch holding a CUDA tensor; a host-only batch goes to it.

        The worker prepares on a CUDA stream that does not wait for the caller's, so it could read a CUDA input before
        the kernels writing it finish. The CUDA stream API is replaced, so this checks the routing alone on any host.
        """
        mocker.patch.object(torch.cuda, "Stream")
        mocker.patch.object(torch.cuda, "Event")
        mocker.patch.object(torch.cuda, "stream")
        mocker.patch.object(torch.cuda, "current_stream")
        mocker.patch.object(pipelines, "_decode_batch", side_effect=lambda prepared, _: [prepared.label])
        cuda_image = mocker.Mock(spec=torch.Tensor, is_cuda=True)
        caller_thread = threading.current_thread().name
        prepare_threads: list[str] = []

        def prepare(images: list[Any]) -> SimpleNamespace:
            """Record the thread preparing ``images`` and return a batch whose tensors accept ``record_stream``."""
            prepare_threads.append(threading.current_thread().name)
            label = "cuda" if images[0] is cuda_image else "host"
            return SimpleNamespace(
                context=SimpleNamespace(run=lambda batch: {}),
                batch_tensor=mocker.Mock(),
                target_sizes=mocker.Mock(),
                label=label,
            )

        results = pipelines._predict_prepared_ahead(
            iter([[cuda_image], ["host.png"]]), prepare, torch.device("cuda"), in_worker=True
        )

        assert list(results) == ["cuda", "host"]
        assert prepare_threads[0] == caller_thread
        assert prepare_threads[1].startswith("rfdetr-prepare")

    @pytest.fixture
    def caller_stream(self, mocker: MockerFixture) -> Mock:
        """Replace the CUDA stream API so the worker path runs on any host, and return the caller's stream."""
        mocker.patch.object(torch.cuda, "Stream")
        mocker.patch.object(torch.cuda, "Event")
        mocker.patch.object(torch.cuda, "stream")
        mocker.patch.object(pipelines, "_decode_batch", side_effect=lambda prepared, _: [prepared.label])
        return mocker.patch.object(torch.cuda, "current_stream").return_value

    @staticmethod
    def _stream_batch(label: str, events: list[str]) -> SimpleNamespace:
        """Return a prepared batch named ``label`` whose tensors accept ``record_stream`` and whose run is recorded.

        Examples:
            >>> events = []
            >>> prepared = TestPreparedAhead._stream_batch("a", events)
            >>> prepared.context.run(prepared.batch_tensor), events
            ({}, ['run a'])
        """

        def run(batch: Any) -> dict[str, torch.Tensor]:
            """Record the forward pass of the batch named ``label``."""
            events.append(f"run {label}")
            return {}

        return SimpleNamespace(context=SimpleNamespace(run=run), batch_tensor=Mock(), target_sizes=Mock(), label=label)

    def test_worker_batch_is_used_only_after_the_caller_stream_waits_for_it(self, caller_stream: Mock) -> None:
        """The caller's stream waits for the worker stream's event before the forward pass reads the prepared input.

        Both tensors are allocated on the worker's stream, so each is also recorded on the caller's stream: otherwise
        the allocator could hand their memory back to the worker's stream while the forward pass still reads it.
        """
        events: list[str] = []
        caller_stream.wait_event.side_effect = lambda ready: events.append("wait")
        prepared = self._stream_batch("a", events)
        results = pipelines._predict_prepared_ahead(
            iter([["a"]]), lambda images: prepared, torch.device("cuda"), in_worker=True
        )

        assert list(results) == ["a"]
        assert events == ["wait", "run a"]
        caller_stream.wait_event.assert_called_once_with(torch.cuda.Event.return_value)
        prepared.batch_tensor.record_stream.assert_called_once_with(caller_stream)
        prepared.target_sizes.record_stream.assert_called_once_with(caller_stream)

    @pytest.mark.parametrize("in_worker", [False, True])
    @pytest.mark.parametrize(
        "batches,expected",
        [
            pytest.param([], [], id="empty-source"),
            pytest.param([["a"]], ["a"], id="single-batch"),
            pytest.param([["a", "b"], ["c"]], ["a+b", "c"], id="partial-final-batch"),
        ],
    )
    def test_every_batch_is_predicted_once_in_order(
        self, caller_stream: Mock, in_worker: bool, batches: list[list[str]], expected: list[str]
    ) -> None:
        """Preparing ahead yields each batch's results once, in source order, whatever the number and size of batches.

        An empty source must end without preparing anything, and a short last batch must not wait for a batch after it.
        """
        results = pipelines._predict_prepared_ahead(
            iter(batches),
            lambda images: self._stream_batch("+".join(images), []),
            torch.device("cuda"),
            in_worker=in_worker,
        )

        assert list(results) == expected

    @pytest.mark.parametrize("in_worker", [False, True])
    @pytest.mark.parametrize(
        "error", [pytest.param(ValueError, id="value-error"), pytest.param(OSError, id="os-error")]
    )
    def test_error_preparing_first_batch_is_raised_before_any_result(
        self, caller_stream: Mock, in_worker: bool, error: type[Exception]
    ) -> None:
        """An unreadable first batch raises its own error, of any type, before a result is yielded.

        The serial loop raised it at the first batch too; a worker must hand the error back, not swallow it.
        """

        def prepare(images: list[str]) -> SimpleNamespace:
            """Fail to read every batch."""
            raise error(f"unreadable {images[0]}")

        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"]]), prepare, torch.device("cuda"), in_worker=in_worker
        )

        with pytest.raises(error, match="unreadable a"):
            next(results)

    @pytest.mark.parametrize(
        "error", [pytest.param(ValueError, id="value-error"), pytest.param(OSError, id="os-error")]
    )
    def test_worker_error_preparing_next_batch_is_raised_after_current_results(
        self, caller_stream: Mock, error: type[Exception]
    ) -> None:
        """An error the worker raised for the next batch reaches the caller after the current batch's results.

        The serial loop raised it when it reached that batch, after yielding every result before it.
        """

        def prepare(images: list[str]) -> SimpleNamespace:
            """Prepare batch ``a`` and fail to read batch ``b``."""
            (label,) = images
            if label == "b":
                raise error("unreadable b")
            return self._stream_batch(label, [])

        results = pipelines._predict_prepared_ahead(iter([["a"], ["b"]]), prepare, torch.device("cuda"), in_worker=True)

        assert next(results) == "a"
        with pytest.raises(error, match="unreadable b"):
            next(results)

    def test_closing_early_ends_the_worker_thread(self, caller_stream: Mock) -> None:
        """Closing the results after the first batch stops the worker thread instead of leaving it running.

        The worker is already preparing the second batch then, so closing waits for it and shuts the worker down.
        """
        prepare_threads: list[threading.Thread] = []

        def prepare(images: list[str]) -> SimpleNamespace:
            """Record the preparing thread and return the batch."""
            prepare_threads.append(threading.current_thread())
            return self._stream_batch(images[0], [])

        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"], ["c"]]), prepare, torch.device("cuda"), in_worker=True
        )
        next(results)
        results.close()

        assert prepare_threads
        assert not any(thread.is_alive() for thread in prepare_threads)

    def test_consumer_error_ends_the_worker_thread(self, caller_stream: Mock) -> None:
        """An error raised into the results by their consumer stops the worker thread and reaches the caller.

        A consumer failing mid-stream is the other way prediction ends early, and must leave no thread behind either.
        """
        prepare_threads: list[threading.Thread] = []

        def prepare(images: list[str]) -> SimpleNamespace:
            """Record the preparing thread and return the batch."""
            prepare_threads.append(threading.current_thread())
            return self._stream_batch(images[0], [])

        results = pipelines._predict_prepared_ahead(
            iter([["a"], ["b"], ["c"]]), prepare, torch.device("cuda"), in_worker=True
        )
        next(results)

        with pytest.raises(RuntimeError, match="consumer failed"):
            results.throw(RuntimeError("consumer failed"))
        assert not any(thread.is_alive() for thread in prepare_threads)

    @pytest.mark.parametrize(
        "device,backend,prepare_ahead,expected",
        [
            pytest.param("cuda", "pytorch", True, [False], id="cuda-native"),
            pytest.param("cuda", "tensorrt", True, [True], id="cuda-export"),
            pytest.param("cuda", "pytorch", False, [], id="cuda-borrowed"),
            pytest.param("cpu", "pytorch", True, [], id="cpu-native"),
            pytest.param("cpu", "onnxruntime", True, [], id="cpu-export"),
            pytest.param("mps", "pytorch", True, [], id="mps-native"),
        ],
    )
    def test_only_cuda_contexts_prepare_ahead(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        device: str,
        backend: str,
        prepare_ahead: bool,
        expected: list[bool],
    ) -> None:
        """Only an owned CUDA context prepares ahead, on a worker for an exported runtime; all others run serially.

        A borrowed live model must read its state when each batch starts, so it runs serially on CUDA too. The
        prediction functions are replaced, so no device is touched: this checks the routing alone.
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
            prepare_ahead=prepare_ahead,
        )

        assert list(results) == []
        assert in_worker_flags == expected

    @pytest.mark.parametrize(
        "device,backend,prepare_ahead,expected_calls",
        [
            pytest.param("cuda", "pytorch", True, 1, id="cuda-native"),
            pytest.param("cuda", "tensorrt", True, 1, id="cuda-export"),
            pytest.param("cuda", "pytorch", False, 1, id="cuda-borrowed"),
            pytest.param("cpu", "onnxruntime", True, 1, id="cpu-export"),
            pytest.param("cpu", "pytorch", True, 0, id="cpu-native"),
        ],
    )
    def test_source_is_classified_at_most_once(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        device: str,
        backend: str,
        prepare_ahead: bool,
        expected_calls: int,
    ) -> None:
        """Deciding to decode or prepare ahead checks the source on disk once, and never for a native CPU model.

        Classifying a source checks every path, directory and glob in it, so checking it for each decision doubled that
        work on CUDA. A native CPU model reads one batch at a time and needs no check.
        """
        Image.new("RGB", (8, 6)).save(tmp_path / "a.png")
        context = dataclasses.replace(
            _DummyRFDETR()._prediction_context(),
            device=torch.device(device),
            runtime_info={"backend": backend, "device": device},
        )
        mocker.patch.object(pipelines, "_predict_prepared_ahead", return_value=iter([]))
        mocker.patch.object(pipelines, "_predict_batch", return_value=[])
        classify = mocker.spy(pipelines, "has_only_finite_images")
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
            prepare_ahead=prepare_ahead,
        )

        assert list(results) == []
        assert classify.call_count == expected_calls

    @pytest.mark.parametrize(
        "build_predictor,expected_prepared_ahead",
        [
            pytest.param(RFDETRInference, 0, id="borrowed"),
            pytest.param(RFDETRInference._from_owned_model, 1, id="owned"),
        ],
    )
    def test_only_an_owned_predictor_prepares_ahead_on_cuda(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        build_predictor: Callable[[_DummyRFDETR], RFDETRInference],
        expected_prepared_ahead: int,
    ) -> None:
        """A borrowed live model runs each CUDA batch serially; a predictor owning its model prepares the next ahead.

        Preparing ahead reads the next batch's model state before the current batch's results reach the caller, who may
        change a borrowed model in between. The CUDA device is only reported and the prediction functions are replaced,
        so this checks the routing alone on any host.
        """
        Image.new("RGB", (8, 6)).save(tmp_path / "a.png")
        model = _DummyRFDETR()
        context = dataclasses.replace(model._prediction_context(), device=torch.device("cuda"))
        mocker.patch.object(model, "_prediction_context", return_value=context)
        prepared_ahead = mocker.patch.object(pipelines, "_predict_prepared_ahead", return_value=iter([]))
        serial = mocker.patch.object(pipelines, "_predict_batch", return_value=[])

        build_predictor(model).predict(tmp_path, batch=1)

        assert prepared_ahead.call_count == expected_prepared_ahead
        assert serial.call_count == 1 - expected_prepared_ahead

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("backend", ["pytorch", "tensorrt"])
    def test_cuda_directory_matches_one_serial_batch(self, tmp_path: Path, mocker: MockerFixture, backend: str) -> None:
        """Preparing ahead, on this thread or on a worker's stream, returns exactly what one serial batch returns.

        ``tensorrt`` makes the native dummy model report an exported runtime, which routes it to the worker path. The
        predictor owns the model, as one loaded from a path does; a borrowed model would not prepare ahead at all.
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
        predictor = RFDETRInference._from_owned_model(model)
        paths = sorted(str(path) for path in images.iterdir())
        expected = core._predict_batch(predictor._prediction_context, paths, 0.5, None, None, True, None)

        actual = predictor.predict(images, batch=2)

        assert _source_pixels(actual) == _source_pixels(expected)
        assert [result.xyxy.tolist() for result in actual] == [result.xyxy.tolist() for result in expected]
        assert [result.confidence.tolist() for result in actual] == [result.confidence.tolist() for result in expected]

    @pytest.mark.gpu
    @_requires_cuda
    def test_cuda_tensor_still_being_written_matches_one_serial_batch(self, mocker: MockerFixture) -> None:
        """A CUDA input whose write is still queued on the caller's stream reaches an exported runtime with its values.

        The caller queues a long kernel and then the write, so the input is not written yet when ``predict`` starts. A
        worker preparing it on its own stream would read it before the write, and its source pixels would differ.
        ``tensorrt`` makes the native dummy model report an exported runtime, and owning the model lets the predictor
        prepare ahead, which routes it to the worker path.
        """
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda")
        native_context = model._prediction_context
        mocker.patch.object(
            model,
            "_prediction_context",
            side_effect=lambda: dataclasses.replace(native_context(), runtime_info={"backend": "tensorrt"}),
        )
        predictor = RFDETRInference._from_owned_model(model)
        values = torch.linspace(0.1, 0.9, 4, device="cuda").view(4, 1, 1, 1).expand(4, 3, 12, 16).contiguous()
        images = torch.zeros_like(values)
        torch.cuda.synchronize()
        # Keeps the caller's stream busy for a while, so the copy queued behind it has not run when prediction starts.
        torch.cuda._sleep(1 << 30)
        images.copy_(values)

        actual = predictor.predict(images, batch=2)

        torch.cuda.synchronize()
        expected = core._predict_batch(
            predictor._prediction_context, list(values.unbind(0)), 0.5, None, None, True, None
        )
        assert _source_pixels(actual) == _source_pixels(expected)
