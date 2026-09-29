# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the shared latency and memory benchmark helpers."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.export._benchmark import (
    BenchmarkResult,
    _artifact_size_mb,
    _decode_batch,
    _enable_notebook_inline_matplotlib,
    _measure_cuda,
    _result_row,
    _sampled_delta_mb,
    _tile_batch,
    measure_latency,
    measure_memory,
    visualize_detections,
)


class TestMeasureLatency:
    """Check timer dispatch, sample counts, and result statistics."""

    def test_wall_clock_counts_warmups_and_returns_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wall-clock path excludes warmups and returns population statistics."""
        clock = Mock(side_effect=[10.0, 10.002, 11.0, 11.006])
        monkeypatch.setattr("rfdetr.export._benchmark.time.perf_counter", clock)
        call = Mock()

        result = measure_latency(call, label="cpu", device="cpu", warmup=2, runs=2)

        assert call.call_count == 4
        assert clock.call_count == 4
        assert result.label == "cpu"
        assert result.mean_ms == pytest.approx(4.0, abs=1e-9)
        assert result.std_ms == pytest.approx(2.0, abs=1e-9)
        assert result.fps == pytest.approx(250.0)

    def test_cuda_events_include_warmups_and_synchronize_each_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA timing orders events around each call and synchronizes before reading elapsed time."""
        order: list[str] = []
        start = Mock()
        end = Mock()
        start.record.side_effect = lambda: order.append("start")
        end.record.side_effect = lambda: order.append("end")
        start.elapsed_time.side_effect = [1.0, 3.0]
        event_factory = Mock(side_effect=[start, end])
        synchronize = Mock(side_effect=lambda: order.append("sync"))
        call = Mock(side_effect=lambda: order.append("call"))
        monkeypatch.setattr(torch.cuda, "Event", event_factory)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        mean_ms, std_ms = _measure_cuda(call, warmup=2, runs=2)

        assert order == [
            "call",
            "call",
            "sync",
            "start",
            "call",
            "end",
            "sync",
            "start",
            "call",
            "end",
            "sync",
        ]
        assert event_factory.call_count == 2
        assert all(call.kwargs == {"enable_timing": True} for call in event_factory.call_args_list)
        assert call.call_count == 4
        assert synchronize.call_count == 3
        assert [call.args for call in start.elapsed_time.call_args_list] == [(end,), (end,)]
        assert mean_ms == pytest.approx(2.0)
        assert std_ms == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("warmup", "runs"),
        [pytest.param(-1, 1, id="negative-warmup"), pytest.param(0, 0, id="zero-runs")],
    )
    def test_invalid_sample_counts_raise(self, warmup: int, runs: int) -> None:
        """Negative warmups and empty measurement sets must be rejected."""
        with pytest.raises(ValueError, match="warmup|runs"):
            measure_latency(lambda: None, label="invalid", warmup=warmup, runs=runs)

    def test_zero_mean_latency_has_infinite_fps(self) -> None:
        """A zero-duration sample maps to infinite FPS without division failure."""
        assert BenchmarkResult("instant", 0.0, 0.0).fps == float("inf")


def _stepping_reader(readings: list[int]) -> tuple[Callable[[], int], threading.Event]:
    """Build a memory reader that walks *readings* once, then repeats the last value forever.

    The returned event is set once only the final value remains, which lets a test block until the
    sampler has definitely consumed every interesting reading — without that handshake the watcher
    thread's timing decides which values it sees, and any peak assertion becomes a race.

    Args:
        readings: Byte values to hand out in order; the last one is repeated indefinitely.

    Returns:
        The reader callable and the event marking the sequence exhausted.

    Examples:
        >>> read, exhausted = _stepping_reader([1, 2])
        >>> read(), read(), read()
        (1, 2, 2)
        >>> exhausted.is_set()
        True
    """
    remaining = list(readings)
    exhausted = threading.Event()
    lock = threading.Lock()

    def read() -> int:
        with lock:
            if len(remaining) > 1:
                return remaining.pop(0)
            exhausted.set()
            return remaining[0]

    return read, exhausted


class TestSampledDelta:
    """Check how net and peak growth are derived from a stream of memory readings."""

    def test_peak_exceeds_net_when_memory_is_released_inside_the_block(self) -> None:
        """Memory allocated and freed inside the block raises ``peak_mb`` but not ``delta_mb``.

        This is the case an endpoint-only reading cannot see, and the reason the sampler exists: a CoreML load was
        measured peaking 141 MB above where it settled, all of it compile scratch released before the block closed. A
        regression here silently reports the settled figure as though it were the cost of bringing the runtime up.
        """
        read, exhausted = _stepping_reader([100_000_000, 180_000_000, 120_000_000])

        with contextmanager(_sampled_delta_mb)(read) as result:
            assert exhausted.wait(timeout=5.0), "sampler never consumed the seeded readings"

        assert result.peak_mb == pytest.approx(80.0)
        assert result.delta_mb == pytest.approx(20.0)
        assert result.samples >= 2

    def test_net_is_negative_when_the_block_ends_below_its_baseline(self) -> None:
        """A block ending with less memory in use reports a negative ``delta_mb`` and zero ``peak_mb``.

        Observed for real when the OS reclaimed an earlier section's Neural Engine buffers during a later section's
        bracket. The negative value is information, not an error, so it must survive to the caller rather than being
        clamped away.
        """
        read, exhausted = _stepping_reader([100_000_000, 40_000_000])

        with contextmanager(_sampled_delta_mb)(read) as result:
            assert exhausted.wait(timeout=5.0), "sampler never consumed the seeded readings"

        assert result.delta_mb == pytest.approx(-60.0)
        assert result.peak_mb == pytest.approx(0.0)

    def test_watcher_thread_is_joined_before_the_block_returns(self) -> None:
        """The sampler thread is stopped and joined on exit, leaking nothing into later tests.

        A daemon thread left polling would keep calling into a torn-down mock for the rest of the session, producing
        failures attributed to whichever test ran next.
        """
        read, _exhausted = _stepping_reader([1_000_000, 2_000_000])

        with contextmanager(_sampled_delta_mb)(read):
            pass

        assert [t for t in threading.enumerate() if t.name == "measure_memory"] == []


class TestMeasureMemory:
    """Check host and device reader wiring, including exceptional block exit."""

    def test_host_delta_is_recorded_when_block_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The host memory sample is finalized while the original block error propagates.

        Export cells raise often while a notebook is being written, and the partially filled result is what tells the
        author how far the runtime got before failing.
        """
        process = Mock()
        process.memory_info.side_effect = lambda: SimpleNamespace(rss=5_000_000)
        monkeypatch.setattr("psutil.Process", Mock(return_value=process))

        with pytest.raises(RuntimeError, match="benchmark block failed"):
            with measure_memory() as result:
                raise RuntimeError("benchmark block failed")

        assert result.delta_mb == pytest.approx(0.0)

    def test_cuda_reads_device_bytes_in_use_with_synchronization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CUDA path derives bytes-in-use from ``mem_get_info`` and synchronizes before reading.

        Without the sync, an async kernel's allocation may not be visible yet, so a TensorRT or ONNX Runtime CUDA row
        would under-report whatever was still in flight.
        """
        synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(side_effect=lambda: (26_000_000, 50_000_000)))

        with measure_memory(device="cuda") as result:
            pass

        assert synchronize.call_count >= 2
        assert result.delta_mb == pytest.approx(0.0)

    def test_cuda_delta_is_recorded_when_block_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CUDA memory sample is finalized while the original block error propagates.

        Mirrors the host case: a failed engine build should still leave a readable result rather than
        an untouched default.
        """
        monkeypatch.setattr(torch.cuda, "synchronize", Mock())
        monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(side_effect=lambda: (17_000_000, 30_000_000)))

        with pytest.raises(RuntimeError, match="benchmark block failed"):
            with measure_memory(device="cuda") as result:
                raise RuntimeError("benchmark block failed")

        assert result.delta_mb == pytest.approx(0.0)


class TestResultRow:
    """Check cookbook results-table row construction, including the FPS derivation."""

    def test_fps_scales_by_batch_size(self) -> None:
        """FPS is images-per-second, so it scales linearly with batch size at fixed per-call latency.

        A batch-4 call that takes proportionally longer than batch-1 must still report the same throughput at both rows;
        only multiplying by ``batch`` before dividing by ``mean_ms`` gives that.
        """
        end2end = BenchmarkResult("onnx", mean_ms=20.0, std_ms=1.0)

        row = _result_row("ONNX", batch=4, config="CUDA EP", forward=None, end2end=end2end, memory_mb=12.5)

        assert row["FPS [img/s] (end2end)"] == pytest.approx(200.0)
        assert row == {
            "Format": "ONNX",
            "Batch": 4,
            "Config": "CUDA EP",
            "forward [ms]": "—",
            "end2end [ms]": "20.00 ± 1.00",
            "FPS [img/s] (end2end)": pytest.approx(200.0),
            "Memory [MB]": "12.5",
        }

    def test_zero_mean_latency_reports_infinite_fps_without_raising(self) -> None:
        """A zero-latency scope routes through ``BenchmarkResult.fps`` and reports ``inf``, never raises.

        Regression guard for the pre-fix formula (``batch * 1000 / scope.mean_ms``), which raised ``ZeroDivisionError``
        on a zero-latency row instead of reporting infinite throughput.
        """
        end2end = BenchmarkResult("instant", mean_ms=0.0, std_ms=0.0)

        row = _result_row("ONNX", batch=4, config="CUDA EP", forward=None, end2end=end2end, memory_mb=None)

        assert row["FPS [img/s] (end2end)"] == float("inf")

    def test_prefers_end2end_scope_over_forward_for_fps(self) -> None:
        """When both scopes are given, FPS is derived from ``end2end``, not ``forward``."""
        forward = BenchmarkResult("forward", mean_ms=5.0, std_ms=0.0)
        end2end = BenchmarkResult("end2end", mean_ms=10.0, std_ms=0.0)

        row = _result_row("ONNX", batch=1, config="CPU", forward=forward, end2end=end2end, memory_mb=None)

        assert row["FPS [img/s] (end2end)"] == pytest.approx(100.0)
        assert row["forward [ms]"] == "5.00 ± 0.00"
        assert row["end2end [ms]"] == "10.00 ± 0.00"

    def test_missing_both_scopes_raises(self) -> None:
        """A row needs at least one timing scope to derive FPS from."""
        with pytest.raises(ValueError, match="needs at least one of forward/end2end"):
            _result_row("ONNX", batch=1, config="CPU", forward=None, end2end=None, memory_mb=None)


class TestTileBatch:
    """Check that tiling repeats one preprocessed image along the batch axis."""

    def test_stacks_copies_along_batch_axis(self) -> None:
        """Tiling a ``(1, C, H, W)`` array *batch* times produces a ``(batch, C, H, W)`` array of copies."""
        single = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)

        tiled = _tile_batch(single, batch=3)

        assert tiled.shape == (3, 3, 2, 2)
        for i in range(3):
            np.testing.assert_array_equal(tiled[i], single[0])


class TestDecodeBatch:
    """Check that decoding dispatches once per image in the batch."""

    def test_calls_decode_detections_once_per_image_with_matching_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each image's boxes/logits row is decoded independently, in batch order."""
        decode = Mock()
        monkeypatch.setattr("rfdetr.export._benchmark.decode_detections", decode)
        boxes = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
        logits = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

        _decode_batch(boxes, logits, image_size=(640, 480), batch=2, threshold=0.5)

        assert decode.call_count == 2
        for i, call_args in enumerate(decode.call_args_list):
            args, kwargs = call_args
            np.testing.assert_array_equal(args[0], boxes[i])
            np.testing.assert_array_equal(args[1], logits[i])
            assert args[2] == (640, 480)
            assert kwargs == {"threshold": 0.5}


class TestArtifactSizeMb:
    """Check on-disk size summation across single files, directories, and multiple paths."""

    def test_single_file_size(self, tmp_path: Path) -> None:
        """A single file's size in bytes converts to megabytes."""
        file_path = tmp_path / "model.onnx"
        file_path.write_bytes(b"0" * 2_000_000)

        assert _artifact_size_mb(file_path) == pytest.approx(2.0)

    def test_directory_size_is_recursive(self, tmp_path: Path) -> None:
        """A directory's size sums every file it contains, including nested subdirectories."""
        bundle = tmp_path / "model.mlpackage"
        (bundle / "nested").mkdir(parents=True)
        (bundle / "top.bin").write_bytes(b"0" * 1_000_000)
        (bundle / "nested" / "weights.bin").write_bytes(b"0" * 500_000)

        assert _artifact_size_mb(bundle) == pytest.approx(1.5)

    def test_multiple_paths_are_summed(self, tmp_path: Path) -> None:
        """A bundle export passed as several paths (e.g. OpenVINO's ``.xml`` + ``.bin``) sums across all of them."""
        xml_path = tmp_path / "model.xml"
        bin_path = tmp_path / "model.bin"
        xml_path.write_bytes(b"0" * 100_000)
        bin_path.write_bytes(b"0" * 900_000)

        assert _artifact_size_mb(xml_path, bin_path) == pytest.approx(1.0)


class TestEnableNotebookInlineMatplotlib:
    """Check IPython detection and the inline-backend magics it enables.

    Every case injects a synthetic ``IPython`` module into ``sys.modules`` instead of importing
    or patching the real package: the real IPython, first-imported inside a torch-loaded
    pytest-xdist worker on this platform, intermittently SIGABRTs at interpreter teardown with a
    native ``recursive_mutex lock failed`` error — reproduced in isolation (~1 in 3 runs) and never
    on unrelated tests in this file, so it is specific to that first real import, not a logic bug.
    A fake module sidesteps the real import path entirely and is deterministic on every platform.
    """

    def test_noop_when_ipython_is_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Outside a notebook/IPython kernel, the ``ImportError`` path is a silent no-op."""
        monkeypatch.setitem(sys.modules, "IPython", None)

        _enable_notebook_inline_matplotlib()

    def test_noop_when_ipython_installed_but_no_active_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain Python process has IPython importable but no active shell, so ``get_ipython()`` returns ``None``."""
        fake_ipython = ModuleType("IPython")
        fake_ipython.get_ipython = Mock(return_value=None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)

        _enable_notebook_inline_matplotlib()

    def test_enables_inline_backend_when_ipython_shell_is_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inside an active IPython shell, both documented magics are invoked in order."""
        shell = Mock()
        fake_ipython = ModuleType("IPython")
        fake_ipython.get_ipython = Mock(return_value=shell)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)

        _enable_notebook_inline_matplotlib()

        assert shell.run_line_magic.call_args_list == [
            call("matplotlib", "inline"),
            call("config", "InlineBackend.close_figures = True"),
        ]


class TestVisualizeDetections:
    """Check detection annotation, label sourcing, and optional saving."""

    @pytest.fixture
    def image(self) -> Image.Image:
        """A small solid-color RGB image to annotate."""
        return Image.new("RGB", (64, 48), color=(10, 20, 30))

    @pytest.fixture
    def detections(self) -> sv.Detections:
        """One detection with confidence and class_id set, no ``class_name`` metadata."""
        return sv.Detections(
            xyxy=np.array([[5.0, 5.0, 30.0, 30.0]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([1]),
        )

    def test_smoke_runs_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, image: Image.Image, detections: sv.Detections
    ) -> None:
        """Annotating and displaying valid detections completes without raising."""
        monkeypatch.setattr("rfdetr.export._benchmark.sv.plot_image", Mock())

        visualize_detections(detections, image)

    def test_falls_back_to_coco_classes_for_label_text(
        self, monkeypatch: pytest.MonkeyPatch, image: Image.Image, detections: sv.Detections
    ) -> None:
        """Detections with no ``class_name`` metadata get their label text from ``COCO_CLASSES``."""
        plot_image = Mock()
        monkeypatch.setattr("rfdetr.export._benchmark.sv.plot_image", plot_image)
        label_annotator = Mock(wraps=sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=4).annotate)
        monkeypatch.setattr(sv.LabelAnnotator, "annotate", lambda self, **kwargs: label_annotator(**kwargs))

        visualize_detections(detections, image)

        expected_label = f"{COCO_CLASSES[1]} 0.90"
        assert label_annotator.call_args.kwargs["labels"] == [expected_label]
        plot_image.assert_called_once()

    def test_missing_class_id_raises(self, image: Image.Image) -> None:
        """Detections without a ``class_id`` array cannot be labeled, so the call is rejected up front."""
        detections = sv.Detections(
            xyxy=np.array([[5.0, 5.0, 30.0, 30.0]], dtype=np.float32),
            confidence=np.array([0.9]),
        )

        with pytest.raises(ValueError, match="class_id and detections.confidence"):
            visualize_detections(detections, image)

    def test_missing_confidence_raises(self, image: Image.Image) -> None:
        """Detections without a ``confidence`` array cannot be labeled, so the call is rejected up front."""
        detections = sv.Detections(
            xyxy=np.array([[5.0, 5.0, 30.0, 30.0]], dtype=np.float32),
            class_id=np.array([1]),
        )

        with pytest.raises(ValueError, match="class_id and detections.confidence"):
            visualize_detections(detections, image)

    def test_saves_annotated_image_when_save_path_given(
        self,
        monkeypatch: pytest.MonkeyPatch,
        image: Image.Image,
        detections: sv.Detections,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When ``save_path`` is given, the annotated image is written to disk and the path is announced."""
        monkeypatch.setattr("rfdetr.export._benchmark.sv.plot_image", Mock())
        save_path = tmp_path / "annotated.png"

        visualize_detections(detections, image, save_path=save_path)

        assert save_path.is_file()
        assert str(save_path) in capsys.readouterr().out
