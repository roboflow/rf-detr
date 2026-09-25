# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the shared latency and memory benchmark helpers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rfdetr.export._benchmark import (
    BenchmarkResult,
    _measure_cuda,
    _sampled_delta_mb,
    measure_latency,
    measure_memory,
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
