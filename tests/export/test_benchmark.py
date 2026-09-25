# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the shared latency and memory benchmark helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rfdetr.export._benchmark import BenchmarkResult, _measure_cuda, measure_latency, measure_memory


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


class TestMeasureMemory:
    """Check host and device memory deltas, including exceptional block exit."""

    def test_host_delta_is_recorded_when_block_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The host memory sample is finalized while the original block error propagates."""
        process = Mock()
        process.memory_info.side_effect = [SimpleNamespace(rss=2_000_000), SimpleNamespace(rss=5_000_000)]
        monkeypatch.setattr("psutil.Process", Mock(return_value=process))

        with pytest.raises(RuntimeError, match="benchmark block failed"):
            with measure_memory() as result:
                raise RuntimeError("benchmark block failed")

        assert result.delta_mb == pytest.approx(3.0)
        assert process.memory_info.call_count == 2

    def test_cuda_delta_synchronizes_and_reads_device_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CUDA path measures device-wide free-memory shrinkage with synchronization."""
        synchronize = Mock()
        free_memory = Mock(side_effect=[(30_000_000, 50_000_000), (26_000_000, 50_000_000)])
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)

        with measure_memory(device="cuda") as result:
            pass

        assert synchronize.call_count == 2
        assert free_memory.call_count == 2
        assert result.delta_mb == pytest.approx(4.0)

    def test_cuda_delta_is_recorded_when_block_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CUDA memory sample is finalized while the original block error propagates."""
        synchronize = Mock()
        free_memory = Mock(side_effect=[(20_000_000, 30_000_000), (17_000_000, 30_000_000)])
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)

        with pytest.raises(RuntimeError, match="benchmark block failed"):
            with measure_memory(device="cuda") as result:
                raise RuntimeError("benchmark block failed")

        assert synchronize.call_count == 2
        assert free_memory.call_count == 2
        assert result.delta_mb == pytest.approx(3.0)
