# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Latency benchmarking helpers shared by the per-hardware export cookbooks.

Two timer strategies exist because GPU kernels execute asynchronously: CUDA events measure actual device-side execution,
while ``time.perf_counter`` is correct wall-clock timing for anything that blocks the calling thread — CPU inference,
and every non-CUDA runtime (CoreML, Core AI, ExecuTorch, TensorFlow Lite, LiteRT, OpenVINO) has no CUDA stream to
desynchronize from in the first place.

Private module: no compatibility guarantee across versions. Formerly duplicated per export format (see the removed
``rfdetr.export._onnx.inference._onnx_runtime``); this is the single home for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class BenchmarkResult(NamedTuple):
    """One latency measurement: mean and standard deviation across timed runs, in milliseconds."""

    label: str
    mean_ms: float
    std_ms: float

    @property
    def fps(self) -> float:
        """Frames per second implied by ``mean_ms``.

        Examples:
            >>> BenchmarkResult("cpu", 10.0, 0.5).fps
            100.0
        """
        return 1000.0 / self.mean_ms


def _mean_std(timings: list[float]) -> tuple[float, float]:
    """Mean and population standard deviation of a list of millisecond timings.

    Examples:
        >>> [round(v, 2) for v in _mean_std([1.0, 2.0, 3.0])]
        [2.0, 0.82]
    """
    arr = np.array(timings)
    return float(arr.mean()), float(arr.std())


def _measure_cuda(fn: Callable[[], object], warmup: int, runs: int) -> tuple[float, float]:
    """Time ``fn`` with CUDA events — captures device-side kernel execution, not Python overhead."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    timings: list[float] = []
    for _ in range(runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return _mean_std(timings)


def _measure_wall_clock(fn: Callable[[], object], warmup: int, runs: int) -> tuple[float, float]:
    """Time ``fn`` with ``time.perf_counter`` — correct for any call with no CUDA stream to sync."""
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return _mean_std(timings)


def measure_latency(
    fn: Callable[[], object],
    *,
    label: str,
    device: str = "cpu",
    warmup: int = 20,
    runs: int = 100,
) -> BenchmarkResult:
    """Measure the latency of a zero-argument callable.

    ``device="cuda"`` times with CUDA events; any other value times with ``time.perf_counter``. Pass
    a thunk wrapping only the runtime's forward call to measure ``forward_ms``, or a thunk wrapping
    preprocess + forward + postprocess to measure ``end2end_ms`` — the caller chooses the scope by
    what it wraps, this function only times whatever it is given.

    Args:
        fn: Zero-argument callable to time.
        label: Name for the resulting :class:`BenchmarkResult` row, e.g. ``"TensorRT forward"``.
        device: ``"cuda"`` selects the CUDA-event timer; any other value uses ``perf_counter``.
        warmup: Untimed warm-up calls before measurement starts, to skip first-call JIT/lazy-init cost.
        runs: Timed calls used to compute the mean and standard deviation.

    Returns:
        A :class:`BenchmarkResult` with ``mean_ms``, ``std_ms``, and the derived ``fps``.

    Examples:
        >>> result = measure_latency(lambda: sum(range(1000)), label="sum", warmup=1, runs=3)
        >>> result.label
        'sum'
        >>> result.mean_ms >= 0.0
        True
    """
    measure = _measure_cuda if device == "cuda" else _measure_wall_clock
    mean_ms, std_ms = measure(fn, warmup, runs)
    return BenchmarkResult(label, mean_ms, std_ms)
