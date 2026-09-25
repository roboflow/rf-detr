# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Latency and memory benchmarking helpers shared by the per-hardware export cookbooks.

Two timer strategies exist because GPU kernels execute asynchronously: CUDA events measure actual device-side execution,
while ``time.perf_counter`` is correct wall-clock timing for anything that blocks the calling thread — CPU inference,
and every non-CUDA runtime (CoreML, Core AI, ExecuTorch, TensorFlow Lite, LiteRT, OpenVINO) has no CUDA stream to
desynchronize from in the first place. The same split applies to :func:`measure_memory`: device-side weights and buffers
on a CUDA GPU don't show up in host resident memory, so it reads ``torch.cuda.mem_get_info()`` there instead of process
RSS.

Private module: no compatibility guarantee across versions. Formerly duplicated per export format (see the removed
``rfdetr.export._onnx.inference._onnx_runtime``); this is the single home for it.
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
        return float("inf") if self.mean_ms == 0.0 else 1000.0 / self.mean_ms


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
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")

    measure = _measure_cuda if device == "cuda" else _measure_wall_clock
    mean_ms, std_ms = measure(fn, warmup, runs)
    return BenchmarkResult(label, mean_ms, std_ms)


#: Seconds between memory samples taken by the background watcher thread.
_SAMPLE_INTERVAL_S = 0.005


@dataclass
class MemoryResult:
    """Mutable holder for the memory measurements taken by :func:`measure_memory`.

    All three fields keep their defaults until the ``with`` block exits. ``delta_mb`` and ``peak_mb``
    answer different questions and routinely differ by a lot: bringing up a CoreML model was measured
    at a 548.7 MB peak but only a 407.2 MB net change, the 141 MB gap being compile scratch space
    released before the block closed.

    Attributes:
        delta_mb: Net change between the start and the end of the block — what the runtime still
            holds once it is up. May be *negative*, which means the block ended with less memory in
            use than it started with, usually because the OS reclaimed an earlier allocation.
        peak_mb: Largest growth above the starting level seen at any sample during the block — what
            it costs to bring the runtime up, including transient scratch space. Never negative.
        samples: Number of samples the watcher thread took. ``0`` means it never got scheduled, so
            ``peak_mb`` is only as good as the two endpoint reads and should not be trusted.
    """

    delta_mb: float = 0.0
    peak_mb: float = 0.0
    samples: int = 0


def _sampled_delta_mb(read_bytes_in_use: Callable[[], int]) -> Iterator[MemoryResult]:
    """Track net and peak growth of ``read_bytes_in_use()`` across a block, sampling in a thread.

    Sampling rather than reading only the endpoints is what makes ``peak_mb`` meaningful: memory
    allocated and released inside the block is invisible to an endpoint-only diff.

    Args:
        read_bytes_in_use: Returns the current bytes-in-use figure for whichever memory is measured.

    Yields:
        The :class:`MemoryResult` filled in when the block exits.
    """
    baseline = read_bytes_in_use()
    peak = baseline
    samples = 0
    stop = threading.Event()

    def watch() -> None:
        nonlocal peak, samples
        while not stop.is_set():
            peak = max(peak, read_bytes_in_use())
            samples += 1
            stop.wait(_SAMPLE_INTERVAL_S)

    watcher = threading.Thread(target=watch, name="measure_memory", daemon=True)
    watcher.start()
    result = MemoryResult()
    try:
        yield result
    finally:
        stop.set()
        watcher.join(timeout=1.0)
        final = read_bytes_in_use()
        result.delta_mb = (final - baseline) / 1e6
        result.peak_mb = (max(peak, final) - baseline) / 1e6
        result.samples = samples


def _rss_delta_mb() -> Iterator[MemoryResult]:
    """Measure host resident memory across a block via ``psutil``."""
    import psutil  # type: ignore[import-untyped]

    gc.collect()
    process = psutil.Process()
    yield from _sampled_delta_mb(lambda: int(process.memory_info().rss))


def _cuda_free_delta_mb() -> Iterator[MemoryResult]:
    """Measure device memory in use across a block via ``torch.cuda.mem_get_info``.

    Device-wide, unlike ``torch.cuda.memory_allocated()`` — it also captures allocations made outside PyTorch's own
    caching allocator, such as ONNX Runtime's CUDA execution provider or a TensorRT engine's own ``cudaMalloc`` calls.
    """
    import torch

    def device_bytes_in_use() -> int:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return int(total - free)

    yield from _sampled_delta_mb(device_bytes_in_use)


@contextmanager
def measure_memory(*, device: str = "cpu") -> Iterator[MemoryResult]:
    """Measure the memory growth caused by the code inside a ``with`` block.

    ``device="cuda"`` reads free-device-memory shrinkage via ``torch.cuda.mem_get_info()``; any
    other value reads host resident-memory growth via ``psutil``. Bracket both the runtime's
    construction *and* its first inference call — several runtimes allocate lazily (an ONNX
    Runtime session grows its arena on first ``run``, an ExecuTorch CoreML program compiles on
    first ``execute``), so closing the block right after construction undercounts the real
    footprint.

    A background thread samples memory every 5 ms for the duration of the block, so
    :attr:`MemoryResult.peak_mb` sees transient scratch space that an endpoint-only reading misses.
    Check :attr:`MemoryResult.samples` before trusting ``peak_mb``: a block that finishes in under a
    few milliseconds, or one that never releases the GIL, can collect no samples at all.

    Args:
        device: ``"cuda"`` selects the device-memory reader; any other value uses host RSS.

    Yields:
        A :class:`MemoryResult` filled in once the block exits.

    Note:
        Both figures are measurements, not guarantees, and neither is a per-runtime sandbox. In one
        shared process the host reader can report ``0.0`` for a large allocation, because RSS counts
        *resident* pages and the allocator may satisfy the request from pages it already holds — no
        sampling rate fixes that, and it cannot be detected from inside this helper. ``delta_mb`` can
        also read negative when the OS reclaims an earlier section's memory during this block. Report
        what comes back; never assert a lower bound on it.

    Note:
        Wrap construction and correctness checks, not the timed loop: the sampler thread adds
        ``psutil`` syscalls that would perturb :func:`measure_latency`'s numbers.

    Examples:
        ``delta_mb`` stays at its ``0.0`` default while the block is open and holds the measurement
        once the block exits, so read it after the ``with``, never inside:

        >>> with measure_memory() as mem:
        ...     reading_inside_the_block = mem.delta_mb
        >>> reading_inside_the_block
        0.0
        >>> isinstance(mem.delta_mb, float)
        True
    """
    reader = _cuda_free_delta_mb if device == "cuda" else _rss_delta_mb
    yield from reader()
