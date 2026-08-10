# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Benchmark pageable and pinned CPU-to-CUDA image transfers used by prediction.

The benchmark transfers a list of CHW CPU tensors, matching the per-image transfer
shape used by ``RFDETR.predict``. It compares blocking pageable copies, pageable
``non_blocking`` copies, and explicit ``pin_memory`` plus ``non_blocking`` copies.
It emits JSON so CUDA hosts can retain the raw samples and system context needed to
compare results without relying on a prose summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

TransferFunction = Callable[[Sequence[Tensor], torch.device], list[Tensor]]


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer.

    Args:
        value: Text supplied to an argparse option.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If *value* is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, received {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, received {value!r}")
    return parsed


def _blocking_pageable(inputs: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    """Transfer pageable tensors with the default blocking behavior.

    Args:
        inputs: CPU tensors to transfer.
        device: CUDA destination device.

    Returns:
        CUDA tensors copied from *inputs*.
    """
    return [image.to(device, non_blocking=False) for image in inputs]


def _pageable_non_blocking(inputs: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    """Transfer pageable tensors with ``non_blocking=True``.

    Args:
        inputs: CPU tensors to transfer.
        device: CUDA destination device.

    Returns:
        CUDA tensors copied from *inputs*.
    """
    return [image.to(device, non_blocking=True) for image in inputs]


def _pinned_non_blocking(inputs: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    """Pin each CPU tensor then transfer it with ``non_blocking=True``.

    The timed region intentionally includes ``pin_memory()``: this is the complete
    per-call cost of the prediction implementation rather than an allocator-only
    transfer microbenchmark.

    Args:
        inputs: CPU tensors to pin and transfer.
        device: CUDA destination device.

    Returns:
        CUDA tensors copied from pinned versions of *inputs*.
    """
    return [image.pin_memory().to(device, non_blocking=True) for image in inputs]


def _linux_memory_kib() -> dict[str, int | None]:
    """Read Linux process memory fields when ``/proc`` makes them available.

    ``VmRSS`` is current resident memory, ``VmHWM`` is its process peak, and
    ``VmLck`` is the amount of memory locked by the process. Other platforms return
    ``None`` values rather than emulating incompatible units.

    Returns:
        KiB values for ``VmRSS``, ``VmHWM``, and ``VmLck``, or ``None`` when unavailable.
    """
    fields = {"VmRSS": None, "VmHWM": None, "VmLck": None}
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return fields

    for line in status_path.read_text(encoding="utf-8").splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in fields:
            value, _, unit = remainder.strip().partition(" ")
            if unit == "kB" and value.isdecimal():
                fields[name] = int(value)
    return fields


def _summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Summarize latency samples with robust and range statistics.

    Args:
        samples_ms: Per-repetition synchronized latencies in milliseconds.

    Returns:
        Minimum, median, maximum, and median absolute deviation in milliseconds.
    """
    median_ms = statistics.median(samples_ms)
    return {
        "min": min(samples_ms),
        "median": median_ms,
        "max": max(samples_ms),
        "median_absolute_deviation": statistics.median(abs(sample - median_ms) for sample in samples_ms),
    }


def _run_mode(
    transfer: TransferFunction,
    inputs: Sequence[Tensor],
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """Run one transfer mode with synchronized warmup and measured repetitions.

    Args:
        transfer: One CPU-to-CUDA transfer implementation.
        inputs: Identical pageable CPU tensors used by every mode for this batch size.
        device: CUDA destination device.
        warmup: Number of untimed repetitions.
        repetitions: Number of recorded repetitions.

    Returns:
        Raw millisecond samples, robust summary, and Linux memory snapshots.
    """
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        transferred = transfer(inputs, device)
        torch.cuda.synchronize(device)
        del transferred

    samples_ms: list[float] = []
    memory_before = _linux_memory_kib()
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        transferred = transfer(inputs, device)
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1_000)
        del transferred
    memory_after = _linux_memory_kib()

    return {
        "samples_ms": samples_ms,
        "summary_ms": _summary(samples_ms),
        "memory_kib_before": memory_before,
        "memory_kib_after": memory_after,
    }


def _cpu_inputs(batch_size: int, channels: int, height: int, width: int, seed: int) -> list[Tensor]:
    """Create deterministic pageable CPU image tensors for one simulated predict batch.

    Args:
        batch_size: Number of independent CHW images.
        channels: Number of image channels.
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed used only for benchmark input generation.

    Returns:
        Pageable float32 CPU tensors with shape ``(channels, height, width)``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (channels, height, width)
    return [torch.rand(shape, generator=generator) for _ in range(batch_size)]


def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser.

    Returns:
        Parser with reproducible workload and output options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=_positive_int, default=[1, 8], metavar="BATCH_SIZE")
    parser.add_argument("--channels", type=_positive_int, default=3)
    parser.add_argument("--height", type=_positive_int, default=640)
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repetitions", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, help="Optional file to receive the JSON report.")
    return parser


def _device_details(device: torch.device) -> dict[str, Any]:
    """Collect CUDA device metadata included with every benchmark result.

    Args:
        device: Valid CUDA device selected by the command line.

    Returns:
        CUDA hardware and runtime details.
    """
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested_device": str(device),
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": torch.cuda.driver_version if hasattr(torch.cuda, "driver_version") else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute all transfer comparisons and return a JSON-serializable report.

    Args:
        args: Parsed command-line arguments from :func:`_build_parser`.

    Returns:
        Report containing the configuration, environment, raw samples, and summaries.

    Raises:
        RuntimeError: If CUDA is unavailable or the requested device cannot be selected.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a CUDA-enabled PyTorch runtime and GPU.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"--device must name a CUDA device, received {args.device!r}.")

    torch.cuda.set_device(device)
    modes: dict[str, TransferFunction] = {
        "blocking_pageable": _blocking_pageable,
        "pageable_non_blocking": _pageable_non_blocking,
        "pinned_non_blocking": _pinned_non_blocking,
    }
    result: dict[str, Any] = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "configuration": {
            "batches": args.batches,
            "channels": args.channels,
            "height": args.height,
            "width": args.width,
            "dtype": "torch.float32",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": _device_details(device),
        },
        "methodology": (
            "Each sample transfers one list of pageable CHW float32 CPU tensors. "
            "Each mode uses the same tensors for a given batch size, synchronizes before timing "
            "and after the transfers, and includes pin_memory() in pinned_non_blocking timing."
        ),
        "results": {},
    }

    for batch_size in args.batches:
        inputs = _cpu_inputs(batch_size, args.channels, args.height, args.width, args.seed)
        batch_result: dict[str, Any] = {
            "input_shape": [args.channels, args.height, args.width],
            "input_count": batch_size,
            "cpu_inputs_pinned": [image.is_pinned() for image in inputs],
            "modes": {},
        }
        for name, transfer in modes.items():
            batch_result["modes"][name] = _run_mode(transfer, inputs, device, args.warmup, args.repetitions)
        result["results"][str(batch_size)] = batch_result

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark.

    Args:
        argv: Optional argument sequence excluding the executable name.

    Returns:
        Zero on a completed CUDA benchmark; two for invalid or unavailable CUDA setup.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except RuntimeError as error:
        parser.error(str(error))

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
