# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Measure legacy and preallocated segmentation-mask postprocessing.

Purpose:
    Provide a small, reproducible before/after harness for the chunked mask resize change. The benchmark measures the
    same interpolation and threshold operations used by ``PostProcess._postprocess_masks`` and emits machine-readable
    JSON so reviewers can retain raw output instead of relying on prose summaries.

Scope:
    This script compares only the mask-resize allocation strategies. It does not load a model, download weights, or
    claim end-to-end inference performance. Each invocation measures one strategy and one K value; run separate
    invocations for ``legacy`` and ``preallocated`` and for K={100,200,300}. CPU RSS is a process high-water value;
    CUDA memory is PyTorch's peak allocated bytes after input creation. The methods are intentionally reported.

Usage:
    From the repository root, run ``uv run --no-sync python benchmarks/postprocess_mask_memory.py --variant legacy
    --k 100`` and repeat for both variants and K values. Use ``--height 1080 --width 1920`` for the PR's large
    synthetic case, or smaller dimensions for a quick smoke measurement. Set ``--device cuda`` only on a CUDA host.

Outputs:
    One JSON object on stdout containing the seed, tensor shape, torch/Python/platform versions, synchronization
    policy, warmup/repeat counts, median/min/max wall time, output true-count checksum, peak-memory value, and the
    exact peak-memory method. The output is suitable for preserving as a review artifact alongside the command.

Failure:
    The command exits non-zero for an unknown variant, unavailable requested CUDA device, invalid dimensions, or a
    runtime failure in either implementation. It never suppresses PyTorch errors, so unsupported backend behavior is
    visible to the caller and cannot be mistaken for a successful benchmark.

Used by:
    Maintainers and reviewers evaluating PR #1374 or later changes to ``PostProcess._postprocess_masks``. The script
    is deliberately standalone and has no package import side effects; the production implementation remains the
    source of behavior, while this file supplies a repeatable comparison harness and explicit measurement metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_MASK_CHUNK = 32
_SEED = 0


def _synchronize(device: torch.device) -> None:
    """Synchronize the measured device before and after timed work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _legacy_resize(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize masks with the list-plus-concatenate strategy used before preallocation."""
    chunks = [
        F.interpolate(
            masks[start : start + _MASK_CHUNK].unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        > 0.0
        for start in range(0, masks.shape[0], _MASK_CHUNK)
    ]
    return torch.cat(chunks, dim=0)


def _preallocated_resize(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize masks into one preallocated boolean destination."""
    result = masks.new_empty((masks.shape[0], 1, height, width), dtype=torch.bool)
    for start in range(0, masks.shape[0], _MASK_CHUNK):
        interpolated = F.interpolate(
            masks[start : start + _MASK_CHUNK].unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        torch.gt(interpolated, 0.0, out=result[start : start + _MASK_CHUNK])
        del interpolated
    return result


def _process_max_rss_bytes() -> int:
    """Return process high-water RSS in bytes for the current platform."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _measure(
    variant: str,
    device: torch.device,
    num_masks: int,
    height: int,
    width: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure one resize strategy and return raw timing and memory metadata."""
    resize: Callable[[torch.Tensor, int, int], torch.Tensor]
    if variant == "legacy":
        resize = _legacy_resize
    elif variant == "preallocated":
        resize = _preallocated_resize
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    torch.manual_seed(_SEED)
    masks = torch.randn(num_masks, 192, 192, device=device)
    _synchronize(device)
    for _ in range(warmup):
        output = resize(masks, height, width)
        _synchronize(device)
        del output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms: list[float] = []
    output_true_count = -1
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        output = resize(masks, height, width)
        _synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        output_true_count = int(output.sum().item())
        del output

    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        peak_memory_method = "torch.cuda.max_memory_allocated after input creation"
    else:
        peak_memory = _process_max_rss_bytes()
        peak_memory_method = "resource.getrusage(RUSAGE_SELF).ru_maxrss process high-water RSS"

    return {
        "variant": variant,
        "device": str(device),
        "num_masks": num_masks,
        "height": height,
        "width": width,
        "mask_head_shape": [192, 192],
        "chunk_size": _MASK_CHUNK,
        "seed": _SEED,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "output_true_count": output_true_count,
        "peak_memory_bytes": peak_memory,
        "peak_memory_method": peak_memory_method,
        "synchronized": True,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("legacy", "preallocated"), required=True)
    parser.add_argument("--k", type=int, required=True, dest="num_masks")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.num_masks <= 0 or args.height <= 0 or args.width <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("K, height, width, and repeats must be positive; warmup must be non-negative")
    return args


def main() -> None:
    """Run one benchmark case and print its JSON result."""
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    result = _measure(
        args.variant,
        device,
        args.num_masks,
        args.height,
        args.width,
        args.warmup,
        args.repeats,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
