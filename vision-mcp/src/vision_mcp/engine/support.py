"""Small validation and platform helpers for engine tool dispatch."""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import BaseModel

from vision_mcp.api_contract import GpuDevice, GpuMetrics, HistoricalQuery
from vision_mcp.errors import ErrorCode, VisionError

HISTORICAL_TO_QUERY = {
    "get_counts_by_class": "counts_by_class",
    "get_unique_object_count": "unique_objects",
    "get_detection_rate": "detection_rate",
    "get_confidence_distribution": "confidence",
    "get_entry_exit_counts": "entry_exit",
    "get_dwell_times": "dwell",
    "get_line_crossing_events": "crossings",
    "get_inference_latency": "latency",
    "get_processing_throughput": "throughput",
    "get_frame_drop_rate": "drop_rate",
}


def required(args: BaseModel, name: str) -> str:
    """Read a required non-empty string argument."""
    item = getattr(args, name, None)
    if not isinstance(item, str) or not item:
        raise VisionError(ErrorCode.INVALID_ARGUMENT, f"{name} is required.", {"field": name})
    return item


def value(args: BaseModel, name: str, default: Any = None) -> Any:
    """Read an argument from a Pydantic open model."""
    return getattr(args, name, default)


def optional_string(args: BaseModel, name: str) -> str | None:
    """Read an optional non-empty string."""
    item = value(args, name)
    return item if isinstance(item, str) and item else None


def optional_float(args: BaseModel, name: str) -> float | None:
    """Read and validate an optional numeric argument."""
    item = value(args, name)
    if item is None:
        return None
    if not isinstance(item, (int, float)):
        raise VisionError(ErrorCode.INVALID_ARGUMENT, f"{name} must be a number.")
    return float(item)


def optional_strings(args: BaseModel, name: str) -> list[str] | None:
    """Read and validate an optional list of strings."""
    item = value(args, name)
    if item is None:
        return None
    if not isinstance(item, list) or not all(isinstance(entry, str) for entry in item):
        raise VisionError(ErrorCode.INVALID_ARGUMENT, f"{name} must be a list of strings.")
    return item


def limit(args: BaseModel) -> int:
    """Read a bounded result limit."""
    item = value(args, "limit", 50)
    if not isinstance(item, int) or not 1 <= item <= 500:
        raise VisionError(ErrorCode.INVALID_ARGUMENT, "limit must be between 1 and 500.")
    return item


def period(args: BaseModel) -> HistoricalQuery:
    """Validate only the shared historical fields from a larger tool request."""
    return HistoricalQuery.model_validate(
        {
            "time_window": value(args, "time_window", "15m"),
            "interval": value(args, "interval", "1m"),
        }
    )


def gpu_metrics() -> GpuMetrics:
    """Read NVIDIA state when available; absence is a normal structured response."""
    try:
        import pynvml
    except ImportError:
        return GpuMetrics(available=False, reason="pynvml is not installed", device="nvidia", devices=[])
    try:
        pynvml.nvmlInit()
        devices = [_gpu_device(pynvml, index) for index in range(pynvml.nvmlDeviceGetCount())]
        return GpuMetrics(available=True, reason=None, device="nvidia", devices=devices)
    except pynvml.NVMLError as exc:
        return GpuMetrics(available=False, reason=str(exc), device="nvidia", devices=[])
    finally:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


def _gpu_device(pynvml: Any, index: int) -> GpuDevice:
    """Read one NVIDIA device."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilisation = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature: float | None = None
    with contextlib.suppress(pynvml.NVMLError):
        temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    return GpuDevice(
        index=index,
        name=str(pynvml.nvmlDeviceGetName(handle)),
        memory_used_mb=round(memory.used / 1024**2, 1),
        memory_total_mb=round(memory.total / 1024**2, 1),
        utilisation_percent=float(utilisation.gpu),
        temperature_c=temperature,
    )
