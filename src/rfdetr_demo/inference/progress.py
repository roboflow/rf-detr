# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Progress reporting and preview throttling for video export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rfdetr_demo.inference.preview_util import resize_bgr_for_preview
from rfdetr_demo.inference.types import PreviewCallback, ProgressCallback


def compose_progress_callback(
    progress_callback: ProgressCallback | None,
    progress_file: Path | None,
) -> ProgressCallback | None:
    """Combine optional UI callback with JSON progress file writes."""
    if progress_callback is None and progress_file is None:
        return None

    def combined(current: int, total: int, stats: dict[str, int]) -> None:
        if progress_file is not None:
            payload = {"current": current, "total": total, "stats": stats}
            progress_file.write_text(json.dumps(payload), encoding="utf-8")
        if progress_callback is not None:
            progress_callback(current, total, stats)

    return combined


class PreviewThrottle:
    """Rate-limit preview callbacks so GUI updates do not slow inference."""

    def __init__(
        self,
        callback: PreviewCallback,
        *,
        stride: int = 1,
        min_interval_sec: float = 0.12,
        max_width: int = 400,
    ) -> None:
        if stride < 1:
            raise ValueError(f"preview stride must be >= 1, got {stride}")
        if min_interval_sec < 0:
            raise ValueError(f"preview min_interval_sec must be >= 0, got {min_interval_sec}")
        self._callback = callback
        self._stride = stride
        self._min_interval_sec = min_interval_sec
        self._max_width = max_width
        self._last_emit_at = 0.0

    def maybe_emit(self, annotated_bgr: np.ndarray, frame_index: int, processed: int) -> None:
        """Emit a downscaled preview frame when stride and interval allow."""
        if processed % self._stride != 0:
            return
        now = time.perf_counter()
        if self._min_interval_sec > 0 and (now - self._last_emit_at) < self._min_interval_sec:
            return
        self._last_emit_at = now
        preview_bgr = resize_bgr_for_preview(annotated_bgr, self._max_width)
        self._callback(preview_bgr, frame_index, processed)


def compose_preview_callback(
    preview_callback: PreviewCallback | None,
    *,
    preview_stride: int | None,
    preview_min_interval_sec: float,
    preview_max_width: int,
    frame_stride: int,
) -> PreviewThrottle | None:
    """Build a throttled preview emitter for annotated frames."""
    if preview_callback is None:
        return None
    stride = preview_stride if preview_stride is not None else max(1, frame_stride)
    return PreviewThrottle(
        preview_callback,
        stride=stride,
        min_interval_sec=preview_min_interval_sec,
        max_width=preview_max_width,
    )


_compose_progress_callback = compose_progress_callback
_compose_preview_callback = compose_preview_callback
