# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video file I/O helpers for the demo pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from rfdetr_demo.inference.progress import PreviewThrottle
from rfdetr_demo.inference.types import ProgressCallback, VideoProcessingCancelledError


def probe_video_size(source_path: Path) -> tuple[int, int, float]:
    """Return width, height, and fps for a video file."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {source_path}: {width}x{height}")
    return width, height, float(fps)


def finalize_video_path(partial_path: Path, target_path: Path) -> None:
    """Atomically promote a finished partial MP4 to the final output path."""
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        raise RuntimeError(f"Partial video output is missing or empty: {partial_path}")
    if target_path.exists():
        target_path.unlink()
    partial_path.replace(target_path)


def cleanup_partial_video(partial_path: Path) -> None:
    """Remove a partial MP4 after a failed export."""
    if partial_path.is_file():
        partial_path.unlink()


def partial_video_path(target_path: Path) -> Path:
    """Return a sidecar path used while encoding is in progress."""
    return target_path.with_name(f"{target_path.stem}.partial{target_path.suffix}")


def effective_source_frame_limit(
    total_frames: int,
    fps: float,
    max_source_seconds: float | None,
) -> int:
    """Return how many source frames to encode before stopping."""
    if max_source_seconds is None:
        return total_frames if total_frames > 0 else 0
    if max_source_seconds <= 0:
        msg = f"max_source_seconds must be > 0, got {max_source_seconds}"
        raise ValueError(msg)
    by_seconds = max(1, int(max_source_seconds * fps))
    if total_frames <= 0:
        return by_seconds
    return min(total_frames, by_seconds)


def count_inference_targets(
    total_frames: int,
    frame_stride: int,
    max_frames: int | None,
    max_source_seconds: float | None = None,
    fps: float = 30.0,
) -> int:
    """Estimate how many frames will run through the inference callback."""
    effective_frames = effective_source_frame_limit(total_frames, fps, max_source_seconds)
    if effective_frames <= 0:
        return 0
    inferred = (effective_frames + frame_stride - 1) // frame_stride
    if max_frames is not None:
        return min(max_frames, inferred)
    return inferred


def process_video(
    source_path: Path,
    target_path: Path,
    callback: Callable[[np.ndarray, int], np.ndarray],
    frame_stride: int,
    max_frames: int | None,
    stats: dict[str, int] | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_throttle: PreviewThrottle | None = None,
    cancel_event: Event | None = None,
    max_source_seconds: float | None = None,
) -> None:
    """Decode video, run ``callback`` on selected frames, and write annotated MP4."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions for {source_path}: {width}x{height}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    source_frame_limit = effective_source_frame_limit(total_frames, fps, max_source_seconds)
    inference_targets = count_inference_targets(
        total_frames,
        frame_stride,
        max_frames,
        max_source_seconds=max_source_seconds,
        fps=fps,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer: {target_path}")

    frame_index = 0
    processed = 0
    last_annotated: np.ndarray | None = None
    progress_stats = stats if stats is not None else {}

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise VideoProcessingCancelledError("Video export cancelled by user.")

            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                last_annotated = callback(frame_bgr, frame_index)
                processed += 1
                if preview_throttle is not None and last_annotated is not None:
                    preview_throttle.maybe_emit(last_annotated, frame_index, processed)
                if progress_callback is not None:
                    total = inference_targets if inference_targets > 0 else processed
                    progress_callback(processed, total, progress_stats)

            output_frame = last_annotated if last_annotated is not None else frame_bgr
            writer.write(output_frame)
            frame_index += 1

            if max_source_seconds is not None:
                if source_frame_limit > 0 and frame_index >= source_frame_limit:
                    break
            elif max_frames is not None and processed >= max_frames:
                break
    finally:
        capture.release()
        writer.release()


# Backward-compatible aliases
_probe_video_size = probe_video_size
_finalize_video_path = finalize_video_path
_cleanup_partial_video = cleanup_partial_video
_partial_video_path = partial_video_path
_process_video = process_video
