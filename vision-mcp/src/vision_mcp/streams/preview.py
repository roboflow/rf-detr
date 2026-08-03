# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The MJPEG debug preview.

This is a debugging aid, not a product surface. It is off unless `debug.preview` is set, it serves on loopback only, and
it runs on a copy of the last processed frame at its own frame rate — a browser that cannot keep up slows nothing but
itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import cv2
import numpy as np

from vision_mcp.clock import monotonic
from vision_mcp.config import DebugConfig, StreamEntry
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.logging_setup import get_logger
from vision_mcp.streams.annotate import Annotator, Overlay
from vision_mcp.streams.runtime import PreviewFrame, StreamRuntime

logger = get_logger("vision-mcp.preview")

BOUNDARY = "visionmcpframe"
"""Multipart boundary; the HTTP layer must advertise the same token."""

CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
"""Response content type for an MJPEG stream."""

_IDLE_POLL_SECONDS = 0.05
"""How long to wait before re-checking for a frame that has not arrived yet."""


class PreviewService:
    """Turns processed frames into an MJPEG byte stream on demand."""

    def __init__(self, debug: DebugConfig) -> None:
        self._debug = debug
        self._annotator = Annotator()
        self._clients = 0

    @property
    def enabled(self) -> bool:
        """Whether the operator turned the preview on."""
        return self._debug.preview

    @property
    def clients(self) -> int:
        """How many browsers are watching right now."""
        return self._clients

    def require_enabled(self) -> None:
        """Guard the preview routes.

        Raises:
            VisionError: INVALID_ARGUMENT when the preview is disabled in config.
        """
        if not self._debug.preview:
            raise VisionError(
                ErrorCode.INVALID_ARGUMENT,
                "The debug preview is disabled. Set debug.preview to true to enable it.",
            )

    def render(self, runtime: StreamRuntime, frame: PreviewFrame) -> bytes:
        """Annotate one frame and encode it as JPEG."""
        overlay = _overlay(runtime, frame)
        canvas = self._annotator.render(frame.array, frame.raw, frame.detections, overlay)
        return encode_jpeg(canvas, self._debug.jpeg_quality)

    async def snapshot(self, runtime: StreamRuntime) -> bytes:
        """A single annotated JPEG.

        Raises:
            VisionError: STREAM_DISCONNECTED when no frame has been processed yet.
        """
        self.require_enabled()
        frame = runtime.preview_frame()
        if frame is None:
            raise VisionError(
                ErrorCode.STREAM_DISCONNECTED,
                "No frame has been processed for this stream yet.",
                {"stream_id": runtime.stream_id},
            )
        return await asyncio.to_thread(self.render, runtime, frame)

    async def mjpeg(self, runtime: StreamRuntime) -> AsyncIterator[bytes]:
        """Yield multipart JPEG parts until the client disconnects."""
        self.require_enabled()
        interval = 1.0 / self._debug.preview_fps
        self._clients += 1
        logger.info(
            "preview client connected",
            extra={"stream_id": runtime.stream_id, "clients": self._clients},
        )
        last_index = -1
        try:
            while True:
                started = monotonic()
                frame = runtime.preview_frame()
                if frame is None or frame.index == last_index:
                    await asyncio.sleep(_IDLE_POLL_SECONDS)
                    continue
                last_index = frame.index
                jpeg = await asyncio.to_thread(self.render, runtime, frame)
                yield _part(jpeg)
                await asyncio.sleep(max(0.0, interval - (monotonic() - started)))
        finally:
            self._clients -= 1
            logger.info(
                "preview client disconnected",
                extra={"stream_id": runtime.stream_id, "clients": self._clients},
            )


def _overlay(runtime: StreamRuntime, frame: PreviewFrame) -> Overlay:
    """Resolve the stream's zones and lines to pixels and build the heads-up display."""
    height, width = frame.array.shape[:2]
    return Overlay(
        zones=resolve_zones(runtime.entry, width, height),
        lines=resolve_lines(runtime.entry, width, height),
        zone_counts=frame.zone_counts,
        line_counts=frame.line_counts,
        hud=_hud(runtime, frame),
    )


def _hud(runtime: StreamRuntime, frame: PreviewFrame) -> list[str]:
    """The stats panel: what an operator needs to see that the pipeline is honest."""
    status = runtime.status()
    height, width = frame.array.shape[:2]
    return [
        f"{runtime.stream_id}  [{status.state}/{status.health}]",
        f"{status.model}  {width}x{height}",
        f"{status.processed_fps:.1f}/{status.target_fps:.1f} fps   {frame.inference_ms:.0f} ms",
        f"objects {status.current_objects}   tracks {status.active_tracks}",
        f"queue {status.queue_depth}/{status.queue_capacity}   dropped {status.dropped_frames}",
    ]


def resolve_zones(entry: StreamEntry, width: int, height: int) -> dict[str, list[tuple[int, int]]]:
    """Convert normalised zone polygons to pixel coordinates for this frame size."""
    return {
        name: [_to_pixels(point, width, height) for point in zone.polygon]
        for name, zone in entry.zones.items()
    }


def resolve_lines(
    entry: StreamEntry, width: int, height: int
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Convert normalised counting lines to pixel coordinates for this frame size."""
    return {
        name: (_to_pixels(line.start, width, height), _to_pixels(line.end, width, height))
        for name, line in entry.lines.items()
    }


def _to_pixels(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    """Normalised (0-1) coordinates to pixels, clamped inside the frame."""
    x = min(width - 1, max(0, round(point[0] * width)))
    y = min(height - 1, max(0, round(point[1] * height)))
    return x, y


def encode_jpeg(canvas: np.ndarray[Any, Any], quality: int) -> bytes:
    """Encode an RGB frame as JPEG bytes.

    Raises:
        VisionError: INFERENCE_FAILED when OpenCV cannot encode the frame.
    """
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VisionError(ErrorCode.INFERENCE_FAILED, "Could not encode the preview frame.")
    return bytes(buffer)


def _part(jpeg: bytes) -> bytes:
    """One multipart chunk of an MJPEG response."""
    header = (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n").encode(
        "ascii"
    )
    return header + jpeg + b"\r\n"
