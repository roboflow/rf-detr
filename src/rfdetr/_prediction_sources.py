# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Source expansion for the prediction facade."""

import csv
import glob
import os
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image

from rfdetr._prediction_streams import iter_live_frames

ImageInput = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
PredictionSource = ImageInput | int
PredictionInput = PredictionSource | list[PredictionSource] | tuple[PredictionSource, ...]

_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".ico",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_VIDEO_SUFFIXES = {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv"}
_MANIFEST_SUFFIXES = {".txt", ".csv", ".streams"}


def is_expanded_source(source: PredictionInput) -> bool:
    """Identify inputs that require source expansion."""
    if isinstance(source, (list, tuple)):
        return any(is_expanded_source(item) for item in source)
    if isinstance(source, torch.Tensor):
        return source.ndim == 4
    if is_live_source(source):
        return True
    if not isinstance(source, (str, os.PathLike)):
        return False
    path = os.fspath(source)
    url = urlparse(path)
    if url.scheme in ("http", "https"):
        return Path(url.path).suffix.lower() in _VIDEO_SUFFIXES
    return (
        Path(path).is_dir()
        or (glob.has_magic(path) and not Path(path).is_file())
        or Path(path).suffix.lower() in _VIDEO_SUFFIXES | _MANIFEST_SUFFIXES
    )


def is_live_source(source: PredictionInput) -> bool:
    """Identify sources that can produce an unbounded number of frames."""
    if isinstance(source, (list, tuple)):
        return any(is_live_source(item) for item in source)
    if isinstance(source, int):
        return True
    if not isinstance(source, (str, os.PathLike)):
        return False
    path = os.fspath(source)
    url = urlparse(path)
    return (
        path.isdecimal()
        or path.split(" ", 1)[0] == "screen"
        or Path(path).suffix.lower() == ".streams"
        or url.scheme.lower() in ("rtsp", "rtsps", "rtmp", "tcp")
        or (
            url.scheme.lower() in ("http", "https")
            and Path(url.path).suffix.lower() not in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES
        )
    )


def iter_source_batches(
    source: PredictionInput, *, batch: int = 1, vid_stride: int = 1, stream_buffer: bool = False
) -> Generator[list[ImageInput], None, None]:
    """Batch finite images and preserve simultaneous live-stream batches."""
    groups = _iter_source_groups(source, vid_stride, stream_buffer, ())
    pending: list[ImageInput] = []
    try:
        for images, simultaneous in groups:
            if simultaneous:
                if pending:
                    yield pending
                    pending = []
                yield images
            else:
                for image in images:
                    pending.append(image)
                    if len(pending) == batch:
                        yield pending
                        pending = []
        if pending:
            yield pending
    finally:
        groups.close()


def _iter_source_groups(
    source: PredictionInput, vid_stride: int, stream_buffer: bool, manifests: tuple[Path, ...]
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Expand nested sources while retaining live-stream batch boundaries."""
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from _iter_source_groups(item, vid_stride, stream_buffer, manifests)
        return
    if isinstance(source, torch.Tensor) and source.ndim == 4:
        for image in source:
            yield [image], False
        return
    if isinstance(source, int):
        if isinstance(source, bool) or source < 0:
            raise ValueError("The webcam index must be a non-negative integer.")
        yield from _live_groups([source], vid_stride, stream_buffer)
        return
    if not isinstance(source, (str, os.PathLike)):
        yield [source], False
        return
    path = os.fspath(source)
    url = urlparse(path)
    remote = url.scheme in ("http", "https", "rtsp", "rtsps", "rtmp", "tcp")
    if not remote and Path(path).suffix.lower() in _MANIFEST_SUFFIXES:
        manifest = Path(path).resolve()
        if manifest in manifests:
            raise ValueError(f"Prediction manifest cycle: {manifest}.")
        entries = _manifest_entries(manifest)
        try:
            if manifest.suffix.lower() == ".streams":
                sources = [int(item) if item.isdecimal() else _resolve_youtube(item)[0] for item in entries]
                if not sources:
                    raise ValueError("The stream manifest contains no sources.")
                yield from _live_groups(sources, vid_stride, stream_buffer)
            else:
                for entry in entries:
                    yield from _iter_source_groups(entry, vid_stride, stream_buffer, (*manifests, manifest))
        finally:
            entries.close()
        return
    if path.split(" ", 1)[0] == "screen":
        screenshots = _iter_screenshots(path)
        try:
            for image in screenshots:
                yield [image], True
        finally:
            screenshots.close()
        return
    if is_live_source(path):
        resolved, live = _resolve_youtube(path)
        if live:
            yield from _live_groups([int(path) if path.isdecimal() else resolved], vid_stride, stream_buffer)
        else:
            frames = _iter_video_frames(resolved, vid_stride)
            try:
                for frame in frames:
                    yield [frame], False
            finally:
                frames.close()
        return
    paths = None
    if not remote:
        if Path(path).is_dir():
            paths = sorted(Path(path).iterdir())
        elif glob.has_magic(path) and not Path(path).is_file():
            paths = sorted(Path(match) for match in glob.iglob(path, recursive=True))
    if paths is not None:
        media = [item for item in paths if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES]
        if not media:
            raise FileNotFoundError(f"No supported images or videos found for {path!r}.")
        for item in media:
            yield from _iter_source_groups(item, vid_stride, stream_buffer, manifests)
        return
    if Path(url.path if remote else path).suffix.lower() in _VIDEO_SUFFIXES:
        if not remote and not Path(path).is_file():
            raise FileNotFoundError(f"Video file does not exist: {path!r}.")
        frames = _iter_video_frames(path, vid_stride)
        try:
            for frame in frames:
                yield [frame], False
        finally:
            frames.close()
        return
    yield [path], False


def _live_groups(
    sources: list[str | int], vid_stride: int, stream_buffer: bool
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Close background readers when prediction stops."""
    batches = iter_live_frames(sources, vid_stride=vid_stride, stream_buffer=stream_buffer)
    try:
        for frames in batches:
            yield list(frames), True
    finally:
        batches.close()


def _manifest_entries(path: Path) -> Generator[str, None, None]:
    """Read manifest entries relative to the containing directory."""
    with path.open(encoding="utf-8-sig", newline="") as manifest:
        rows = csv.reader(manifest) if path.suffix.lower() == ".csv" else ([line] for line in manifest)
        for index, row in enumerate(rows):
            if index == 0 and len(row) == 1 and row[0].strip().lower() in ("source", "path"):
                continue
            for cell in row:
                entry = cell.strip()
                if not entry or entry.startswith("#"):
                    continue
                if (
                    not urlparse(entry).scheme
                    and not entry.isdecimal()
                    and entry.split(" ", 1)[0] != "screen"
                    and not Path(entry).is_absolute()
                ):
                    entry = str(path.parent / entry)
                yield entry


def _resolve_youtube(source: str) -> tuple[str, bool]:
    """Resolve a YouTube page to a decodable video URL."""
    host = (urlparse(source).hostname or "").lower()
    if host != "youtu.be" and host != "youtube.com" and not host.endswith(".youtube.com"):
        return source, True
    try:
        yt_dlp = import_module("yt_dlp")
    except ImportError as error:
        raise ImportError("YouTube prediction requires yt-dlp. Install rfdetr[stream].") from error
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True, "noplaylist": True}) as downloader:
        info = downloader.extract_info(source, download=False)
    if not info or not isinstance(info.get("url"), str):
        raise ValueError("Could not resolve a video URL from the YouTube source.")
    return info["url"], bool(info.get("is_live", False))


def _iter_screenshots(source: str) -> Generator[np.ndarray[Any, Any], None, None]:
    """Capture RGB screenshots until the caller closes the generator."""
    try:
        values = [int(value) for value in source.split()[1:]]
    except ValueError as error:
        raise ValueError("Screen coordinates and monitor indexes must be integers.") from error
    if len(values) not in (0, 1, 4, 5):
        raise ValueError("Use screen [monitor] or screen [monitor] left top width height.")
    monitor = values[0] if len(values) in (1, 5) else 0
    region = values[-4:] if len(values) >= 4 else None
    if monitor < 0 or (region is not None and (region[2] <= 0 or region[3] <= 0)):
        raise ValueError("Screen monitor must be non-negative and dimensions must be positive.")
    try:
        import mss
    except ImportError as error:
        raise ImportError("Screenshot prediction requires mss. Install rfdetr[stream].") from error
    with mss.mss() as capture:
        if monitor >= len(capture.monitors):
            raise ValueError(f"Screen monitor {monitor} does not exist.")
        bounds = dict(capture.monitors[monitor])
        if region is not None:
            left, top, width, height = region
            bounds = {"left": bounds["left"] + left, "top": bounds["top"] + top, "width": width, "height": height}
        while True:
            yield cv2.cvtColor(np.asarray(capture.grab(bounds)), cv2.COLOR_BGRA2RGB)


def _iter_video_frames(source: str | int, vid_stride: int = 1) -> Generator[np.ndarray[Any, Any], None, None]:
    """Release the capture on exhaustion, errors, or generator closure."""
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the video source.")
        while True:
            for _ in range(vid_stride):
                success, frame = capture.read()
                if not success:
                    return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
