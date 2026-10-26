# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Source expansion for the prediction facade."""

import csv
import glob
import os
import warnings
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

import numpy as np
import torch

from rfdetr._prediction_streams import iter_live_frames
from rfdetr.prediction import ImageInput, PredictionInput

#: Image extensions included when expanding directories and globs.
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".gif",
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
#: Extensions that identify finite video sources.
_VIDEO_SUFFIXES = {".asf", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv"}
#: Supported source manifest formats.
_MANIFEST_SUFFIXES = {".txt", ".csv", ".streams"}


#: Source categories used by routing and manifest validation.
_SourceKind: TypeAlias = Literal[
    "image", "video", "http_video", "camera", "live", "youtube", "screen", "directory", "glob", "manifest", "streams"
]


def _classify_source(
    source: str, *, relative_to: Path | None = None, stream_entry: bool = False
) -> tuple[str, _SourceKind]:
    """Resolve path spelling and classify a source without opening readers or resolving YouTube URLs.

    Relative files use the manifest directory when supplied. Existing files take precedence over screen commands.
    Extensionless HTTP URLs are images unless used as simultaneous stream entries.
    """
    if source.isdecimal():
        return source, "camera"
    url = urlparse(source)
    if url.scheme in ("rtsp", "rtsps", "rtmp", "tcp"):
        return source, "live"
    if url.scheme in ("http", "https"):
        host = (url.hostname or "").lower()
        if host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"):
            return source, "youtube"
        suffix = Path(url.path).suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            return source, "http_video"
        if suffix in {".m3u8", ".mjpg", ".mjpeg"} or (stream_entry and suffix not in _IMAGE_SUFFIXES):
            return source, "live"
        return source, "image"

    path = Path(source)
    if relative_to is not None and not url.scheme and not path.is_absolute():
        path = relative_to / path
    if path.suffix.lower() in _MANIFEST_SUFFIXES:
        return str(path), "streams" if path.suffix.lower() == ".streams" else "manifest"
    if source.split(" ", 1)[0] == "screen" and not path.is_file():
        return source, "screen"
    if path.is_dir():
        return str(path), "directory"
    if glob.has_magic(str(path)) and not path.is_file():
        return str(path), "glob"
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return str(path), "video"
    return str(path), "image"


def is_expanded_source(source: PredictionInput) -> bool:
    """Identify inputs that require source expansion."""
    if isinstance(source, (list, tuple)):
        return any(is_expanded_source(item) for item in source)
    if isinstance(source, torch.Tensor):
        return source.ndim == 4
    if isinstance(source, int):
        return True
    if isinstance(source, (str, os.PathLike)):
        _, kind = _classify_source(os.fspath(source))
        return kind != "image"
    return False


def iter_source_batches(
    source: PredictionInput,
    *,
    batch: int = 1,
    vid_stride: int = 1,
    stream_buffer: bool = False,
    warn_on_live: bool = False,
) -> Generator[list[ImageInput], None, None]:
    """Batch finite images and preserve simultaneous live-stream batches."""
    groups = _iter_source_groups(
        source, vid_stride=vid_stride, stream_buffer=stream_buffer, manifests=(), warn_on_live=warn_on_live
    )
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
    source: PredictionInput, *, vid_stride: int, stream_buffer: bool, manifests: tuple[Path, ...], warn_on_live: bool
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Expand nested sources while retaining live-stream batch boundaries."""
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from _iter_source_groups(
                item, vid_stride=vid_stride, stream_buffer=stream_buffer, manifests=manifests, warn_on_live=warn_on_live
            )
        return
    if isinstance(source, torch.Tensor) and source.ndim == 4:
        for image in source:
            yield [image], False
        return
    if isinstance(source, int):
        if isinstance(source, bool) or source < 0:
            raise ValueError("The webcam index must be a non-negative integer.")
        yield from _live_groups([source], vid_stride=vid_stride, stream_buffer=stream_buffer, warn_on_live=warn_on_live)
        return
    if not isinstance(source, (str, os.PathLike)):
        yield [source], False
        return
    path, kind = _classify_source(os.fspath(source))
    if kind in ("manifest", "streams"):
        manifest = Path(path).resolve()
        if manifest in manifests:
            raise ValueError(f"Prediction manifest cycle: {manifest}.")
        entries = _manifest_entries(manifest)
        try:
            if kind == "streams":
                resolved_sources = [_stream_entry(item) for item in entries]
                sources = [item for item, _ in resolved_sources]
                if not sources:
                    raise ValueError("The stream manifest contains no sources.")
                yield from _live_groups(
                    sources,
                    finite_sources=frozenset(item for item, live in resolved_sources if not live),
                    vid_stride=vid_stride,
                    stream_buffer=stream_buffer,
                    warn_on_live=warn_on_live and any(live for _, live in resolved_sources),
                )
            else:
                for entry in entries:
                    yield from _iter_source_groups(
                        entry,
                        vid_stride=vid_stride,
                        stream_buffer=stream_buffer,
                        manifests=(*manifests, manifest),
                        warn_on_live=warn_on_live,
                    )
        finally:
            entries.close()
        return
    if kind == "screen":
        if warn_on_live:
            _warn_live_accumulation()
        screenshots = _iter_screenshots(path)
        try:
            for image in screenshots:
                yield [image], True
        finally:
            screenshots.close()
        return
    if kind == "youtube":
        path, live = _resolve_youtube(path)
        kind = "live" if live else "http_video"
    if kind in ("camera", "live"):
        yield from _live_groups(
            [int(path) if kind == "camera" else path],
            vid_stride=vid_stride,
            stream_buffer=stream_buffer,
            warn_on_live=warn_on_live,
        )
        return
    paths = None
    if kind == "directory":
        paths = sorted(Path(path).iterdir())
    elif kind == "glob":
        paths = sorted(Path(match) for match in glob.iglob(path, recursive=True))
    if paths is not None:
        media = [item for item in paths if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES]
        if not media:
            raise FileNotFoundError(f"No supported images or videos found for {path!r}.")
        for item in media:
            yield from _iter_source_groups(
                item, vid_stride=vid_stride, stream_buffer=stream_buffer, manifests=manifests, warn_on_live=warn_on_live
            )
        return
    if kind in ("video", "http_video"):
        if kind == "video" and not Path(path).is_file():
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
    sources: list[str | int],
    *,
    vid_stride: int,
    stream_buffer: bool,
    warn_on_live: bool,
    finite_sources: frozenset[str | int] = frozenset(),
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Close background readers when prediction stops."""
    if warn_on_live:
        _warn_live_accumulation()
    batches = iter_live_frames(
        sources, vid_stride=vid_stride, stream_buffer=stream_buffer, finite_sources=finite_sources
    )
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
                resolved, _ = _classify_source(entry, relative_to=path.parent)
                yield resolved


def _resolve_youtube(source: str) -> tuple[str, bool]:
    """Resolve a YouTube page to a decodable video URL."""
    try:
        yt_dlp = import_module("yt_dlp")
    except ImportError as error:
        raise ImportError("YouTube prediction requires yt-dlp. Install rfdetr[stream].") from error
    # Prefer H.264 because some OpenCV builds cannot decode YouTube's AV1 formats.
    with yt_dlp.YoutubeDL(
        {
            "format": "bestvideo[vcodec^=avc1]/best[vcodec^=avc1]/bestvideo[ext=mp4]/best[ext=mp4]/bestvideo/best",
            "quiet": True,
            "noplaylist": True,
        }
    ) as downloader:
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
        mss = import_module("mss")
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
            yield np.asarray(capture.grab(bounds))[:, :, 2::-1].copy()


def _iter_video_frames(source: str | int, vid_stride: int = 1) -> Generator[np.ndarray[Any, Any], None, None]:
    """Release the capture on exhaustion, errors, or generator closure."""
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Video prediction requires OpenCV. Install rfdetr[stream].") from error

    if isinstance(source, str) and urlparse(source).scheme in ("http", "https"):
        capture = cv2.VideoCapture(
            source,
            cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )
    else:
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


def _stream_entry(source: str) -> tuple[str | int, bool]:
    """Validate one simultaneous capture source before any capture opens."""
    source, kind = _classify_source(source, stream_entry=True)
    if kind == "camera":
        return int(source), True
    if kind == "youtube":
        return _resolve_youtube(source)
    if kind == "live":
        return source, True
    if kind in ("video", "http_video"):
        if kind == "video" and not Path(source).is_file():
            raise FileNotFoundError(f"Video file does not exist: {source!r}.")
        return source, False
    raise ValueError(f"Invalid stream manifest entry: {source!r}. Use a camera index, video file, or stream URL.")


def _warn_live_accumulation() -> None:
    """Warn before an eager prediction starts an unbounded source."""
    # Attribute this to the source-opening site; manifest nesting makes caller depth variable.
    warnings.warn(
        "Live results accumulate in memory with stream=False. Use stream=True to limit memory use.",
        UserWarning,
        stacklevel=2,
    )
