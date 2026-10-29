# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Source expansion for the prediction facade.

Empty-source policy: an empty collection is quiet, an empty location is loud.

- A collection holds no inputs when it lists none: a manifest (``.txt`` or ``.csv``) with only blank lines, comments or
  a CSV header, a four-dimensional tensor with no frames, or an empty list or tuple. ``stream=True`` yields nothing for
  each, so a pipeline may feed a possibly empty queue. An eager call returns ``[]`` for the manifest and the tensor; an
  empty list or tuple keeps the long-standing eager ``ValueError`` (``At least one image is required.``) because a
  batch with nothing to predict is a caller bug.
- A location that must hold media or sources raises in both modes, because an empty match almost always means a wrong
  path, glob or manifest: a directory or glob without supported images or videos raises ``FileNotFoundError``, and a
  ``.streams`` manifest without sources raises ``ValueError``.

A URL scheme in ``LIVE_STREAM_SCHEMES`` (RTSP, RTSPS, RTMP, TCP, UDP and SRT) names a live stream. Any other scheme is
not a live source and falls through to image handling.
"""

import contextlib
import csv
import glob
import inspect
import operator
import os
import warnings
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from rfdetr._prediction_streams import LIVE_STREAM_SCHEMES, _capture_args, _import_cv2, iter_live_frames
from rfdetr.prediction import ImageInput, PredictionInput
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Image extensions included when expanding directories and globs; HEIF is added by :func:`_image_suffixes`.
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
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
#: Estimated RGB source-image bytes an eager prediction may retain before it warns (1 GiB).
_RETAINED_FRAME_WARNING_BYTES = 2**30

#: Largest webcam index OpenCV accepts.
_MAX_CAMERA_INDEX = 2**31 - 1
#: First-row CSV cells that name the source column of a manifest.
_MANIFEST_HEADERS = ("source", "path")
#: HEIF extensions Pillow opens only after a plugin such as pillow-heif registers them.
_PLUGIN_IMAGE_SUFFIXES = (".heic", ".heif")
#: Tallest YouTube video, in pixels, that resolution selects; frames are resized to the model resolution anyway.
_YOUTUBE_MAX_HEIGHT = 1080
#: Seconds yt-dlp waits on an unresponsive host while resolving a YouTube page.
_YOUTUBE_SOCKET_TIMEOUT = 10.0


def _image_suffixes() -> set[str]:
    """List the image extensions Pillow can open now, adding HEIF only while a plugin has registered an opener.

    A plugin registers at runtime, so the check runs on each call. Only the HEIF extensions are checked against
    ``Image.OPEN``: Pillow also registers save-only extensions (``.pdf``), and it opens ``.mpo`` through the JPEG
    opener, so deriving the whole list from Pillow would add or drop the wrong files.

    Returns:
        Lower-case extensions with a leading dot.

    Examples:
        >>> ".png" in _image_suffixes()
        True
        >>> ".dng" in _image_suffixes()
        False
    """
    registered = Image.registered_extensions()
    return _IMAGE_SUFFIXES | {suffix for suffix in _PLUGIN_IMAGE_SUFFIXES if registered.get(suffix) in Image.OPEN}


class _SourceKind(str, Enum):
    """Source categories used by routing and manifest validation."""

    IMAGE = "image"
    VIDEO = "video"
    HTTP_VIDEO = "http_video"
    CAMERA = "camera"
    LIVE = "live"
    YOUTUBE = "youtube"
    SCREEN = "screen"
    DIRECTORY = "directory"
    GLOB = "glob"
    MANIFEST = "manifest"
    STREAMS = "streams"


def _camera_index(value: int | np.integer[Any]) -> int:
    """Validate a webcam index before any capture opens.

    Args:
        value: Python or NumPy integer naming a camera.

    Returns:
        The index as a Python ``int``.

    Raises:
        ValueError: If the value is a boolean or outside the 32-bit range OpenCV accepts.

    Examples:
        >>> _camera_index(np.int64(2))
        2
        >>> type(_camera_index(np.uint8(0)))
        <class 'int'>
    """
    if isinstance(value, bool):
        raise ValueError("The webcam index must be an integer, not a boolean.")
    index = operator.index(value)
    # OpenCV takes a 32-bit index, including CAP_* backend offsets such as 700 + n.
    if not 0 <= index <= _MAX_CAMERA_INDEX:
        raise ValueError(f"The webcam index must be a non-negative integer no greater than {_MAX_CAMERA_INDEX}.")
    return index


def _classify_source(
    source: str, *, relative_to: Path | None = None, stream_entry: bool = False
) -> tuple[str, _SourceKind]:
    """Resolve path spelling and classify a source without opening readers or resolving YouTube URLs.

    Relative files use the manifest directory when supplied. An existing local file or directory takes precedence over
    camera indexes, screen commands, glob patterns and manifest suffixes. Only ASCII digits spell a camera index, and a
    ``screen`` command never ends in a media or manifest suffix. Extensionless HTTP URLs are images unless used as
    simultaneous stream entries.

    Args:
        source: Path, URL, camera index spelled as digits, or ``screen`` command.
        relative_to: Directory (a manifest's parent) against which a relative local path is resolved. URLs and absolute
            paths ignore it.
        stream_entry: Treat the source as a simultaneous stream-manifest entry, so an HTTP URL whose suffix is not an
            image becomes a live stream instead of an image.

    Returns:
        The source to use from here on and its category. Local paths come back as ``str`` of the resolved path; URLs,
        camera indexes and screen commands come back unchanged.

    Raises:
        ValueError: If the source is empty or whitespace, which would otherwise name the working directory, or if a
            digit-only source is not a valid camera index.
    """
    if not source.strip():
        raise ValueError("The prediction source is empty. Pass an image, path, URL, or camera index.")
    url = urlparse(source)
    if url.scheme in LIVE_STREAM_SCHEMES:
        return source, _SourceKind.LIVE
    if url.scheme in ("http", "https"):
        host = (url.hostname or "").lower()
        if host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"):
            return source, _SourceKind.YOUTUBE
        suffix = Path(url.path).suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            return source, _SourceKind.HTTP_VIDEO
        if suffix in {".m3u8", ".mjpg", ".mjpeg"} or (stream_entry and suffix not in _image_suffixes()):
            return source, _SourceKind.LIVE
        return source, _SourceKind.IMAGE

    path = Path(source)
    if relative_to is not None and not url.scheme and not path.is_absolute():
        path = relative_to / path
    suffix = path.suffix.lower()
    if path.is_dir():
        return str(path), _SourceKind.DIRECTORY
    if not path.is_file():
        if source.isascii() and source.isdigit():
            _camera_index(int(source))
            return source, _SourceKind.CAMERA
        if (
            source.split(" ", 1)[0] == "screen"
            and suffix not in _image_suffixes() | _VIDEO_SUFFIXES | _MANIFEST_SUFFIXES
        ):
            return source, _SourceKind.SCREEN
        if glob.has_magic(str(path)):
            return str(path), _SourceKind.GLOB
    if suffix in _MANIFEST_SUFFIXES:
        return str(path), _SourceKind.STREAMS if suffix == ".streams" else _SourceKind.MANIFEST
    if suffix in _VIDEO_SUFFIXES:
        return str(path), _SourceKind.VIDEO
    return str(path), _SourceKind.IMAGE


def is_expanded_source(source: PredictionInput) -> bool:
    """Identify inputs that require source expansion."""
    if isinstance(source, (list, tuple)):
        return any(is_expanded_source(item) for item in source)
    if isinstance(source, torch.Tensor):
        return source.ndim == 4
    if isinstance(source, (int, np.integer)):
        return True
    if isinstance(source, (str, os.PathLike)):
        _, kind = _classify_source(os.fspath(source))
        return kind != _SourceKind.IMAGE
    return False


def iter_source_batches(
    source: PredictionInput,
    *,
    batch: int = 1,
    vid_stride: int = 1,
    stream_buffer: bool = False,
    warn_on_live: bool = False,
    num_channels: int = 3,
    warn_on_retained_frames: bool = False,
) -> Generator[list[ImageInput], None, None]:
    """Batch finite images and preserve simultaneous live-stream batches.

    Captured video, camera and screen frames are converted for the model's ``num_channels``.

    ``warn_on_retained_frames`` warns before a video, directory, or glob whose estimated RGB source pixels exceed the
    retained-frame threshold; set it when every result keeps its source image.
    """
    groups = _iter_source_groups(
        source,
        vid_stride=vid_stride,
        stream_buffer=stream_buffer,
        manifests=(),
        warn_on_live=warn_on_live,
        num_channels=num_channels,
        warn_on_retained_frames=warn_on_retained_frames,
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
    source: PredictionInput,
    *,
    vid_stride: int,
    stream_buffer: bool,
    manifests: tuple[Path, ...],
    warn_on_live: bool,
    num_channels: int,
    warn_on_retained_frames: bool,
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Expand nested sources recursively into groups of images, retaining live-stream batch boundaries.

    Lists, tuples, directories, globs and non-stream manifests recurse into their items; a 4-D tensor yields one image
    per slice; cameras, live URLs, ``.streams`` manifests, YouTube pages and screens go to live capture; videos yield
    one decoded frame per group; anything else is passed through as a single image. Nothing is opened until the
    generator is advanced, and closing it closes the inner readers.

    Args:
        source: One input or a nested collection of inputs.
        vid_stride: Read every Nth video or live frame.
        stream_buffer: Queue live frames instead of retaining only the latest.
        manifests: Resolved manifest files currently being expanded, used to reject cycles.
        warn_on_live: Warn before an unbounded live source is accumulated eagerly.
        num_channels: Channel count the model expects, applied to captured frames.
        warn_on_retained_frames: Warn when decoded source pixels of a directory, glob or video would exceed the
            retained-frame threshold.

    Yields:
        ``(images, simultaneous)``. ``simultaneous`` is True when the images were captured together from several live
        sources and must stay in one batch; False groups may be merged into larger batches by the caller.

    Raises:
        ValueError: If a manifest includes itself (directly or through nested manifests) or a ``.streams`` manifest
            holds no sources.
        FileNotFoundError: If a directory or glob matches no supported media, or a video file does not exist.
    """
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from _iter_source_groups(
                item,
                vid_stride=vid_stride,
                stream_buffer=stream_buffer,
                manifests=manifests,
                warn_on_live=warn_on_live,
                num_channels=num_channels,
                warn_on_retained_frames=warn_on_retained_frames,
            )
        return
    if isinstance(source, torch.Tensor) and source.ndim == 4:
        for image in source:
            yield [image], False
        return
    if isinstance(source, (int, np.integer)):
        yield from _live_groups(
            [_camera_index(source)],
            vid_stride=vid_stride,
            stream_buffer=stream_buffer,
            warn_on_live=warn_on_live,
            num_channels=num_channels,
        )
        return
    if not isinstance(source, (str, os.PathLike)):
        # The prediction entry point already read other iterables into a list, so this is one image.
        yield [cast("ImageInput", source)], False
        return
    path, kind = _classify_source(os.fspath(source))
    if kind in (_SourceKind.MANIFEST, _SourceKind.STREAMS):
        manifest = Path(path).resolve()
        if manifest in manifests:
            raise ValueError(f"Prediction manifest cycle: {manifest}.")
        entries = _manifest_entries(manifest)
        if kind == _SourceKind.STREAMS:
            resolved_sources = _resolve_stream_entries(entries)
            sources = [item for item, _ in resolved_sources]
            if not sources:
                raise ValueError("The stream manifest contains no sources.")
            yield from _live_groups(
                sources,
                finite_sources=frozenset(item for item, live in resolved_sources if not live),
                vid_stride=vid_stride,
                stream_buffer=stream_buffer,
                warn_on_live=warn_on_live and any(live for _, live in resolved_sources),
                num_channels=num_channels,
            )
        else:
            for entry in entries:
                yield from _iter_source_groups(
                    entry,
                    vid_stride=vid_stride,
                    stream_buffer=stream_buffer,
                    manifests=(*manifests, manifest),
                    warn_on_live=warn_on_live,
                    num_channels=num_channels,
                    warn_on_retained_frames=warn_on_retained_frames,
                )
        return
    if kind == _SourceKind.SCREEN:
        if warn_on_live:
            _warn_live_accumulation()
        screenshots = _iter_screenshots(path)
        try:
            for image in screenshots:
                yield [_frame_input(image, num_channels)], True
        finally:
            screenshots.close()
        return
    if kind == _SourceKind.YOUTUBE:
        path, live = _resolve_youtube(path)
        kind = _SourceKind.LIVE if live else _SourceKind.HTTP_VIDEO
    if kind in (_SourceKind.CAMERA, _SourceKind.LIVE):
        yield from _live_groups(
            [int(path) if kind == _SourceKind.CAMERA else path],
            vid_stride=vid_stride,
            stream_buffer=stream_buffer,
            warn_on_live=warn_on_live,
            num_channels=num_channels,
        )
        return
    paths = None
    if kind == _SourceKind.DIRECTORY:
        paths = sorted(Path(path).iterdir())
    elif kind == _SourceKind.GLOB:
        paths = sorted(Path(match) for match in glob.iglob(path, recursive=True))
    if paths is not None:
        suffixes = _image_suffixes() | _VIDEO_SUFFIXES
        media = [item for item in paths if item.is_file() and item.suffix.lower() in suffixes]
        if not media:
            raise FileNotFoundError(f"No supported images or videos found for {path!r}.")
        if warn_on_retained_frames:
            _warn_retained_images([item for item in media if item.suffix.lower() in _IMAGE_SUFFIXES], path)
        for item in media:
            yield from _iter_source_groups(
                item,
                vid_stride=vid_stride,
                stream_buffer=stream_buffer,
                manifests=manifests,
                warn_on_live=warn_on_live,
                num_channels=num_channels,
                warn_on_retained_frames=warn_on_retained_frames,
            )
        return
    if kind in (_SourceKind.VIDEO, _SourceKind.HTTP_VIDEO):
        if kind == _SourceKind.VIDEO and not Path(path).is_file():
            raise FileNotFoundError(f"Video file does not exist: {path!r}.")
        frames = _iter_video_frames(path, vid_stride, warn_on_retained_frames=warn_on_retained_frames)
        try:
            for frame in frames:
                yield [_frame_input(frame, num_channels)], False
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
    num_channels: int,
    finite_sources: frozenset[str | int] = frozenset(),
) -> Generator[tuple[list[ImageInput], bool], None, None]:
    """Yield frames captured together from live sources, closing background readers when prediction stops.

    Args:
        sources: Camera indexes, stream URLs or files, in batch order.
        vid_stride: Read every Nth frame.
        stream_buffer: Queue frames instead of retaining only the latest.
        warn_on_live: Warn first that eager results of a live source accumulate in memory.
        num_channels: Channel count the model expects, applied to each captured frame.
        finite_sources: Sources known to end, so they stop at end of file instead of reconnecting.

    Yields:
        ``(images, True)`` with one image per source; the flag marks the images as one simultaneous batch.
    """
    if warn_on_live:
        _warn_live_accumulation()
    batches = iter_live_frames(
        sources, vid_stride=vid_stride, stream_buffer=stream_buffer, finite_sources=finite_sources
    )
    try:
        for frames in batches:
            yield [_frame_input(frame, num_channels) for frame in frames], True
    finally:
        batches.close()


def _frame_input(frame: np.ndarray[Any, Any], num_channels: int) -> ImageInput:
    """Hand a captured RGB frame to preprocessing so a one-channel model receives grayscale input.

    A PIL image takes the same luminance conversion as image files; three-channel models keep the array without a copy.

    Args:
        frame: Captured RGB frame in HWC layout.
        num_channels: Channel count the model expects.

    Returns:
        The frame itself, or an RGB PIL image of it for a one-channel model.

    Examples:
        >>> frame = np.zeros((2, 3, 3), dtype=np.uint8)
        >>> _frame_input(frame, 3) is frame
        True
        >>> _frame_input(frame, 1).size
        (3, 2)
    """
    return Image.fromarray(frame) if num_channels == 1 else frame


def _manifest_entries(path: Path) -> list[str | int]:
    """Read every manifest entry, resolved against the manifest directory, before any entry is opened.

    Text and stream manifests hold one source per line. A CSV whose first row has a ``source`` or ``path`` cell treats
    that row as a header and reads only that column, so other columns such as labels are not sources; a CSV without such
    a header treats every cell as a source. Blank entries and ``#`` comments are skipped. The file is closed before the
    first entry is predicted, so a long live entry does not keep it open. Camera entries become integers so that later
    classification cannot resolve them against the working directory.

    Args:
        path: Manifest file.

    Returns:
        Resolved paths, URLs and screen commands, plus camera indexes as integers, in file order.
    """
    with path.open(encoding="utf-8-sig", newline="") as manifest:
        is_csv = path.suffix.lower() == ".csv"
        rows = list(csv.reader(manifest)) if is_csv else [[line] for line in manifest]
    if is_csv and rows:
        header = [cell.strip().lower() for cell in rows[0]]
        column = next((index for index, cell in enumerate(header) if cell in _MANIFEST_HEADERS), None)
        if column is not None:
            # A row too short to reach the source column has no source.
            rows = [row[column : column + 1] for row in rows[1:]]
    entries: list[str | int] = []
    for row in rows:
        for cell in row:
            entry = cell.strip()
            if not entry or entry.startswith("#"):
                continue
            resolved, kind = _classify_source(entry, relative_to=path.parent)
            entries.append(int(resolved) if kind == _SourceKind.CAMERA else resolved)
    return entries


def _youtube_format(max_height: int) -> str:
    """Build a yt-dlp format chain that prefers H.264 and skips videos taller than ``max_height``.

    Some OpenCV builds cannot decode YouTube's AV1 formats, so the chain tries H.264, then MP4, then any codec. Every
    step carries the height cap, because a 4K or 8K stream costs decode time and memory that the model resolution
    discards. ``<=?`` keeps formats whose height is unknown, so an unusual extractor result still resolves.

    Args:
        max_height: Tallest video, in pixels, the chain may select.

    Returns:
        A yt-dlp format selector.

    Examples:
        >>> _youtube_format(720).split("/")[0]
        'bestvideo[height<=?720][vcodec^=avc1]'
        >>> _youtube_format(720).count("[height<=?720]")
        6
    """
    cap = f"[height<=?{max_height}]"
    return (
        f"bestvideo{cap}[vcodec^=avc1]/best{cap}[vcodec^=avc1]/bestvideo{cap}[ext=mp4]/best{cap}[ext=mp4]"
        f"/bestvideo{cap}/best{cap}"
    )


def _resolve_youtube(source: str) -> tuple[str, bool]:
    """Resolve a YouTube page to a decodable video URL.

    Args:
        source: YouTube page URL.

    Returns:
        The direct video URL and whether the video is a live stream.

    Raises:
        ImportError: If yt-dlp is not installed.
        ValueError: If yt-dlp cannot resolve the page (unavailable, private or region-locked video, network failure) or
            returns no video URL.
    """
    try:
        yt_dlp = import_module("yt_dlp")
    except ImportError as error:
        raise ImportError("YouTube prediction requires yt-dlp. Install rfdetr[stream].") from error
    try:
        with yt_dlp.YoutubeDL(
            {
                "format": _youtube_format(_YOUTUBE_MAX_HEIGHT),
                "quiet": True,
                "noplaylist": True,
                "socket_timeout": _YOUTUBE_SOCKET_TIMEOUT,
            }
        ) as downloader:
            info = downloader.extract_info(source, download=False)
    except yt_dlp.utils.DownloadError as error:
        raise ValueError(f"Could not resolve a video URL for the YouTube source {source!r}: {error}") from error
    if not info or not isinstance(info.get("url"), str):
        raise ValueError("Could not resolve a video URL from the YouTube source.")
    return info["url"], bool(info.get("is_live", False))


def _iter_screenshots(source: str) -> Generator[np.ndarray[Any, Any], None, None]:
    """Capture RGB screenshots of a monitor or region until the caller closes the generator.

    The command is ``screen [monitor] [left top width height]``. Region offsets are relative to the chosen monitor.
    Arguments are validated when the generator is first advanced, before any capture starts.

    Args:
        source: Screen command, e.g. ``"screen"``, ``"screen 1"`` or ``"screen 0 10 20 640 480"``.

    Yields:
        One RGB frame in HWC layout per grab, with no frame limit.

    Raises:
        ValueError: If a coordinate is not an integer, the argument count is wrong, the monitor is negative or
            missing, or the region size is not positive.
        ImportError: If ``mss`` is not installed.
    """
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


def _iter_video_frames(
    source: str | int, vid_stride: int = 1, *, warn_on_retained_frames: bool = False
) -> Generator[np.ndarray[Any, Any], None, None]:
    """Decode a finite video into RGB frames, releasing the capture on exhaustion, errors, or generator closure.

    With ``warn_on_retained_frames``, estimate the retained RGB pixels from the reported frame count and size.

    A video whose first frame cannot be read raises. A network video logs a warning when it stops, because OpenCV cannot
    distinguish its end from a read timeout.

    Args:
        source: Video file path or HTTP(S) video URL.
        vid_stride: Keep frames 0, N, 2N, ...; the frames between are skipped without decoding.
        warn_on_retained_frames: Warn when the estimated retained RGB pixels exceed the retained-frame threshold.

    Yields:
        One RGB frame in HWC layout per kept frame.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the video cannot be opened or its first frame cannot be read.
    """
    cv2 = _import_cv2("Video")
    network = isinstance(source, str) and urlparse(source).scheme in ("http", "https")
    capture = cv2.VideoCapture(*_capture_args(source))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the video source.")
        # Keep frames 0, N, 2N, ..., matching live capture.
        success, frame = capture.read()
        if not success:
            raise ValueError("Could not read the first frame from the video source.")
        if warn_on_retained_frames:
            properties = (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)
            values = [capture.get(prop) for prop in properties]
            # Network streams and some containers report no frame count; skip the estimate when a value is unknown.
            if all(isinstance(value, (int, float)) and value > 0 for value in values):
                count, height, width = (int(value) for value in values)
                _warn_retained_frames(-(-count // vid_stride), height, width, str(source))
        while success:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # grab() advances past skipped frames without decoding them; all() stops at the first failure.
            success = all(capture.grab() for _ in range(vid_stride - 1))
            if success:
                success, frame = capture.read()
        if network:
            logger.warning(
                "Finite network stream stopped at end of file or a read failure. "
                "OpenCV cannot distinguish EOF from a timeout; results may be incomplete."
            )
    finally:
        capture.release()


def _stream_entry(source: str | int) -> tuple[str | int, bool]:
    """Validate one simultaneous capture source before any capture opens.

    Args:
        source: Camera index, video file, video URL, live stream URL or YouTube page from a stream manifest.

    Returns:
        The source to open (a YouTube page becomes its direct video URL) and whether it is live; False marks a finite
        video that stops at its end.

    Raises:
        FileNotFoundError: If a local video file does not exist.
        ValueError: If the entry is not a camera, video or stream, or a YouTube page cannot be resolved.
        ImportError: If a YouTube entry needs yt-dlp and it is not installed.
    """
    if isinstance(source, int):
        return source, True
    source, kind = _classify_source(source, stream_entry=True)
    if kind == _SourceKind.CAMERA:
        return int(source), True
    if kind == _SourceKind.YOUTUBE:
        return _resolve_youtube(source)
    if kind == _SourceKind.LIVE:
        return source, True
    if kind in (_SourceKind.VIDEO, _SourceKind.HTTP_VIDEO):
        if kind == _SourceKind.VIDEO and not Path(source).is_file():
            raise FileNotFoundError(f"Video file does not exist: {source!r}.")
        return source, False
    raise ValueError(f"Invalid stream manifest entry: {source!r}. Use a camera index, video file, or stream URL.")


def _resolve_stream_entries(entries: list[str | int]) -> list[tuple[str | int, bool]]:
    """Validate every simultaneous source, resolving slow YouTube pages concurrently.

    Results keep manifest order and the first invalid entry in that order raises. Every lookup already started finishes
    before the error surfaces; none of them holds a capture.

    Args:
        entries: Manifest entries, each a camera index or a path or URL.

    Returns:
        One ``(source, is_live)`` pair per entry, in manifest order, as produced by :func:`_stream_entry`.

    Raises:
        FileNotFoundError: If a local video file does not exist.
        ValueError: If an entry is not a camera, video or stream, or a YouTube page cannot be resolved.
    """
    if len(entries) < 2:
        return [_stream_entry(item) for item in entries]
    with ThreadPoolExecutor(max_workers=len(entries), thread_name_prefix="rfdetr-stream-entry") as pool:
        return list(pool.map(_stream_entry, entries))


def _warn_retained_images(images: list[Path], source: str) -> None:
    """Estimate retained directory or glob pixels from the first image header without decoding every file.

    The estimate assumes every image has the first image's size and warns through :func:`_warn_retained_frames`. It
    does nothing for an empty list, and an unreadable header skips the warning instead of blocking prediction.

    Args:
        images: Image files the eager call will keep source pixels for.
        source: Directory or glob pattern named in the warning message.
    """
    if not images:
        return
    # The estimate is advisory; an unreadable or plugin-only header must not block prediction.
    with contextlib.suppress(OSError):
        with Image.open(images[0]) as first:
            width, height = first.size
        _warn_retained_frames(len(images), height, width, source)


def _warn_retained_frames(frame_count: int, height: int, width: int, source: str) -> None:
    """Warn when eager results would keep more RGB source pixels than the retained-frame threshold.

    Args:
        frame_count: Number of source frames or images that would be kept.
        height: Height of each frame in pixels.
        width: Width of each frame in pixels.
        source: Source name shown in the warning message.
    """
    retained = frame_count * height * width * 3
    if retained >= _RETAINED_FRAME_WARNING_BYTES:
        warnings.warn(
            f"Eager prediction keeps about {retained / 2**30:.1f} GiB of source images for {frame_count} frames "
            f"from {source!r}. Use stream=True or include_source_image=False to limit memory use.",
            UserWarning,
            stacklevel=2,
        )


def _stacklevel_outside_package(package: str = "rfdetr") -> int:
    """Count the stack levels from the caller of this helper to the first frame outside ``package``.

    ``warnings.warn(..., stacklevel=n)`` attributes a warning to the ``n``-th frame, counting the function that calls
    ``warn`` as 1. Generator frames, manifest nesting, and the two public ``predict`` entry points make the distance
    from ``warn`` to user code vary, so the depth is measured instead of fixed. ``skip_file_prefixes`` does this on
    Python 3.12 and later, but the project supports 3.10.

    Args:
        package: Top-level package whose frames, including its submodules, are skipped.

    Returns:
        The stacklevel naming the first frame outside the package. Falls back to 2, the caller of the function that
        warns, when every frame belongs to the package or the interpreter exposes no frames.

    Examples:
        >>> _stacklevel_outside_package("a_package_that_is_not_on_the_stack")
        1
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    level = 1
    try:
        while caller is not None:
            name = caller.f_globals.get("__name__", "")
            if name != package and not name.startswith(f"{package}."):
                return level
            caller = caller.f_back
            level += 1
        return 2
    finally:
        # Drop the frame references so they do not form a cycle with this function's locals.
        del frame, caller


def _warn_live_accumulation() -> None:
    """Warn before an eager prediction starts an unbounded source.

    The warning names the first caller outside ``rfdetr``, not an internal generator frame, so it points at the user's
    ``predict`` call.
    """
    warnings.warn(
        "Live results accumulate in memory with stream=False. Use stream=True to limit memory use.",
        UserWarning,
        stacklevel=_stacklevel_outside_package(),
    )
