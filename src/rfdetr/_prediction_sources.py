# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Source expansion for the prediction facade."""

import glob
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image

ImageInput = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
PredictionSource = ImageInput | int
PredictionInput = PredictionSource | list[PredictionSource] | tuple[PredictionSource, ...]

_IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv"}


def is_expanded_source(source: PredictionInput) -> bool:
    """Identify inputs that require one prediction per source image."""
    if isinstance(source, (list, tuple)):
        return any(is_expanded_source(item) for item in source)
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
        or Path(path).suffix.lower() in _VIDEO_SUFFIXES
    )


def is_live_source(source: PredictionInput) -> bool:
    """Identify camera inputs that can produce an unbounded number of frames."""
    if isinstance(source, (list, tuple)):
        return any(is_live_source(item) for item in source)
    return isinstance(source, int) or (isinstance(source, str) and urlparse(source).scheme in ("rtsp", "rtsps"))


def iter_source_images(source: PredictionInput) -> Generator[ImageInput, None, None]:
    """Expand collections in order without loading images into memory."""
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from iter_source_images(item)
        return
    if isinstance(source, int):
        if isinstance(source, bool) or source < 0:
            raise ValueError("The webcam index must be a non-negative integer.")
        yield from _iter_video_frames(source)
        return
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        url = urlparse(path)
        if url.scheme in ("rtsp", "rtsps"):
            yield from _iter_video_frames(path)
            return
        remote = url.scheme in ("http", "https")
        paths = None
        if not remote:
            if Path(path).is_dir():
                paths = sorted(Path(path).iterdir())
            elif glob.has_magic(path) and not Path(path).is_file():
                paths = sorted(Path(match) for match in glob.iglob(path, recursive=True))
        if paths is not None:
            media = [
                item for item in paths if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES
            ]
            if not media:
                raise FileNotFoundError(f"No supported images or videos found for {path!r}.")
            for item in media:
                yield from iter_source_images(str(item))
            return
        if Path(url.path if remote else path).suffix.lower() in _VIDEO_SUFFIXES:
            if not remote and not Path(path).is_file():
                raise FileNotFoundError(f"Video file does not exist: {path!r}.")
            yield from _iter_video_frames(path)
            return
        yield path
        return
    yield source


def _iter_video_frames(source: str | int) -> Generator[np.ndarray[Any, Any], None, None]:
    """Release the capture on exhaustion, prediction errors, or generator closure."""
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the video source.")
        while True:
            success, frame = capture.read()
            if not success:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
