# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Calibration data for the static-quantization export paths.

Static post-training quantization derives activation ranges from data, so ONNX and OpenVINO both need the same thing:
representative images, preprocessed exactly as inference preprocesses them. That is one job with one correct answer, so
it lives here rather than once per format -- and it stays free of both formats' heavy optional dependencies, which is
what lets each import it without dragging in the other's runtime.

The accepted forms mirror ``RFDETR.export``'s *calibration_data* keyword: a directory of images, a ``.npy`` file, or an
array. A directory is the normal case and the only one where preprocessing happens here; arrays are taken as already
prepared.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.preprocess import preprocess_to_nchw
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Image suffixes read from a *calibration_data* directory.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

#: Fewest calibration samples both INT8 paths accept without a warning. A conservative heuristic floor, not a measured
#: threshold: min/max activation ranges taken from a handful of images rarely cover what the model sees in deployment,
#: and the resulting model still loads and runs, so the accuracy loss is otherwise silent.
MIN_CALIBRATION_SAMPLES: int = 32


def warn_if_too_few_samples(count: int) -> None:
    """Log a warning when calibration is about to run on fewer than :data:`MIN_CALIBRATION_SAMPLES` samples.

    Args:
        count: Number of calibration samples the quantizer will use.

    Examples:
        >>> warn_if_too_few_samples(MIN_CALIBRATION_SAMPLES)  # at the floor: silent
    """
    if count < MIN_CALIBRATION_SAMPLES:
        logger.warning(
            f"Calibrating INT8 quantization on only {count} sample(s). Fewer than {MIN_CALIBRATION_SAMPLES} "
            "representative images often yields activation ranges that cost accuracy; pass more calibration data."
        )


def _image_paths(directory: Path, max_images: int) -> list[Path]:
    """Return up to *max_images* image files from *directory*, sorted by name.

    Only regular files with an image suffix count: a subdirectory named like an image is skipped, and so are macOS
    AppleDouble sidecars (``._name.jpg``), which carry an image suffix but hold resource-fork metadata, not pixels.

    Args:
        directory: Directory to read images from.
        max_images: Maximum number of files to return.

    Returns:
        Sorted image paths, at most *max_images* of them.

    Raises:
        ValueError: If *directory* holds no readable image.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _ = (Path(tmp) / "b.jpg").write_bytes(b"")
        ...     _ = (Path(tmp) / "a.png").write_bytes(b"")
        ...     [p.name for p in _image_paths(Path(tmp), max_images=5)]
        ['a.png', 'b.jpg']
    """
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._") and p.is_file()
    )
    if not paths:
        raise ValueError(f"No calibration images found in {directory}. Supported suffixes: {sorted(IMAGE_SUFFIXES)}.")
    return paths[:max_images]


def _arrays_from_samples(
    array: NDArray[Any], height: int, width: int, channels: int = 3
) -> Iterator[NDArray[np.float32]]:
    """Yield one ``(1, C, H, W)`` float32 batch per sample of a pre-normalized calibration array.

    Args:
        array: Samples shaped ``(N, C, H, W)``, already normalized the way the model expects.
        height: Spatial height the graph was exported at.
        width: Spatial width the graph was exported at.
        channels: Channel count the graph expects.

    Yields:
        One single-sample batch per row of *array*.

    Raises:
        ValueError: If *array* is not rank 4, is not floating point (raw ``uint8`` pixels are not normalized), or its
            channel count or spatial dimensions do not match the graph.

    Examples:
        >>> import numpy as np
        >>> batches = list(_arrays_from_samples(np.zeros((2, 3, 4, 4), dtype=np.float32), height=4, width=4))
        >>> len(batches), batches[0].shape
        (2, (1, 3, 4, 4))
    """
    if array.ndim != 4:
        raise ValueError(f"Calibration array must be rank 4 (N, C, H, W); got shape {array.shape}.")
    expected = f"(N, {channels}, {height}, {width})"
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"Calibration array must be floating point and already normalized, shaped {expected}; got dtype "
            f"{array.dtype}. Pass a directory of images instead to have them preprocessed for you."
        )
    if array.shape[1] != channels:
        raise ValueError(
            f"Calibration array has {array.shape[1]} channel(s) but the graph expects {channels}: expected shape "
            f"{expected}, got {array.shape}."
        )
    if array.shape[2] != height or array.shape[3] != width:
        raise ValueError(
            f"Calibration array is {array.shape[2]}x{array.shape[3]} but the graph expects {height}x{width}. "
            "Pass a directory of images instead to have them resized for you."
        )
    for sample in array:
        yield np.ascontiguousarray(sample[None], dtype=np.float32)


def calibration_batches(
    calibration_data: str | Path | NDArray[Any],
    *,
    height: int,
    width: int,
    channels: int = 3,
    max_images: int = 100,
) -> Iterator[NDArray[np.float32]]:
    """Yield ``(1, C, H, W)`` float32 batches to calibrate activation ranges with.

    A directory of images is the normal case: each is preprocessed exactly as
    :meth:`~rfdetr.detr.RFDETR.predict` would, so the ranges reflect what the model sees at inference. A ``.npy``
    path or array is taken as already preprocessed and is passed through unchanged. A ``.npy`` file is memory-mapped
    rather than read up front, so only the samples being converted are paged in.

    Both quantizers materialize the whole sequence before calibrating (ONNX Runtime rewinds its reader, NNCF takes a
    sized dataset), so peak memory grows with the number of samples: roughly ``N * C * H * W * 4`` bytes.

    Args:
        calibration_data: Directory of images, path to a ``.npy`` file, or a preprocessed array.
        height: Spatial height the graph was exported at.
        width: Spatial width the graph was exported at.
        channels: Channel count the graph expects.
        max_images: Maximum images read from a directory. An array or ``.npy`` file is used whole and is not capped
            by this value -- slice it before passing it in to calibrate on fewer samples.

    Yields:
        One preprocessed single-image batch at a time.

    Raises:
        ValueError: If *calibration_data* names a path that does not exist or holds no usable image, if a ``.npy`` file
            holds pickled objects rather than a numeric array, or if a directory entry with an image suffix cannot be
            identified as an image by Pillow (the message names the file).

    Examples:
        >>> import numpy as np
        >>> len(list(calibration_batches(np.zeros((3, 3, 8, 8), dtype=np.float32), height=8, width=8)))
        3
    """
    if isinstance(calibration_data, np.ndarray):
        yield from _arrays_from_samples(calibration_data, height, width, channels)
        return

    path = Path(calibration_data)
    if not path.exists():
        raise ValueError(f"Calibration data path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Calibration file must be a .npy array; got {path.name}.")
        try:
            # allow_pickle=False: a calibration file is data, never code -- refuse object arrays outright.
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as exc:
            raise ValueError(f"Calibration file {path} is not a plain numeric .npy array: {exc}") from exc
        yield from _arrays_from_samples(array, height, width, channels)
        return

    from PIL import Image, UnidentifiedImageError  # Pillow is an inference-time dependency, not an import-time one.

    for image_path in _image_paths(path, max_images):
        try:
            image = Image.open(image_path)
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Calibration image {image_path} is not a readable image. Remove it from the directory or replace it."
            ) from exc
        with image:
            image.load()
            yield preprocess_to_nchw(image, height=height, width=width, channels=channels)
