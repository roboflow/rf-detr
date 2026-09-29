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

#: Image suffixes read from a *calibration_data* directory.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _image_paths(directory: Path, max_images: int) -> list[Path]:
    """Return up to *max_images* image files from *directory*, sorted by name.

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
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No calibration images found in {directory}. Supported suffixes: {sorted(IMAGE_SUFFIXES)}.")
    return paths[:max_images]


def _arrays_from_samples(array: NDArray[Any], height: int, width: int) -> Iterator[NDArray[np.float32]]:
    """Yield one ``(1, C, H, W)`` float32 batch per sample of a pre-normalized calibration array.

    Args:
        array: Samples shaped ``(N, C, H, W)``, already normalized the way the model expects.
        height: Spatial height the graph was exported at.
        width: Spatial width the graph was exported at.

    Yields:
        One single-sample batch per row of *array*.

    Raises:
        ValueError: If *array* is not rank 4 or its spatial dimensions do not match the graph.

    Examples:
        >>> import numpy as np
        >>> batches = list(_arrays_from_samples(np.zeros((2, 3, 4, 4), dtype=np.float32), height=4, width=4))
        >>> len(batches), batches[0].shape
        (2, (1, 3, 4, 4))
    """
    if array.ndim != 4:
        raise ValueError(f"Calibration array must be rank 4 (N, C, H, W); got shape {array.shape}.")
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
    path or array is taken as already preprocessed and is passed through unchanged.

    Args:
        calibration_data: Directory of images, path to a ``.npy`` file, or a preprocessed array.
        height: Spatial height the graph was exported at.
        width: Spatial width the graph was exported at.
        channels: Channel count the graph expects.
        max_images: Maximum images read from a directory.

    Yields:
        One preprocessed single-image batch at a time, so calibration never holds the whole set in memory.

    Raises:
        ValueError: If *calibration_data* names a path that does not exist, or holds no usable image.

    Examples:
        >>> import numpy as np
        >>> len(list(calibration_batches(np.zeros((3, 3, 8, 8), dtype=np.float32), height=8, width=8)))
        3
    """
    if isinstance(calibration_data, np.ndarray):
        yield from _arrays_from_samples(calibration_data, height, width)
        return

    path = Path(calibration_data)
    if not path.exists():
        raise ValueError(f"Calibration data path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Calibration file must be a .npy array; got {path.name}.")
        yield from _arrays_from_samples(np.load(path), height, width)
        return

    from PIL import Image  # Pillow is an inference-time dependency, not an import-time one.

    for image_path in _image_paths(path, max_images):
        with Image.open(image_path) as image:
            image.load()
            yield preprocess_to_nchw(image, height=height, width=width, channels=channels)
