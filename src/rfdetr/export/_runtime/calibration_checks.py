# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Configuration-time checks on *calibration_data*, before anything is traced.

Static quantization reads *calibration_data* only after the forward pass and the conversion have run, so a mistyped
path or an array of the wrong rank would otherwise surface minutes into an export. What can be judged without the graph
-- does the path exist, does the directory hold an image, is the file a ``.npy``, is the array a non-empty
four-dimensional float array, is *max_images* a positive integer -- is judged here, from metadata alone: nothing is
decoded, loaded, or opened. What needs the graph (the spatial size, the channel count) stays with
:func:`~rfdetr.export._runtime.calibration.calibration_batches`.

No format's runtime is imported, which is what lets the ONNX, OpenVINO and TensorRT exporters all call this from
``_check_capabilities``.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import IMAGE_SUFFIXES, is_image_file


def _is_positive_integer(value: object) -> bool:
    """Return whether *value* is an integer of at least 1, NumPy integers included and booleans excluded.

    Examples:
        >>> _is_positive_integer(np.int64(5)), _is_positive_integer(True), _is_positive_integer(0)
        (True, False, False)
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return operator.index(value) >= 1  # type: ignore[arg-type]
    except TypeError:
        return False


def calibration_array_problem(array: NDArray[Any]) -> str | None:
    """Say why a calibration array cannot calibrate, judging its shape and dtype only.

    Returned rather than raised so that a caller holding a memory-mapped array can release it before raising, and
    the error's traceback does not keep the file mapped.

    Args:
        array: Calibration samples, expected shaped ``(N, C, H, W)``, floating point and already normalized.

    Returns:
        The reason, worded as :func:`~rfdetr.export._runtime.calibration.calibration_batches` words it, or ``None``
        when the array has a usable form.

    Examples:
        >>> calibration_array_problem(np.zeros((1, 3, 8, 8), np.float32)) is None
        True
        >>> calibration_array_problem(np.zeros((0, 3, 8, 8), np.float32))
        'Calibration array must hold at least one image; got shape (0, 3, 8, 8).'
    """
    if array.ndim != 4:
        return f"Calibration array must be rank 4 (N, C, H, W); got shape {array.shape}."
    if not np.issubdtype(array.dtype, np.floating):
        return (
            f"Calibration array must be floating point and already normalized, shaped (N, C, H, W); got dtype "
            f"{array.dtype}. Pass a directory of images instead to have them preprocessed for you."
        )
    if array.shape[0] == 0:
        return f"Calibration array must hold at least one image; got shape {array.shape}."
    return None


def check_calibration_data(calibration_data: str | Path | NDArray[Any], *, max_images: int = 100) -> None:
    """Refuse *calibration_data* that cannot possibly calibrate, using nothing but its metadata.

    The messages match the ones :func:`~rfdetr.export._runtime.calibration.calibration_batches` raises for the same
    mistakes later, so the wording a caller sees does not depend on when the mistake is caught.

    Args:
        calibration_data: Directory of images, path to a ``.npy`` file, or a preprocessed array.
        max_images: The cap the caller passes on to ``calibration_batches``.

    Raises:
        ValueError: If *max_images* is not a positive integer (``bool`` excluded); if *calibration_data* is neither a
            non-empty path nor an array; if an array is not rank 4 ``(N, C, H, W)``, not floating point, or empty; if
            a path does not exist; if a file is not a ``.npy``; or if a directory holds no supported image.

    Examples:
        >>> check_calibration_data(np.zeros((2, 3, 8, 8), dtype=np.float32))
        >>> check_calibration_data(np.zeros((3, 8, 8), dtype=np.float32))
        Traceback (most recent call last):
            ...
        ValueError: Calibration array must be rank 4 (N, C, H, W); got shape (3, 8, 8).
        >>> check_calibration_data("/definitely/not/here")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Calibration data path does not exist: ...here
        >>> check_calibration_data(np.zeros((2, 3, 8, 8), dtype=np.float32), max_images=0)
        Traceback (most recent call last):
            ...
        ValueError: max_images must be a positive integer, got 0.
    """
    if not _is_positive_integer(max_images):
        raise ValueError(f"max_images must be a positive integer, got {max_images!r}.")
    if isinstance(calibration_data, np.ndarray):
        problem = calibration_array_problem(calibration_data)
        if problem is not None:
            raise ValueError(problem)
        return
    if not isinstance(calibration_data, (str, os.PathLike)) or not os.fspath(calibration_data):
        # An empty path would otherwise be read as the working directory.
        described = "an empty path" if isinstance(calibration_data, (str, os.PathLike)) else type(calibration_data)
        raise ValueError(
            "Expected calibration_data to be a directory of images, a .npy path, or a preprocessed (N, C, H, W) float "
            f"array; got {described}."
        )

    path = Path(calibration_data)
    if not path.exists():
        raise ValueError(f"Calibration data path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Calibration file must be a .npy array; got {path.name}.")
        return
    # The reader's own predicate, so a directory accepted here is one calibration_batches finds an image in.
    if not any(is_image_file(child) for child in path.iterdir()):
        raise ValueError(f"No calibration images found in {path}. Supported suffixes: {sorted(IMAGE_SUFFIXES)}.")
