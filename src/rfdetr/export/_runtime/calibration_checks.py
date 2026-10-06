# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Configuration-time checks on *calibration_data*, before anything is traced.

Static quantization reads *calibration_data* only after the forward pass and the conversion have run, so a mistyped
path or an array of the wrong rank would otherwise surface minutes into an export. What can be judged without the graph
-- does the path exist, does the directory hold an image, is the file a ``.npy``, is the array four-dimensional --
is judged here, from metadata alone: nothing is decoded, loaded, or opened. What needs the graph (the spatial size, the
channel count) stays with :func:`~rfdetr.export._runtime.calibration.calibration_batches`.

Neither ONNX nor OpenVINO is imported, which is what lets both exporters call this from ``_check_capabilities``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import IMAGE_SUFFIXES


def check_calibration_data(calibration_data: str | Path | NDArray[Any]) -> None:
    """Refuse *calibration_data* that cannot possibly calibrate, using nothing but its metadata.

    The messages match the ones :func:`~rfdetr.export._runtime.calibration.calibration_batches` raises for the same
    mistakes later, so the wording a caller sees does not depend on when the mistake is caught.

    Args:
        calibration_data: Directory of images, path to a ``.npy`` file, or a preprocessed array.

    Raises:
        ValueError: If an array is not rank 4 ``(N, C, H, W)``; if a path does not exist; if a file is not a ``.npy``;
            or if a directory holds no supported image.

    Examples:
        >>> import numpy as np
        >>> check_calibration_data(np.zeros((2, 3, 8, 8), dtype=np.float32))
        >>> check_calibration_data(np.zeros((3, 8, 8), dtype=np.float32))
        Traceback (most recent call last):
            ...
        ValueError: Calibration array must be rank 4 (N, C, H, W); got shape (3, 8, 8).
        >>> check_calibration_data("/definitely/not/here")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Calibration data path does not exist: ...here
    """
    if isinstance(calibration_data, np.ndarray):
        if calibration_data.ndim != 4:
            raise ValueError(f"Calibration array must be rank 4 (N, C, H, W); got shape {calibration_data.shape}.")
        return

    path = Path(calibration_data)
    if not path.exists():
        raise ValueError(f"Calibration data path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Calibration file must be a .npy array; got {path.name}.")
        return
    if not any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir()):
        raise ValueError(f"No calibration images found in {path}. Supported suffixes: {sorted(IMAGE_SUFFIXES)}.")
