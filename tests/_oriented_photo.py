# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Photo writer for the EXIF orientation tests of the dataset readers, the decoders and the export helpers."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

#: EXIF tag that records how the camera was held when the photo was taken.
_ORIENTATION_TAG = 0x0112


def write_oriented_photo(path: Path, orientation: int, width: int = 29, height: int = 17) -> None:
    """Write noise stored as ``width`` x ``height`` RGB pixels to ``path``, tagged with an EXIF ``orientation``.

    The format follows the suffix of ``path``; JPEG files use quality 90. Every pixel differs from its mirror images,
    so a wrong flip or turn of the decoded photo changes its pixels, not only its size.

    Args:
        path: File to write.
        orientation: Value of the EXIF orientation tag; a viewer shows the stored pixels turned by it.
        width: Stored width in pixels.
        height: Stored height in pixels.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     photo = Path(tmp) / "photo.jpg"
        ...     write_oriented_photo(photo, 6)
        ...     with Image.open(photo) as image:
        ...         print(image.size, image.getexif()[_ORIENTATION_TAG])
        (29, 17) 6
    """
    pixels = ((np.arange(height * width * 3).reshape(height, width, 3) * 37 + 13) % 256).astype(np.uint8)
    exif = Image.Exif()
    exif[_ORIENTATION_TAG] = orientation
    options = {"quality": 90} if path.suffix.lower() in (".jpg", ".jpeg") else {}
    Image.fromarray(pixels).save(path, exif=exif.tobytes(), **options)


def upright_reference(path: Path) -> np.ndarray:
    """Decode ``path`` through Pillow alone to RGB, turned upright with ``ImageOps.exif_transpose``.

    Args:
        path: Image file to decode.

    Returns:
        The upright ``(H, W, 3)`` uint8 pixels.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     photo = Path(tmp) / "photo.png"
        ...     write_oriented_photo(photo, 6)
        ...     upright_reference(photo).shape
        (29, 17, 3)
    """
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
