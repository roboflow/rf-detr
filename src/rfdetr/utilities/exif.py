# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""EXIF orientation helpers shared by the dataset readers and ``predict()``.

A camera stores a photo's pixels in sensor order and records how it was held in the EXIF orientation tag. Image viewers,
annotation tools, OpenCV's ``imread`` and Supervision's dataset loaders show the photo turned by that tag, so boxes,
masks and keypoints are drawn on the upright image. RF-DETR decodes image files upright the same way, through Pillow's
transposes, so a label always lands on the pixels it was drawn on.
"""

from __future__ import annotations

import os
import struct

from PIL import Image, PngImagePlugin

#: EXIF tag that records how the camera was held when the photo was taken.
_ORIENTATION_TAG = 0x0112

#: The eight bytes every PNG file starts with.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Most bytes of a PNG eXIf chunk read for its orientation, which sits in the first IFD near the start of the block.
#: A corrupt chunk length then cannot make the walk allocate gigabytes.
_MAX_PNG_EXIF_BYTES = 1 << 20

#: Transpose that turns stored pixels upright for each orientation, the table ``PIL.ImageOps.exif_transpose`` uses.
#: Orientation 1, a missing tag and a value outside 1 to 8 need no transpose.
_UPRIGHT_TRANSPOSES = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}

#: Orientations whose transpose swaps the image's width and height.
_QUARTER_TURNS = frozenset({5, 6, 7, 8})


def exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation still to apply to an opened image, or ``1`` when it is upright as Pillow decodes it.

    Reads metadata only and never decodes the pixels. Pillow parses a PNG's chunks only up to the pixel data when it
    opens the file, and ``PngImageFile.getexif`` decodes the whole image to reach an eXIf chunk stored after it, so
    ``PIL.Image.Image.getexif`` is called directly and such a chunk is found by :func:`_png_exif_chunk`, which reads
    one 8-byte header per chunk. OpenCV's ``imread`` and ``PIL.ImageOps.exif_transpose`` turn that file as well.
    Pillow turns a TIFF upright itself, reporting the upright size when it opens the file and turning the pixels when
    it loads them, so a TIFF has nothing left to apply.

    Args:
        image: Image opened with Pillow; its pixels need not be loaded.

    Returns:
        The orientation, from 1 to 8.

    Examples:
        >>> import io
        >>> exif = Image.Exif()
        >>> exif[_ORIENTATION_TAG] = 6
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="JPEG", exif=exif.tobytes())
        >>> exif_orientation(Image.open(encoded))
        6
        >>> exif_orientation(Image.new("RGB", (4, 2)))
        1
    """
    if image.format == "TIFF":
        return 1
    try:
        orientation = Image.Image.getexif(image).get(_ORIENTATION_TAG)
        if orientation is None and isinstance(image, PngImagePlugin.PngImageFile) and "exif" not in image.info:
            trailing = _png_exif_chunk(image)
            if trailing is not None:
                exif = Image.Exif()
                exif.load(trailing)
                orientation = exif.get(_ORIENTATION_TAG)
    except Exception:
        # EXIF is optional metadata: a block Pillow cannot parse must not stop the pixels from decoding.
        return 1
    if isinstance(orientation, int) and orientation in _UPRIGHT_TRANSPOSES:
        return orientation
    return 1


def _png_exif_chunk(image: PngImagePlugin.PngImageFile) -> bytes | None:
    """Return the data of a PNG's eXIf chunk without decoding its pixels, for a chunk Pillow has not read yet.

    The PNG specification puts eXIf before the pixel data, where Pillow reads it on open. When a file stores it after
    the pixels instead, Pillow reads it only while decoding them. This reads the 8-byte header of each chunk from the
    start of the file, seeks over the chunk's data and CRC, stops at IEND, and puts the stream back where it was.

    Args:
        image: PNG opened with Pillow.

    Returns:
        The eXIf data, at most ``_MAX_PNG_EXIF_BYTES`` of it, or ``None`` when the file has no eXIf chunk or is cut
        short, when Pillow has already closed the file (``image.fp`` is ``None``), or for an animated PNG, whose later
        chunks belong to frames Pillow has not loaded.

    Raises:
        OSError: If the stream cannot be read or moved.
        ValueError: If the stream is closed.

    Examples:
        >>> import io
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="PNG")
        >>> _png_exif_chunk(Image.open(encoded)) is None
        True
    """
    stream = image.fp
    if stream is None or image.is_animated:
        return None
    position = stream.tell()
    try:
        stream.seek(0)
        if stream.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
            return None
        while len(header := stream.read(8)) == 8:
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type == b"eXIf":
                return stream.read(min(length, _MAX_PNG_EXIF_BYTES))
            if chunk_type == b"IEND":
                return None
            stream.seek(length + 4, os.SEEK_CUR)
        return None
    finally:
        stream.seek(position)


def swaps_axes(orientation: int) -> bool:
    """Return whether turning an image upright for ``orientation`` swaps its width and height.

    Args:
        orientation: EXIF orientation, as :func:`exif_orientation` returns it.

    Returns:
        ``True`` for the quarter turns, orientations 5 to 8.

    Examples:
        >>> swaps_axes(6), swaps_axes(3)
        (True, False)
    """
    return orientation in _QUARTER_TURNS


def upright_image(image: Image.Image, orientation: int) -> Image.Image:
    """Turn decoded pixels upright for their EXIF orientation, returning ``image`` itself when it already is.

    Args:
        image: Decoded image in the order its file stores the pixels.
        orientation: The file's EXIF orientation, as :func:`exif_orientation` returns it.

    Returns:
        The upright image; a new image only when ``orientation`` needs a transpose.

    Examples:
        >>> upright_image(Image.new("RGB", (4, 2)), 6).size
        (2, 4)
    """
    transpose = _UPRIGHT_TRANSPOSES.get(orientation)
    return image if transpose is None else image.transpose(transpose)
