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

from PIL import Image

#: EXIF tag that records how the camera was held when the photo was taken.
_ORIENTATION_TAG = 0x0112

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

    Reads only the metadata Pillow parsed while opening the file, so it costs microseconds before the pixel decode.
    ``PIL.Image.Image.getexif`` is called directly because ``PngImageFile.getexif`` decodes the whole image to look
    for an eXIf chunk after the pixel data; the chunk PNG writers place before the pixels is still read. Pillow turns a
    TIFF upright itself, reporting the upright size when it opens the file and turning the pixels when it loads them,
    so a TIFF has nothing left to apply.

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
        orientation = Image.Image.getexif(image).get(_ORIENTATION_TAG, 1)
    except Exception:
        # EXIF is optional metadata: a block Pillow cannot parse must not stop the pixels from decoding.
        return 1
    if isinstance(orientation, int) and orientation in _UPRIGHT_TRANSPOSES:
        return orientation
    return 1


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
