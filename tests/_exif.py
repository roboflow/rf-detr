# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared PNG builder for the EXIF orientation tests of the utilities, the dataset readers and ``predict()``."""

import struct

#: The eight bytes every PNG file starts with.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def move_exif_after_pixels(png: bytes) -> bytes:
    """Move a PNG's eXIf chunk from before its pixel data to just before IEND, keeping every other chunk in order.

    Pillow writes the chunk before the first IDAT, where the PNG specification puts it, and reads it when it opens the
    file; it reads a chunk stored after the pixels only while decoding them.

    Args:
        png: PNG bytes with one eXIf chunk.

    Returns:
        The same chunks with the eXIf chunk moved after the last IDAT.

    Raises:
        ValueError: If ``png`` does not hold exactly one eXIf chunk or does not end with IEND.

    Examples:
        >>> import io
        >>> from PIL import Image
        >>> exif = Image.Exif()
        >>> exif[0x0112] = 6
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="PNG", exif=exif.tobytes())
        >>> reopened = Image.open(io.BytesIO(move_exif_after_pixels(encoded.getvalue())))
        >>> "exif" in reopened.info
        False
        >>> reopened.getexif()[0x0112]
        6
    """
    chunks = []
    position = len(_PNG_SIGNATURE)
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        chunks.append(png[position : position + 12 + length])
        position += 12 + length
    exif = [chunk for chunk in chunks if chunk[4:8] == b"eXIf"]
    others = [chunk for chunk in chunks if chunk[4:8] != b"eXIf"]
    if len(exif) != 1 or others[-1][4:8] != b"IEND":
        raise ValueError("expected a PNG with one eXIf chunk that ends with IEND")
    return png[: len(_PNG_SIGNATURE)] + b"".join(others[:-1]) + exif[0] + others[-1]
