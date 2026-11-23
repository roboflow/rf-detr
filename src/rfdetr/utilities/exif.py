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

That agreement covers the EXIF block. Pillow, and so RF-DETR and ``PIL.ImageOps.exif_transpose``, also reads
``tiff:Orientation`` from the XMP packet when the EXIF block records no orientation, and a PNG's
``Raw profile type exif`` text chunk when the file has no EXIF block. OpenCV's ``imread``, which Supervision's dataset
loaders call, reads neither and shows such a file as stored.
"""

from __future__ import annotations

import os
import re
import struct

from PIL import Image, PngImagePlugin

from rfdetr.utilities.logger import get_logger

logger = get_logger()

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

#: The two ways an XMP packet spells the orientation, the patterns ``PIL.ImageOps.exif_transpose`` removes.
_XMP_ORIENTATION_PATTERNS = (r'tiff:Orientation="([0-9])"', r"<tiff:Orientation>([0-9])</tiff:Orientation>")

#: ``Image.info`` keys of the XMP packet, which Pillow reads the orientation from when the EXIF block has none.
_XMP_KEYS = ("XML:com.adobe.xmp", "xmp")


def exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation still to apply to an opened image, or ``1`` when it is upright as Pillow decodes it.

    Reads metadata only and never decodes the pixels. Pillow parses a PNG's chunks only up to the pixel data when it
    opens the file, and ``PngImageFile.getexif`` decodes the whole image to reach an eXIf chunk stored after it, so
    ``PIL.Image.Image.getexif`` is called directly and such a chunk is found by :func:`_png_exif_chunk`, which reads
    one 8-byte header per chunk. OpenCV's ``imread`` and ``PIL.ImageOps.exif_transpose`` turn that file as well.
    Pillow turns a TIFF upright itself, reporting the upright size when it opens the file and turning the pixels when
    it loads them, so a TIFF has nothing left to apply.

    A metadata block Pillow cannot parse counts as upright. A PNG whose chunks cannot be read while looking for an eXIf
    chunk after the pixels counts as upright too, with a warning that names the file, so the read is not silently
    left in stored order.

    Where the pixels are decoded anyway, :func:`load_with_orientation` reads the same orientation without walking the
    chunks.

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
    orientation = _opened_orientation(image)
    if orientation is None and isinstance(image, PngImagePlugin.PngImageFile) and "exif" not in image.info:
        orientation = _trailing_png_orientation(image)
    return _turn_or_upright(orientation)


def load_with_orientation(image: Image.Image) -> int:
    """Decode an opened image's pixels and return the EXIF orientation still to apply, as :func:`exif_orientation` does.

    The orientation comes from the same places, but a PNG's eXIf chunk stored after the pixels is taken from what
    Pillow read while decoding them instead of from a walk over the chunk headers. Use it where the pixels are decoded
    anyway; :func:`exif_orientation` stays the read for metadata-only checks that must not decode them.

    Only an eXIf chunk counts once the pixels are decoded: an XMP packet or a raw-profile text chunk stored after them
    is ignored, as :func:`exif_orientation` ignores it, so both functions give every file the same orientation.

    Args:
        image: Image opened with Pillow whose pixels are not loaded yet; this loads them.

    Returns:
        The orientation, from 1 to 8.

    Examples:
        >>> import io
        >>> exif = Image.Exif()
        >>> exif[_ORIENTATION_TAG] = 6
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="PNG", exif=exif.tobytes())
        >>> image = Image.open(encoded)
        >>> load_with_orientation(image), image.getpixel((0, 0))
        (6, (0, 0, 0))
    """
    if image.format == "TIFF":
        image.load()
        return 1
    read_on_open = "exif" in image.info
    orientation = _opened_orientation(image)
    image.load()
    still_png = isinstance(image, PngImagePlugin.PngImageFile) and not image.is_animated
    if orientation is None and still_png and not read_on_open and "exif" in image.info:
        # Pillow stores an eXIf chunk it meets after the pixels in ``info`` while decoding them.
        orientation = _block_orientation(image.info["exif"])
    return _turn_or_upright(orientation)


def _opened_orientation(image: Image.Image) -> object:
    """Return the orientation tag in the metadata Pillow parsed when it opened the image, without decoding pixels.

    Args:
        image: Image opened with Pillow.

    Returns:
        The tag's value, ``None`` when there is none, or ``1`` when the metadata cannot be parsed.

    Examples:
        >>> _opened_orientation(Image.new("RGB", (4, 2))) is None
        True
    """
    try:
        # The base class method: ``PngImageFile.getexif`` decodes the pixels to reach a chunk stored after them.
        return Image.Image.getexif(image).get(_ORIENTATION_TAG)
    except Exception:
        # EXIF is optional metadata: a block Pillow cannot parse must not stop the pixels from decoding.
        return 1


def _block_orientation(data: bytes) -> object:
    """Return the orientation tag of a raw EXIF block.

    Args:
        data: The block, with or without the six-byte ``Exif`` header a JPEG stores before it.

    Returns:
        The tag's value, ``None`` when there is none, or ``1`` when the block cannot be parsed.

    Examples:
        >>> exif = Image.Exif()
        >>> exif[_ORIENTATION_TAG] = 3
        >>> _block_orientation(exif.tobytes()), _block_orientation(b"not exif")
        (3, 1)
    """
    try:
        exif = Image.Exif()
        exif.load(data)
        return exif.get(_ORIENTATION_TAG)
    except Exception:
        # EXIF is optional metadata: a block Pillow cannot parse must not stop the pixels from decoding.
        return 1


def _turn_or_upright(orientation: object) -> int:
    """Return ``orientation`` when it names a transpose, else ``1``: a missing tag or a value outside 2 to 8.

    Args:
        orientation: The tag's value as Pillow read it, ``None`` when there is none.

    Returns:
        The orientation, from 1 to 8.

    Examples:
        >>> _turn_or_upright(6), _turn_or_upright(None), _turn_or_upright(9)
        (6, 1, 1)
    """
    if isinstance(orientation, int) and orientation in _UPRIGHT_TRANSPOSES:
        return orientation
    return 1


def _trailing_png_orientation(image: PngImagePlugin.PngImageFile) -> object:
    """Return the orientation tag of a PNG eXIf chunk stored after the pixels, read without decoding them.

    Reading the file and parsing the block fail differently. A block Pillow cannot parse is optional metadata gone
    bad, read as upright like any other. A file that cannot be read (an I/O error on network storage, a stream the
    caller closed) is logged as a warning naming the file before it is read as upright, because a tagged photo then
    stays in its stored order and its labels land on the wrong pixels.

    Args:
        image: PNG opened with Pillow, with no orientation in the metadata Pillow parsed on open.

    Returns:
        The tag's value, ``None`` when the file has no such chunk or it holds no orientation, or ``1`` when the file
        cannot be read or the block cannot be parsed.

    Examples:
        >>> import io
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="PNG")
        >>> _trailing_png_orientation(Image.open(encoded)) is None
        True
    """
    try:
        trailing = _png_exif_chunk(image)
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not read %s to look for an EXIF orientation stored after its pixels, so it is read as stored: %s",
            repr(image.filename) if image.filename else "a PNG stream",
            error,
        )
        return 1
    return None if trailing is None else _block_orientation(trailing)


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

    The turned image no longer records the orientation it was turned for. Its metadata is copied from ``image``, so
    the tag is removed from it as ``PIL.ImageOps.exif_transpose`` removes it, and code that later calls
    ``exif_transpose`` does not turn the image a second time. ``image`` keeps its own metadata.

    Args:
        image: Decoded image in the order its file stores the pixels.
        orientation: The file's EXIF orientation, as :func:`exif_orientation` returns it.

    Returns:
        The upright image; a new image only when ``orientation`` needs a transpose.

    Examples:
        >>> from PIL import ImageOps
        >>> exif = Image.Exif()
        >>> exif[_ORIENTATION_TAG] = 6
        >>> stored = Image.new("RGB", (4, 2))
        >>> stored.info["exif"] = exif.tobytes()
        >>> upright = upright_image(stored, 6)
        >>> upright.size, ImageOps.exif_transpose(upright).size
        ((2, 4), (2, 4))
        >>> stored.info["exif"] == exif.tobytes()
        True
    """
    transpose = _UPRIGHT_TRANSPOSES.get(orientation)
    if transpose is None:
        return image
    upright = image.transpose(transpose)
    _remove_orientation(upright)
    return upright


def _remove_orientation(image: Image.Image) -> None:
    """Remove the orientation from the metadata of an image that has been turned upright.

    Pillow reads the orientation from ``info["exif"]``, else from a PNG's ``info["Raw profile type exif"]``, and else
    from ``tiff:Orientation`` in the XMP packet, so it is removed from all three, as ``PIL.ImageOps.exif_transpose``
    removes it. An EXIF block Pillow cannot parse or write back is dropped instead of raising: the pixels are already
    upright, and a block that may still hold the tag must not turn them again.

    Args:
        image: Image whose pixels are upright; its ``info`` is updated in place.

    Examples:
        >>> from PIL import ImageOps
        >>> image = Image.new("RGB", (4, 2))
        >>> image.info["XML:com.adobe.xmp"] = '<rdf:Description tiff:Orientation="6"/>'
        >>> _remove_orientation(image)
        >>> image.info["XML:com.adobe.xmp"], ImageOps.exif_transpose(image).size
        ('<rdf:Description />', (4, 2))
    """
    try:
        exif = image.getexif()
        if _ORIENTATION_TAG in exif:
            del exif[_ORIENTATION_TAG]
            if "exif" in image.info:
                image.info["exif"] = exif.tobytes()
            elif "Raw profile type exif" in image.info:
                image.info["Raw profile type exif"] = exif.tobytes().hex()
    except Exception:
        image.info.pop("exif", None)
        image.info.pop("Raw profile type exif", None)
    for key in _XMP_KEYS:
        # Pillow stores a packet as text, as bytes, or as a tuple of byte packets.
        packet = image.info.get(key)
        if isinstance(packet, tuple):
            image.info[key] = tuple(_without_xmp_orientation(part) for part in packet)
        elif isinstance(packet, (str, bytes)):
            image.info[key] = _without_xmp_orientation(packet)


def _without_xmp_orientation(packet: str | bytes) -> str | bytes:
    """Return an XMP packet with its ``tiff:Orientation`` attribute or element removed, keeping its type.

    Args:
        packet: The packet, as text or bytes.

    Returns:
        The packet without the orientation.

    Examples:
        >>> _without_xmp_orientation(b"<tiff:Orientation>6</tiff:Orientation><tiff:Make>x</tiff:Make>")
        b'<tiff:Make>x</tiff:Make>'
    """
    for pattern in _XMP_ORIENTATION_PATTERNS:
        packet = re.sub(pattern, "", packet) if isinstance(packet, str) else re.sub(pattern.encode(), b"", packet)
    return packet
