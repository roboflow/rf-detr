# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Image decoding shared by the dataset readers, independent of any annotation format.

The module exposes two output-type families sharing one decoder policy: :func:`decode_image` and
:func:`decode_image_bytes` return NumPy arrays, while :func:`decode_pil_image` and :func:`decode_pil_image_bytes` return
PIL images; see below for why the split exists.  Most of the array-out speedup :func:`decode_image` describes comes from
skipping Pillow's ``convert("RGB")`` and ``np.array`` copies rather than from a faster codec: the raw libjpeg-turbo
decode itself is only about 12% faster, since Pillow's wheels link the same library.  Readers whose transforms take a
PIL image, ``CocoDetection`` and the WebDataset reader, use :func:`decode_pil_image` and :func:`decode_pil_image_bytes`
instead: same policy, but a Pillow-decoded image is returned as is, because copying it into an array only for the caller
to copy it back with ``Image.fromarray`` slowed their data loading on large PNG and BMP files (#1544). ``YoloDetection``
still wraps the arrays of ``_LazyYoloDetectionDataset`` with ``Image.fromarray``, as it did before ``simplejpeg``
support; that keeps its non-JPEG files (and any JPEG ``simplejpeg`` rejects) paying the same copy into an array and back
that :func:`decode_pil_image` avoids for its own callers, a scope call deferred rather than folded into this fix.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

try:
    import simplejpeg  # type: ignore[import-untyped,import-not-found,unused-ignore]
except ImportError:  # optional (``rfdetr[train]``); JPEG decoding falls back to Pillow without it
    simplejpeg = None
from PIL import Image

_JPEG_SOI = b"\xff\xd8"


def _jpeg_draft_reduction(width: int, height: int, draft_size: int | None) -> int:
    """Return the power-of-two factor ``PIL.Image.draft`` would reduce a JPEG by for a square ``draft_size`` box.

    Mirrors ``JpegImageFile.draft``: the largest of 8, 4, 2 that keeps both axes at or above ``draft_size``, else 1.
    Pinning this factor explicitly matters because libjpeg-turbo can also scale by 3/8, 5/8, ... and would otherwise
    pick a finer, slower reduction than Pillow for the same box.

    Examples:
        >>> _jpeg_draft_reduction(2000, 2000, 700)
        2
        >>> _jpeg_draft_reduction(2000, 2000, None)
        1
        >>> _jpeg_draft_reduction(513, 1024, 512)
        1
    """
    if draft_size is None:
        return 1
    scale = min(width // draft_size, height // draft_size)
    return next((factor for factor in (8, 4, 2) if scale >= factor), 1)


def _check_decompression_bomb(width: int, height: int) -> None:
    """Apply Pillow's decompression-bomb guard to a header size before any pixel buffer is allocated.

    Mirrors ``PIL.Image._decompression_bomb_check`` through public names only.  ``PIL.Image.MAX_IMAGE_PIXELS`` is
    read at call time, so a process-wide override (``rfdetr.datasets.o365`` raises it) still applies and ``None``
    disables the guard; more than twice the limit raises and more than the limit warns, with Pillow's own wording so
    callers filtering on the message see the same text from either decoder.

    Args:
        width: Full-size image width from the header.
        height: Full-size image height from the header.

    Raises:
        PIL.Image.DecompressionBombError: If ``width * height`` exceeds twice ``PIL.Image.MAX_IMAGE_PIXELS``.

    Examples:
        >>> _check_decompression_bomb(10, 10)
    """
    limit = Image.MAX_IMAGE_PIXELS
    if limit is None:
        return

    pixels = max(1, width) * max(1, height)

    if pixels > 2 * limit:
        msg = (
            f"Image size ({pixels} pixels) exceeds limit of {2 * limit} pixels, could be decompression bomb DOS attack."
        )
        raise Image.DecompressionBombError(msg)

    if pixels > limit:
        warnings.warn(
            f"Image size ({pixels} pixels) exceeds limit of {limit} pixels, could be decompression bomb DOS attack.",
            Image.DecompressionBombWarning,
        )


def _decode_with_pillow(source: Path | IO[bytes], draft_size: int | None) -> tuple[Image.Image, tuple[float, float]]:
    """Decode a file or an in-memory stream through Pillow into an RGB image with its decode scales.

    The Pillow half of the decoder policy every entry point shares: ``PIL.Image.draft`` applies the JPEG reduction when
    ``draft_size`` is set (a no-op for other formats) and ``convert("RGB")`` loads the pixels into an image that
    outlives the closed source.

    Args:
        source: Image file path, or a binary stream positioned at the start of the encoded bytes.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded RGB image and its horizontal/vertical decode scales, both ``1.0`` when the decoder did not reduce.

    Examples:
        >>> encoded = io.BytesIO()
        >>> Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(encoded, format="JPEG")
        >>> image, scales = _decode_with_pillow(io.BytesIO(encoded.getvalue()), draft_size=16)
        >>> image.mode, image.size, scales
        ('RGB', (32, 16), (0.5, 0.5))
    """
    with Image.open(source) as image:
        full_width, full_height = image.size
        if draft_size is not None:
            image.draft("RGB", (draft_size, draft_size))
        rgb_image = image.convert("RGB")
    return rgb_image, (rgb_image.width / full_width, rgb_image.height / full_height)


def _decode_with_simplejpeg(
    data: bytes, draft_size: int | None
) -> tuple[NDArray[np.uint8], tuple[float, float]] | None:
    """Decode JPEG bytes with ``simplejpeg``, or return ``None`` to leave them to Pillow.

    ``None`` when ``simplejpeg`` is not installed, when ``data`` does not start with a JPEG marker, or when
    ``simplejpeg`` rejects it as corrupt or unsupported, so that Pillow decodes it or raises its usual error.

    Args:
        data: Encoded image bytes.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded ``(H, W, 3)`` uint8 RGB pixels and their horizontal/vertical decode scales, or ``None``.

    Raises:
        PIL.Image.DecompressionBombError: If the image has more than twice ``PIL.Image.MAX_IMAGE_PIXELS`` pixels.

    Examples:
        >>> _decode_with_simplejpeg(b"not a jpeg", draft_size=None) is None
        True
    """
    if simplejpeg is None or not data.startswith(_JPEG_SOI):
        return None
    try:
        header = simplejpeg.decode_jpeg_header(data)
        full_height, full_width = int(header[0]), int(header[1])
        # Pillow's guard on public names, so the limit and its warn/raise tiers stay those of ``Image.open``.
        _check_decompression_bomb(full_width, full_height)
        reduction = _jpeg_draft_reduction(full_width, full_height, draft_size)
        pixels = simplejpeg.decode_jpeg(
            data,
            colorspace="RGB",
            min_height=-(-full_height // reduction),
            min_width=-(-full_width // reduction),
        )
    except ValueError:
        return None  # corrupt or unsupported JPEG: let Pillow decode it or raise its usual error
    return pixels, (pixels.shape[1] / full_width, pixels.shape[0] / full_height)


def _read_simplejpeg_candidate(path: Path) -> bytes | None:
    """Return the file's bytes when ``simplejpeg`` is installed and the file starts with a JPEG marker, else ``None``.

    The marker decides, not the extension, so a mislabelled file still reaches the decoder that can read it. Without
    ``simplejpeg`` nothing is read here, and Pillow streams the file itself.

    Args:
        path: Image file to inspect.

    Returns:
        The encoded bytes for :func:`_decode_with_simplejpeg`, or ``None`` when the file is Pillow's to decode.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     png_path = Path(tmp) / "img.png"
        ...     Image.new("RGB", (4, 4)).save(png_path)
        ...     _read_simplejpeg_candidate(png_path) is None
        True
    """
    if simplejpeg is None:
        return None
    with path.open("rb") as encoded:
        if encoded.read(len(_JPEG_SOI)) != _JPEG_SOI:
            return None
        encoded.seek(0)
        return encoded.read()


def decode_image(path: Path, draft_size: int | None = None) -> tuple[NDArray[np.uint8], tuple[float, float]]:
    """Decode an image file to RGB, optionally downscaling during the JPEG discrete cosine transform (DCT) decode.

    JPEG files go through ``simplejpeg`` (libjpeg-turbo straight into a NumPy buffer) when it is installed.  Its
    output matches Pillow's to within decoder rounding: both wrap libjpeg-turbo, but the two packages can bundle
    different builds, which can introduce small rounding differences.  Everything else falls back to Pillow and
    copies its image into an array: non-JPEG files, a missing ``simplejpeg``, and any JPEG it rejects, so error
    behavior does not depend on the optional package either.  That includes Pillow's decompression-bomb guard
    (``PIL.Image.MAX_IMAGE_PIXELS``), which the ``simplejpeg`` path applies to the full-size header dimensions
    before allocating, exactly as ``PIL.Image.open`` would.

    Returning an array rather than a PIL image lets a caller that wants an array skip a round trip, which is where the
    speedup lands: 1.3-1.8x over Pillow at the decode stage for array consumers such as ``_LazyYoloDetectionDataset``,
    varying with image size and with how much high-frequency detail the JPEG carries.

    When ``draft_size`` is set, both decoders apply the same power-of-two reduction ``PIL.Image.draft`` would choose to
    keep the image at least ``draft_size`` on both axes; it is a no-op for non-JPEG files.

    Args:
        path: Image file to decode.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded ``(H, W, 3)`` uint8 RGB pixels and their horizontal/vertical decode scales, both ``1.0`` when the
        decoder did not reduce.

    Raises:
        PIL.Image.DecompressionBombError: If the image has more than twice ``PIL.Image.MAX_IMAGE_PIXELS`` pixels.
    """
    data = _read_simplejpeg_candidate(path)
    if data is not None:
        return decode_image_bytes(data, draft_size)
    image, scales = _decode_with_pillow(path, draft_size)
    return np.array(image), scales


def decode_image_bytes(data: bytes, draft_size: int | None = None) -> tuple[NDArray[np.uint8], tuple[float, float]]:
    """Decode encoded image bytes to RGB, optionally downscaling during the JPEG DCT (discrete cosine transform) decode.

    Holds the decoder policy :func:`decode_image` applies to encoded bytes already in memory; :func:`decode_image`
    routes JPEG files through it, and :func:`decode_pil_image_bytes` is its counterpart that returns a PIL image.

    Args:
        data: Encoded image bytes.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded ``(H, W, 3)`` uint8 RGB pixels and their horizontal/vertical decode scales, both ``1.0`` when the
        decoder did not reduce.

    Raises:
        PIL.Image.DecompressionBombError: If the image has more than twice ``PIL.Image.MAX_IMAGE_PIXELS`` pixels.
    """
    decoded = _decode_with_simplejpeg(data, draft_size)
    if decoded is not None:
        return decoded
    image, scales = _decode_with_pillow(io.BytesIO(data), draft_size)
    return np.array(image), scales


def decode_pil_image(path: Path, draft_size: int | None = None) -> tuple[Image.Image, tuple[float, float]]:
    """Decode an image file to an RGB PIL image under the same decoder policy as :func:`decode_image`.

    For readers whose transforms take a PIL image, such as ``CocoDetection``.  A JPEG that ``simplejpeg`` decodes is
    wrapped with ``Image.fromarray``; everything Pillow decodes is returned as Pillow produced it, without the copy
    into an array and back that :func:`decode_image` plus ``Image.fromarray`` would make.  Pixels and decode scales are
    the ones :func:`decode_image` returns for the same file.  The default Albumentations CPU training wrapper still
    round-trips a decoded JPEG through its own ``np.array``/``Image.fromarray`` pair on top of this, independent of
    this policy.  The returned image's ``info`` dict is populated only when Pillow decoded it; a JPEG that
    ``simplejpeg`` decoded goes through ``Image.fromarray``, which always starts with an empty ``info``, so ``info``
    is not part of this function's contract.

    Args:
        path: Image file to decode.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded RGB image and its horizontal/vertical decode scales, both ``1.0`` when the decoder did not reduce.

    Raises:
        PIL.Image.DecompressionBombError: If the image has more than twice ``PIL.Image.MAX_IMAGE_PIXELS`` pixels.
    """
    data = _read_simplejpeg_candidate(path)
    if data is not None:
        return decode_pil_image_bytes(data, draft_size)
    return _decode_with_pillow(path, draft_size)


def decode_pil_image_bytes(data: bytes, draft_size: int | None = None) -> tuple[Image.Image, tuple[float, float]]:
    """Decode encoded image bytes to an RGB PIL image under the same decoder policy as :func:`decode_image_bytes`.

    The in-memory counterpart of :func:`decode_pil_image`, for readers such as the WebDataset loader that hold a shard
    member's bytes and feed PIL-based transforms.  As in :func:`decode_pil_image`, the returned image's ``info`` dict
    is populated only when Pillow decoded it, not when ``simplejpeg`` did (``Image.fromarray`` starts with an empty
    ``info``); ``info`` is not part of this function's contract.

    Args:
        data: Encoded image bytes.
        draft_size: Smallest extent the caller can consume without upscaling, or ``None`` for full resolution.

    Returns:
        Decoded RGB image and its horizontal/vertical decode scales, both ``1.0`` when the decoder did not reduce.

    Raises:
        PIL.Image.DecompressionBombError: If the image has more than twice ``PIL.Image.MAX_IMAGE_PIXELS`` pixels.
    """
    decoded = _decode_with_simplejpeg(data, draft_size)
    if decoded is not None:
        pixels, scales = decoded
        return Image.fromarray(pixels), scales
    return _decode_with_pillow(io.BytesIO(data), draft_size)
