# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Decode one image with a process-local Pillow pixel limit."""

import io
import json
import sys
import warnings
from pathlib import Path
from typing import IO

from PIL import Image, PngImagePlugin, UnidentifiedImageError

#: EXIF orientation tag; the parent process turns the pixels upright, so this worker only reports it.
_ORIENTATION_TAG = 0x0112

#: PIL mode an image is decoded to for each input channel count a dataset can supply.
CHANNEL_MODES = {1: "L", 3: "RGB", 4: "RGBA"}


def image_mode(num_channels: int) -> str:
    """Return the PIL mode datasets decode to for a model with *num_channels* input channels.

    Args:
        num_channels: Input channel count of the model (``ModelConfig.num_channels``).

    Raises:
        ValueError: If image files cannot supply *num_channels* channels.

    Examples:
        >>> image_mode(4)
        'RGBA'
    """
    if num_channels not in CHANNEL_MODES:
        raise ValueError(
            f"num_channels={num_channels} cannot be read from image files. Datasets supply 1 (grayscale), 3 (RGB) or "
            "4 (RGBA) channels."
        )
    return CHANNEL_MODES[num_channels]


def open_image(source: Path | IO[bytes]) -> Image.Image:
    """Open an image with ``Image.open``, reporting a header Pillow rejects with ``ValueError`` as unidentifiable.

    ``Image.open`` lets a ``ValueError`` raised by a format plugin through, for example for a PNG ``pHYs`` chunk shorter
    than its 9 bytes, while it reports other header damage as ``PIL.UnidentifiedImageError``, an ``OSError``.  This
    raises that type for both, so callers that skip corrupt files catch one type.

    Args:
        source: Image file path, or a binary stream positioned at the start of the encoded bytes.

    Returns:
        The opened image, with its pixels not yet decoded.

    Raises:
        PIL.UnidentifiedImageError: If no Pillow plugin recognises the image, or the one that does rejects its header.

    Examples:
        >>> encoded = io.BytesIO()
        >>> Image.new("RGB", (4, 2)).save(encoded, format="PNG")
        >>> open_image(io.BytesIO(encoded.getvalue())).size
        (4, 2)
    """
    try:
        return Image.open(source)
    except ValueError as exc:
        raise UnidentifiedImageError(f"cannot identify image file {source!r}: {exc}") from exc


def convert_image(image: Image.Image, mode: str, name: str) -> Image.Image:
    """Convert an opened image to *mode* without inventing a fourth channel or clipping deep grayscale.

    The channel checks run before any pixel is decoded.  Pillow reports some corrupt files with a ``SyntaxError`` or a
    ``ValueError`` while it loads the pixels, for example a PNG whose chunk header between two image data chunks is cut
    short, or whose ``pHYs`` chunk after them is too short.  That error is raised as an ``OSError`` chained to it, the
    type Pillow uses for every other corrupt or truncated file, so callers catch one type and a corrupt file never
    passes for a channel mismatch.

    Args:
        image: Opened image.
        mode: Target mode, one of the :data:`CHANNEL_MODES` values.
        name: How to name the image in an error message, usually its path.

    Returns:
        The converted image, with its pixels loaded.

    Raises:
        ValueError: If *mode* is ``RGBA`` and the image stores no fourth channel, or *mode* is ``L`` and the image
            holds pixels deeper than 8 bits (``I;16``, ``I``, ``F``), which ``convert`` would saturate at 255.
        OSError: If the pixels cannot be decoded.

    Examples:
        >>> convert_image(Image.new("LA", (2, 2)), "RGBA", "image").mode
        'RGBA'
    """
    if mode == "L" and _is_deep_gray(image.mode):
        raise ValueError(
            f"{name} stores {image.mode!r} pixels deeper than 8 bits, which a model with num_channels=1 cannot read "
            "without saturating them at 255. Convert the file to 8-bit grayscale first."
        )
    if mode == "RGBA" and not _has_fourth_channel(image):
        raise ValueError(_no_fourth_channel_message(image, name))
    try:
        image.load()
    except (SyntaxError, ValueError) as exc:
        raise OSError(str(exc)) from exc
    return image.convert(mode)


def _is_deep_gray(mode: str) -> bool:
    """Return whether *mode* holds single-band pixels deeper than 8 bits (``I;16``, ``I``, ``F``).

    ``Image.convert("L")`` clips these to 255 without an error.

    Examples:
        >>> [_is_deep_gray(mode) for mode in ("I;16", "I", "F", "L", "LA", "RGB")]
        [True, True, True, False, False, False]
    """
    return mode.startswith("I") or mode == "F"


def _has_fourth_channel(image: Image.Image) -> bool:
    """Return whether *image* stores a real fourth channel that ``convert("RGBA")`` carries over.

    A mode ending in ``A`` or ``a`` (``RGBA``, ``LA``, ``PA``, ``RGBa``) always qualifies. The ``transparency`` info
    key counts only for mode ``P``, where it is a palette alpha. On ``RGB`` and ``L`` images it is a tRNS colour key, so
    the alpha ``convert("RGBA")`` derives from it is a function of the stored colour and no fourth channel.

    Associated-alpha (premultiplied) TIFF files, ``ExtraSamples=1``, are admitted too: Pillow reads their ``RGBa``
    raw mode as ``RGBA`` and un-premultiplies the colour bands by the alpha plane, so stored ``(100, 50, 25, 128)``
    decodes to ``(199, 99, 49, 128)``. That is the TIFF-defined meaning of real alpha. A fourth plane that carries data
    rather than transparency must be saved with ``ExtraSamples=2`` (unassociated) to keep its colour bands unchanged.

    Examples:
        >>> _has_fourth_channel(Image.new("RGBA", (1, 1))), _has_fourth_channel(Image.new("RGB", (1, 1)))
        (True, False)
    """
    return image.mode.endswith(("A", "a")) or (image.mode == "P" and "transparency" in image.info)


def _no_fourth_channel_message(image: Image.Image, name: str) -> str:
    """Return the error for an image without a fourth channel, explaining an unspecified TIFF extra sample.

    Pillow opens a TIFF whose fourth sample has ``ExtraSamples=0`` (unspecified) as ``RGB`` and drops that sample,
    so the file does store four samples although the mode shows three.

    Examples:
        >>> "no fourth channel" in _no_fourth_channel_message(Image.new("RGB", (1, 1)), "image")
        True
    """
    samples = getattr(image, "tag_v2", {}).get(277, 0)  # SamplesPerPixel, TIFF only
    if image.mode == "RGB" and samples >= 4:
        return (
            f"{name} stores {samples} samples per pixel, but the extra sample is tagged unspecified "
            "(ExtraSamples=0), so Pillow drops it and reads mode 'RGB'. Re-save the file with ExtraSamples=2 "
            "(unassociated alpha) so the fourth channel is read; a model with num_channels=4 needs RGBA images."
        )
    return f"{name} has no fourth channel (mode {image.mode!r}). A model with num_channels=4 needs RGBA images."


def main() -> None:
    """Decode the requested image and return a JSON header followed by its pixel bytes in stored order."""
    limit = int(sys.argv[1])
    draft_size = int(sys.argv[2]) or None
    mode = sys.argv[3]
    allow_warning = sys.argv[4] == "allow"
    source = Path(sys.argv[5]) if len(sys.argv) > 5 else io.BytesIO(sys.stdin.buffer.read())
    Image.MAX_IMAGE_PIXELS = limit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore" if allow_warning else "error", Image.DecompressionBombWarning)
        try:
            with open_image(source) as image:
                full_width, full_height = image.size
                if draft_size is not None:
                    image.draft("RGB", (draft_size, draft_size))
                # Read as ``rfdetr.utilities.exif.exif_orientation`` reads it, so both processes decode alike: Pillow
                # turns a TIFF itself, and a still PNG's eXIf chunk counts even when it is stored after the pixels.
                # That function walks the chunk headers to find it; here Pillow reads it while decoding the pixels.
                read_on_open = "exif" in image.info
                try:
                    orientation = 1 if image.format == "TIFF" else Image.Image.getexif(image).get(_ORIENTATION_TAG)
                except Exception:
                    orientation = 1  # EXIF is optional metadata; an unparsable block must not stop the decode
                rgb = convert_image(image, mode, str(source) if isinstance(source, Path) else "The image")
                still_png = isinstance(image, PngImagePlugin.PngImageFile) and not image.is_animated
                if orientation is None and still_png and not read_on_open and "exif" in image.info:
                    try:
                        trailing = Image.Exif()
                        trailing.load(image.info["exif"])
                        orientation = trailing.get(_ORIENTATION_TAG)
                    except Exception:
                        orientation = 1
            header = {
                "size": rgb.size,
                "scales": [rgb.width / full_width, rgb.height / full_height],
                "orientation": orientation if isinstance(orientation, int) else 1,
                "warnings": [[str(w.message), w.category.__name__] for w in caught],
            }
            pixels = rgb.tobytes()
        except Image.DecompressionBombWarning as warning:
            header = {"warning": str(warning)}
            pixels = b""
        except Exception as error:
            error_number = error.errno if isinstance(error, OSError) else None
            header = {"error": type(error).__name__, "message": str(error), "errno": error_number}
            pixels = b""
    sys.stdout.buffer.write(json.dumps(header).encode("utf-8") + b"\n" + pixels)


if __name__ == "__main__":
    main()
