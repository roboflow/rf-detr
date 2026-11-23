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

from PIL import Image

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


def convert_image(image: Image.Image, mode: str, name: str) -> Image.Image:
    """Convert an opened image to *mode* without inventing a fourth channel or clipping deep grayscale.

    Args:
        image: Opened image.
        mode: Target mode, one of the :data:`CHANNEL_MODES` values.
        name: How to name the image in an error message, usually its path.

    Returns:
        The converted image, with its pixels loaded.

    Raises:
        ValueError: If *mode* is ``RGBA`` and the image stores no fourth channel, or *mode* is ``L`` and the image
            holds pixels deeper than 8 bits (``I;16``, ``I``, ``F``), which ``convert`` would saturate at 255.

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
    """Decode the requested image and return a JSON header followed by its pixel bytes."""
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
            with Image.open(source) as image:
                full_width, full_height = image.size
                if draft_size is not None:
                    image.draft("RGB", (draft_size, draft_size))
                rgb = convert_image(image, mode, str(source) if isinstance(source, Path) else "The image")
            header = {
                "size": rgb.size,
                "scales": [rgb.width / full_width, rgb.height / full_height],
                "warnings": [[str(w.message), w.category.__name__] for w in caught],
            }
            pixels = rgb.tobytes()
        except Image.DecompressionBombWarning as warning:
            header = {"warning": str(warning)}
            pixels = b""
        except Exception as error:
            header = {"error": type(error).__name__, "message": str(error)}
            pixels = b""
    sys.stdout.buffer.write(json.dumps(header).encode("utf-8") + b"\n" + pixels)


if __name__ == "__main__":
    main()
