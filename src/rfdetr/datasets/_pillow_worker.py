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
    """Convert an opened image to *mode* without inventing a fourth channel.

    Args:
        image: Opened image.
        mode: Target mode, one of the :data:`CHANNEL_MODES` values.
        name: How to name the image in an error message, usually its path.

    Returns:
        The converted image, with its pixels loaded.

    Raises:
        ValueError: If *mode* is ``RGBA`` and the image stores no fourth channel.

    Examples:
        >>> convert_image(Image.new("LA", (2, 2)), "RGBA", "image").mode
        'RGBA'
    """
    if mode == "RGBA" and not (image.mode.endswith(("A", "a")) or "transparency" in image.info):
        raise ValueError(
            f"{name} has no fourth channel (mode {image.mode!r}). A model with num_channels=4 needs RGBA images."
        )
    return image.convert(mode)


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
