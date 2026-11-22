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

from PIL import Image, PngImagePlugin

#: EXIF orientation tag; the parent process turns the pixels upright, so this worker only reports it.
_ORIENTATION_TAG = 0x0112


def main() -> None:
    """Decode the requested image and return a JSON header followed by RGB bytes in stored order."""
    limit = int(sys.argv[1])
    draft_size = int(sys.argv[2]) or None
    allow_warning = sys.argv[3] == "allow"
    source = Path(sys.argv[4]) if len(sys.argv) > 4 else io.BytesIO(sys.stdin.buffer.read())
    Image.MAX_IMAGE_PIXELS = limit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore" if allow_warning else "error", Image.DecompressionBombWarning)
        try:
            with Image.open(source) as image:
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
                rgb = image.convert("RGB")
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
            header = {"error": type(error).__name__, "message": str(error)}
            pixels = b""
    sys.stdout.buffer.write(json.dumps(header).encode("utf-8") + b"\n" + pixels)


if __name__ == "__main__":
    main()
