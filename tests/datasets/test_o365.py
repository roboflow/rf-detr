# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the Object365 dataset module."""

import subprocess
import sys


def test_importing_rfdetr_keeps_process_pixel_limit() -> None:
    """Importing ``rfdetr`` must not change ``PIL.Image.MAX_IMAGE_PIXELS`` (#1716).

    Runs in a fresh interpreter: the package is already imported here, so an import-time side effect is not observable.
    """
    code = (
        "import PIL.Image; "
        "PIL.Image.MAX_IMAGE_PIXELS = 10_000; "
        "import rfdetr; "
        "assert PIL.Image.MAX_IMAGE_PIXELS == 10_000, PIL.Image.MAX_IMAGE_PIXELS"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
