# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the Object365 dataset module."""

import PIL.Image


def test_o365_import_keeps_finite_decompression_bomb_guard() -> None:
    """Importing ``o365`` must not disable PIL's decompression-bomb guard process-wide."""
    from rfdetr.datasets import o365  # noqa: F401

    assert PIL.Image.MAX_IMAGE_PIXELS is not None, (
        "o365 must set a finite MAX_IMAGE_PIXELS cap, not None (which disables the guard globally)"
    )
    assert PIL.Image.MAX_IMAGE_PIXELS >= 178_956_970, (
        "the cap must stay above PIL's default so legitimate large O365 images still load"
    )
