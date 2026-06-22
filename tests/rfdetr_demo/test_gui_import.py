# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke test for GUI module imports."""

from __future__ import annotations


def test_main_window_imports() -> None:
    from rfdetr_demo.gui.main_window import VideoDemoGuiApp

    assert VideoDemoGuiApp.__name__ == "VideoDemoGuiApp"
