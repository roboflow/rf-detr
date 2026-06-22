# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecated facade — use :mod:`rfdetr_demo.inference.runner` and :mod:`rfdetr_demo.cli.run_video`."""

from rfdetr_demo.cli.run_video import main, parse_args
from rfdetr_demo.inference.runner import run_demo

__all__ = ["main", "parse_args", "run_demo"]
