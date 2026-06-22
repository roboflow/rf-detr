# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixins for the video demo window."""

from rfdetr_demo.gui.panels.compute import ComputePanelMixin
from rfdetr_demo.gui.panels.io_task import IoTaskPanelMixin
from rfdetr_demo.gui.panels.job_runner import JobRunnerMixin
from rfdetr_demo.gui.panels.preview import PreviewPanelMixin

__all__ = [
    "ComputePanelMixin",
    "IoTaskPanelMixin",
    "JobRunnerMixin",
    "PreviewPanelMixin",
]
