# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightning callbacks for RF-DETR training."""

from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.training.callbacks.drop_schedule import DropPathCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.callbacks.metrics_plot import MetricsPlotCallback

__all__ = [
    "BestModelCallback",
    "DropPathCallback",
    "MetricsPlotCallback",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
]
