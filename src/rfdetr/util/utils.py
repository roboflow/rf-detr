# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities`` or ``rfdetr.training.model_ema`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.utils is deprecated; use rfdetr.utilities instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations.
from rfdetr.training.model_ema import BestMetricHolder, BestMetricSingle, ModelEma  # noqa: E402
from rfdetr.utilities.reproducibility import seed_all  # noqa: E402
from rfdetr.utilities.state_dict import clean_state_dict  # noqa: E402

__all__ = [
    "seed_all",
    "clean_state_dict",
    "ModelEma",
    "BestMetricSingle",
    "BestMetricHolder",
]
