# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities`` or ``rfdetr.training.model_ema`` instead."""

from rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.utils", "rfdetr.utilities")

# Re-export from new locations.
from rfdetr.training.model_ema import BestMetricHolder, BestMetricSingle, ModelEma  # noqa: F401, E402
from rfdetr.utilities.reproducibility import seed_all  # noqa: F401, E402
from rfdetr.utilities.state_dict import clean_state_dict  # noqa: F401, E402
