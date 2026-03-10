# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.util.get_param_dicts is deprecated; use rfdetr.training.param_groups."""

import warnings

warnings.warn(
    "rfdetr.util.get_param_dicts is deprecated; use rfdetr.training.param_groups instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.training.param_groups import (  # noqa: F401, E402
    get_param_dict,
    get_vit_lr_decay_rate,
    get_vit_weight_decay_rate,
)
