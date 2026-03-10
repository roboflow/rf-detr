# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — rfdetr.deploy is deprecated; use rfdetr.export."""

import sys
import warnings

warnings.warn(
    "rfdetr.deploy is deprecated; use rfdetr.export instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Make old submodule paths still importable without submodule files
import rfdetr.export.benchmark as _benchmark  # noqa: E402
import rfdetr.export.main as _export_main  # noqa: E402
from rfdetr.export import *  # noqa: F401, F403, E402

sys.modules.setdefault("rfdetr.deploy.benchmark", _benchmark)
sys.modules.setdefault("rfdetr.deploy.export", _export_main)
