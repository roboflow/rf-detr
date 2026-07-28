# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CoreML export availability.

Import converters from submodules, not this package root.
"""

try:
    import coremltools  # noqa: F401

    _IS_COREMLTOOLS_AVAILABLE = True
except ImportError:
    _IS_COREMLTOOLS_AVAILABLE = False
