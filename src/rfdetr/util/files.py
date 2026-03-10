# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities.files`` instead."""

import warnings

warnings.warn(
    "rfdetr.util.files is deprecated; use rfdetr.utilities.files instead.",
    DeprecationWarning,
    stacklevel=2,
)

from rfdetr.utilities.files import (  # noqa: E402, F401
    _compute_file_md5,
    _download_file,
    _validate_file_md5,
)

