# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""RF-DETR CLI package.

The ``rfdetr`` console script and ``python -m rfdetr`` both invoke :func:`main`,
which delegates to the Lightning CLI (:class:`~rfdetr.training.cli.RFDETRCli`).

The legacy ``rfdetr.cli.main:trainer`` entry point is still importable but emits
a :class:`DeprecationWarning` on every call.
"""

from rfdetr.training.cli import main

__all__ = ["main"]
