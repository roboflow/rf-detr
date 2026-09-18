# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR CLI package.

The ``rfdetr`` console script and ``python -m rfdetr`` both invoke :func:`main`, which runs
:class:`~rfdetr.training.cli.RFDETRCli` (Lightning CLI with jsonargparse).
"""

__all__ = ["main"]


def main() -> None:
    """Run the training CLI, loading its optional dependencies only when invoked."""
    # Packing subcommands must remain usable without the training extra.
    from rfdetr.training.cli import main as training_main

    training_main()
