#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Self-contained entry point for Vast.ai remote video jobs.

Copied to the remote instance together with the ``rfdetr_demo`` package tree.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``rfdetr_demo.cli`` (requires ``PYTHONPATH`` to include package root)."""
    from rfdetr_demo.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
