# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Analyze keypoint quality on a short clip."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr_demo.tuning.analyze_clip import main as analyze_clip_main


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``analyze-clip`` subcommand."""
    parser = subparsers.add_parser(
        "analyze-clip",
        help="Analyze keypoint inference quality on a short video clip",
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-threshold", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Delegate to ``rfdetr_demo.tuning.analyze_clip``."""
    argv: list[str] = []
    if args.source is not None:
        argv.extend(["--source", str(args.source)])
    argv.extend(["--seconds", str(args.seconds)])
    argv.extend(["--frame-stride", str(args.frame_stride)])
    argv.extend(["--threshold", str(args.threshold)])
    argv.extend(["--keypoint-threshold", str(args.keypoint_threshold)])
    if args.json_out is not None:
        argv.extend(["--json-out", str(args.json_out)])
    return analyze_clip_main(argv)
