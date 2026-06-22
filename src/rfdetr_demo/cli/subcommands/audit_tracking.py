# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidential center-person tracking audit subcommand."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rfdetr_demo.media.tracking_audit import run_center_tracking_audit
from rfdetr_demo.paths import resolve_default_source

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``audit-tracking`` subcommand."""
    parser = subparsers.add_parser(
        "audit-tracking",
        help="Audit center-person tracking to confidential/audit/",
    )
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--interval", type=int, default=20, help="Save JPG every N source frames")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames")
    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute audit-tracking and print evaluation summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source = args.source or resolve_default_source()
    summary = run_center_tracking_audit(
        source_path=source,
        sample_interval=args.interval,
        threshold=args.threshold,
        max_frames=args.max_frames,
    )

    print(json.dumps(summary.evaluation, indent=2, ensure_ascii=False))
    print(f"\nrun_id: {summary.run_id}")
    print(f"images: {summary.images_saved} -> {summary.run_dir_relpath}/")
    print("jsonl: confidential/audit/tracking-audit.jsonl")
    return 0
