# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR demo CLI entry point with subcommands."""

from __future__ import annotations

import argparse
import sys

from rfdetr_demo.cli.run_video import main as video_demo_main
from rfdetr_demo.cli.subcommands import analyze_clip, audit_tracking, compare_reid, probe_count

SUBCOMMANDS = frozenset({"probe-count", "audit-tracking", "analyze-clip", "compare-reid", "video"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfdetr-demo",
        description="RF-DETR video demo and diagnostic tools.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    probe_count.add_parser(subparsers)
    audit_tracking.add_parser(subparsers)
    analyze_clip.add_parser(subparsers)
    compare_reid.add_parser(subparsers)
    subparsers.add_parser(
        "video",
        help="Run RF-DETR on a video and export an annotated MP4 (same as default)",
        add_help=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to video demo or a registered subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return video_demo_main(argv)

    if argv[0] == "video":
        return video_demo_main(argv[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


__all__ = ["main"]
