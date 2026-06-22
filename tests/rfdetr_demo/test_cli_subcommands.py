# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr-demo CLI routing."""

from __future__ import annotations

from unittest.mock import patch

from rfdetr_demo.cli.main import SUBCOMMANDS, main


def test_default_routes_to_video_demo() -> None:
    with patch("rfdetr_demo.cli.main.video_demo_main", return_value=0) as video_main:
        assert main(["--help"]) == 0
        video_main.assert_called_once_with(["--help"])


def test_probe_count_subcommand() -> None:
    with patch("rfdetr_demo.cli.subcommands.probe_count.run", return_value=0) as probe_run:
        with patch("rfdetr_demo.cli.main._build_parser") as build_parser:
            namespace = type("NS", (), {"_handler": probe_run})()
            build_parser.return_value.parse_args.return_value = namespace
            assert main(["probe-count", "--frames", "5"]) == 0
            build_parser.return_value.parse_args.assert_called_once_with(
                ["probe-count", "--frames", "5"],
            )


def test_video_subcommand_passes_remaining_args() -> None:
    with patch("rfdetr_demo.cli.main.video_demo_main", return_value=0) as video_main:
        assert main(["video", "--task", "keypoint"]) == 0
        video_main.assert_called_once_with(["--task", "keypoint"])


def test_subcommand_names_are_stable() -> None:
    assert SUBCOMMANDS == frozenset(
        {"probe-count", "audit-tracking", "analyze-clip", "video"},
    )
