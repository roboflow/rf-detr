# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the video CLI ReID / track flag wiring."""

from __future__ import annotations

from unittest.mock import patch

from rfdetr_demo.cli import run_video


def _summary() -> dict[str, object]:
    return {"target": "out.mp4", "task": "detect", "processed_frames": 1, "total_detections": 1, "elapsed_sec": 0.1}


def test_reid_model_flag_enables_embedding_reid() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        code = run_video.main(
            ["--task", "detect", "--person-only", "--track", "--reid-model", "reid.onnx", "--source", "x.mp4"],
        )
    assert code == 0
    kwargs = run_demo.call_args.kwargs
    assert kwargs["detect_track"] is True
    assert kwargs["reid_enabled"] is True
    assert kwargs["reid_model"] == "reid.onnx"


def test_no_reid_flag_leaves_reid_disabled() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        run_video.main(["--task", "detect", "--person-only", "--track", "--source", "x.mp4"])
    kwargs = run_demo.call_args.kwargs
    assert kwargs["reid_enabled"] is False
    assert kwargs["reid_model"] is None
