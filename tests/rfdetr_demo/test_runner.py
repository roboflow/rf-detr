# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for inference runner validation and wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rfdetr_demo.inference.runner import run_demo


def test_run_demo_rejects_missing_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError, match="Source video not found"):
        run_demo(
            source_path=missing,
            target_path=tmp_path / "out.mp4",
            task="detect",
            model_size="nano",
            threshold=0.5,
            frame_stride=1,
            max_frames=None,
            person_only=False,
        )


def test_run_demo_rejects_invalid_frame_stride(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-video")
    with pytest.raises(ValueError, match="frame_stride"):
        run_demo(
            source_path=source,
            target_path=tmp_path / "out.mp4",
            task="detect",
            model_size="nano",
            threshold=0.5,
            frame_stride=0,
            max_frames=None,
            person_only=False,
        )


@patch("rfdetr_demo.inference.runner.finalize_video_path")
@patch("rfdetr_demo.inference.runner.process_video")
@patch("rfdetr_demo.inference.runner.probe_video_size", return_value=(640, 480, 30.0))
@patch("rfdetr_demo.inference.runner.build_detection_model")
def test_run_demo_detection_returns_summary(
    mock_build_model: MagicMock,
    _mock_probe: MagicMock,
    mock_process: MagicMock,
    _mock_finalize: MagicMock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    target = tmp_path / "out.mp4"
    mock_build_model.return_value = MagicMock()

    summary = run_demo(
        source_path=source,
        target_path=target,
        task="detect",
        model_size="nano",
        threshold=0.5,
        frame_stride=1,
        max_frames=2,
        person_only=True,
    )

    assert summary["task"] == "detect"
    assert summary["processed_frames"] == 0
    assert summary["target"] == str(target.resolve())
    mock_process.assert_called_once()
