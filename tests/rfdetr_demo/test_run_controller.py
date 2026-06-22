# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for GUI RunController."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rfdetr_demo.gui.controllers.run_controller import RunController
from rfdetr_demo.gui.state.job_state import RunConfig, StartJobError, TuneJobState


def _sample_config(
    *,
    source: Path,
    compute_backend: str = "local",
    tune_mode: bool = False,
) -> RunConfig:
    return RunConfig(
        source_path=source,
        output_path=source.with_suffix(".out.mp4"),
        task="keypoint",
        model_size="nano",
        threshold=0.5,
        frame_stride=1,
        max_frames=None,
        person_only=False,
        keypoint_threshold=0.25,
        keypoint_uncertainty_style="heatmap",
        keypoint_uncertainty_enabled=True,
        ellipse_sigma=1.5,
        max_ellipse_axis=None,
        heatmap_opacity=0.38,
        heatmap_decay=3.0,
        vertex_radius=4,
        compute_backend=compute_backend,
        tune_mode=tune_mode,
        tune_preview_seconds=2.0,
        preview_enabled=True,
        motion_filter_enabled=True,
        motion_max_speed_fraction=0.35,
        motion_ema_alpha=0.55,
        motion_oscillation_enabled=True,
    )


def test_prepare_start_rejects_missing_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    result = RunController.prepare_start(
        _sample_config(source=missing),
        tune_state=TuneJobState.IDLE,
    )
    assert isinstance(result, StartJobError)


def test_prepare_start_tune_preview_plan(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    result = RunController.prepare_start(
        _sample_config(source=source, tune_mode=True),
        tune_state=TuneJobState.IDLE,
    )
    assert not isinstance(result, StartJobError)
    assert result.is_tune_preview_run is True
    assert result.max_source_seconds == 2.0
    assert result.output_path.name.endswith("_tune_preview.mp4")


@patch("rfdetr_demo.gui.controllers.run_controller.run_demo", return_value={"processed_frames": 1})
def test_run_local_delegates_to_runner(mock_run_demo: MagicMock) -> None:
    summary = RunController.run_local(source_path=Path("a.mp4"), target_path=Path("b.mp4"))
    assert summary["processed_frames"] == 1
    mock_run_demo.assert_called_once()
