# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for GUI TuneController."""

from __future__ import annotations

from rfdetr_demo.gui.controllers.tune_controller import TuneController
from rfdetr_demo.gui.state.job_state import TuneJobState
from rfdetr_demo.gui.state.tune_parameters import TuneParameters
from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.inference.types import TaskName
from tests.rfdetr_demo.helpers import populate_tune_cache


def _sample_parameters(*, task: TaskName = "keypoint") -> TuneParameters:
    return TuneParameters(
        task=task,
        threshold=0.5,
        keypoint_threshold=0.25,
        person_only=False,
        motion_filter_enabled=True,
        motion_max_speed_fraction=0.35,
        motion_ema_alpha=0.55,
        motion_oscillation_enabled=True,
        ellipse_sigma=1.5,
        heatmap_opacity=0.38,
        heatmap_decay=3.0,
        vertex_radius=4,
        max_ellipse_axis=None,
        keypoint_uncertainty_enabled=True,
        uncertainty_style="heatmap",
    )


def test_start_button_label_states() -> None:
    assert (
        TuneController.start_button_label(
            tune_state=TuneJobState.TUNE_PAUSED,
            tune_mode=True,
            compute_backend="local",
        )
        == "本番実行"
    )
    assert (
        TuneController.start_button_label(
            tune_state=TuneJobState.IDLE,
            tune_mode=True,
            compute_backend="local",
        )
        == "試走開始"
    )
    assert (
        TuneController.start_button_label(
            tune_state=TuneJobState.IDLE,
            tune_mode=False,
            compute_backend="local",
        )
        == "開始"
    )


def test_run_auto_tune_rejects_empty_cache() -> None:
    cache = TunePreviewCache(task="keypoint", fps=30.0, frame_stride=1)
    outcome = TuneController.run_auto_tune(cache, _sample_parameters(), apply=False)
    assert outcome.proposed is None
    assert any("試走キャッシュがありません" in line.message for line in outcome.log_lines)


def test_run_auto_tune_rejects_non_keypoint_task() -> None:
    cache = TunePreviewCache(task="detect", fps=30.0, frame_stride=1)
    populate_tune_cache(cache, person_counts=[1, 1])
    outcome = TuneController.run_auto_tune(
        cache,
        _sample_parameters(task="detect"),
        apply=False,
    )
    assert outcome.proposed is None
    assert any("キーポイントタスク専用" in line.message for line in outcome.log_lines)


def test_plan_tune_preview_complete_live_preview_path() -> None:
    parameters = _sample_parameters()
    summary = {
        "max_source_seconds": 2.0,
        "processed_frames": 10,
        "total_detections": 20,
        "elapsed_sec": 1.2,
        "target": "/tmp/out_tune_preview.mp4",
        "motion_speed_rejections": 1,
        "motion_oscillation_corrections": 2,
    }
    plan = TuneController.plan_tune_preview_complete(
        summary,
        cache_count=10,
        live_preview_enabled=True,
        auto_tune_enabled=True,
        task="keypoint",
        parameters=parameters,
    )
    assert plan.run_auto_tune is True
    assert plan.refresh_live_preview is True
    assert "試走完了" in plan.progress_text
    assert plan.status_metrics.startswith("σ=")
    assert any("リアルタイムプレビュー" in line.message for line in plan.log_lines)


def test_plan_tune_preview_complete_manual_adjust_path() -> None:
    plan = TuneController.plan_tune_preview_complete(
        {
            "max_source_seconds": 2.0,
            "processed_frames": 5,
            "total_detections": 8,
            "elapsed_sec": 0.8,
            "target": "/tmp/out_tune_preview.mp4",
        },
        cache_count=0,
        live_preview_enabled=False,
        auto_tune_enabled=False,
        task="keypoint",
        parameters=_sample_parameters(),
    )
    assert plan.run_auto_tune is False
    assert plan.refresh_live_preview is False
    assert any("本番実行" in line.message for line in plan.log_lines)
