# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI job lifecycle state and run configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from rfdetr_demo.inference.types import KeypointUncertaintyStyle, ModelSize, TaskName

ComputeBackend = Literal["local", "vast"]


class TuneJobState(str, Enum):
    """State machine for tune-preview → full-run workflow."""

    IDLE = "idle"
    TUNE_RUNNING = "tune_running"
    TUNE_PAUSED = "tune_paused"
    FULL_RUNNING = "full_running"
    DONE = "done"

    def may_start_tune(self) -> bool:
        return self in {TuneJobState.IDLE, TuneJobState.TUNE_PAUSED}

    def may_start_full(self) -> bool:
        return self in {TuneJobState.TUNE_PAUSED, TuneJobState.DONE}

    def transition_tune_start(self) -> TuneJobState:
        if not self.may_start_tune():
            msg = f"Cannot start tune from state {self.value}"
            raise ValueError(msg)
        return TuneJobState.TUNE_RUNNING

    def transition_tune_complete(self) -> TuneJobState:
        if self != TuneJobState.TUNE_RUNNING:
            msg = f"Cannot complete tune from state {self.value}"
            raise ValueError(msg)
        return TuneJobState.TUNE_PAUSED

    def transition_full_start(self) -> TuneJobState:
        if not self.may_start_full():
            msg = f"Cannot start full run from state {self.value}"
            raise ValueError(msg)
        return TuneJobState.FULL_RUNNING

    def transition_cancel(self) -> TuneJobState:
        return TuneJobState.IDLE

    def transition_done(self) -> TuneJobState:
        return TuneJobState.DONE


@dataclass(frozen=True)
class RunConfig:
    """Snapshot of GUI form values for one inference job."""

    source_path: Path
    output_path: Path
    task: TaskName
    model_size: ModelSize
    threshold: float
    frame_stride: int
    max_frames: int | None
    person_only: bool
    keypoint_threshold: float
    keypoint_uncertainty_style: KeypointUncertaintyStyle
    keypoint_uncertainty_enabled: bool
    ellipse_sigma: float
    max_ellipse_axis: float | None
    heatmap_opacity: float
    heatmap_decay: float
    vertex_radius: int
    compute_backend: ComputeBackend
    tune_mode: bool
    tune_preview_seconds: float
    preview_enabled: bool
    motion_filter_enabled: bool
    motion_max_speed_fraction: float
    motion_ema_alpha: float
    motion_oscillation_enabled: bool
    vast_offer_id: int | None = None
    vast_api_key: str | None = None
    vast_destroy_on_finish: bool = True


@dataclass(frozen=True)
class StartJobPlan:
    """Validated plan to start local or Vast execution."""

    config: RunConfig
    output_path: Path
    is_tune_preview_run: bool
    is_full_run_after_tune: bool
    max_source_seconds: float | None
    effective_max_frames: int | None


@dataclass(frozen=True)
class StartJobError:
    """User-facing validation error before job start."""

    title: str
    message: str


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable summary of a completed job for insight panels."""

    processed_frames: int
    total_detections: int
    elapsed_sec: float
    target: str
    compute: str
    tune_preview: bool = False
    max_source_seconds: float | None = None

    @classmethod
    def from_summary(cls, summary: dict[str, object]) -> JobSnapshot:
        return cls(
            processed_frames=int(summary.get("processed_frames", 0)),
            total_detections=int(summary.get("total_detections", 0)),
            elapsed_sec=float(summary.get("elapsed_sec", 0.0)),
            target=str(summary.get("target", "")),
            compute=str(summary.get("compute", "local")),
            tune_preview=bool(summary.get("tune_preview")),
            max_source_seconds=(
                float(summary["max_source_seconds"])
                if summary.get("max_source_seconds") is not None
                else None
            ),
        )
