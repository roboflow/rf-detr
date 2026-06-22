# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared types and constants for Vast.ai integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rfdetr_demo.vast.start_phases import VastProgressUpdate

VAST_DOCS_URL = "https://vast.ai/"
VAST_CLI_DOCS_URL = "https://docs.vast.ai/cli/hello-world"
DEFAULT_VAST_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
REMOTE_JOB_DIR = "/workspace/rfdetr_job"
REMOTE_PROGRESS_PATH = f"{REMOTE_JOB_DIR}/progress.json"
REMOTE_OUTPUT_NAME = "output.mp4"

VastLogCallback = Callable[[str], None]
VastPhaseCallback = Callable[[VastProgressUpdate], None]


class VastRunnerError(RuntimeError):
    """Raised when a Vast.ai workflow step fails."""


class VastRunnerCancelledError(VastRunnerError):
    """Raised when the user cancels a Vast.ai job."""


class VastPhase(str, Enum):
    """High-level phases for remote job progress reporting."""

    SEARCHING = "searching"
    CREATING = "creating"
    BOOTING = "booting"
    UPLOADING = "uploading"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    CLEANUP = "cleanup"
    DONE = "done"


@dataclass(frozen=True)
class VastGpuOffer:
    """A rentable GPU offer returned by ``vastai search offers``."""

    offer_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    dph_total: float
    reliability: float
    cuda_max_good: float

    @property
    def label(self) -> str:
        """Human-readable label for GUI list boxes."""
        return (
            f"{self.gpu_name} x{self.num_gpus} | "
            f"${self.dph_total:.2f}/hr | "
            f"{self.gpu_ram_gb:.0f}GB | rel {self.reliability:.2f}"
        )


@dataclass(frozen=True)
class VastVideoJobConfig:
    """Parameters for a remote video demo run on Vast.ai."""

    source_path: Path
    target_path: Path
    task: str
    model_size: str
    threshold: float
    frame_stride: int
    max_frames: int | None
    person_only: bool
    keypoint_threshold: float
    keypoint_uncertainty_style: str
    ellipse_sigma: float
    max_ellipse_axis: float | None
    offer_id: int
    api_key: str | None = None
    destroy_on_finish: bool = True
    disk_gb: int = 50
    docker_image: str = DEFAULT_VAST_IMAGE
    user_acknowledged: bool = False
