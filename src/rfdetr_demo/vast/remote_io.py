# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Remote file transfer and command helpers for Vast.ai jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rfdetr_demo.vast.cli import run_vast_cli
from rfdetr_demo.vast.instance import execute
from rfdetr_demo.vast.types import (
    REMOTE_JOB_DIR,
    REMOTE_OUTPUT_NAME,
    REMOTE_PROGRESS_PATH,
    VastRunnerError,
    VastVideoJobConfig,
)

REMOTE_PACKAGE_DIR = f"{REMOTE_JOB_DIR}/package"
REMOTE_RUNNER_PATH = f"{REMOTE_JOB_DIR}/remote_runner.py"


def vast_copy(local_path: Path, remote_spec: str, *, api_key: str) -> None:
    """Copy a local file or directory to a remote Vast.ai path."""
    local_spec = f"local:{local_path.resolve()}"
    run_vast_cli(["copy", local_spec, remote_spec], api_key=api_key)


def vast_copy_from_remote(remote_spec: str, local_path: Path, *, api_key: str) -> None:
    """Copy a remote Vast.ai path to the local filesystem."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_spec = f"local:{local_path.resolve()}"
    run_vast_cli(["copy", remote_spec, local_spec], api_key=api_key)


def build_remote_command(config: VastVideoJobConfig) -> str:
    """Build the remote shell command that installs deps and runs the demo."""
    uncertainty_flags = ""
    if config.task == "keypoint" and config.keypoint_uncertainty_style != "none":
        uncertainty_flags = (
            f" --keypoint-uncertainty"
            f" --keypoint-uncertainty-style {config.keypoint_uncertainty_style}"
            f" --ellipse-sigma {config.ellipse_sigma}"
        )
        if config.max_ellipse_axis is not None:
            uncertainty_flags += f" --max-ellipse-axis {config.max_ellipse_axis}"

    person_flag = " --person-only" if config.person_only else ""
    max_frames_flag = f" --max-frames {config.max_frames}" if config.max_frames is not None else ""

    return (
        "set -euo pipefail; "
        "python -m pip install -q --upgrade pip; "
        "python -m pip install -q rfdetr supervision opencv-python-headless "
        "--no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124; "
        f"export PYTHONPATH={REMOTE_PACKAGE_DIR}; "
        f"python {REMOTE_RUNNER_PATH} "
        f"--source {REMOTE_JOB_DIR}/input{config.source_path.suffix} "
        f"--output {REMOTE_JOB_DIR}/{REMOTE_OUTPUT_NAME} "
        f"--task {config.task} "
        f"--model {config.model_size} "
        f"--threshold {config.threshold} "
        f"--frame-stride {config.frame_stride} "
        f"--keypoint-threshold {config.keypoint_threshold} "
        f"--progress-file {REMOTE_PROGRESS_PATH} "
        f"{person_flag}{max_frames_flag}{uncertainty_flags}"
    )


def read_remote_progress(instance_id: int, *, api_key: str) -> dict[str, Any] | None:
    """Fetch remote progress JSON if present."""
    try:
        raw = execute(instance_id, f"cat {REMOTE_PROGRESS_PATH} 2>/dev/null || true", api_key=api_key)
    except VastRunnerError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
