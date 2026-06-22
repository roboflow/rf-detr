# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run RF-DETR video demo jobs on Vast.ai GPU instances.

This module re-exports the split implementation for backward compatibility.
"""

from __future__ import annotations

from rfdetr_demo.vast.api_config import resolve_vast_api_key
from rfdetr_demo.vast.cli import (
    ensure_vast_cli_or_raise,
    is_vast_cli_available,
)
from rfdetr_demo.vast.offers import search_gpu_offers
from rfdetr_demo.vast.types import (
    DEFAULT_VAST_IMAGE,
    REMOTE_JOB_DIR,
    REMOTE_OUTPUT_NAME,
    REMOTE_PROGRESS_PATH,
    VAST_CLI_DOCS_URL,
    VAST_DOCS_URL,
    VastGpuOffer,
    VastLogCallback,
    VastPhase,
    VastPhaseCallback,
    VastRunnerCancelledError,
    VastRunnerError,
    VastVideoJobConfig,
)
from rfdetr_demo.vast.video_job import run_video_demo_on_vast

__all__ = [
    "DEFAULT_VAST_IMAGE",
    "REMOTE_JOB_DIR",
    "REMOTE_OUTPUT_NAME",
    "REMOTE_PROGRESS_PATH",
    "VAST_CLI_DOCS_URL",
    "VAST_DOCS_URL",
    "VastGpuOffer",
    "VastLogCallback",
    "VastPhase",
    "VastPhaseCallback",
    "VastRunnerCancelledError",
    "VastRunnerError",
    "VastVideoJobConfig",
    "ensure_vast_cli_or_raise",
    "is_vast_cli_available",
    "resolve_vast_api_key",
    "run_video_demo_on_vast",
    "search_gpu_offers",
]
