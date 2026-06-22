# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Repository and media path constants for the RF-DETR demo layer."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"

CONFIDENTIAL_ROOT: Path = REPO_ROOT / "confidential"
CONFIDENTIAL_INPUT: Path = CONFIDENTIAL_ROOT / "media" / "input"
CONFIDENTIAL_OUTPUT: Path = CONFIDENTIAL_ROOT / "media" / "output"
CONFIDENTIAL_AUDIT: Path = CONFIDENTIAL_ROOT / "audit"
VAST_CONSENT_FILE: Path = CONFIDENTIAL_ROOT / ".vast-consent"

SAMPLE_DIR: Path = REPO_ROOT / "sample"
ARTIFACTS_DEMO: Path = REPO_ROOT / "artifacts" / "demo"

DEFAULT_MN1_INPUT: Path = CONFIDENTIAL_INPUT / "mn1-2.mov"
LEGACY_MN1_INPUT: Path = SAMPLE_DIR / "mn1-2.mov"
SAMPLE_DANCE: Path = SAMPLE_DIR / "mzoo.mov"

FLASHFIND_POTATO: Path = (
    REPO_ROOT.parent / "FlashFind" / "frontend" / "public" / "demo" / "potato_conveyor.mov"
)
LEGACY_POTATO: Path = REPO_ROOT.parent / "ジャガイモ動画.mov"


def flashfind_potato_path() -> Path:
    """Return FlashFind demo video path (env override supported)."""
    override = os.environ.get("RFDETR_FLASHFIND_POTATO")
    if override:
        return Path(override).expanduser().resolve()
    return FLASHFIND_POTATO


def resolve_default_source() -> Path:
    """Pick the best default input video available on disk."""
    candidates = [
        DEFAULT_MN1_INPUT,
        LEGACY_MN1_INPUT,
        SAMPLE_DANCE,
        flashfind_potato_path(),
        LEGACY_POTATO,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return SAMPLE_DANCE


def default_output_path(
    source_path: Path,
    task: str,
    *,
    keypoint_uncertainty: bool = False,
    keypoint_uncertainty_style: str = "none",
) -> Path:
    """Derive an output path from the source stem and task."""
    if task == "keypoint":
        if not keypoint_uncertainty:
            suffix = "keypoints"
        elif keypoint_uncertainty_style == "heatmap":
            suffix = "keypoints_uncertainty_heatmap"
        elif keypoint_uncertainty_style == "magnitude":
            suffix = "keypoints_uncertainty_magnitude"
        else:
            suffix = "keypoints_uncertainty"
    elif task == "segment":
        suffix = "segmented"
    else:
        suffix = "detected"
    root = CONFIDENTIAL_OUTPUT if is_under_confidential(source_path) else ARTIFACTS_DEMO
    return root / f"{source_path.stem}_{suffix}.mp4"


def is_under_confidential(path: Path) -> bool:
    """Return True when *path* resolves under ``confidential/``."""
    try:
        path.resolve().relative_to(CONFIDENTIAL_ROOT.resolve())
        return True
    except ValueError:
        return False
