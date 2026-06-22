# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr_demo.paths."""

from __future__ import annotations

from pathlib import Path

from rfdetr_demo.paths import (
    ARTIFACTS_DEMO,
    CONFIDENTIAL_INPUT,
    default_output_path,
    is_under_confidential,
    resolve_default_source,
)


def test_resolve_default_source_returns_existing_or_fallback(tmp_path: Path, monkeypatch: object) -> None:
    import rfdetr_demo.paths as paths_module

    sample = tmp_path / "sample.mov"
    sample.write_bytes(b"fake")
    monkeypatch.setattr(paths_module, "DEFAULT_MN1_INPUT", sample, raising=False)
    monkeypatch.setattr(paths_module, "LEGACY_MN1_INPUT", tmp_path / "missing.mov", raising=False)
    monkeypatch.setattr(paths_module, "SAMPLE_DANCE", tmp_path / "missing2.mov", raising=False)
    assert resolve_default_source() == sample


def test_default_output_path_keypoint_suffix() -> None:
    source = Path("video.mov")
    output = default_output_path(source, "keypoint", keypoint_uncertainty=True, keypoint_uncertainty_style="heatmap")
    assert output.name == "video_keypoints_uncertainty_heatmap.mp4"
    assert output.parent == ARTIFACTS_DEMO


def test_is_under_confidential() -> None:
    inside = CONFIDENTIAL_INPUT / "clip.mov"
    outside = ARTIFACTS_DEMO / "clip.mp4"
    assert is_under_confidential(inside) is True
    assert is_under_confidential(outside) is False
