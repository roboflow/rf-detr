# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for confidential frame audit logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfdetr_demo.media.frame_audit import (
    ConfidentialFrameAuditLogger,
    FrameAuditRecord,
    evaluate_frame_audit,
)
from rfdetr_demo.paths import REPO_ROOT


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rfdetr_demo.media.frame_audit.CONFIDENTIAL_AUDIT", tmp_path / "audit")
    monkeypatch.setattr("rfdetr_demo.media.frame_audit._FRAME_AUDIT_ROOT", tmp_path / "audit" / "frame-runs")
    monkeypatch.setattr("rfdetr_demo.media.frame_audit._FRAME_AUDIT_JSONL", tmp_path / "audit" / "frame-audit.jsonl")
    monkeypatch.setattr("rfdetr_demo.media.frame_audit.REPO_ROOT", tmp_path)
    return tmp_path / "audit"


def test_evaluate_frame_audit_ok_for_stable_keypoints() -> None:
    records = [
        FrameAuditRecord(1, 0, "keypoint", 4, "img.jpg"),
        FrameAuditRecord(2, 1, "keypoint", 5, "img.jpg"),
        FrameAuditRecord(3, 2, "keypoint", 4, "img.jpg"),
    ]
    result = evaluate_frame_audit(records, task="keypoint")
    assert "OK" in result["verdict"]
    assert result["avg_instances"] == pytest.approx(4.33, abs=0.1)


def test_evaluate_frame_audit_flags_all_zero() -> None:
    records = [FrameAuditRecord(i, i - 1, "keypoint", 0, "img.jpg") for i in range(1, 4)]
    result = evaluate_frame_audit(records, task="keypoint")
    assert result["zero_frame_count"] == 3
    assert "要確認" in result["verdict"]


def test_confidential_frame_audit_logger_writes_files(audit_dir: Path) -> None:
    source = REPO_ROOT / "confidential" / "media" / "input" / "demo.mov"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake")

    audit_logger = ConfidentialFrameAuditLogger(source_path=source, task="keypoint", run_id="test-run")
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    record = audit_logger.maybe_record(
        frame_bgr=frame,
        annotated_bgr=frame,
        frame_index=0,
        processed_count=1,
        instance_count=3,
    )
    assert record is not None
    summary = audit_logger.finalize()
    assert summary is not None
    assert summary.frames_logged == 1
    assert (audit_dir / "frame-runs" / "test-run" / "summary.json").is_file()
