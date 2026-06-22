# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for confidential media governance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfdetr_demo.media.guard import (
    MediaGuardError,
    assert_vast_transfer_allowed,
    is_vast_transfer_allowed,
    log_transfer_audit,
    resolve_media_path,
)
from rfdetr_demo.paths import CONFIDENTIAL_AUDIT, CONFIDENTIAL_INPUT, REPO_ROOT


def test_resolve_media_path_rejects_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mov"
    outside.write_bytes(b"x")
    with pytest.raises(MediaGuardError):
        resolve_media_path(outside)


def test_vast_transfer_allowlist_blocks_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_file = REPO_ROOT / "sample" / "test-block.mov"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_bytes(b"demo")
    try:
        monkeypatch.delenv("RFDETR_VAST_ALLOW_ANY_SOURCE", raising=False)
        assert not is_vast_transfer_allowed(sample_file)
        with pytest.raises(MediaGuardError):
            assert_vast_transfer_allowed(sample_file, user_acknowledged=True)
    finally:
        if sample_file.is_file():
            sample_file.unlink()


def test_vast_transfer_requires_acknowledgement(tmp_path: Path) -> None:
    confidential_input = CONFIDENTIAL_INPUT
    confidential_input.mkdir(parents=True, exist_ok=True)
    media = confidential_input / "ack-test.mov"
    media.write_bytes(b"secret")
    try:
        with pytest.raises(MediaGuardError):
            assert_vast_transfer_allowed(media, user_acknowledged=False)
        assert_vast_transfer_allowed(media, user_acknowledged=True)
    finally:
        if media.is_file():
            media.unlink()


def test_log_transfer_audit_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_dir = CONFIDENTIAL_AUDIT
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_file = audit_dir / "vast-transfers.jsonl"
    if log_file.is_file():
        log_file.unlink()
    media = CONFIDENTIAL_INPUT / "audit-test.mov"
    CONFIDENTIAL_INPUT.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"audit")
    try:
        log_transfer_audit("upload_start", media, "remote:123:/input.mov", instance_id=123)
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "upload_start"
        assert record["instance_id"] == 123
        assert record["sha256"]
    finally:
        if media.is_file():
            media.unlink()
        if log_file.is_file():
            log_file.unlink()


def test_allow_any_source_env(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_file = REPO_ROOT / "sample" / "env-override.mov"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_bytes(b"demo")
    try:
        monkeypatch.setenv("RFDETR_VAST_ALLOW_ANY_SOURCE", "1")
        assert is_vast_transfer_allowed(sample_file)
        assert_vast_transfer_allowed(sample_file, user_acknowledged=True)
    finally:
        if sample_file.is_file():
            sample_file.unlink()
