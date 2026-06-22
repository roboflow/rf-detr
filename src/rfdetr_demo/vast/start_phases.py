# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai Pod / job start phases (ported from FlashFind gpuPodStartPhases.ts)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

PreflightStatus = Literal["pass", "warn", "fail"]


class VastJobPhase(str, Enum):
    """High-level phases for external GPU jobs."""

    IDLE = "idle"
    REQUESTING = "requesting"
    BOOTING = "booting"
    SSH_READY = "ssh_ready"
    UPLOADING = "uploading"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    CLEANUP = "cleanup"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class VastJobStep:
    id: str
    label: str


# FlashFind POD_START_STEPS + rf-detr job steps (upload / infer / download / cleanup)
VAST_JOB_STEPS: tuple[VastJobStep, ...] = (
    VastJobStep("request", "Vast API に起動リクエスト"),
    VastJobStep("boot", "GPU インスタンスのブート"),
    VastJobStep("ssh", "SSH ポートの割当"),
    VastJobStep("upload", "入力動画のアップロード"),
    VastJobStep("run", "リモート GPU で解析"),
    VastJobStep("download", "結果動画のダウンロード"),
    VastJobStep("cleanup", "インスタンスの破棄"),
)

BOOT_STATUSES = frozenset(
    {
        "loading",
        "created",
        "scheduling",
        "pending",
        "starting",
        "rebooting",
    },
)


@dataclass(frozen=True)
class VastProgressUpdate:
    """Structured progress event for GUI step display."""

    phase: VastJobPhase
    message: str
    percent: float
    vast_status: str | None = None
    instance_id: int | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    dph_total: float | None = None
    error: str | None = None
    error_hint: str | None = None


def phase_from_runner_phase(
    runner_phase: str,
    *,
    vast_status: str | None,
    request_sent: bool,
    ssh_port: int | None,
) -> VastJobPhase:
    """Map ``vast_ai_runner.VastPhase`` string to GUI job phase."""
    if runner_phase == "creating":
        return VastJobPhase.REQUESTING if not request_sent else VastJobPhase.BOOTING
    if runner_phase == "booting":
        return phase_from_instance_status(vast_status, ssh_port=ssh_port, request_sent=True)
    if runner_phase == "uploading":
        return VastJobPhase.UPLOADING
    if runner_phase == "running":
        return VastJobPhase.RUNNING
    if runner_phase == "downloading":
        return VastJobPhase.DOWNLOADING
    if runner_phase == "cleanup":
        return VastJobPhase.CLEANUP
    if runner_phase == "done":
        return VastJobPhase.DONE
    return VastJobPhase.BOOTING


def phase_from_instance_status(
    vast_status: str | None,
    *,
    ssh_port: int | None,
    request_sent: bool,
) -> VastJobPhase:
    """FlashFind-compatible boot phase from Vast instance status."""
    if not request_sent:
        return VastJobPhase.REQUESTING

    vast = (vast_status or "unknown").lower()
    if vast == "running":
        if ssh_port is not None:
            return VastJobPhase.SSH_READY
        return VastJobPhase.BOOTING
    if vast in BOOT_STATUSES or vast in {"stopped", "exited"}:
        return VastJobPhase.BOOTING
    return VastJobPhase.BOOTING


def completed_step_index(phase: VastJobPhase) -> int:
    """Number of completed steps (0..len(VAST_JOB_STEPS))."""
    mapping = {
        VastJobPhase.IDLE: -1,
        VastJobPhase.REQUESTING: 1,
        VastJobPhase.BOOTING: 2,
        VastJobPhase.SSH_READY: 3,
        VastJobPhase.UPLOADING: 4,
        VastJobPhase.RUNNING: 5,
        VastJobPhase.DOWNLOADING: 6,
        VastJobPhase.CLEANUP: 7,
        VastJobPhase.DONE: len(VAST_JOB_STEPS),
        VastJobPhase.FAILED: -1,
    }
    return mapping.get(phase, -1)


def phase_label(phase: VastJobPhase, vast_status: str | None) -> str:
    """Human-readable phase message (FlashFind gpuPodStartPhases.phaseLabel)."""
    if phase == VastJobPhase.IDLE:
        return ""
    if phase == VastJobPhase.REQUESTING:
        return "起動リクエストを送信中…"
    if phase == VastJobPhase.BOOTING:
        return (
            f"インスタンス起動中（Vast: {vast_status}）…"
            if vast_status
            else "インスタンス起動中…"
        )
    if phase == VastJobPhase.SSH_READY:
        return "SSH 接続情報を取得中…"
    if phase == VastJobPhase.UPLOADING:
        return "入力動画とスクリプトをアップロード中…"
    if phase == VastJobPhase.RUNNING:
        return "リモート GPU で解析中…"
    if phase == VastJobPhase.DOWNLOADING:
        return "結果動画をダウンロード中…"
    if phase == VastJobPhase.CLEANUP:
        return "インスタンスを破棄中（課金停止）…"
    if phase == VastJobPhase.DONE:
        return "外部 GPU ジョブ完了"
    if phase == VastJobPhase.FAILED:
        return "外部 GPU ジョブ失敗"
    return ""


def format_elapsed(sec: int) -> str:
    minutes = sec // 60
    seconds = sec % 60
    if minutes > 0:
        return f"{minutes}:{seconds:02d}"
    return f"{seconds}秒"
