# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai GUI operations (testable without Tk)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfdetr_demo.gui.theme import preflight_icon_for_status, preflight_style_for_status
from rfdetr_demo.gui.vast_preflight import (
    preflight_blocks_start,
    preflight_overall_status,
    run_gui_vast_preflight,
)
from rfdetr_demo.media.guard import is_vast_transfer_allowed
from rfdetr_demo.paths import VAST_CONSENT_FILE
from rfdetr_demo.vast.api_config import resolve_vast_api_key, resolve_vast_api_key_info, save_local_vast_api_key
from rfdetr_demo.vast.cli import is_vast_cli_available
from rfdetr_demo.vast.preflight import PreflightCheck
from rfdetr_demo.vast.runner import (
    VAST_DOCS_URL,
    VastGpuOffer,
    VastRunnerError,
    ensure_vast_cli_or_raise,
    search_gpu_offers,
)
from rfdetr_demo.vast.safety import VastSafetySettings, cleanup_orphan_instances
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate

_LOG_PHASES = frozenset(
    {
        VastJobPhase.REQUESTING,
        VastJobPhase.BOOTING,
        VastJobPhase.SSH_READY,
        VastJobPhase.UPLOADING,
        VastJobPhase.DOWNLOADING,
        VastJobPhase.CLEANUP,
        VastJobPhase.DONE,
        VastJobPhase.FAILED,
    },
)


@dataclass(frozen=True)
class VastApiKeyLoadOutcome:
    """Result of resolving a Vast API key from GUI input."""

    success: bool
    key: str | None = None
    source_label: str = ""
    log_line: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PreflightHeaderView:
    """Preflight summary line for the GUI header."""

    overall_text: str
    blocks_start: bool


@dataclass(frozen=True)
class PreflightRowView:
    """One preflight check row for the GUI checklist."""

    icon: str
    line: str
    style: str
    fix_hint: str | None = None


@dataclass(frozen=True)
class OfferSearchUiOutcome:
    """GPU offer search result formatted for GUI application."""

    labels: list[str]
    default_label: str
    log_lines: list[tuple[str, str]]
    show_empty_info_dialog: bool


@dataclass(frozen=True)
class VastProgressUiState:
    """Progress bar and status text derived from a Vast phase update."""

    percent: float
    progress_text: str
    status_message: str
    status_metrics: str
    show_progress_panel: bool
    phase_log_line: str | None = None


@dataclass(frozen=True)
class VastJobStartError:
    """User-facing error when a Vast job cannot start."""

    title: str
    message: str


class VastController:
    """Stateless Vast.ai GUI orchestration."""

    @staticmethod
    def normalize_api_key_input(explicit: str) -> str | None:
        stripped = explicit.strip()
        return stripped or None

    @staticmethod
    def should_skip_transfer_prompt(source_path: Path) -> bool:
        return VAST_CONSENT_FILE.is_file() and is_vast_transfer_allowed(source_path)

    @staticmethod
    def build_transfer_prompt_message(source_path: Path) -> str:
        allowed_hint = (
            "confidential/media/input/" if is_vast_transfer_allowed(source_path) else "（許可リスト外）"
        )
        return (
            "この動画は Vast.ai リモート GPU に転送されます。\n\n"
            f"パス: {source_path.resolve()}\n"
            f"転送: {allowed_hint}\n\n"
            "機密データの外部送信に同意しますか？"
        )

    @staticmethod
    def persist_transfer_consent() -> bool:
        try:
            VAST_CONSENT_FILE.parent.mkdir(parents=True, exist_ok=True)
            VAST_CONSENT_FILE.write_text("acknowledged\n", encoding="utf-8")
        except OSError:
            return False
        return True

    @staticmethod
    def load_api_key(explicit: str | None) -> VastApiKeyLoadOutcome:
        try:
            info = resolve_vast_api_key_info(explicit)
        except ValueError as error:
            return VastApiKeyLoadOutcome(
                success=False,
                source_label=f"未設定 — {error}",
                error_message=str(error),
            )
        source_label = f"読み込み元: {info.source} ({info.masked})"
        return VastApiKeyLoadOutcome(
            success=True,
            key=info.key,
            source_label=source_label,
            log_line=f"Vast API キー: {info.source}",
        )

    @staticmethod
    def save_api_key_local(key: str) -> tuple[str, str]:
        stripped = key.strip()
        if not stripped:
            raise ValueError("保存する API キーを入力してください。")
        save_local_vast_api_key(stripped)
        return (
            "読み込み元: artifacts/vast/vast-config.local.json",
            "API キーを artifacts/vast/vast-config.local.json に保存しました",
        )

    @staticmethod
    def run_preflight(
        *,
        explicit_api_key: str | None,
        offer_selected: bool,
    ) -> list[PreflightCheck]:
        return run_gui_vast_preflight(
            explicit_api_key=explicit_api_key,
            offer_selected=offer_selected,
        )

    @staticmethod
    def build_preflight_view(checks: list[PreflightCheck]) -> tuple[PreflightHeaderView, list[PreflightRowView]]:
        overall = preflight_overall_status(checks)
        pass_count = sum(1 for check in checks if check.status == "pass")
        overall_label = {"pass": "OK", "warn": "注意", "fail": "要対応"}[overall]
        header = PreflightHeaderView(
            overall_text=f"総合: {overall_label} ({pass_count}/{len(checks)})",
            blocks_start=preflight_blocks_start(checks),
        )
        rows: list[PreflightRowView] = []
        for check in checks:
            icon = preflight_icon_for_status(check.status)
            rows.append(
                PreflightRowView(
                    icon=icon,
                    line=f"{icon}  {check.name}",
                    style=preflight_style_for_status(check.status),
                    fix_hint=check.fix_hint if check.status in {"fail", "warn"} else None,
                ),
            )
        return header, rows

    @staticmethod
    def find_offer(offers: list[VastGpuOffer], label: str) -> VastGpuOffer | None:
        stripped = label.strip()
        if not stripped:
            return None
        for offer in offers:
            if offer.label == stripped:
                return offer
        return None

    @staticmethod
    def search_offers(
        *,
        api_key: str | None,
        max_dph: float,
        gpu_name: str,
    ) -> list[VastGpuOffer]:
        return search_gpu_offers(
            api_key=api_key,
            max_dph=max_dph,
            gpu_name=gpu_name,
        )

    @staticmethod
    def build_offer_search_ui(offers: list[VastGpuOffer]) -> OfferSearchUiOutcome:
        labels = [offer.label for offer in offers]
        if labels:
            return OfferSearchUiOutcome(
                labels=labels,
                default_label=labels[0],
                log_lines=[("info", f"Vast.ai: {len(labels)} 件の GPU オファーを取得")],
                show_empty_info_dialog=False,
            )
        return OfferSearchUiOutcome(
            labels=[],
            default_label="",
            log_lines=[("info", "Vast.ai: 条件に合う GPU が見つかりませんでした")],
            show_empty_info_dialog=True,
        )

    @staticmethod
    def startup_log_lines() -> list[str]:
        lines: list[str] = []
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                lines.append(f"ローカル CUDA 利用可: {gpu_name}")
            else:
                lines.append("ローカル CUDA 不可 — CPU 実行、または Vast.ai GPU を選択")
        except ImportError:
            lines.append("PyTorch 未検出 — ローカル実行は制限される可能性があります")
        if is_vast_cli_available():
            lines.append("vastai CLI 検出済み")
        else:
            lines.append("vastai CLI 未検出 — `uv pip install vastai` で Vast.ai 実行を有効化")
        safety = VastSafetySettings.from_env()
        lines.append(
            f"Vast.ai 安全設定: 最大セッション {safety.max_session_sec / 3600:.1f}h, "
            f"最大実行 {safety.max_execute_sec / 3600:.1f}h, "
            f"ラベル {safety.instance_label_prefix}-*",
        )
        lines.append(f"Vast.ai: {VAST_DOCS_URL}")
        return lines

    @staticmethod
    def startup_orphan_cleanup(*, explicit_api_key: str | None) -> list[str]:
        settings = VastSafetySettings.from_env()
        if not settings.auto_cleanup_orphans_on_start or not is_vast_cli_available():
            return []
        try:
            api_key = resolve_vast_api_key(explicit_api_key)
            destroyed = cleanup_orphan_instances(api_key=api_key, settings=settings)
        except VastRunnerError:
            return []
        if destroyed:
            return [f"前回残っていた Vast.ai インスタンスを破棄: {destroyed}"]
        return []

    @staticmethod
    def progress_ui_state(update: VastProgressUpdate) -> VastProgressUiState:
        percent = min(100.0, max(0.0, update.percent))
        phase_log_line: str | None = None
        if update.phase in _LOG_PHASES:
            status_suffix = f" [{update.vast_status}]" if update.vast_status else ""
            extra = ""
            if update.ssh_port is not None and update.ssh_host:
                extra = f" | ssh -p {update.ssh_port} root@{update.ssh_host}"
                if update.dph_total is not None:
                    extra += f" (~${update.dph_total:.2f}/h)"
            phase_log_line = f"[Vast:{update.phase.value}] {update.message}{status_suffix}{extra}"
        return VastProgressUiState(
            percent=percent,
            progress_text=f"{percent:.0f}%  ·  {update.message}",
            status_message="外部 GPU 実行中",
            status_metrics=update.message,
            show_progress_panel=update.phase != VastJobPhase.IDLE,
            phase_log_line=phase_log_line,
        )

    @staticmethod
    def validate_job_start(
        *,
        api_key: str | None,
        offer_selected: bool,
    ) -> VastJobStartError | None:
        try:
            ensure_vast_cli_or_raise()
            resolve_vast_api_key(api_key)
        except (VastRunnerError, ValueError) as error:
            return VastJobStartError(title="Vast.ai", message=str(error))
        checks = VastController.run_preflight(
            explicit_api_key=api_key,
            offer_selected=offer_selected,
        )
        if preflight_blocks_start(checks):
            return VastJobStartError(
                title="Preflight 未完了",
                message=(
                    "事前チェックで要対応項目があります。\n"
                    "Vast API キーと vastai CLI を確認してください。"
                ),
            )
        return None
