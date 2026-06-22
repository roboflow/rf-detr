# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai preflight checks (FlashFind PreflightChecklist pattern)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rfdetr_demo.vast.api_config import _FLASHFIND_ENV, VAST_API_KEY_DOCS_URL, resolve_vast_api_key_info
from rfdetr_demo.vast.cli import is_vast_cli_available

PreflightStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class PreflightCheck:
    id: str
    name: str
    status: PreflightStatus
    detail: str
    fix_hint: str | None = None


def run_vast_preflight(
    *,
    explicit_api_key: str | None,
    vast_cli_available: bool,
    offer_selected: bool,
) -> list[PreflightCheck]:
    """Run preflight checks before starting an external GPU job."""
    checks: list[PreflightCheck] = []

    if vast_cli_available:
        checks.append(
            PreflightCheck(
                id="vast_cli",
                name="vastai CLI",
                status="pass",
                detail="vastai コマンド利用可能",
            ),
        )
    else:
        checks.append(
            PreflightCheck(
                id="vast_cli",
                name="vastai CLI",
                status="fail",
                detail="vastai CLI が見つかりません",
                fix_hint="uv pip install vastai",
            ),
        )

    try:
        info = resolve_vast_api_key_info(explicit_api_key)
        checks.append(
            PreflightCheck(
                id="api_key",
                name="Vast API キー",
                status="pass",
                detail=f"読み込み済み ({info.masked}) — {info.source}",
            ),
        )
    except ValueError as error:
        checks.append(
            PreflightCheck(
                id="api_key",
                name="Vast API キー",
                status="fail",
                detail="未設定",
                fix_hint=f"FlashFind/backend/.env に FLASHFIND_VAST_API_KEY を設定、または {VAST_API_KEY_DOCS_URL}",
            ),
        )
        _ = error

    if offer_selected:
        checks.append(
            PreflightCheck(
                id="gpu_offer",
                name="外部 GPU 選択",
                status="pass",
                detail="GPU オファーが選択されています",
            ),
        )
    else:
        checks.append(
            PreflightCheck(
                id="gpu_offer",
                name="外部 GPU 選択",
                status="warn",
                detail="GPU オファー未選択 — 「GPU 検索」で一覧を取得してください",
                fix_hint="最大 $/時間 や GPU フィルタを調整",
            ),
        )

    flashfind_env_exists = _FLASHFIND_ENV.is_file()
    checks.append(
        PreflightCheck(
            id="flashfind_env",
            name="FlashFind 共有設定",
            status="pass" if flashfind_env_exists else "warn",
            detail=(
                f"FlashFind .env 検出: {_FLASHFIND_ENV}"
                if flashfind_env_exists
                else "FlashFind .env なし（環境変数または GUI 入力を使用）"
            ),
        ),
    )

    return checks


def overall_preflight_status(checks: list[PreflightCheck]) -> PreflightStatus:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "pass"


__all__ = [
    "PreflightCheck",
    "PreflightStatus",
    "is_vast_cli_available",
    "overall_preflight_status",
    "run_vast_preflight",
]
