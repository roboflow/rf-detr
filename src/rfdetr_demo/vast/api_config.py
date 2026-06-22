# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai API key resolution (FlashFind-compatible sources)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rfdetr_demo.paths import REPO_ROOT

_ER_FLOWSCAN_ROOT = REPO_ROOT.parent
_FLASHFIND_ENV = _ER_FLOWSCAN_ROOT / "FlashFind" / "backend" / ".env"
_VASTAI_KEY_FILE = Path.home() / ".config" / "vastai" / "vast_api_key"
_LOCAL_CONFIG = REPO_ROOT / "artifacts" / "vast" / "vast-config.local.json"

VAST_API_KEY_DOCS_URL = "https://cloud.vast.ai/manage-keys/"


@dataclass(frozen=True)
class VastApiKeyInfo:
    """Resolved API key and its provenance (for UI display)."""

    key: str
    source: str

    @property
    def masked(self) -> str:
        if len(self.key) <= 8:
            return "***"
        return f"{self.key[:4]}…{self.key[-4:]}"


def parse_dotenv_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE dotenv file (no export syntax)."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", stripped)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2).strip()
        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            raw_value = raw_value[1:-1]
        if raw_value:
            values[key] = raw_value
    return values


def load_local_vast_config() -> dict[str, str]:
    """Load optional local override from ``artifacts/vast/vast-config.local.json``."""
    if not _LOCAL_CONFIG.is_file():
        return {}
    try:
        payload = json.loads(_LOCAL_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items() if v}


def save_local_vast_api_key(api_key: str) -> None:
    """Persist API key to local config (artifacts/, gitignored)."""
    _LOCAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": api_key.strip()}
    _LOCAL_CONFIG.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_vast_api_key_info(explicit: str | None = None) -> VastApiKeyInfo:
    """Resolve Vast.ai API key using the same sources as FlashFind + rf-detr."""
    if explicit and explicit.strip():
        return VastApiKeyInfo(key=explicit.strip(), source="GUI 入力")

    local_cfg = load_local_vast_config()
    local_key = local_cfg.get("api_key", "").strip()
    if local_key:
        return VastApiKeyInfo(key=local_key, source="artifacts/vast/vast-config.local.json")

    for env_name in ("RFDETR_VAST_API_KEY", "VAST_API_KEY", "FLASHFIND_VAST_API_KEY"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return VastApiKeyInfo(key=env_key, source=f"環境変数 {env_name}")

    flashfind_env = parse_dotenv_file(_FLASHFIND_ENV)
    ff_key = flashfind_env.get("FLASHFIND_VAST_API_KEY", "").strip()
    if ff_key:
        return VastApiKeyInfo(
            key=ff_key,
            source="FlashFind/backend/.env (FLASHFIND_VAST_API_KEY)",
        )

    if _VASTAI_KEY_FILE.is_file():
        key = _VASTAI_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return VastApiKeyInfo(key=key, source="~/.config/vastai/vast_api_key (vastai set api-key)")

    raise ValueError(
        "Vast.ai API キーが見つかりません。\n"
        f"• FlashFind と同様に { _FLASHFIND_ENV } に FLASHFIND_VAST_API_KEY を設定\n"
        f"• または {VAST_API_KEY_DOCS_URL} でキー発行後 `vastai set api-key`\n"
        "• または GUI の API キー欄に直接入力"
    )


def resolve_vast_api_key(explicit: str | None = None) -> str:
    """Return the resolved API key string."""
    return resolve_vast_api_key_info(explicit).key
