# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai CLI helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from threading import Event
from typing import Any

from rfdetr_demo.vast.types import VAST_DOCS_URL, VastRunnerError

logger = logging.getLogger(__name__)


def is_vast_cli_available() -> bool:
    """Return True when the ``vastai`` CLI is on PATH."""
    return shutil.which("vastai") is not None


def run_vast_cli(
    args: list[str],
    *,
    api_key: str | None = None,
    cancel_event: Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a ``vastai`` CLI command and return the completed process."""
    if cancel_event is not None and cancel_event.is_set():
        from rfdetr_demo.vast.types import VastRunnerCancelledError

        raise VastRunnerCancelledError("Cancelled before vastai CLI invocation.")
    if not is_vast_cli_available():
        raise VastRunnerError(
            "vastai CLI が見つかりません。`uv pip install vastai` または "
            f"`pip install vastai` でインストールしてください。{VAST_DOCS_URL}",
        )
    command = ["vastai", *args]
    env = os.environ.copy()
    if api_key:
        env["VAST_API_KEY"] = api_key
    logger.info("Running: %s", " ".join(command))
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=None,
        )
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        stdout = (error.stdout or "").strip()
        detail = stderr or stdout or str(error)
        raise VastRunnerError(f"vastai {' '.join(args[:2])} failed: {detail}") from error


def parse_json_output(raw_output: str) -> Any:
    """Parse JSON emitted by ``vastai ... --raw``."""
    text = raw_output.strip()
    if not text:
        raise VastRunnerError("vastai CLI returned empty output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise VastRunnerError(f"Failed to parse vastai JSON output: {text[:500]}") from error


def ensure_vast_cli_or_raise() -> None:
    """Validate that the Vast.ai CLI is available."""
    if not is_vast_cli_available():
        raise VastRunnerError(
            "vastai CLI が見つかりません。"
            " `uv pip install vastai` を実行するか、"
            f"{VAST_DOCS_URL} から CLI をセットアップしてください。",
        )


