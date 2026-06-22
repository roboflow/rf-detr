# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared Vast.ai preflight helpers for the GUI."""

from __future__ import annotations

from rfdetr_demo.vast.cli import is_vast_cli_available
from rfdetr_demo.vast.preflight import (
    PreflightCheck,
    PreflightStatus,
    overall_preflight_status,
    run_vast_preflight,
)


def run_gui_vast_preflight(
    *,
    explicit_api_key: str | None,
    offer_selected: bool,
) -> list[PreflightCheck]:
    """Run preflight checks using the current CLI availability."""
    return run_vast_preflight(
        explicit_api_key=explicit_api_key,
        vast_cli_available=is_vast_cli_available(),
        offer_selected=offer_selected,
    )


def preflight_blocks_start(checks: list[PreflightCheck]) -> bool:
    """Return True when preflight status is fail (start should be disabled)."""
    return overall_preflight_status(checks) == "fail"


def preflight_overall_status(checks: list[PreflightCheck]) -> PreflightStatus:
    """Return overall preflight status: pass, warn, or fail."""
    return overall_preflight_status(checks)
