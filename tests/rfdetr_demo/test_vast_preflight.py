# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Vast.ai preflight checks."""

from __future__ import annotations

from rfdetr_demo.vast.preflight import overall_preflight_status, run_vast_preflight


def test_preflight_fail_without_cli_or_key() -> None:
    checks = run_vast_preflight(
        explicit_api_key=None,
        vast_cli_available=False,
        offer_selected=False,
    )
    assert overall_preflight_status(checks) == "fail"
    assert any(check.id == "vast_cli" and check.status == "fail" for check in checks)


def test_preflight_pass_with_cli_and_key(monkeypatch: object) -> None:
    from rfdetr_demo.vast import api_config

    class _Info:
        key = "test-key"
        masked = "****"
        source = "test"

    monkeypatch.setattr(api_config, "resolve_vast_api_key_info", lambda _explicit: _Info())
    checks = run_vast_preflight(
        explicit_api_key="test-key",
        vast_cli_available=True,
        offer_selected=True,
    )
    assert overall_preflight_status(checks) == "pass"
