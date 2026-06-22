# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for GUI VastController."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rfdetr_demo.gui.controllers.vast_controller import VastController
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate
from rfdetr_demo.vast.types import VastGpuOffer


def _sample_offer(*, offer_id: int = 1, label_suffix: str = "") -> VastGpuOffer:
    return VastGpuOffer(
        offer_id=offer_id,
        gpu_name=f"RTX4090{label_suffix}",
        num_gpus=1,
        gpu_ram_gb=24.0,
        dph_total=0.55,
        reliability=0.98,
        cuda_max_good=12.0,
    )


def test_normalize_api_key_input() -> None:
    assert VastController.normalize_api_key_input("  abc  ") == "abc"
    assert VastController.normalize_api_key_input("   ") is None


def test_find_offer_by_label() -> None:
    offers = [_sample_offer(), _sample_offer(offer_id=2, label_suffix="b")]
    selected = VastController.find_offer(offers, offers[1].label)
    assert selected is not None
    assert selected.offer_id == 2
    assert VastController.find_offer(offers, "") is None


def test_build_offer_search_ui_with_results() -> None:
    offers = [_sample_offer(), _sample_offer(offer_id=2)]
    ui = VastController.build_offer_search_ui(offers)
    assert len(ui.labels) == 2
    assert ui.default_label == offers[0].label
    assert ui.show_empty_info_dialog is False


def test_build_offer_search_ui_empty() -> None:
    ui = VastController.build_offer_search_ui([])
    assert ui.labels == []
    assert ui.show_empty_info_dialog is True


def test_progress_ui_state_includes_phase_log() -> None:
    update = VastProgressUpdate(
        phase=VastJobPhase.BOOTING,
        message="インスタンス起動中",
        percent=12.0,
        vast_status="loading",
        ssh_host="1.2.3.4",
        ssh_port=22022,
        dph_total=0.55,
    )
    state = VastController.progress_ui_state(update)
    assert state.percent == 12.0
    assert state.show_progress_panel is True
    assert state.phase_log_line is not None
    assert "booting" in state.phase_log_line


@patch("rfdetr_demo.gui.controllers.vast_controller.is_vast_transfer_allowed", return_value=True)
@patch("rfdetr_demo.gui.controllers.vast_controller.VAST_CONSENT_FILE")
def test_should_skip_transfer_prompt(consent_file: MagicMock, _allowed: MagicMock) -> None:
    consent_file.is_file.return_value = True
    assert VastController.should_skip_transfer_prompt(Path("confidential/media/input/clip.mp4")) is True


@patch("rfdetr_demo.gui.controllers.vast_controller.ensure_vast_cli_or_raise")
@patch("rfdetr_demo.gui.controllers.vast_controller.resolve_vast_api_key")
@patch("rfdetr_demo.gui.controllers.vast_controller.run_gui_vast_preflight")
def test_validate_job_start_blocks_on_preflight_fail(
    mock_preflight: MagicMock,
    _resolve_key: MagicMock,
    _ensure_cli: MagicMock,
) -> None:
    from rfdetr_demo.vast.preflight import PreflightCheck

    mock_preflight.return_value = [
        PreflightCheck(
            id="vast_cli",
            name="vastai CLI",
            status="fail",
            detail="missing",
            fix_hint="install",
        ),
    ]
    error = VastController.validate_job_start(api_key="key", offer_selected=True)
    assert error is not None
    assert error.title == "Preflight 未完了"
