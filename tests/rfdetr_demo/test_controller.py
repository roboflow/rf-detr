# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for tune job state controller."""

from __future__ import annotations

import pytest

from rfdetr_demo.gui.controller import TuneJobState


def test_tune_state_transitions() -> None:
    state = TuneJobState.IDLE
    state = state.transition_tune_start()
    assert state == TuneJobState.TUNE_RUNNING
    state = state.transition_tune_complete()
    assert state == TuneJobState.TUNE_PAUSED
    state = state.transition_full_start()
    assert state == TuneJobState.FULL_RUNNING
    state = state.transition_done()
    assert state == TuneJobState.DONE


def test_cannot_start_full_from_idle() -> None:
    with pytest.raises(ValueError):
        TuneJobState.IDLE.transition_full_start()
