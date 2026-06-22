# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Vast.ai CLI helpers."""

from __future__ import annotations

import json

import pytest

from rfdetr_demo.vast.cli import VastRunnerError, parse_json_output


def test_parse_json_output_valid() -> None:
    payload = parse_json_output('{"ok": true}')
    assert payload == {"ok": True}


def test_parse_json_output_empty_raises() -> None:
    with pytest.raises(VastRunnerError, match="empty"):
        parse_json_output("   ")


def test_parse_json_output_invalid_json_raises() -> None:
    with pytest.raises(VastRunnerError, match="Failed to parse"):
        parse_json_output("not-json")


def test_parse_json_output_list() -> None:
    payload = parse_json_output(json.dumps([1, 2, 3]))
    assert payload == [1, 2, 3]
