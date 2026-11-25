# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for the asset tests."""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Provide an empty temporary home directory with all cache environment variables cleared."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("RF_HOME", raising=False)
    monkeypatch.delenv("ROBOFLOW_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return home
