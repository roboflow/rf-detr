# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the CLI entry point configuration."""

import pathlib
import re

import pytest


@pytest.fixture
def entry_point() -> str:
    """Return the rfdetr console_scripts value from pyproject.toml."""
    root = pathlib.Path(__file__).parent.parent.parent
    content = (root / "pyproject.toml").read_text()
    m = re.search(r"\[project\.scripts\].*?rfdetr\s*=\s*\"([^\"]+)\"", content, re.DOTALL)
    assert m, "rfdetr entry not found in [project.scripts]"
    return m.group(1)


class TestEntryPoint:
    """[project.scripts] in pyproject.toml uses the correct CLI entry point."""

    def test_is_cli_main(self, entry_point: str) -> None:
        """The rfdetr console script points at rfdetr.cli:main."""
        assert entry_point == "rfdetr.cli:main"

    def test_is_not_legacy_trainer(self, entry_point: str) -> None:
        """The rfdetr console script no longer references rfdetr.cli.main:trainer."""
        assert entry_point != "rfdetr.cli.main:trainer"
