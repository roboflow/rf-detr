# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the CLI entry point configuration."""

import pathlib
import re


class TestEntryPoint:
    """[project.scripts] in pyproject.toml uses the correct CLI entry point."""

    def _read_entry_point(self) -> str:
        """Return the rfdetr console_scripts value from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        content = (root / "pyproject.toml").read_text()
        m = re.search(r"\[project\.scripts\].*?rfdetr\s*=\s*\"([^\"]+)\"", content, re.DOTALL)
        assert m, "rfdetr entry not found in [project.scripts]"
        return m.group(1)

    def test_entry_point_value(self):
        """rfdetr entry point must be rfdetr.cli:main."""
        assert self._read_entry_point() == "rfdetr.cli:main"

    def test_entry_point_not_legacy(self):
        """Entry point must no longer reference rfdetr.cli.main:trainer."""
        assert self._read_entry_point() != "rfdetr.cli.main:trainer"
