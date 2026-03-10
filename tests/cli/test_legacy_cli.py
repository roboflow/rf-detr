# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the legacy CLI entry point migration — PTL Ch4/T5.

Verifies two invariants:
1. pyproject.toml [project.scripts] now points to rfdetr.training.cli:main.
2. The legacy rfdetr.cli.main:trainer emits a DeprecationWarning so users
   who call it programmatically are guided toward the new CLI.
"""

import pathlib
import re
import sys
import unittest.mock as mock
import warnings

# ---------------------------------------------------------------------------
# Entry point config
# ---------------------------------------------------------------------------


class TestEntryPoint:
    """[project.scripts] in pyproject.toml uses the new CLI entry point."""

    def _read_entry_point(self) -> str:
        """Return the rfdetr console_scripts value from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        content = (root / "pyproject.toml").read_text()
        # Match rfdetr = "..." under [project.scripts]
        m = re.search(r"\[project\.scripts\].*?rfdetr\s*=\s*\"([^\"]+)\"", content, re.DOTALL)
        assert m, "rfdetr entry not found in [project.scripts]"
        return m.group(1)

    def test_entry_point_value(self):
        """rfdetr entry point must be rfdetr.training.cli:main."""
        assert self._read_entry_point() == "rfdetr.training.cli:main"

    def test_entry_point_not_legacy(self):
        """Entry point must no longer reference rfdetr.cli.main:trainer."""
        assert self._read_entry_point() != "rfdetr.cli.main:trainer"


# ---------------------------------------------------------------------------
# Legacy trainer() deprecation warning
# ---------------------------------------------------------------------------


class TestLegacyTrainerDeprecation:
    """rfdetr.cli.main:trainer emits a DeprecationWarning on every call."""

    def _call_trainer_with_coco_dir(self):
        """Call trainer() with --coco_dir /tmp mocked to avoid real training."""
        from rfdetr.cli.main import trainer

        with (
            mock.patch(
                "sys.argv",
                ["rfdetr", "--coco_dir", "/tmp"],
            ),
            mock.patch("rfdetr.cli.main.train_from_coco_dir"),
        ):
            trainer()

    def test_deprecation_warning_emitted(self):
        """trainer() emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_trainer_with_coco_dir()

        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warns, "trainer() must emit a DeprecationWarning"

    def test_deprecation_message_mentions_new_cli(self):
        """DeprecationWarning message references rfdetr.training.cli:main."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_trainer_with_coco_dir()

        dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("rfdetr.training.cli" in m for m in dep_msgs), (
            f"Expected 'rfdetr.training.cli' in deprecation message; got: {dep_msgs}"
        )

    def test_deprecation_message_mentions_version(self):
        """DeprecationWarning message references the deprecation version (1.5.1)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_trainer_with_coco_dir()

        dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("1.5.1" in m for m in dep_msgs), f"Expected '1.5.1' in deprecation message; got: {dep_msgs}"

    def test_trainer_still_functional(self):
        """trainer() still calls train_from_coco_dir when --coco_dir is given."""
        from rfdetr.cli.main import trainer

        with (
            mock.patch("sys.argv", ["rfdetr", "--coco_dir", "/tmp"]),
            mock.patch("rfdetr.cli.main.train_from_coco_dir") as mock_train,
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            trainer()

        mock_train.assert_called_once_with("/tmp")
