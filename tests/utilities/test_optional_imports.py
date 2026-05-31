# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for optional-dependency import helpers."""

import sys

import pytest

from rfdetr.utilities.optional_imports import import_supervision


class TestImportSupervision:
    """Tests for ``import_supervision()``."""

    def test_returns_module_when_installed(self) -> None:
        """When ``supervision`` is installed, the helper returns the module."""
        sv = pytest.importorskip("supervision")
        assert import_supervision() is sv

    def test_raises_with_install_hint_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``supervision`` cannot be imported, a friendly ImportError with an install hint is raised."""
        # Setting an entry to None makes the import machinery raise ImportError for that name.
        monkeypatch.setitem(sys.modules, "supervision", None)

        with pytest.raises(ImportError, match="pip install supervision"):
            import_supervision()
