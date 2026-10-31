# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for headless packaging and headless environment behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from rfdetr._prediction.streams import _import_cv2
from rfdetr.visualize.training import _ensure_headless_backend

_PYPROJECT_PATH = Path(__file__).resolve().parents[2].joinpath("pyproject.toml")


def _load_pyproject() -> dict:
    with open(_PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


class TestHeadlessPackaging:
    """Test packaging configuration for headless environments."""

    def test_pyproject_declares_headless_extra(self) -> None:
        """The headless extra must be declared under optional-dependencies."""
        pyproject = _load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "headless" in extras, "Missing 'headless' extra in pyproject.toml"

        headless_deps = extras["headless"]
        assert any("opencv-python-headless" in dep for dep in headless_deps), (
            "The 'headless' extra must include opencv-python-headless"
        )
        assert not any(dep.startswith("opencv-python") and "headless" not in dep for dep in headless_deps), (
            "The 'headless' extra must not depend on GUI opencv-python"
        )

    def test_core_dependencies_do_not_require_gui_opencv(self) -> None:
        """Core dependencies must not require GUI opencv-python."""
        pyproject = _load_pyproject()
        core_deps = pyproject.get("project", {}).get("dependencies", [])
        for dep in core_deps:
            assert not dep.startswith("opencv-python") or "headless" in dep, (
                f"Core dependency {dep!r} requires GUI opencv-python; use optional extra instead"
            )


class TestHeadlessRuntime:
    """Test runtime helpers under headless conditions."""

    def test_import_cv2_error_mentions_headless_extra(self) -> None:
        """When cv2 is missing, _import_cv2 points to both stream and headless extras."""
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError) as exc_info:
                _import_cv2("Video")

            message = str(exc_info.value)
            assert "rfdetr[headless]" in message, f"Expected 'rfdetr[headless]' in error message: {message}"
            assert "rfdetr[stream]" in message, f"Expected 'rfdetr[stream]' in error message: {message}"

    def test_ensure_headless_backend_sets_agg_when_display_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Matplotlib is switched to Agg when DISPLAY is absent and backend is not interactive."""
        monkeypatch.delenv("DISPLAY", raising=False)

        mock_matplotlib = MagicMock()
        mock_matplotlib.get_backend.return_value = "TkAgg"
        mock_matplotlib.is_interactive.return_value = False

        _ensure_headless_backend(mock_matplotlib)
        mock_matplotlib.use.assert_called_once_with("Agg")

    def test_ensure_headless_backend_skips_when_display_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Matplotlib preserves its backend when DISPLAY is present."""
        monkeypatch.setenv("DISPLAY", ":0")

        mock_matplotlib = MagicMock()
        mock_matplotlib.get_backend.return_value = "TkAgg"
        mock_matplotlib.is_interactive.return_value = False

        _ensure_headless_backend(mock_matplotlib)
        mock_matplotlib.use.assert_not_called()

    def test_ensure_headless_backend_skips_when_already_agg(self) -> None:
        """Matplotlib preserves its backend when already set to Agg."""
        mock_matplotlib = MagicMock()
        mock_matplotlib.get_backend.return_value = "agg"

        _ensure_headless_backend(mock_matplotlib)
        mock_matplotlib.use.assert_not_called()
