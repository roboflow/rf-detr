# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.platform._IS_RFDETR_PLUS_AVAILABLE availability check.

Regression tests for the bug where `importlib.util.find_spec("rfdetr_plus.models")`
returns None when `rfdetr_plus` exposes its `models` namespace via __init__.py
(i.e. does not have a standalone models.py file), causing _IS_RFDETR_PLUS_AVAILABLE
to be False even when `rfdetr_plus` is importable.
"""

import importlib
import importlib.util
import sys
import types
import warnings
from unittest.mock import patch


class TestIsRfdetrPlusAvailable:
    """Verify _IS_RFDETR_PLUS_AVAILABLE reflects actual rfdetr_plus importability."""

    def _reload_platform(self) -> bool:
        """Reload rfdetr.platform and return the current flag value.

        Returns:
            The value of _IS_RFDETR_PLUS_AVAILABLE after reloading.
        """
        sys.modules.pop("rfdetr.platform", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            platform_mod = importlib.import_module("rfdetr.platform")
        return platform_mod._IS_RFDETR_PLUS_AVAILABLE

    def test_flag_false_when_rfdetr_plus_absent(self) -> None:
        """_IS_RFDETR_PLUS_AVAILABLE must be False when rfdetr_plus is not installed.

        Uses a targeted find_spec stub so that only the "rfdetr_plus" lookup returns None — avoids the None-sentinel
        sys.modules trick whose behaviour under find_spec is an implementation detail.
        """
        _real_find_spec = importlib.util.find_spec

        def _stub(name: str, *args, **kwargs):
            if name == "rfdetr_plus":
                return None
            return _real_find_spec(name, *args, **kwargs)

        with patch("importlib.util.find_spec", side_effect=_stub):
            available = self._reload_platform()

        assert available is False

    def test_flag_true_when_rfdetr_plus_present_as_namespace_package(self) -> None:
        """_IS_RFDETR_PLUS_AVAILABLE must be True when rfdetr_plus is importable.

        Regression test: when rfdetr_plus exposes `models` via its __init__ (not
        as a file rfdetr_plus/models.py), find_spec("rfdetr_plus.models") returns
        None while find_spec("rfdetr_plus") returns a valid spec.

        This test reproduces the exact failure: a fake rfdetr_plus package is
        registered in sys.modules WITHOUT a "rfdetr_plus.models" submodule entry.
        find_spec("rfdetr_plus") finds the top-level package (True), while
        find_spec("rfdetr_plus.models") cannot locate the submodule (None → False).
        """
        # Build a minimal fake rfdetr_plus package with a proper package spec
        # (is_package=True ensures submodule_search_locations is set, which is
        # required for find_spec to recognise it as a package).
        fake_pkg = types.ModuleType("rfdetr_plus")
        fake_pkg.__path__ = []  # type: ignore[assignment]
        fake_pkg.__package__ = "rfdetr_plus"
        fake_pkg_spec = importlib.util.spec_from_loader("rfdetr_plus", loader=None, is_package=True)
        if fake_pkg_spec is not None:
            fake_pkg_spec.submodule_search_locations = fake_pkg.__path__  # type: ignore[assignment]
        fake_pkg.__spec__ = fake_pkg_spec  # type: ignore[assignment]

        # Verify the test precondition: find_spec("rfdetr_plus.models") must return
        # None for the fake package (no standalone models.py), while
        # find_spec("rfdetr_plus") must return the spec (package is present).
        # Only patch the rfdetr_plus key — clear=False leaves the rest of
        # sys.modules intact, keeping the test lightweight.
        with patch.dict(sys.modules, {"rfdetr_plus": fake_pkg}):  # type: ignore[dict-item]
            spec_top = importlib.util.find_spec("rfdetr_plus")
            spec_sub = importlib.util.find_spec("rfdetr_plus.models")

        assert spec_top is not None, "precondition: find_spec('rfdetr_plus') must find the fake package"
        assert spec_sub is None, "precondition: find_spec('rfdetr_plus.models') must return None (no models.py)"

        # Now verify that the production flag is True (checks top-level package,
        # not the missing submodule).
        with patch.dict(sys.modules, {"rfdetr_plus": fake_pkg}):  # type: ignore[dict-item]
            available = self._reload_platform()

        assert available is True, (
            "_IS_RFDETR_PLUS_AVAILABLE should be True when rfdetr_plus is importable, "
            "even if rfdetr_plus.models is not a separate submodule file."
        )
