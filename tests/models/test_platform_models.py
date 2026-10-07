# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the optional plus-model import block in ``rfdetr.platform.models``."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

import rfdetr.platform.models as platform_models_module
from rfdetr.platform import _PLUS_EXPORTS

_XLARGE_SYMBOLS = frozenset({"RFDETR2XLarge", "RFDETRXLarge"})
_PE_SYMBOLS = frozenset({"RFDETRAtto", "RFDETRFemto", "RFDETRPico"})


def _load_platform_models(
    monkeypatch: pytest.MonkeyPatch,
    plus_symbols: frozenset[str],
    pe_error: ImportError | None = None,
) -> types.ModuleType:
    """Execute a fresh copy of ``rfdetr.platform.models`` against a fake ``rfdetr_plus``.

    The shipped module decides what to export while it is imported, so each case needs its own copy; the fake
    package stands in for whichever ``rfdetr_plus`` release the case describes.

    Args:
        monkeypatch: Patches ``sys.modules`` and the plus-availability flag for the duration of the test.
        plus_symbols: Model names the fake ``rfdetr_plus.models`` provides.
        pe_error: When given, the fake ``rfdetr_plus.models`` raises it on any PE-model attribute lookup, like a
            plus release whose PE import fails on a broken dependency.

    Returns:
        The freshly executed module, not registered in ``sys.modules``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as patch:
        ...     module = _load_platform_models(patch, frozenset({"RFDETRXLarge", "RFDETR2XLarge"}))
        ...     sorted(module.__all__)
        ['RFDETR2XLarge', 'RFDETRXLarge']
    """
    fake_models = types.ModuleType("rfdetr_plus.models")
    for symbol in plus_symbols:
        setattr(fake_models, symbol, type(symbol, (), {}))
    if pe_error is not None:

        def _raise_for_pe(name: str) -> object:
            if name in _PE_SYMBOLS:
                raise pe_error
            raise AttributeError(name)

        fake_models.__getattr__ = _raise_for_pe  # type: ignore[attr-defined]
    fake_package = types.ModuleType("rfdetr_plus")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rfdetr_plus", fake_package)
    monkeypatch.setitem(sys.modules, "rfdetr_plus.models", fake_models)
    monkeypatch.setattr("rfdetr.platform._IS_RFDETR_PLUS_AVAILABLE", True)

    spec = importlib.util.spec_from_file_location(
        "_rfdetr_platform_models_under_test", Path(platform_models_module.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPlusImportBlock:
    """What ``rfdetr.platform.models`` exports for each rfdetr_plus release shape."""

    def test_current_rfdetr_plus_exports_every_plus_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A current rfdetr_plus exposes every name in ``_PLUS_EXPORTS``."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS | _PE_SYMBOLS)

        assert set(module.__all__) == _PLUS_EXPORTS

    def test_rfdetr_plus_predating_pe_models_keeps_xlarge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A release that ships XLarge but not the PE models still exports XLarge.

        The ``rfdetr[plus]`` floor admits such releases, so the PE import failing must not hide the XLarge models.
        """
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS)

        assert set(module.__all__) == _XLARGE_SYMBOLS
        assert module.RFDETRXLarge.__name__ == "RFDETRXLarge"

    def test_failed_pe_import_keeps_xlarge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A PE import that fails for another reason also leaves the XLarge exports intact."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS, pe_error=ImportError("broken dependency"))

        assert set(module.__all__) == _XLARGE_SYMBOLS


class TestMissingPeModelAccess:
    """Accessing a PE model the installed rfdetr_plus lacks."""

    @pytest.mark.parametrize("symbol", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_old_rfdetr_plus_raises_upgrade_hint(self, monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
        """Asking an old rfdetr_plus for a PE model raises the upgrade hint, not a bare AttributeError."""
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS)

        with pytest.raises(ImportError, match="predates it"):
            getattr(module, symbol)

    def test_upgrade_hint_chains_the_failed_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The upgrade hint carries the swallowed import error as its cause.

        A PE import that fails on a broken dependency reads like an outdated install; the chained cause is what tells
        the two apart in the traceback.
        """
        cause = ImportError("broken dependency")
        module = _load_platform_models(monkeypatch, _XLARGE_SYMBOLS, pe_error=cause)

        with pytest.raises(ImportError, match="predates it") as raised:
            module.RFDETRAtto

        assert raised.value.__cause__ is cause
