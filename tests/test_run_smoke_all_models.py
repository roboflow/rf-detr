# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the all-models smoke runner."""

import builtins
import importlib.metadata
import runpy
from pathlib import Path

import pytest

import tests.run_smoke_all_models as smoke


class TestPlusImports:
    """Check that installed Plus models cannot disappear from the smoke run."""

    def test_missing_plus_is_optional(self, monkeypatch):
        """Continue without Plus models when the distribution is absent."""
        original_import = builtins.__import__

        def fail_plus_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rfdetr" and {"RFDETRXLarge", "RFDETRAtto"}.intersection(fromlist or ()):
                raise ImportError("Plus unavailable")
            return original_import(name, globals, locals, fromlist, level)

        def missing_version(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(builtins, "__import__", fail_plus_import)
        monkeypatch.setattr(importlib.metadata, "version", missing_version)
        result = runpy.run_path(str(Path(smoke.__file__)), run_name="smoke_missing_plus_test")
        assert result["RFDETRXLarge"] is None
        assert result["RFDETRAtto"] is None

    def test_installed_plus_import_error_is_fatal(self, monkeypatch):
        """Propagate an import failure when the Plus distribution is installed."""
        original_import = builtins.__import__

        def fail_legacy_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rfdetr" and "RFDETRXLarge" in (fromlist or ()):
                raise ImportError("broken Plus import")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fail_legacy_import)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.1.0")
        with pytest.raises(ImportError, match="broken Plus import"):
            runpy.run_path(str(Path(smoke.__file__)), run_name="smoke_import_test")


class TestPredictionSmoke:
    """Check that every PE size exercises its prediction path."""

    @pytest.mark.parametrize("name", ["RFDETRAtto", "RFDETRFemto", "RFDETRPico"])
    def test_pe_model_predicts_at_default_resolution(self, name, mocker):
        """Run prediction once for each PE model at its default resolution."""
        instance = mocker.Mock()
        model_class = mocker.Mock(size=name.removeprefix("RFDETR").lower(), return_value=instance)
        mocker.patch.object(smoke, name, model_class)
        mocker.patch.object(smoke, "MODELS_TO_TEST", [model_class])
        mocker.patch.object(smoke, "_test_from_checkpoint")
        with pytest.raises(SystemExit) as exit_info:
            smoke.main()
        assert exit_info.value.code == 0
        instance.predict.assert_called_once()
