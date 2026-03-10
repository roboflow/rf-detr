# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for RFDETRCli — PTL Ch4/T4.

Verifies that the CLI module is correctly structured: importable, subclasses
LightningCLI, overrides add_arguments_to_parser, and exposes a callable
main() entry point.  CLI integration / smoke tests (--help subprocess, YAML
roundtrip) live in T4-7.
"""

import pytest

# ---------------------------------------------------------------------------
# Structure and importability
# ---------------------------------------------------------------------------


class TestRFDETRCliStructure:
    """RFDETRCli is correctly structured and importable."""

    def test_cli_module_importable(self):
        """rfdetr.training.cli imports without error."""
        import rfdetr.training.cli  # noqa: F401

    def test_rfdetr_cli_importable(self):
        """RFDETRCli can be imported from rfdetr.training.cli."""
        from rfdetr.training.cli import RFDETRCli  # noqa: F401

    def test_main_importable(self):
        """main() can be imported from rfdetr.training.cli."""
        from rfdetr.training.cli import main  # noqa: F401

    def test_rfdetr_cli_is_lightning_cli_subclass(self):
        """RFDETRCli must subclass pytorch_lightning LightningCLI."""
        from pytorch_lightning.cli import LightningCLI

        from rfdetr.training.cli import RFDETRCli

        assert issubclass(RFDETRCli, LightningCLI)

    def test_main_is_callable(self):
        """main must be a callable (function, not e.g. a string)."""
        from rfdetr.training.cli import main

        assert callable(main)

    def test_add_arguments_to_parser_is_overridden(self):
        """RFDETRCli overrides add_arguments_to_parser from LightningCLI."""
        from pytorch_lightning.cli import LightningCLI

        from rfdetr.training.cli import RFDETRCli

        assert RFDETRCli.add_arguments_to_parser is not LightningCLI.add_arguments_to_parser

    def test_exported_from_lit_package(self):
        """RFDETRCli is exported from rfdetr.training (appears in __all__)."""
        import rfdetr.training as lit

        assert hasattr(lit, "RFDETRCli")
        assert "RFDETRCli" in lit.__all__


# ---------------------------------------------------------------------------
# Argument linking
# ---------------------------------------------------------------------------


class TestRFDETRCliArgumentLinking:
    """add_arguments_to_parser registers the expected argument links."""

    def _collect_links(self):
        """Instantiate a minimal parser and collect registered link sources."""
        import unittest.mock as mock

        from rfdetr.training.cli import RFDETRCli

        captured = []

        class _FakeParser:
            def link_arguments(self, source, target, **kwargs):
                captured.append({"source": source, "target": target, **kwargs})

            # LightningArgumentParser methods that may be called during setup
            def __getattr__(self, name):
                return mock.MagicMock()

        cli = RFDETRCli.__new__(RFDETRCli)
        cli.add_arguments_to_parser(_FakeParser())
        return captured

    def test_model_config_link_registered(self):
        """model.model_config is linked to data.model_config."""
        links = self._collect_links()
        sources = [lnk["source"] for lnk in links]
        assert "model.model_config" in sources

    def test_train_config_link_registered(self):
        """model.train_config is linked to data.train_config."""
        links = self._collect_links()
        sources = [lnk["source"] for lnk in links]
        assert "model.train_config" in sources

    @pytest.mark.parametrize(
        "source, expected_target",
        [
            pytest.param("model.model_config", "data.model_config", id="model_config"),
            pytest.param("model.train_config", "data.train_config", id="train_config"),
        ],
    )
    def test_link_target(self, source, expected_target):
        """Each link points to the correct data.* target."""
        links = self._collect_links()
        match = next((lnk for lnk in links if lnk["source"] == source), None)
        assert match is not None, f"No link registered for source {source!r}"
        assert match["target"] == expected_target
