# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``rfdetr export`` CLI subcommand and dispatcher.

Covers:
* ``export_main()`` forwards flags (including the TFLite quantization /
  calibration flags) to ``RFDETR.export``.
* The ``rfdetr`` dispatcher routes ``export`` to ``rfdetr.cli.export``
  without importing the training stack.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


class TestExportMainForwarding:
    """``export_main`` loads the checkpoint and forwards flags to ``export``."""

    @staticmethod
    def _run(**overrides) -> dict:
        """Call export_main with mocked from_checkpoint; return export kwargs."""
        captured: dict = {}
        fake_model = mock.MagicMock()
        fake_model.export.side_effect = lambda **kw: captured.update(kw)

        from rfdetr.cli.export import export_main

        with mock.patch("rfdetr.from_checkpoint", return_value=fake_model) as from_ckpt:
            export_main("ckpt.pth", **overrides)
        captured["__from_checkpoint_arg__"] = from_ckpt.call_args.args[0]
        return captured

    def test_checkpoint_is_loaded_via_from_checkpoint(self) -> None:
        """The positional checkpoint path is passed to from_checkpoint."""
        captured = self._run()
        assert captured["__from_checkpoint_arg__"] == "ckpt.pth"

    def test_tflite_quantization_flags_forwarded(self) -> None:
        """TFLite quantization/calibration flags reach RFDETR.export."""
        captured = self._run(
            format="tflite",
            quantization="int8",
            calibration_data="images/",
            max_images=50,
        )
        assert captured["format"] == "tflite"
        assert captured["quantization"] == "int8"
        assert captured["calibration_data"] == "images/"
        assert captured["max_images"] == 50

    def test_onnx_is_default_format(self) -> None:
        """Format defaults to onnx when not specified."""
        captured = self._run()
        assert captured["format"] == "onnx"


class TestDispatcher:
    """The ``rfdetr`` console entry point routes subcommands correctly."""

    def test_export_subcommand_routes_to_export_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``rfdetr export ...`` invokes rfdetr.cli.export.main and strips the subcommand."""
        import rfdetr.cli as cli

        seen_argv: dict = {}
        monkeypatch.setattr(
            "rfdetr.cli.export.main",
            lambda: seen_argv.update(argv=list(sys.argv)),
        )
        monkeypatch.setattr(sys, "argv", ["rfdetr", "export", "--checkpoint", "x.pth"])

        cli.main()

        # The "export" token is stripped so jsonargparse sees only the flags.
        assert seen_argv["argv"] == ["rfdetr", "--checkpoint", "x.pth"]

    def test_export_routing_does_not_import_pytorch_lightning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Routing to export must not import the training stack (pytorch_lightning)."""
        import rfdetr.cli as cli

        monkeypatch.setattr("rfdetr.cli.export.main", lambda: None)
        monkeypatch.setattr(sys, "argv", ["rfdetr", "export"])
        # Make ``rfdetr.cli.train`` un-importable so an accidental eager import would raise.
        monkeypatch.setitem(sys.modules, "rfdetr.cli.train", None)

        cli.main()  # must not raise

    def test_root_level_option_before_command_routes_to_train(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LightningCLI root-option ordering (e.g. `rfdetr -c cfg fit`) delegates to the train backend with argv left
        intact."""
        import rfdetr.cli as cli

        seen_argv: dict = {}
        monkeypatch.setattr("rfdetr.cli.train.main", lambda: seen_argv.update(argv=list(sys.argv)))
        monkeypatch.setattr(sys, "argv", ["rfdetr", "-c", "cfg.yaml", "fit"])

        cli.main()

        # LightningCLI parses root options + command itself, so argv is unchanged.
        assert seen_argv["argv"] == ["rfdetr", "-c", "cfg.yaml", "fit"]

    def test_top_level_help_is_root_owned(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`rfdetr --help` is rendered by the RF-DETR root: it lists every command and does not stitch in LightningCLI's
        root usage."""
        import rfdetr.cli as cli

        monkeypatch.setattr(sys, "argv", ["rfdetr", "--help"])

        cli.main()

        out = capsys.readouterr().out
        for command in ("fit", "validate", "test", "predict", "export"):
            assert command in out, f"{command!r} missing from root help"
        # LightningCLI's stitched root usage must not leak into the top-level help.
        assert "{fit,validate,test,predict}" not in out

    def test_unknown_command_exits_2(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """An unrecognized command prints an error to stderr and exits with code 2."""
        import rfdetr.cli as cli

        monkeypatch.setattr(sys, "argv", ["rfdetr", "bogus"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 2
        assert "invalid command" in capsys.readouterr().err
