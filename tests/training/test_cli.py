# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETRCli — PTL Ch4/T4.

Verifies that the CLI module is correctly structured: importable, subclasses LightningCLI, overrides
add_arguments_to_parser, and exposes a callable main() entry point.  CLI integration / smoke tests (--help subprocess,
YAML roundtrip) live in T4-7.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from jsonargparse import Namespace
from pytorch_lightning import LightningDataModule, LightningModule

from rfdetr.config import ModelConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.training.cli import RFDETRCli
from rfdetr.training.param_groups import _PARAM_GROUP_SETTINGS, _resolve_resumed_param_group_settings


class ConfigModel(LightningModule):
    """Exercise actual CLI config instantiation without constructing the detection network."""

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        """Store the configs the parser instantiated.

        Examples:
            >>> ConfigModel(RFDETRNanoConfig(), TrainConfig(dataset_dir="data")).train_config.batch_size
            4
        """
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config


class ConfigData(LightningDataModule):
    """Receive the CLI's linked training config."""

    def __init__(self, model_config: dict[str, Any], train_config: dict[str, Any]) -> None:
        """Store the linked instantiated configs.

        Examples:
            >>> ConfigData(RFDETRNanoConfig().model_dump(), {"dataset_dir": "data"}).train_config.batch_size
            4
        """
        super().__init__()
        self.model_config = ModelConfig(**model_config)
        self.train_config = TrainConfig(**train_config)


class ParseOnlyCli(RFDETRCli):
    """Run the real subcommand parsing and checkpoint merge without starting training."""

    def _run_subcommand(self, subcommand: str) -> None:
        """Stop after the instantiated configs are available.

        Examples:
            >>> ParseOnlyCli._run_subcommand(None, "fit")
        """


class TestCliResumeIntent:
    """Caller intent survives defaults, config files and Lightning checkpoint hyperparameters."""

    def test_without_subcommand_preserves_explicit_default(self) -> None:
        """Programmatic CLI construction has the same sparse-intent contract."""
        cli = RFDETRCli(
            ConfigModel,
            ConfigData,
            run=False,
            args={
                "model": {
                    "model_config": RFDETRNanoConfig().model_dump(mode="json"),
                    "train_config": {"dataset_dir": "data", "lr": 1e-4},
                }
            },
            save_config_callback=None,
            auto_configure_optimizers=False,
        )
        assert set(_PARAM_GROUP_SETTINGS) & cli.model.train_config.model_fields_set == {"lr"}
        resolved, _, _ = _resolve_resumed_param_group_settings(cli.model.train_config, {"args": {"lr": 0.123}})
        assert resolved.lr == 1e-4

    @pytest.mark.parametrize("setting", _PARAM_GROUP_SETTINGS)
    @pytest.mark.parametrize("input_kind", ["list", "dict", "namespace", "yaml"])
    @pytest.mark.parametrize("explicit", [False, True])
    def test_checkpoint_merge_preserves_optimizer_intent(
        self, tmp_path: Path, setting: str, input_kind: str, explicit: bool
    ) -> None:
        """Check values and explicit membership on actual module/data config instances."""
        checkpoint = tmp_path / "last.ckpt"
        recorded = TrainConfig(dataset_dir="old", **{name: 0.123 for name in _PARAM_GROUP_SETTINGS})
        torch.save(
            {
                "hyper_parameters": {
                    "train_config": {"dataset_dir": "old", **{name: 0.123 for name in _PARAM_GROUP_SETTINGS}}
                }
            },
            checkpoint,
        )
        train: dict[str, Any] = {"dataset_dir": "data", "batch_size": 2}
        if explicit:
            train[setting] = TrainConfig.model_fields[setting].default
        config = {
            "fit": {
                "model": {"model_config": RFDETRNanoConfig().model_dump(mode="json"), "train_config": train},
                "ckpt_path": str(checkpoint),
            }
        }
        args: Any = config
        if input_kind == "namespace":
            args = Namespace(**config)
        elif input_kind == "list":
            args = [
                "fit",
                "--model.model_config=" + json.dumps(config["fit"]["model"]["model_config"]),
                "--model.train_config.dataset_dir=data",
                "--model.train_config.batch_size=2",
                f"--ckpt_path={checkpoint}",
            ]
            if explicit:
                args.append(f"--model.train_config.{setting}={train[setting]}")
        elif input_kind == "yaml":
            path = tmp_path / "config.yaml"
            path.write_text(yaml.safe_dump(config["fit"]))
            args = ["fit", f"--config={path}"]
        cli = ParseOnlyCli(
            ConfigModel, ConfigData, args=args, save_config_callback=None, auto_configure_optimizers=False
        )
        for component in (cli.model, cli.datamodule):
            parsed = component.train_config
            assert set(_PARAM_GROUP_SETTINGS) & parsed.model_fields_set == ({setting} if explicit else set())
            assert "batch_size" in parsed.model_fields_set
            resolved, _, _ = _resolve_resumed_param_group_settings(parsed, {"args": recorded.model_dump()})
            assert getattr(resolved, setting) == (train[setting] if explicit else 0.123)


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
        """Main() can be imported from rfdetr.training.cli."""
        from rfdetr.training.cli import main  # noqa: F401

    def test_rfdetr_cli_is_lightning_cli_subclass(self):
        """RFDETRCli must subclass pytorch_lightning LightningCLI."""
        from pytorch_lightning.cli import LightningCLI

        from rfdetr.training.cli import RFDETRCli

        assert issubclass(RFDETRCli, LightningCLI)

    def test_main_is_callable(self):
        """Main must be a callable (function, not e.g. a string)."""
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
