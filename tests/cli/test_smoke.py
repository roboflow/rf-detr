# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLI smoke tests and YAML roundtrip tests — PTL Ch4/T7.

Smoke tests run RFDETRCli in-process with args=['--help'] / ['fit', '--help'] / ['validate', '--help'] and assert
SystemExit(0) — no subprocess needed.

YAML roundtrip tests load each example config with yaml.safe_load, import the class_path, construct the config object
with the YAML init_args, and verify every specified field survived the round-trip.
"""

import importlib
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

CONFIGS_DIR = pathlib.Path(__file__).parent.parent.parent / "configs"

ALL_CONFIGS = [
    "rfdetr_nano",
    "rfdetr_small",
    "rfdetr_medium",
    "rfdetr_base",
    "rfdetr_large",
    "rfdetr_seg_nano",
    "rfdetr_seg_small",
    "rfdetr_seg_medium",
    "rfdetr_seg_large",
    "rfdetr_seg_xlarge",
    "rfdetr_seg_2xlarge",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> int:
    """Run RFDETRCli in-process with the given args; return the SystemExit code.

    The doctest is skipped because the CLI prints its help text to stdout.

    >>> _run_cli("--help")  # doctest: +SKIP
    0
    """
    from rfdetr.training.cli import RFDETRCli
    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule

    with pytest.raises(SystemExit) as exc_info:
        RFDETRCli(RFDETRModelModule, RFDETRDataModule, args=list(args))
    return exc_info.value.code


def _load(name: str) -> dict:
    """Load a config YAML file by stem name.

    >>> _load("rfdetr_small")["model"]["model_config"]["class_path"]
    'rfdetr.config.RFDETRSmallConfig'
    """
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())


def _instantiate(class_path: str, init_args: dict) -> object:
    """Import class_path and construct an instance with init_args.

    >>> config = _instantiate("rfdetr.config.RFDETRSmallConfig", {})
    >>> type(config).__name__
    'RFDETRSmallConfig'
    """
    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**init_args)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestCLIEntrypoint:
    """Module entrypoint and CLI import tests."""

    @pytest.mark.flaky(reruns=3, condition=sys.platform == "darwin")
    def test_module_help_exits_zero(self) -> None:
        """`python -m rfdetr --help` exits 0 and mentions rfdetr."""
        result = subprocess.run([sys.executable, "-m", "rfdetr", "--help"], capture_output=True, text=True, timeout=30)

        assert result.returncode == 0
        assert "rfdetr" in result.stdout.lower() or "rfdetr" in result.stderr.lower()

    def test_main_is_callable(self) -> None:
        """The `rfdetr.cli` module is importable and exposes a callable `main`."""
        mod = importlib.import_module("rfdetr.cli")

        assert callable(getattr(mod, "main", None))


class TestCLIHelp:
    """`rfdetr --help` and subcommand `--help` exit 0."""

    @pytest.mark.parametrize("command", ["--help", "fit --help", "validate --help"])
    def test_help_exits_zero(self, command: str) -> None:
        """Each help invocation exits with code 0."""
        assert _run_cli(*command.split()) == 0

    @pytest.mark.parametrize("section", ["model_config", "train_config"])
    def test_fit_help_lists_section(self, section: str, capsys: pytest.CaptureFixture[str]) -> None:
        """`rfdetr fit --help` output lists the model's `model_config` and `train_config` arguments."""
        _run_cli("fit", "--help")

        assert section in capsys.readouterr().out


class TestShippedConfigsParse:
    """Every shipped config parses through the CLI, as ``rfdetr fit --config configs/<name>.yaml`` documents."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_fit_parses(self, name: str, capsys: pytest.CaptureFixture[str]) -> None:
        """``fit --print_config`` exits 0 and keeps the config's own ``ModelConfig`` subclass."""
        exit_code = _run_cli("fit", "--config", str(CONFIGS_DIR / f"{name}.yaml"), "--print_config")

        printed = capsys.readouterr()
        assert exit_code == 0, printed.err[-600:]
        assert f"class_path: {_load(name)['model']['model_config']['class_path']}" in printed.out


# ---------------------------------------------------------------------------
# YAML roundtrip — model_config
# ---------------------------------------------------------------------------


class TestModelConfigRoundtrip:
    """model_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_fields_survive_roundtrip(self, name: str) -> None:
        """Every field in model_config.init_args is preserved after instantiation."""
        mc_section = _load(name)["model"]["model_config"]
        init_args = mc_section.get("init_args", {})

        mc = _instantiate(mc_section["class_path"], init_args)

        assert {field: getattr(mc, field) for field in init_args} == init_args, f"{name}.yaml: model_config mismatch"


# ---------------------------------------------------------------------------
# YAML roundtrip — train_config
# ---------------------------------------------------------------------------


class TestTrainConfigRoundtrip:
    """train_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_fields_survive_roundtrip(self, name: str, tmp_path: pathlib.Path) -> None:
        """Every field in train_config.init_args is preserved after instantiation.

        dataset_dir is rewritten to tmp_path so path expansion doesn't fail on the placeholder /data/coco value.
        TrainConfig.expand_paths() converts relative path strings to absolute, so expected path-like strings are
        normalised with os.path.abspath before comparison.
        """
        tc_section = _load(name)["model"]["train_config"]
        init_args = {**tc_section.get("init_args", {}), "dataset_dir": str(tmp_path)}
        expected = {
            field: os.path.abspath(value) if isinstance(value, str) and (os.sep in value or "/" in value) else value
            for field, value in init_args.items()
        }

        tc = _instantiate(tc_section["class_path"], init_args)

        assert {field: getattr(tc, field) for field in init_args} == expected, f"{name}.yaml: train_config mismatch"
