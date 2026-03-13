# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CLI smoke tests and YAML roundtrip tests — PTL Ch4/T7.

Smoke tests run RFDETRCli in-process with args=['--help'] / ['fit', '--help'] /
['validate', '--help'] and assert SystemExit(0) — no subprocess needed.

YAML roundtrip tests load each example config with yaml.safe_load, import the
class_path, construct the config object with the YAML init_args, and verify
every specified field survived the round-trip.
"""

import importlib
import pathlib

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
    """Run RFDETRCli in-process with the given args; return the SystemExit code."""
    from rfdetr.training.cli import RFDETRCli
    from rfdetr.training.datamodule import RFDETRDataModule
    from rfdetr.training.module import RFDETRModule

    with pytest.raises(SystemExit) as exc_info:
        RFDETRCli(RFDETRModule, RFDETRDataModule, args=list(args))
    return exc_info.value.code


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())


def _instantiate(class_path: str, init_args: dict) -> object:
    """Import class_path and construct an instance with init_args."""
    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**init_args)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """rfdetr --help and subcommand --help must exit 0."""

    def test_top_level_help(self):
        """rfdetr --help exits with code 0."""
        assert _run_cli("--help") == 0

    def test_fit_help(self):
        """rfdetr fit --help exits with code 0."""
        assert _run_cli("fit", "--help") == 0

    def test_validate_help(self):
        """rfdetr validate --help exits with code 0."""
        assert _run_cli("validate", "--help") == 0

    def test_fit_help_exposes_model_config(self):
        """rfdetr fit --help output lists model.model_config arguments."""
        import io
        import sys

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _run_cli("fit", "--help")
        finally:
            sys.stdout = old_stdout
        assert "model_config" in buf.getvalue()

    def test_fit_help_exposes_train_config(self):
        """rfdetr fit --help output lists model.train_config arguments."""
        import io
        import sys

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _run_cli("fit", "--help")
        finally:
            sys.stdout = old_stdout
        assert "train_config" in buf.getvalue()


# ---------------------------------------------------------------------------
# YAML roundtrip — model_config
# ---------------------------------------------------------------------------


class TestModelConfigRoundtrip:
    """model_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_config_fields_survive_roundtrip(self, name):
        """Every field in model_config.init_args is preserved after instantiation."""
        data = _load(name)
        mc_section = data["model"]["model_config"]
        class_path = mc_section["class_path"]
        init_args = mc_section.get("init_args", {})

        mc = _instantiate(class_path, init_args)
        for field, value in init_args.items():
            assert getattr(mc, field) == value, (
                f"{name}.yaml: model_config.{field} expected {value!r}, got {getattr(mc, field)!r}"
            )


# ---------------------------------------------------------------------------
# YAML roundtrip — train_config
# ---------------------------------------------------------------------------


class TestTrainConfigRoundtrip:
    """train_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_train_config_fields_survive_roundtrip(self, name, tmp_path):
        """Every field in train_config.init_args is preserved after instantiation.

        dataset_dir is rewritten to tmp_path so path expansion doesn't fail on
        the placeholder /data/coco value.  TrainConfig.expand_paths() converts
        relative paths containing separators to absolute, so both sides are
        normalised with os.path.abspath before comparison.
        """
        import os

        data = _load(name)
        tc_section = data["model"]["train_config"]
        class_path = tc_section["class_path"]
        init_args = dict(tc_section.get("init_args", {}))

        # Substitute placeholder dataset_dir with a real tmp_path.
        init_args["dataset_dir"] = str(tmp_path)

        tc = _instantiate(class_path, init_args)
        for field, value in init_args.items():
            actual = getattr(tc, field)
            # TrainConfig.expand_paths() resolves relative path strings to
            # absolute, so normalise both sides for string path fields.
            if isinstance(value, str) and (os.sep in value or "/" in value):
                value = os.path.abspath(value)
            assert actual == value, f"{name}.yaml: train_config.{field} expected {value!r}, got {actual!r}"
