# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for YAML config files in configs/ — PTL Ch4/T6.

Verifies that every example YAML config file:
  - exists on disk,
  - parses as valid YAML,
  - contains a ``model`` section with ``model_config`` and ``train_config``,
  - references the expected model class_path, and
  - segmentation configs use SegmentationTrainConfig.

Tests for the CLI entry point configuration.

CLI smoke tests and YAML roundtrip tests — PTL Ch4/T7.

Smoke tests run RFDETRCli in-process with args=['--help'] / ['fit', '--help'] / ['validate', '--help'] and assert
SystemExit(0) — no subprocess needed.

YAML roundtrip tests load each example config with yaml.safe_load, import the class_path, construct the config object
with the YAML init_args, and verify every specified field survived the round-trip.

Exercise new and compatibility WebDataset packing commands as subprocesses.

Exercise the ``python -m rfdetr`` entrypoint shim in ``rfdetr.__main__``.

Regression tests for the all-models smoke runner.
"""

import builtins
import importlib
import importlib.metadata
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

import tests.run_smoke_all_models as smoke
from rfdetr.datasets.webdataset.index import read_shard_index

CONFIGS_DIR = pathlib.Path(__file__).parent.parent.parent / "configs"


DETECTION_CONFIGS = [
    "rfdetr_nano",
    "rfdetr_small",
    "rfdetr_medium",
    "rfdetr_base",
    "rfdetr_large",
]


SEGMENTATION_CONFIGS = [
    "rfdetr_seg_nano",
    "rfdetr_seg_small",
    "rfdetr_seg_medium",
    "rfdetr_seg_large",
    "rfdetr_seg_xlarge",
    "rfdetr_seg_2xlarge",
]


ALL_CONFIGS = DETECTION_CONFIGS + SEGMENTATION_CONFIGS


# Maps filename stem → expected model_config class_path.
EXPECTED_MODEL_CLASS = {
    "rfdetr_nano": "rfdetr.config.RFDETRNanoConfig",
    "rfdetr_small": "rfdetr.config.RFDETRSmallConfig",
    "rfdetr_medium": "rfdetr.config.RFDETRMediumConfig",
    "rfdetr_base": "rfdetr.config.RFDETRBaseConfig",
    "rfdetr_large": "rfdetr.config.RFDETRLargeConfig",
    "rfdetr_seg_nano": "rfdetr.config.RFDETRSegNanoConfig",
    "rfdetr_seg_small": "rfdetr.config.RFDETRSegSmallConfig",
    "rfdetr_seg_medium": "rfdetr.config.RFDETRSegMediumConfig",
    "rfdetr_seg_large": "rfdetr.config.RFDETRSegLargeConfig",
    "rfdetr_seg_xlarge": "rfdetr.config.RFDETRSegXLargeConfig",
    "rfdetr_seg_2xlarge": "rfdetr.config.RFDETRSeg2XLargeConfig",
}


def _load(name: str) -> dict:
    """Parse a config file by stem name and return its dict.

    >>> _load("rfdetr_small")["model"]["train_config"]["class_path"]
    'rfdetr.config.TrainConfig'
    """
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())


class TestConfigFilesExist:
    """Every expected YAML config file must be present on disk."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_file_exists(self, name: str) -> None:
        """configs/{name}.yaml must exist."""
        assert (CONFIGS_DIR / f"{name}.yaml").exists(), f"Missing config file: {name}.yaml"


class TestConfigFilesValidYAML:
    """Each file must be parseable as YAML and produce a mapping."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_is_valid_yaml(self, name: str) -> None:
        """yaml.safe_load must succeed and return a dict."""
        data = _load(name)
        assert isinstance(data, dict), f"{name}.yaml did not parse to a dict"


class TestConfigStructure:
    """Each YAML must have a model section with model_config and train_config."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_model_section(self, name: str) -> None:
        """Top-level 'model' key must be present."""
        assert "model" in _load(name), f"{name}.yaml missing 'model' section"

    @pytest.mark.parametrize("key", ["model_config", "train_config"])
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_section_has_key(self, name: str, key: str) -> None:
        """The model section must contain both 'model_config' and 'train_config'."""
        assert key in _load(name)["model"], f"{name}.yaml missing model.{key}"


class TestConfigClassPaths:
    """model_config and train_config class_path must match the expected model variant."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_model_config_class_path(self, name: str) -> None:
        """model.model_config.class_path must match the variant."""
        got = _load(name)["model"]["model_config"]["class_path"]
        want = EXPECTED_MODEL_CLASS[name]
        assert got == want, f"{name}.yaml: expected class_path {want!r}, got {got!r}"

    @pytest.mark.parametrize(
        ("name", "want"),
        [(name, "rfdetr.config.SegmentationTrainConfig") for name in SEGMENTATION_CONFIGS]
        + [(name, "rfdetr.config.TrainConfig") for name in DETECTION_CONFIGS],
    )
    def test_train_config_class_path(self, name: str, want: str) -> None:
        """Segmentation configs must use SegmentationTrainConfig; detection configs TrainConfig (not a subclass)."""
        got = _load(name)["model"]["train_config"]["class_path"]
        assert got == want, f"{name}.yaml: train_config must use {want}, got {got!r}"


@pytest.fixture
def entry_point() -> str:
    """Return the rfdetr console_scripts value from pyproject.toml."""
    root = pathlib.Path(__file__).parent.parent.parent
    content = (root / "pyproject.toml").read_text()
    m = re.search(r"\[project\.scripts\].*?rfdetr\s*=\s*\"([^\"]+)\"", content, re.DOTALL)
    assert m, "rfdetr entry not found in [project.scripts]"
    return m.group(1)


class TestEntryPoint:
    """[project.scripts] in pyproject.toml uses the correct CLI entry point."""

    def test_is_cli_main(self, entry_point: str) -> None:
        """The rfdetr console script points at rfdetr.cli:main."""
        assert entry_point == "rfdetr.cli:main"

    def test_is_not_legacy_trainer(self, entry_point: str) -> None:
        """The rfdetr console script no longer references rfdetr.cli.main:trainer."""
        assert entry_point != "rfdetr.cli.main:trainer"


SMOKE_CONFIGS = [
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


def _load_config(name: str) -> dict:
    """Load a config YAML file by stem name.

    >>> _load_config("rfdetr_small")["model"]["model_config"]["class_path"]
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

    @pytest.mark.parametrize("name", SMOKE_CONFIGS)
    def test_fit_parses(self, name: str, capsys: pytest.CaptureFixture[str]) -> None:
        """``fit --print_config`` exits 0 and keeps the config's own ``ModelConfig`` subclass."""
        exit_code = _run_cli("fit", "--config", str(CONFIGS_DIR / f"{name}.yaml"), "--print_config")

        printed = capsys.readouterr()
        assert exit_code == 0, printed.err[-600:]
        assert f"class_path: {_load_config(name)['model']['model_config']['class_path']}" in printed.out


class TestModelConfigRoundtrip:
    """model_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", SMOKE_CONFIGS)
    def test_fields_survive_roundtrip(self, name: str) -> None:
        """Every field in model_config.init_args is preserved after instantiation."""
        mc_section = _load_config(name)["model"]["model_config"]
        init_args = mc_section.get("init_args", {})

        mc = _instantiate(mc_section["class_path"], init_args)

        assert {field: getattr(mc, field) for field in init_args} == init_args, f"{name}.yaml: model_config mismatch"


class TestTrainConfigRoundtrip:
    """train_config init_args from YAML construct a valid config with matching values."""

    @pytest.mark.parametrize("name", SMOKE_CONFIGS)
    def test_fields_survive_roundtrip(self, name: str, tmp_path: pathlib.Path) -> None:
        """Every field in train_config.init_args is preserved after instantiation.

        dataset_dir is rewritten to tmp_path so path expansion doesn't fail on the placeholder /data/coco value.
        TrainConfig.expand_paths() converts relative path strings to absolute, so expected path-like strings are
        normalised with os.path.abspath before comparison.
        """
        tc_section = _load_config(name)["model"]["train_config"]
        init_args = {**tc_section.get("init_args", {}), "dataset_dir": str(tmp_path)}
        expected = {
            field: os.path.abspath(value) if isinstance(value, str) and (os.sep in value or "/" in value) else value
            for field, value in init_args.items()
        }

        tc = _instantiate(tc_section["class_path"], init_args)

        assert {field: getattr(tc, field) for field in init_args} == expected, f"{name}.yaml: train_config mismatch"


def _run_webdataset(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `python -m rfdetr.cli.webdataset` with the given args and return the completed process.

    >>> _run_webdataset("--help").returncode
    0
    """
    return subprocess.run(
        [sys.executable, "-m", "rfdetr.cli.webdataset", *args], capture_output=True, text=True, timeout=60, check=False
    )


class TestWebDatasetCLI:
    """The WebDataset command preserves packing and argument validation."""

    def test_help(self) -> None:
        """Help succeeds without required packing arguments."""
        result = _run_webdataset("--help")

        assert result.returncode == 0, result.stderr
        assert "--image-dir" in result.stdout
        assert "--category-ids {remap,raw}" in result.stdout

    def test_missing_arguments(self) -> None:
        """Missing required paths produce argparse's usage error."""
        result = _run_webdataset()

        assert result.returncode == 2
        assert "the following arguments are required" in result.stderr
        assert "--annotations" in result.stderr

    def test_help_without_lightning(self) -> None:
        """Packing must not import the optional Lightning training dependency."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; sys.modules['pytorch_lightning'] = None; "
                "module = sys.argv.pop(1); runpy.run_module(module, run_name='__main__')",
                "rfdetr.cli.webdataset",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "--image-dir" in result.stdout

    @pytest.mark.parametrize("category_ids", ["remap", "raw"])
    def test_pack(self, category_ids: str, tmp_path: Path) -> None:
        """Packing preserves image bytes, category policy and output summary."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_path = image_dir / "cat.png"
        Image.new("RGB", (8, 8)).save(image_path)
        annotations = tmp_path / "annotations.json"
        annotations.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "cat.png", "width": 8, "height": 8}],
                    "categories": [{"id": 9, "name": "cat"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 9, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}
                    ],
                }
            )
        )
        output = tmp_path / "shards"

        result = _run_webdataset(
            *("--image-dir", str(image_dir), "--annotations", str(annotations), "--output-dir", str(output)),
            *("--split", "val", "--max-shard-mb", "1", "--category-ids", category_ids),
        )

        assert result.returncode == 0, result.stderr
        index = read_shard_index(output, "val")
        assert index.num_samples == 1
        assert index.category_ids == category_ids
        assert index.categories == ({"id": 9, "name": "cat"},)
        assert index.annotated_category_ids == (9,)
        assert len(index.shards) == 1
        assert result.stdout.splitlines()[-1] == f"1 samples -> 1 shard(s) in {output} (val-index.json)"
        with tarfile.open(output / index.shards[0]) as shard:
            members = shard.getmembers()
            image_member = next(member for member in members if member.name.endswith(".png"))
            image_file = shard.extractfile(image_member)
            assert image_file is not None
            assert image_file.read() == image_path.read_bytes()


#: Import path patched to isolate the CLI entrypoint during module execution.
RFDETR_CLI_MAIN = "rfdetr.cli.main"


class TestMainEntrypoint:
    """The ``__main__`` guard dispatches to ``rfdetr.cli.main`` only when run as a script."""

    def test_dispatches_to_cli_main_when_run_as_script(self, mocker) -> None:
        """Running the module with ``__name__ == "__main__"`` invokes ``rfdetr.cli.main`` exactly once.

        This reproduces what ``python -m rfdetr`` does: runpy executes ``rfdetr/__main__.py`` with
        ``run_name="__main__"``, the same value the interpreter sets for a ``-m`` invocation. ``rfdetr.cli.main`` is
        mocked so the (heavyweight, optional-dependency-gated) training CLI it delegates to is never actually
        constructed.
        """
        mock_main = mocker.patch(RFDETR_CLI_MAIN)

        runpy.run_module("rfdetr", run_name="__main__")

        mock_main.assert_called_once_with()

    def test_skips_cli_main_when_imported_as_a_module(self, mocker) -> None:
        """Executing the module under a non-``__main__`` name leaves ``rfdetr.cli.main`` uncalled.

        This is the state ``rfdetr/__main__.py`` runs in for an ordinary import (``__name__`` set to its dotted module
        path rather than ``"__main__"``), which must leave the guard's body unrun.
        """
        mock_main = mocker.patch(RFDETR_CLI_MAIN)

        runpy.run_module("rfdetr", run_name="rfdetr.__main__")

        mock_main.assert_not_called()


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
