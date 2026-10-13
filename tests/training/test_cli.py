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

import csv
import json
import sys
import warnings
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import yaml
from jsonargparse import Namespace
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, Timer
from pytorch_lightning.cli import SaveConfigCallback

from rfdetr.config import ModelConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.training import BestModelCallback, COCOEvalCallback, RFDETREMACallback
from rfdetr.training.cli import RFDETRCli
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import _PARAM_GROUP_SETTINGS, _resolve_resumed_param_group_settings

from .test_module_model import _fake_criterion, _fake_model, _fake_postprocess

#: The shipped config the trainer tests parse, as ``rfdetr fit --config configs/rfdetr_nano.yaml`` does.
_NANO_CONFIG = Path(__file__).parents[2] / "configs" / "rfdetr_nano.yaml"


class _NoRunCli(RFDETRCli):
    """``RFDETRCli`` whose subcommands return at once, so a test can inspect the trainer each one builds.

    ``LightningCLI`` calls a subcommand method defined on the CLI instead of the trainer's.
    """

    def fit(self, **kwargs: Any) -> None:
        """Skip training."""

    def validate(self, **kwargs: Any) -> None:
        """Skip validation."""

    def test(self, **kwargs: Any) -> None:
        """Skip testing."""

    def predict(self, **kwargs: Any) -> None:
        """Skip prediction."""


def _cli(
    tmp_path: Path, *args: str, subcommand: str | None = "fit", from_sys_argv: bool = False, **cli_kwargs: Any
) -> RFDETRCli:
    """Parse the shipped Nano config as ``rfdetr <subcommand>`` does, around a stub detector so no weights load.

    Args:
        tmp_path: The dataset and output directory.
        *args: Extra command-line arguments.
        subcommand: The subcommand whose trainer to build, without running it; ``None`` uses ``run=False``.
        from_sys_argv: Put the arguments in ``sys.argv``, as the ``rfdetr`` console script has them, instead of
            passing them as ``args``.
        **cli_kwargs: Extra ``RFDETRCli`` arguments, such as ``trainer_defaults``.

    Returns:
        The CLI, with its trainer built.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _cli(Path(tmp)).trainer.max_epochs
        100
    """
    argv = [
        *([subcommand] if subcommand else []),
        "--config",
        str(_NANO_CONFIG),
        "--model.model_config.init_args.pretrain_weights",
        "null",
        "--model.train_config.init_args.dataset_dir",
        str(tmp_path),
        "--model.train_config.init_args.output_dir",
        str(tmp_path),
        *args,
    ]
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_fake_model()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_fake_criterion(), _fake_postprocess()),
        ),
        patch.object(sys, "argv", ["rfdetr", *argv]) if from_sys_argv else nullcontext(),
    ):
        return _NoRunCli(
            RFDETRModelModule,
            RFDETRDataModule,
            args=None if from_sys_argv else argv,
            run=subcommand is not None,
            **cli_kwargs,
        )


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

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        """Store the linked instantiated configs.

        Examples:
            >>> ConfigData(RFDETRNanoConfig(), TrainConfig(dataset_dir="data")).train_config.batch_size
            4
        """
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config


class ParseOnlyCli(RFDETRCli):
    """Run real subcommand parsing and checkpoint merge without starting training."""

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
        """Check values and explicit membership on actual module and datamodule configs."""
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestRFDETRCliTrainer:
    """The CLI trains with ``build_trainer``'s stack, as ``RFDETR.train()`` does; ``--trainer.*`` overrides it."""

    def test_fit_takes_epochs_accumulation_and_clipping_from_train_config(self, tmp_path: Path) -> None:
        """``train_config`` drives the loop instead of Lightning's defaults, such as ``max_epochs=1000``."""
        cli = _cli(
            tmp_path,
            "--model.train_config.init_args.epochs=7",
            "--model.train_config.init_args.grad_accum_steps=2",
            "--model.train_config.init_args.clip_max_norm=0.5",
        )

        trainer = cli.trainer
        assert (trainer.max_epochs, trainer.accumulate_grad_batches, trainer.gradient_clip_val) == (7, 2, 0.5)

    def test_fit_has_each_rfdetr_callback_once(self, tmp_path: Path) -> None:
        """EMA, the COCO metric, best-checkpoint selection and the config saver run, so ``fit`` logs mAP and saves."""
        cli = _cli(tmp_path)

        counts = Counter(type(callback) for callback in cli.trainer.callbacks)
        kinds = (RFDETREMACallback, COCOEvalCallback, BestModelCallback, SaveConfigCallback)
        assert [counts[kind] for kind in kinds] == [1, 1, 1, 1], counts

    def test_run_false_builds_the_fit_trainer(self, tmp_path: Path) -> None:
        """``RFDETRCli(run=False)`` hands the caller the trainer ``fit`` would use."""
        cli = _cli(tmp_path, subcommand=None)

        assert any(isinstance(callback, BestModelCallback) for callback in cli.trainer.callbacks)

    def test_trainer_flag_at_lightning_default_overrides_train_config(self, tmp_path: Path) -> None:
        """``--trainer.accumulate_grad_batches=1`` turns off the accumulation ``grad_accum_steps`` asks for."""
        cli = _cli(tmp_path, "--model.train_config.init_args.grad_accum_steps=2", "--trainer.accumulate_grad_batches=1")

        assert cli.trainer.accumulate_grad_batches == 1

    def test_trainer_flag_in_sys_argv_overrides_train_config(self, tmp_path: Path) -> None:
        """The ``rfdetr`` console script reads ``sys.argv``; a default-equal flag there still overrides."""
        cli = _cli(
            tmp_path,
            "--model.train_config.init_args.grad_accum_steps=2",
            "--trainer.accumulate_grad_batches=1",
            from_sys_argv=True,
        )

        assert cli.trainer.accumulate_grad_batches == 1

    def test_rerun_from_the_saved_config_keeps_train_config(self, tmp_path: Path) -> None:
        """The saved ``config.yaml`` lists every trainer argument, most at Lightning's defaults; they don't override."""
        first = _cli(tmp_path, "--model.train_config.init_args.grad_accum_steps=2")
        save_config = next(callback for callback in first.trainer.callbacks if isinstance(callback, SaveConfigCallback))
        save_config.setup(first.trainer, first.model, stage="fit")

        rerun = _cli(tmp_path, "--config", str(tmp_path / "config.yaml"))

        assert (rerun.trainer.max_epochs, rerun.trainer.accumulate_grad_batches) == (100, 2)

    def test_trainer_group_file_overrides_train_config(self, tmp_path: Path) -> None:
        """``--trainer trainer.yaml``, the group config option ``--help`` lists, applies its non-default values."""
        trainer_file = tmp_path / "trainer.yaml"
        trainer_file.write_text("accumulate_grad_batches: 3\n")
        cli = _cli(tmp_path, "--trainer", str(trainer_file))

        assert cli.trainer.accumulate_grad_batches == 3

    @pytest.mark.parametrize("run_length_flag", ["--trainer.max_epochs=3", "--trainer.max_steps=200"])
    def test_run_length_flag_warns_that_the_schedules_follow_epochs(self, tmp_path: Path, run_length_flag: str) -> None:
        """``--trainer.max_epochs`` or ``max_steps`` ends the run; the schedules follow ``train_config.epochs``."""
        with pytest.warns(UserWarning, match=r"train_config\.epochs=7"):
            _cli(tmp_path, "--model.train_config.init_args.epochs=7", run_length_flag)

    def test_trainer_callbacks_are_added_once_to_the_rfdetr_callbacks(self, tmp_path: Path) -> None:
        """``--trainer.callbacks`` extends the RF-DETR callbacks instead of replacing them."""
        cli = _cli(tmp_path, "--trainer.callbacks+=LearningRateMonitor")

        counts = Counter(type(callback) for callback in cli.trainer.callbacks)
        assert (counts[LearningRateMonitor], counts[COCOEvalCallback]) == (1, 1)

    def test_trainer_model_summary_takes_the_place_of_the_default_one(self, tmp_path: Path) -> None:
        """A ``--trainer.callbacks`` model summary reaches ``Trainer.__init__``, so Lightning adds no second one."""
        cli = _cli(tmp_path, "--trainer.callbacks+=ModelSummary")

        assert sum(isinstance(callback, ModelSummary) for callback in cli.trainer.callbacks) == 1

    def test_trainer_logger_flag_reaches_the_trainer(self, tmp_path: Path) -> None:
        """A logger given as ``--trainer.logger`` replaces the ``train_config`` loggers, instantiated."""
        elsewhere = tmp_path / "elsewhere"
        cli = _cli(tmp_path, "--trainer.logger=CSVLogger", f"--trainer.logger.init_args.save_dir={elsewhere}")

        assert cli.trainer.logger.save_dir == str(elsewhere)

    def test_trainer_defaults_reach_the_trainer(self, tmp_path: Path) -> None:
        """``RFDETRCli(trainer_defaults=...)`` sets trainer arguments and adds callbacks, as in ``LightningCLI``."""
        timer = Timer()
        cli = _cli(tmp_path, trainer_defaults={"limit_train_batches": 3, "callbacks": [timer]})

        assert (cli.trainer.limit_train_batches, timer in cli.trainer.callbacks) == (3, True)

    @pytest.mark.parametrize("subcommand", ["validate", "test", "predict"])
    def test_evaluation_subcommands_get_the_evaluation_only_trainer(self, tmp_path: Path, subcommand: str) -> None:
        """They build the trainer ``RFDETR.evaluate()`` uses: the metric, and nothing that writes to ``output_dir``."""
        cli = _cli(tmp_path, subcommand=subcommand)

        callback_types = {type(callback) for callback in cli.trainer.callbacks}
        writers = {RFDETREMACallback, BestModelCallback, ModelCheckpoint, SaveConfigCallback}
        assert COCOEvalCallback in callback_types and not callback_types & writers, sorted(map(str, callback_types))

    def test_validate_applies_trainer_flags(self, tmp_path: Path) -> None:
        """``rfdetr validate --trainer.limit_val_batches=2`` validates two batches."""
        cli = _cli(tmp_path, "--trainer.limit_val_batches=2", subcommand="validate")

        assert cli.trainer.limit_val_batches == 2

    def test_validate_adds_trainer_callbacks(self, tmp_path: Path) -> None:
        """``--trainer.callbacks`` reaches the evaluation-only trainer too."""
        cli = _cli(tmp_path, "--trainer.callbacks+=LearningRateMonitor", subcommand="validate")

        assert any(isinstance(callback, LearningRateMonitor) for callback in cli.trainer.callbacks)

    def test_validate_does_not_warn_about_the_training_schedules(self, tmp_path: Path) -> None:
        """The run-length warning concerns training; ``validate`` has no learning-rate schedule to size."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _cli(tmp_path, "--trainer.max_epochs=3", subcommand="validate")

        assert not [warning for warning in caught if "schedules follow" in str(warning.message)]

    def test_fit_overwrites_the_config_a_previous_run_saved(self, tmp_path: Path) -> None:
        """A second ``fit`` into one ``output_dir``, as a resume is, overwrites ``config.yaml`` instead of failing."""
        (tmp_path / "config.yaml").write_text("previous run\n")
        cli = _cli(tmp_path)
        save_config = next(callback for callback in cli.trainer.callbacks if isinstance(callback, SaveConfigCallback))

        save_config.setup(cli.trainer, cli.model, stage="fit")

        assert "RFDETRNanoConfig" in (tmp_path / "config.yaml").read_text()

    def test_resume_appends_to_the_metrics_of_earlier_epochs(self, tmp_path: Path) -> None:
        """``fit --ckpt_path`` keeps the ``metrics.csv`` rows already there, as ``RFDETR.train(resume=...)`` does."""
        metrics = tmp_path / "metrics.csv"
        metrics.write_text("epoch,step,val/mAP_50_95\n0,74,0.5\n")
        cli = _cli(tmp_path, f"--ckpt_path={tmp_path / 'last.ckpt'}")

        csv_logger = cli.trainer.logger
        csv_logger.log_metrics({"epoch": 1, "val/mAP_50_95": 0.6}, step=149)
        csv_logger.save()

        rows = csv.DictReader(metrics.read_text().splitlines())
        assert [row["val/mAP_50_95"] for row in rows] == ["0.5", "0.6"]

    def test_fast_dev_run_saves_no_config(self, tmp_path: Path) -> None:
        """A ``fast_dev_run`` smoke test leaves the ``config.yaml`` of a real run in ``output_dir`` alone."""
        cli = _cli(tmp_path, "--trainer.fast_dev_run=true")

        assert not any(isinstance(callback, SaveConfigCallback) for callback in cli.trainer.callbacks)

    @pytest.mark.parametrize(
        ("ckpt_path", "resumed_from"),
        [
            pytest.param(None, "resume.ckpt", id="train_config_resume"),
            pytest.param("given.ckpt", "given.ckpt", id="ckpt_path_wins"),
        ],
    )
    def test_fit_resumes_from_ckpt_path_or_train_config_resume(
        self, tmp_path: Path, ckpt_path: str | None, resumed_from: str
    ) -> None:
        """``--ckpt_path`` wins; otherwise ``resume`` in ``train_config`` resumes, as ``RFDETR.train()`` does."""
        cli = _cli(tmp_path, f"--model.train_config.init_args.resume={tmp_path / 'resume.ckpt'}", subcommand=None)

        with patch.object(cli.trainer, "fit") as trainer_fit:
            RFDETRCli.fit(cli, cli.model, ckpt_path=ckpt_path and str(tmp_path / ckpt_path))

        assert trainer_fit.call_args.kwargs["ckpt_path"] == str(tmp_path / resumed_from)

    @pytest.mark.parametrize("subcommand", ["validate", "test"])
    def test_evaluation_hides_the_missing_logger_warning(self, tmp_path: Path, subcommand: str) -> None:
        """The evaluation-only trainer has no logger; Lightning's warning about that, once per metric, is not shown."""
        cli = _cli(tmp_path, subcommand=subcommand)
        metrics = [{"val/mAP_50_95": 0.5}]

        def run_logging_one_metric(**kwargs: Any) -> list[dict[str, float]]:
            warnings.warn("You called `self.log(...)` but have no logger configured.", UserWarning, stacklevel=1)
            return metrics

        with (
            patch.object(cli.trainer, subcommand, side_effect=run_logging_one_metric),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            results = getattr(RFDETRCli, subcommand)(cli, model=cli.model)

        assert (results, [str(warning.message) for warning in caught]) == (metrics, [])

    def test_custom_trainer_class_is_refused(self, tmp_path: Path) -> None:
        """``build_trainer`` always builds a ``Trainer``, so a custom ``trainer_class`` is refused, not ignored."""
        with pytest.raises(ValueError, match="trainer_class"):
            _cli(tmp_path, trainer_class=type("CustomTrainer", (Trainer,), {}))


class TestRFDETRCliRequirement:
    """The CLI names the install command when the jsonargparse it needs is missing or too old."""

    def test_jsonargparse_before_4_46_gets_the_install_command(self, tmp_path: Path) -> None:
        """Without ``subclasses_enabled``, added in 4.46, the CLI names the extra to install, not a bare TypeError."""
        missing_keyword = TypeError("set_parsing_settings() got an unexpected keyword argument 'subclasses_enabled'")

        with (
            patch("jsonargparse.set_parsing_settings", side_effect=missing_keyword),
            pytest.raises(ModuleNotFoundError, match=r"pip install -U 'rfdetr\[cli\]'"),
        ):
            _cli(tmp_path)
