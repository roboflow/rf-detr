# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningCLI entry point for RF-DETR training and evaluation.

Provides the ``rfdetr`` command with auto-generated subcommands::

    rfdetr fit      --config configs/rfdetr_small.yaml
    rfdetr validate --config configs/rfdetr_small.yaml --ckpt_path output/rfdetr_small/last.ckpt
    rfdetr test     --config configs/rfdetr_small.yaml --ckpt_path output/rfdetr_small/last.ckpt
    rfdetr predict  --config configs/rfdetr_small.yaml --ckpt_path output/rfdetr_small/last.ckpt

Both ``RFDETRModelModule`` and ``RFDETRDataModule`` share the same ``(model_config, train_config)`` constructor
signature.  ``link_arguments`` eliminates the duplication so the user specifies each config group once; the datamodule
receives the same values automatically at parse time. The trainer comes from
:func:`~rfdetr.training.trainer.build_trainer`, the same as for ``RFDETR.train()``.
"""

import sys
import warnings
from collections.abc import Sequence
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import _PARAM_GROUP_SETTINGS
from rfdetr.training.trainer import build_trainer

_PARSER_KWARGS = {
    "prog": "rfdetr",
    "description": "RF-DETR command line interface.",
}

#: Subcommands that train and so get build_trainer's full callback and logger stack. ``None`` is ``run=False``, which
#: builds a trainer for the caller to ``fit`` with. Every other subcommand gets the evaluation-only trainer.
_TRAINING_SUBCOMMANDS: tuple[str | None, ...] = (None, "fit")

#: Lightning warns once per logged metric when the trainer has no logger, which the evaluation-only trainer has not by
#: design. ``RFDETR.evaluate()`` filters the same message.
_NO_LOGGER_WARNING = ".*no logger configured.*"

#: Raised when jsonargparse is missing or predates ``subclasses_enabled``, which 4.46.0 added.
_JSONARGPARSE_REQUIREMENT = (
    "The rfdetr command needs jsonargparse[signatures]>=4.46.0, which the cli extra installs: "
    "pip install -U 'rfdetr[cli]'"
)


def _changed_arguments(parsed: Any, defaults: Any) -> list[str]:
    """Return the names of the arguments in *parsed* whose value differs from their default in *defaults*.

    Keys with no default are not arguments: a group loaded from its own file, as ``--trainer trainer.yaml`` does,
    also carries the file's ``__path__``.

    Args:
        parsed: The parsed ``jsonargparse.Namespace`` of one argument group.
        defaults: The same group's defaults.

    Returns:
        The names, in parse order.

    Examples:
        >>> from jsonargparse import Namespace
        >>> parsed = Namespace(max_epochs=3, devices="auto", __path__="trainer.yaml")
        >>> _changed_arguments(parsed, Namespace(max_epochs=None, devices="auto"))
        ['max_epochs']
    """
    default_values = vars(defaults)
    return [name for name, value in vars(parsed).items() if name in default_values and value != default_values[name]]


def _config_argument_value(
    args: Sequence[Any], index: int, argument: str, attached_short_value: str | None
) -> tuple[str | None, int]:
    """Read a config option's value and return the next argument index.

    Args:
        args: The command-line arguments.
        index: The config option's index.
        argument: The config option, possibly containing its value.
        attached_short_value: The value attached to a short option, if present.

    Returns:
        The value, or ``None`` when missing, and the next argument index.
    """
    if attached_short_value is not None:
        return attached_short_value, index + 1
    if "=" in argument:
        return argument.partition("=")[2], index + 1
    if index + 1 < len(args):
        return str(args[index + 1]), index + 2
    return None, index + 1


def _load_argument_config(action: Any, source_parser: LightningArgumentParser, config_value: str) -> Any:
    """Load a trainer group or full config using its source parser's context.

    Args:
        action: The trainer or config option's parser action.
        source_parser: The parser owning the action.
        config_value: The config path or inline content.

    Returns:
        The parsed trainer group or full config namespace.
    """
    from jsonargparse import Namespace
    from jsonargparse._common import parser_context

    with parser_context(parent_parser=source_parser):
        if action.dest == "trainer":
            return action.check_type(config_value, source_parser)
        config = Namespace()  # type: ignore[no-untyped-call]
        action.apply_config(source_parser, config, action.dest, config_value)
    return config


def _trainer_override_markers(
    source_parser: LightningArgumentParser,
    args: Sequence[Any],
    index: int,
    argument: str,
    has_value: bool,
) -> tuple[set[str] | None, int]:
    """Parse an explicit trainer override marker option and advance past its value.

    Args:
        source_parser: The parser owning the marker option.
        args: The command-line arguments.
        index: The marker option's index.
        argument: The marker option, possibly containing its value.
        has_value: Whether the value is attached to the option.

    Returns:
        The marker names, or ``None`` when the value is missing, and the next argument index.
    """
    if has_value:
        marker_args = [argument]
    elif index + 1 < len(args):
        marker_args = [argument.partition("=")[0], str(args[index + 1])]
    else:
        return None, index + 1
    marker_config = source_parser.parse_args(marker_args, defaults=False, _skip_validation=True)
    marker_value = marker_config.get("rfdetr_cli_trainer_overrides", [])
    marker_names = set(marker_value) if isinstance(marker_value, (list, tuple)) else set()
    return marker_names, index + (1 if has_value else 2)


def _effective_trainer_overrides(
    parser: LightningArgumentParser,
    args: ArgsType,
    subcommand: str | None,
    subcommand_parser: LightningArgumentParser | None,
) -> tuple[set[str], set[str]]:
    """Return trainer flags and saved override markers that survive argument order.

    Args:
        parser: The configured CLI parser.
        args: The parsed arguments, or ``None`` to use ``sys.argv``.
        subcommand: The active LightningCLI subcommand.
        subcommand_parser: The parser for the active subcommand, if one is selected.

    Returns:
        Trainer argument names set on the command line and saved explicit override names loaded from config files.
    """
    if not isinstance(args, (list, tuple)):
        return set(), set()

    from jsonargparse._subcommands import find_parent_action

    flagged: set[str] = set()
    saved: set[str] = set()
    config_parser = subcommand_parser or parser

    def option_action(active_parser: LightningArgumentParser, argument: str) -> Any | None:
        option = argument.partition("=")[0]
        lookup_option = option.removesuffix("+")
        actions = active_parser._option_string_actions
        action = actions.get(lookup_option)
        if action is not None:
            return action
        if lookup_option.startswith("--"):
            matches = {candidate for name, candidate in actions.items() if name.startswith(lookup_option)}
            if len(matches) == 1:
                return matches.pop()
            return find_parent_action(active_parser, lookup_option[2:])
        if lookup_option.startswith("-") and len(lookup_option) > 2:
            return actions.get(lookup_option[:2])
        return None

    subcommand_index = len(args)
    index = 0
    while index < len(args):
        argument = args[index]
        if isinstance(argument, str) and argument == subcommand:
            subcommand_index = index
            break
        if isinstance(argument, str) and argument.startswith("-") and "=" not in argument:
            action = option_action(parser, argument)
            has_attached_short_value = not argument.startswith("--") and len(argument) > 2
            if action is not None and action.nargs != 0 and not has_attached_short_value and index + 1 < len(args):
                index += 2
                continue
        index += 1

    index = 0
    while index < len(args):
        argument_index = index
        argument = args[index]
        if not isinstance(argument, str):
            index += 1
            continue
        active_parser = parser if argument_index < subcommand_index else config_parser
        action = option_action(active_parser, argument)
        if action is None:
            index += 1
            continue

        option = argument.partition("=")[0]
        has_value = "=" in argument
        attached_short_value = (
            argument[2:] if option.startswith("-") and not option.startswith("--") and len(option) > 2 else None
        )
        has_value = has_value or attached_short_value is not None
        group_config = False
        if action.dest.startswith("trainer."):
            name = action.dest.removeprefix("trainer.").split(".", 1)[0].rstrip("+")
            flagged.add(name)
            saved.discard(name)
            index += 1 if has_value else 2
            continue

        source_parser = active_parser
        if action.dest in {"trainer", "config"}:
            config_value, index = _config_argument_value(args, index, argument, attached_short_value)
            if config_value is None:
                continue
            config = _load_argument_config(action, source_parser, config_value)
            group_config = action.dest == "trainer"
        elif action.dest == "rfdetr_cli_trainer_overrides":
            marker_names, index = _trainer_override_markers(source_parser, args, index, argument, has_value)
            if marker_names is None:
                continue
            if option.endswith("+"):
                saved.update(marker_names)
            else:
                saved = marker_names
            continue
        else:
            index += 1
            continue

        if group_config:
            config_values = vars(config)
            markers: set[str] = set()
        else:
            config = config.get(subcommand, config) if subcommand else config
            trainer = config.get("trainer")
            if hasattr(trainer, "__dict__"):
                config_values = vars(trainer)
            elif hasattr(trainer, "keys"):
                config_values = trainer.keys()
            else:
                config_values = trainer or {}
            marker_value = config.get("rfdetr_cli_trainer_overrides", [])
            markers = set(marker_value) if isinstance(marker_value, (list, tuple)) else set()
            marker_config = config
        for name in config_values:
            flagged.discard(name)
            saved.discard(name)
        if not group_config and "rfdetr_cli_trainer_overrides" in vars(marker_config):
            saved = markers
        else:
            saved.update(markers)
    return flagged, saved


def _as_list(value: Any) -> list[Any]:
    """Return a ``callbacks`` value as a list, as ``LightningCLI`` reads it: ``None`` is empty, one item is wrapped.

    Args:
        value: ``None``, one item, or a list of items.

    Returns:
        The items, in a new list.

    Examples:
        >>> _as_list(None), _as_list("a"), _as_list(["a", "b"])
        ([], ['a'], ['a', 'b'])
    """
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


class RFDETRCli(LightningCLI):
    """LightningCLI subclass for RF-DETR training and evaluation.

    Wires ``RFDETRModelModule`` and ``RFDETRDataModule`` under a unified CLI, with argument linking that shares
    ``model_config`` and ``train_config`` between module and datamodule so the user only specifies them once.

    Auto-generated subcommands: ``fit``, ``validate``, ``test``, ``predict``. Each builds its trainer with
    :func:`~rfdetr.training.trainer.build_trainer` from the parsed ``train_config``; see :meth:`instantiate_trainer`.
    """

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Create the parser, letting ``class_path`` name a subclass of ``ModelConfig`` or ``TrainConfig``.

        jsonargparse treats pydantic models as final unless told otherwise, so it rejects the subclass ``class_path``
        that every shipped config uses, such as ``rfdetr.config.RFDETRSmallConfig``. The setting is process-wide: a
        jsonargparse parser built later in the same process also accepts subclasses of both.

        Args:
            **kwargs: Keyword arguments for the parser.

        Returns:
            The parser.

        Raises:
            ModuleNotFoundError: If jsonargparse is not installed, or is older than 4.46.0, the first release with the
                ``subclasses_enabled`` setting.
            ValueError: If the CLI was given a ``trainer_class``: ``build_trainer`` always builds a ``Trainer``.
        """
        if self.trainer_class is not Trainer:
            raise ValueError(
                f"RFDETRCli builds its trainer with build_trainer, which makes a pytorch_lightning.Trainer, so "
                f"trainer_class={self.trainer_class!r} is not supported."
            )
        try:
            from jsonargparse import set_parsing_settings

            set_parsing_settings(subclasses_enabled=[ModelConfig, TrainConfig])
        except (ImportError, TypeError) as exc:
            raise ModuleNotFoundError(_JSONARGPARSE_REQUIREMENT) from exc
        return super().init_parser(**kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register argument links that share configs between module and datamodule.

        Linking ``model.model_config`` → ``data.model_config`` and ``model.train_config`` → ``data.train_config`` means
        the user specifies both config groups once; the datamodule receives the same values automatically.

        Args:
            parser: The jsonargparse ``LightningArgumentParser`` provided by
                ``LightningCLI``.
        """
        parser.link_arguments("model.model_config", "data.model_config", apply_on="parse")
        parser.link_arguments("model.train_config", "data.train_config", apply_on="parse")
        parser.add_argument(
            "--rfdetr_cli_trainer_overrides",
            type=list[str],
            default=[],
            help=(
                "Internal state saved to preserve explicit trainer overrides; older RF-DETR CLIs may reject configs "
                "written by this release. Do not set manually."
            ),
        )

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parse arguments while retaining trainer overrides and explicit optimizer settings.

        Args:
            parser: The parser.
            args: The command line, or ``None`` for ``sys.argv``, or a dict or ``Namespace`` of arguments.
        """
        from jsonargparse import Namespace

        effective_args = sys.argv[1:] if args is None else args
        super().parse_arguments(parser, args)
        subcommand = self.config.get("subcommand")
        self._flagged_trainer_arguments, self._saved_trainer_overrides = _effective_trainer_overrides(
            parser,
            effective_args,
            subcommand,
            self._subcommand_parsers.get(subcommand) if subcommand else None,
        )
        if not isinstance(effective_args, (list, tuple)):
            config = self.config.get(str(subcommand), self.config) if subcommand else self.config
            self._saved_trainer_overrides = set(config.get("rfdetr_cli_trainer_overrides", []))
        # Defaults erase whether an optimizer value was explicitly supplied, which matters when resuming.
        sparse = (
            parser.parse_object(args, defaults=False)
            if isinstance(args, (dict, Namespace))
            else parser.parse_args(args, defaults=False)
        )
        if sparse.get("subcommand"):
            sparse = sparse[sparse.subcommand]
        supplied = sparse.get("model.train_config", {})
        supplied = supplied.get("init_args", supplied)
        self._explicit_optimizer_settings: dict[str, Any] = {
            name: supplied[name] for name in _PARAM_GROUP_SETTINGS if name in supplied
        }

    def before_instantiate_classes(self) -> None:
        """Restore explicitly supplied optimizer settings after Lightning merges checkpoint values."""
        config = self.config[self.subcommand] if self.subcommand else self.config
        for component in ("model", "data"):
            train_config = config.get(f"{component}.train_config")
            if train_config is not None:
                train_config = train_config.get("init_args", train_config)
                for name, value in self._explicit_optimizer_settings.items():
                    train_config[name] = value

    def after_instantiate_classes(self) -> None:
        """Preserve explicit optimizer-setting membership on the instantiated training configs."""
        for component in (self.model, self.datamodule):
            train_config = getattr(component, "train_config", None)
            if isinstance(train_config, TrainConfig):
                train_config.model_fields_set.difference_update(_PARAM_GROUP_SETTINGS)
                train_config.model_fields_set.update(self._explicit_optimizer_settings)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Build the trainer with :func:`~rfdetr.training.trainer.build_trainer`, as ``RFDETR.train()`` does.

        ``train_config`` sets the epochs, precision, devices, gradient accumulation and clipping, callbacks and
        loggers. A ``--trainer.*`` flag overrides the value built from it, and so does a value in a config file's
        ``trainer`` section, in a ``--trainer`` file or in dict arguments, unless it equals Lightning's default: the
        ``config.yaml`` that ``fit`` saves lists every trainer argument, most at their defaults, and running from it
        again must not undo ``train_config``. Arguments apply in order, as in jsonargparse, so a ``--config`` after a
        flag replaces it.

        ``fit`` gets the training stack and saves the parsed config to ``config.yaml`` in ``output_dir``, replacing the
        one a previous run there saved; resumed with ``--ckpt_path`` or ``train_config.resume``, it appends to the
        ``metrics.csv`` already there. A ``max_epochs`` other than ``train_config.epochs``, or any ``max_steps``, warns,
        because the learning-rate and drop-path schedules follow ``train_config.epochs``. ``validate``, ``test`` and
        ``predict`` get the evaluation-only trainer that ``RFDETR.evaluate()`` uses, which writes nothing to
        ``output_dir``.

        Callbacks from ``--trainer.callbacks`` and ``trainer_defaults`` go to ``build_trainer`` as ``extra_callbacks``,
        which adds them to the RF-DETR callbacks before the trainer is built, where Lightning checks them with the rest.
        As in ``LightningCLI``, callbacks in *kwargs* take the place of ``--trainer.callbacks``.

        Args:
            **kwargs: Trainer arguments that take precedence over the parsed ones.

        Returns:
            The trainer.
        """
        parser = self._parser(self.subcommand)
        parsed = self._get(self.config, "trainer")
        defaults = parser.get_defaults(skip_validation=True).get("trainer")
        instantiated = self._get(self.config_init, "trainer")
        changed = set(_changed_arguments(parsed, defaults)) | set(self.trainer_defaults)
        explicit_overrides = set(getattr(self, "_saved_trainer_overrides", set())) | set(
            getattr(self, "_flagged_trainer_arguments", set())
        )
        explicit_overrides.intersection_update(vars(instantiated))
        changed |= set(explicit_overrides)
        overrides = {name: instantiated[name] for name in changed} | kwargs

        config = self.config[str(self.subcommand)] if self.subcommand else self.config
        config["rfdetr_cli_trainer_overrides"] = sorted(explicit_overrides)

        callbacks = _as_list(overrides.pop("callbacks", None))
        callbacks += [self._get(self.config_init, key) for key in parser.callback_keys]
        callbacks += _as_list(self.trainer_defaults.get("callbacks"))
        training = self.subcommand in _TRAINING_SUBCOMMANDS
        train_config = self.model.train_config
        run_length = [
            f"{name}={overrides[name]}"
            for name, unset in (("max_epochs", train_config.epochs), ("max_steps", -1))
            if name in overrides and overrides[name] != unset
        ]
        if training and run_length:
            warnings.warn(
                f"The trainer's {' and '.join(run_length)} sets the run length, but the learning-rate and drop-path "
                f"schedules follow train_config.epochs={train_config.epochs}. Set "
                "--model.train_config.init_args.epochs instead to change both.",
                UserWarning,
                stacklevel=1,
            )
        if training and self.save_config_callback and not overrides.get("fast_dev_run", False):
            callbacks.append(
                self.save_config_callback(
                    parser,
                    self.config.get(str(self.subcommand), self.config),
                    **{"overwrite": True, **self.save_config_kwargs},
                )
            )
        resume = self._get(self.config, "ckpt_path") if training else None
        if resume and not train_config.resume:
            # build_trainer keeps the metrics.csv rows of the epochs before a resume only when train_config.resume
            # names the checkpoint, as RFDETR.train() sets it; the CLI hands the checkpoint to Lightning as ckpt_path.
            train_config = train_config.model_copy(update={"resume": str(resume)})
        return build_trainer(
            train_config,
            self.model.model_config,
            include_training_callbacks=training,
            extra_callbacks=callbacks,
            **overrides,
        )

    def fit(self, model: LightningModule, ckpt_path: str | None = None, **kwargs: Any) -> None:
        """Run ``Trainer.fit``, resuming from ``train_config.resume`` when ``--ckpt_path`` is not given.

        ``RFDETR.train()`` resumes from ``train_config.resume``; ``LightningCLI`` alone would resume only from
        ``--ckpt_path``, which takes precedence here.

        Args:
            model: The module to train.
            ckpt_path: The checkpoint given as ``--ckpt_path``.
            **kwargs: Other arguments for ``Trainer.fit``.
        """
        resume = self.model.train_config.resume
        self.trainer.fit(model, ckpt_path=ckpt_path or (str(resume) if resume else None), **kwargs)

    def validate(self, **kwargs: Any) -> Any:
        """Run ``Trainer.validate`` without Lightning's warning, once per metric, that the trainer has no logger.

        Args:
            **kwargs: Arguments for ``Trainer.validate``.

        Returns:
            The validation metrics.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_NO_LOGGER_WARNING)
            return self.trainer.validate(**kwargs)

    def test(self, **kwargs: Any) -> Any:
        """Run ``Trainer.test`` without Lightning's warning, once per metric, that the trainer has no logger.

        Args:
            **kwargs: Arguments for ``Trainer.test``.

        Returns:
            The test metrics.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_NO_LOGGER_WARNING)
            return self.trainer.test(**kwargs)


def main() -> None:
    """Entry point for the ``rfdetr`` CLI.

    Constructs and runs ``RFDETRCli`` with ``RFDETRModelModule`` and ``RFDETRDataModule``.  Subcommands are auto-
    generated by ``LightningCLI``: ``fit``, ``validate``, ``test``, ``predict``.
    """
    RFDETRCli(RFDETRModelModule, RFDETRDataModule, parser_kwargs=_PARSER_KWARGS)


if __name__ == "__main__":
    main()
