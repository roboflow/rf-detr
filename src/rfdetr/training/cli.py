# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningCLI entry point for RF-DETR training and evaluation.

Provides the ``rfdetr`` command with auto-generated subcommands::

    rfdetr fit      --config configs/rfdetr_base.yaml
    rfdetr validate --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt
    rfdetr test     --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt
    rfdetr predict  --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt

Both ``RFDETRModelModule`` and ``RFDETRDataModule`` share the same ``(model_config, train_config)`` constructor
signature.  ``link_arguments`` eliminates the duplication so the user specifies each config group once; the datamodule
receives the same values automatically at parse time. The trainer comes from
:func:`~rfdetr.training.trainer.build_trainer`, the same as for ``RFDETR.train()``.
"""

import sys
import warnings
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
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


def _command_line_trainer_arguments(args: ArgsType) -> set[str]:
    """Return the names of the trainer arguments that ``--trainer.*`` flags set on a command line.

    Args:
        args: The command line, without the program name. A dict or ``Namespace``, which ``LightningCLI`` also takes,
            has no flags and names none.

    Returns:
        The trainer argument names.

    Examples:
        >>> args = ["fit", "--trainer.max_epochs", "3", "--trainer.callbacks+=Timer", "--trainer.logger.init_args.x=1"]
        >>> sorted(_command_line_trainer_arguments(args))
        ['callbacks', 'logger', 'max_epochs']
        >>> _command_line_trainer_arguments({"trainer": {"max_epochs": 3}})
        set()
    """
    if not isinstance(args, (list, tuple)):
        return set()
    prefix = "--trainer."
    return {
        arg.removeprefix(prefix).split("=", 1)[0].split(".", 1)[0].rstrip("+")
        for arg in args
        if isinstance(arg, str) and arg.startswith(prefix)
    }


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
        that every shipped config uses, such as ``rfdetr.config.RFDETRBaseConfig``. The setting is process-wide: a
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

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parse the arguments, noting which trainer arguments ``--trainer.*`` flags set.

        Args:
            parser: The parser.
            args: The command line, or ``None`` for ``sys.argv``, or a dict or ``Namespace`` of arguments.
        """
        self._flagged_trainer_arguments = _command_line_trainer_arguments(sys.argv[1:] if args is None else args)
        super().parse_arguments(parser, args)

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

        Callbacks from ``--trainer.callbacks`` and ``trainer_defaults`` are added to the RF-DETR callbacks once the
        trainer is built, as the customization guide shows for ``build_trainer``, instead of replacing them. As in
        ``LightningCLI``, callbacks in *kwargs* take the place of ``--trainer.callbacks``.

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
        changed |= getattr(self, "_flagged_trainer_arguments", set()) & set(vars(parsed))
        overrides = {name: instantiated[name] for name in changed} | kwargs

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
        trainer = build_trainer(
            train_config,
            self.model.model_config,
            include_training_callbacks=training,
            **overrides,
        )
        trainer.callbacks.extend(callbacks)  # type: ignore[attr-defined]
        return trainer

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
