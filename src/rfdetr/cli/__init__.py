# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR command-line interface.

``rfdetr`` is the root CLI. It owns the top-level help and dispatches each
command to its backend:

* ``fit`` / ``validate`` / ``test`` / ``predict`` → :mod:`rfdetr.cli.train`
  (``RFDETRCli``, a :class:`pytorch_lightning.cli.LightningCLI` subclass).
* ``export`` → :mod:`rfdetr.cli.export` (jsonargparse wrapper over
  :meth:`rfdetr.detr.RFDETR.export`).

Backends are imported lazily and render their own ``rfdetr <command> --help``,
so ``rfdetr export`` and ``rfdetr --help`` work without the ``[train]`` extra
(no PyTorch Lightning required).
"""

from __future__ import annotations

import importlib
import sys
from typing import TextIO

# Training commands are delegated to ``rfdetr.cli.train`` (LightningCLI); their
# argv is passed through unchanged because LightningCLI expects the command token.
_TRAIN_COMMANDS: dict[str, str] = {
    "fit": "Train a model",
    "validate": "Run the validation loop",
    "test": "Run the test loop",
    "predict": "Run the prediction loop",
}
# Standalone commands live in ``rfdetr.cli.<name>`` (each exposing ``main()``)
# and avoid importing the training stack.
_STANDALONE_COMMANDS: dict[str, str] = {
    "export": "Export a checkpoint to ONNX or TFLite",
}


def _print_root_help(stream: TextIO | None = None) -> None:
    """Render the unified top-level ``rfdetr`` help."""
    out = stream if stream is not None else sys.stdout
    print("usage: rfdetr <command> [options]", file=out)
    print("\nRF-DETR command-line interface.\n", file=out)
    print("commands:", file=out)
    for name, summary in {**_TRAIN_COMMANDS, **_STANDALONE_COMMANDS}.items():
        print(f"  {name:<10} {summary}", file=out)
    print("\nRun 'rfdetr <command> --help' for command-specific options.", file=out)


def main() -> None:
    """Dispatch ``rfdetr <command>`` to its backend.

    The root renders the top-level help itself (it is not delegated to a backend); ``rfdetr <command> --help`` is
    rendered by the backend that owns the command.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return

    command = argv[0]
    if command in _STANDALONE_COMMANDS:
        # Strip the command so the backend parser sees only its own flags.
        sys.argv.pop(1)
        importlib.import_module(f"rfdetr.cli.{command}").main()
        return
    # Training commands, or a LightningCLI root-level option that precedes one
    # (e.g. ``rfdetr -c config.yaml fit`` or ``rfdetr --print_config fit``), are
    # delegated to LightningCLI. It reads sys.argv and parses the root options
    # and command token natively, so argv is left unchanged.
    if command in _TRAIN_COMMANDS or command.startswith("-"):
        from rfdetr.cli.train import main as train_main

        train_main()
        return

    print(f"rfdetr: error: invalid command {command!r}\n", file=sys.stderr)
    _print_root_help(stream=sys.stderr)
    raise SystemExit(2)


__all__ = ["main"]
