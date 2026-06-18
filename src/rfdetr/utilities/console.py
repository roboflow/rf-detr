# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Rich console helper for the training callback stack."""

from typing import Any


def _get_rich_console(trainer: Any) -> Any:
    """Return a Rich Console appropriate for the current training context.

    When a ``RichProgressBar`` callback is active, its ``_console`` is the Rich global
    singleton that owns the Live progress display.  Printing through that same console
    instance ensures output appears correctly above the progress bars rather than
    conflicting with the Live display's cursor positioning (particularly on Windows).

    Falls back to ``Console(force_terminal=True)`` for plain-terminal and TQDM contexts.

    Args:
        trainer: The PTL Trainer (or any object with a ``callbacks`` attribute).

    Returns:
        A Rich ``Console`` instance suitable for printing metric tables.
    """
    from rich.console import Console

    for cb in getattr(trainer, "callbacks", []):
        if cb.__class__.__name__ == "RichProgressBar":
            cb_console = getattr(cb, "_console", None)
            if cb_console is not None:
                return cb_console
    return Console(force_terminal=True)
