# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared logger factory for RF-DETR modules."""

from __future__ import annotations

import logging
import os
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Mapping, TextIO

if TYPE_CHECKING:
    # typeshed declares ``logging.StreamHandler`` as ``Generic[AnyStr]`` for type
    # checkers only; the runtime class does not implement ``__class_getitem__`` on
    # Python 3.10/3.11 (subscripting it raises ``TypeError``), so the parametrized
    # form must stay behind ``TYPE_CHECKING``.
    _StreamHandlerBase = logging.StreamHandler[TextIO]
else:
    _StreamHandlerBase = logging.StreamHandler


class _RedirectAwareStreamHandler(_StreamHandlerBase):
    """``StreamHandler`` that re-resolves ``sys.stdout``/``sys.stderr`` on every emitted record.

    Rich's ``Live`` display (used internally by pytorch_lightning's ``RichProgressBar``, see ``trainer.py``'s
    ``GPUMemoryRichProgressBar`` callback) temporarily replaces *both* the ``sys.stdout`` and ``sys.stderr`` module
    attributes with proxies while training runs (``redirect_stdout``/``redirect_stderr`` default to ``True`` in
    ``rich.live.Live``), so ordinary writes are printed above the live-rendered progress bar instead of corrupting it. A
    plain ``StreamHandler(sys.stdout)`` (or ``sys.stderr``) captures that attribute once, at construction time — which
    happens at import time here, long before any ``Trainer.fit()`` call — so it keeps writing through the pre-redirect
    stream forever, bypassing Rich's coordination. With ``RichProgressBar(leave=True)`` this corruption is no longer
    overwritten by the next refresh and shows up as a duplicated/garbled epoch bar in the terminal history whenever a
    training-time log call fires mid-epoch (e.g. ``BestModelCallback`` logging a new best checkpoint on stdout, or a
    fallback warning on stderr).

    Tracks the target stream by *name* (``"stdout"`` or ``"stderr"``) rather than by the object passed at construction
    time, so ``emit`` always resolves the current ``sys.<name>`` — including Rich's proxy.
    """

    def __init__(self, stream_name: Literal["stdout", "stderr"]) -> None:
        self._stream_name = stream_name
        super().__init__(getattr(sys, stream_name))

    def emit(self, record: logging.LogRecord) -> None:
        """Refresh ``self.stream`` from the current ``sys.<stream_name>`` before writing *record*.

        Falls back to ``sys.stderr`` when the target attribute is ``None`` (e.g. under ``pythonw``/frozen builds, or a
        stream closed during interpreter shutdown), matching ``logging.StreamHandler.__init__``'s own ``stream is None``
        fallback — otherwise a ``None`` stream reaches the base class' ``emit`` and the record is dropped with an
        ``AttributeError`` from ``logging``'s error handler instead of being written.
        """
        stream = getattr(sys, self._stream_name)
        self.stream = stream if stream is not None else sys.stderr
        super().emit(record)


class _RFDETRLogger(logging.Logger):
    """Logger subclass that adds a :meth:`warning_once` helper."""

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self._warned_once: set[str] = set()

    def warning_once(
        self,
        msg: str,
        *args: object,
        exc_info: bool
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | BaseException
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Emit *msg* as a WARNING exactly once per unique message string."""
        if msg not in self._warned_once:
            self._warned_once.add(msg)
            self.warning(
                msg,
                *args,
                exc_info=exc_info,
                stack_info=stack_info,
                stacklevel=stacklevel,
                extra=extra,
            )


def get_logger(name: str = "rf-detr", level: int | None = None) -> _RFDETRLogger:
    """Creates and configures a logger with stdout and stderr handlers.

    This function creates a logger that sends INFO and DEBUG level logs to stdout, and WARNING, ERROR, and CRITICAL
    level logs to stderr. If the logger already has handlers, it returns the existing logger without adding new
    handlers.

    The log level can be specified directly or through the LOG_LEVEL environment variable.

    Args:
        name: The name of the logger. Defaults to "rf-detr".
        level: The logging level to set. If None, uses the LOG_LEVEL environment
            variable, defaulting to INFO if not set.

    Returns:
        A configured _RFDETRLogger instance.
    """
    logger = logging.getLogger(name)

    # If the logger was already registered as a plain Logger before this call,
    # upgrade it in-place so warning_once is always available.
    if not isinstance(logger, _RFDETRLogger):
        logger.__class__ = _RFDETRLogger
        logger._warned_once = set()  # type: ignore[attr-defined]

    first_setup = not logger.handlers
    # Only default the level on first setup; otherwise a bare call would clobber a
    # level the caller set previously. An explicit level always wins.
    if level is not None:
        logger.setLevel(level)
    elif first_setup:
        logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

    if first_setup:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        stdout_handler = _RedirectAwareStreamHandler("stdout")
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda r: r.levelno <= logging.INFO)
        stdout_handler.setFormatter(formatter)

        stderr_handler = _RedirectAwareStreamHandler("stderr")
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
        logger.propagate = False

    return logger  # type: ignore[return-value]
