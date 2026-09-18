# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.logger — the shared logger factory.

``_RedirectAwareStreamHandler`` exists because ``pytorch_lightning``'s ``RichProgressBar`` (installed with
``leave=True`` by ``build_trainer``, see ``training/trainer.py``) relies on Rich's ``Live`` display temporarily swapping
*both* the ``sys.stdout`` and ``sys.stderr`` module attributes for proxies (``redirect_stdout`` and ``redirect_stderr``
default to ``True`` in ``rich.live.Live``) so that ordinary writes print above the live-rendered progress bar instead of
corrupting it. A handler that captured a stale reference to the pre-swap stream at construction time — as a plain
``logging.StreamHandler(sys.stdout)`` does, and ``get_logger()`` is called at import time, well before any redirect
happens — would bypass that coordination and garble the progress bar. This applies equally to stdout (e.g.
``logger.info`` on a new best checkpoint) and stderr (e.g. ``logger.warning`` from ``BestModelCallback`` when EMA
weights are unavailable).
"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from rfdetr.utilities.logger import _RedirectAwareStreamHandler, get_logger


def _make_record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    """Build a minimal LogRecord at *level* carrying *message*.

    Examples:
        >>> record = _make_record("hi")
        >>> record.getMessage()
        'hi'
    """
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


class TestRedirectAwareStreamHandler:
    """_RedirectAwareStreamHandler must always write through the CURRENT sys.stdout/sys.stderr."""

    @pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
    def test_emit_writes_to_stream_present_at_construction(self, stream_name: str) -> None:
        """Baseline: with no reassignment in play, emit writes to whatever sys.<stream_name> already is."""
        buffer = io.StringIO()
        real_stream = getattr(sys, stream_name)
        try:
            setattr(sys, stream_name, buffer)
            handler = _RedirectAwareStreamHandler(stream_name)
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.emit(_make_record("hello"))
        finally:
            setattr(sys, stream_name, real_stream)

        assert "hello" in buffer.getvalue()

    @pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
    def test_emit_follows_reassignment_after_construction(self, stream_name: str) -> None:
        """A sys.<stream_name> swap AFTER construction (Rich's Live redirect) must be honoured, not bypassed."""
        handler = _RedirectAwareStreamHandler(stream_name)
        handler.setFormatter(logging.Formatter("%(message)s"))
        pre_redirect_stream = handler.stream

        redirected_stream = io.StringIO()
        real_stream = getattr(sys, stream_name)
        try:
            setattr(sys, stream_name, redirected_stream)
            handler.emit(_make_record("best checkpoint saved"))
        finally:
            setattr(sys, stream_name, real_stream)

        assert "best checkpoint saved" in redirected_stream.getvalue()
        if isinstance(pre_redirect_stream, io.StringIO):
            assert "best checkpoint saved" not in pre_redirect_stream.getvalue()

    def test_stdout_handler_never_writes_to_stderr(self) -> None:
        """Regression: a stdout handler must track sys.stdout specifically, never sys.stderr."""
        handler = _RedirectAwareStreamHandler("stdout")
        handler.setFormatter(logging.Formatter("%(message)s"))

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        real_stdout, real_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
            handler.emit(_make_record("stdout only"))
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr

        assert "stdout only" in stdout_buffer.getvalue()
        assert "stdout only" not in stderr_buffer.getvalue()

    def test_stdout_handler_falls_back_to_stderr_when_stdout_is_none(self) -> None:
        """A stdout handler must preserve records when Python temporarily removes sys.stdout."""
        handler = _RedirectAwareStreamHandler("stdout")
        handler.setFormatter(logging.Formatter("%(message)s"))

        stderr_buffer = io.StringIO()
        real_stdout, real_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = None, stderr_buffer
            handler.emit(_make_record("fallback to stderr"))
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr

        assert stderr_buffer.getvalue() == "fallback to stderr\n"


class TestGetLoggerUsesRedirectAwareHandlers:
    """get_logger()'s stdout AND stderr handlers must be redirect-aware, not plain StreamHandlers."""

    def test_stdout_handler_is_redirect_aware(self) -> None:
        """The INFO/DEBUG handler installed by get_logger is a _RedirectAwareStreamHandler bound to stdout."""
        logger = get_logger("rf-detr-test-redirect-aware-stdout-handler")

        stdout_handlers = [h for h in logger.handlers if h.level == logging.DEBUG]

        assert len(stdout_handlers) == 1
        assert isinstance(stdout_handlers[0], _RedirectAwareStreamHandler)
        assert stdout_handlers[0]._stream_name == "stdout"

    def test_stderr_handler_is_redirect_aware(self) -> None:
        """The WARNING+ handler installed by get_logger is a _RedirectAwareStreamHandler bound to stderr."""
        logger = get_logger("rf-detr-test-redirect-aware-stderr-handler")

        stderr_handlers = [h for h in logger.handlers if h.level == logging.WARNING]

        assert len(stderr_handlers) == 1
        assert isinstance(stderr_handlers[0], _RedirectAwareStreamHandler)
        assert stderr_handlers[0]._stream_name == "stderr"
