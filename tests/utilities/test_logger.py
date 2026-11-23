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
    def test_emit_writes_to_stream_present_at_construction(self, monkeypatch: pytest.MonkeyPatch, stream_name: str):
        """Baseline: with no reassignment in play, emit writes to whatever sys.<stream_name> already is."""
        buffer = io.StringIO()
        monkeypatch.setattr(sys, stream_name, buffer)
        handler = _RedirectAwareStreamHandler(stream_name)
        handler.setFormatter(logging.Formatter("%(message)s"))

        handler.emit(_make_record("hello"))

        assert "hello" in buffer.getvalue()

    @pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
    def test_emit_follows_reassignment_after_construction(self, monkeypatch: pytest.MonkeyPatch, stream_name: str):
        """A sys.<stream_name> swap AFTER construction (Rich's Live redirect) must be honoured, not bypassed."""
        handler = _RedirectAwareStreamHandler(stream_name)
        handler.setFormatter(logging.Formatter("%(message)s"))
        pre_redirect_stream = handler.stream
        redirected_stream = io.StringIO()
        monkeypatch.setattr(sys, stream_name, redirected_stream)

        handler.emit(_make_record("best checkpoint saved"))

        assert "best checkpoint saved" in redirected_stream.getvalue()
        if isinstance(pre_redirect_stream, io.StringIO):
            assert "best checkpoint saved" not in pre_redirect_stream.getvalue()

    def test_stdout_handler_never_writes_to_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: a stdout handler must track sys.stdout specifically, never sys.stderr."""
        handler = _RedirectAwareStreamHandler("stdout")
        handler.setFormatter(logging.Formatter("%(message)s"))
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout_buffer)
        monkeypatch.setattr(sys, "stderr", stderr_buffer)

        handler.emit(_make_record("stdout only"))

        assert "stdout only" in stdout_buffer.getvalue()
        assert "stdout only" not in stderr_buffer.getvalue()

    def test_stdout_handler_falls_back_to_stderr_when_stdout_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stdout handler must preserve records when Python temporarily removes sys.stdout."""
        handler = _RedirectAwareStreamHandler("stdout")
        handler.setFormatter(logging.Formatter("%(message)s"))
        stderr_buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", stderr_buffer)

        handler.emit(_make_record("fallback to stderr"))

        assert stderr_buffer.getvalue() == "fallback to stderr\n"


class TestGetLoggerUsesRedirectAwareHandlers:
    """get_logger()'s stdout AND stderr handlers must be redirect-aware, not plain StreamHandlers."""

    @pytest.mark.parametrize(
        ("stream_name", "level"),
        [
            pytest.param("stdout", logging.DEBUG, id="stdout-handler-takes-debug-and-up"),
            pytest.param("stderr", logging.WARNING, id="stderr-handler-takes-warning-and-up"),
        ],
    )
    def test_handler_is_redirect_aware(self, stream_name: str, level: int) -> None:
        """The handler get_logger installs at *level* is a _RedirectAwareStreamHandler bound to *stream_name*."""
        logger = get_logger(f"rf-detr-test-redirect-aware-{stream_name}-handler")

        handlers = [h for h in logger.handlers if h.level == level]

        assert len(handlers) == 1
        assert isinstance(handlers[0], _RedirectAwareStreamHandler)
        assert handlers[0]._stream_name == stream_name
