# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Structured JSON logging on stderr, with redaction applied to every record.

stdout belongs to the MCP stdio transport, so nothing here may ever write to it.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from vision_mcp.security import redact, redact_data

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class RedactingJsonFormatter(logging.Formatter):
    """Renders one JSON object per line; message and extras are redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact_data(value)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the stderr JSON handler exactly once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(RedactingJsonFormatter())
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").propagate = False


def get_logger(name: str = "vision-mcp") -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
