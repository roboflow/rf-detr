"""Error codes and the single exception type crossing the engine/MCP boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from vision_mcp.contract.base import SCHEMA_VERSION


class ErrorCode(StrEnum):
    """Every failure returned to a client carries one of these codes."""

    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_TASK_MISMATCH = "MODEL_TASK_MISMATCH"
    STREAM_NOT_FOUND = "STREAM_NOT_FOUND"
    STREAM_DISCONNECTED = "STREAM_DISCONNECTED"
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_SOURCE = "UNSUPPORTED_SOURCE"
    URL_NOT_ALLOWED = "URL_NOT_ALLOWED"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    EVENT_NOT_FOUND = "EVENT_NOT_FOUND"
    ZONE_NOT_FOUND = "ZONE_NOT_FOUND"
    LINE_NOT_FOUND = "LINE_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    GPU_METRICS_UNAVAILABLE = "GPU_METRICS_UNAVAILABLE"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    INVALID_TIME_WINDOW = "INVALID_TIME_WINDOW"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"


#: HTTP status used when an engine endpoint fails with a given code.
_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.STREAM_NOT_FOUND: 404,
    ErrorCode.ARTIFACT_NOT_FOUND: 404,
    ErrorCode.EVENT_NOT_FOUND: 404,
    ErrorCode.ZONE_NOT_FOUND: 404,
    ErrorCode.LINE_NOT_FOUND: 404,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.MODEL_NOT_LOADED: 409,
    ErrorCode.STREAM_DISCONNECTED: 409,
    ErrorCode.GPU_METRICS_UNAVAILABLE: 501,
    ErrorCode.DATABASE_UNAVAILABLE: 503,
    ErrorCode.ENGINE_UNAVAILABLE: 503,
    ErrorCode.INFERENCE_FAILED: 500,
}


class VisionError(Exception):
    """Raised anywhere a request cannot be satisfied; converted to a typed response at the boundary."""

    def __init__(self, code: ErrorCode, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def http_status(self) -> int:
        """HTTP status code the engine should answer with."""
        return _HTTP_STATUS.get(self.code, 400)

    def to_payload(self) -> dict[str, Any]:
        """Serialisable, recursively redacted body carrying the shared schema version."""
        from vision_mcp.security import redact_data

        return {
            "schema_version": SCHEMA_VERSION,
            "code": self.code.value,
            "message": redact_data(self.message),
            "details": redact_data(self.details),
        }
