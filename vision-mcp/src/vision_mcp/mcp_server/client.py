"""HTTP client that validates every engine response against shared contracts."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from vision_mcp.api_contract import (
    ActiveTracks,
    ArtifactResult,
    ClassCounts,
    CompareResult,
    ConfidenceDistribution,
    CountsByClass,
    CropResult,
    CurrentCounts,
    DetectionRate,
    DetectionResult,
    DwellResult,
    EntryExitResult,
    ErrorList,
    ErrorResponse,
    EventList,
    FrameDropRate,
    GpuMetrics,
    LatencyStats,
    LineCrossingResult,
    ModelInfo,
    ModelList,
    ModelStatus,
    QueueMetrics,
    StreamList,
    StreamStatus,
    SystemStatus,
    ThroughputStats,
    UniqueObjectCount,
    WorkerList,
    ZoneOccupancyResult,
)
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.security import redact, validate_artifact_id

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "list_models": ModelList,
    "get_model_info": ModelInfo,
    "detect_objects": DetectionResult,
    "segment_instances": DetectionResult,
    "detect_keypoints": DetectionResult,
    "count_objects": CountsByClass,
    "find_objects": DetectionResult,
    "crop_detections": CropResult,
    "compare_detections": CompareResult,
    "get_system_status": SystemStatus,
    "get_stream_status": StreamStatus,
    "list_active_streams": StreamList,
    "get_model_status": ModelStatus,
    "get_worker_status": WorkerList,
    "get_current_counts": CurrentCounts,
    "get_counts_by_class": ClassCounts,
    "get_unique_object_count": UniqueObjectCount,
    "get_detection_rate": DetectionRate,
    "get_confidence_distribution": ConfidenceDistribution,
    "get_active_tracks": ActiveTracks,
    "get_zone_occupancy": ZoneOccupancyResult,
    "get_entry_exit_counts": EntryExitResult,
    "get_dwell_times": DwellResult,
    "get_line_crossing_events": LineCrossingResult,
    "get_inference_latency": LatencyStats,
    "get_processing_throughput": ThroughputStats,
    "get_frame_drop_rate": FrameDropRate,
    "get_gpu_metrics": GpuMetrics,
    "get_queue_metrics": QueueMetrics,
    "get_latest_annotated_frame": ArtifactResult,
    "get_event_snapshot": ArtifactResult,
    "get_recent_detection_events": EventList,
    "get_recent_errors": ErrorList,
}


class EngineClient:
    """No-state wrapper around one configured engine address."""

    def __init__(self, address: str, timeout: float = 30.0) -> None:
        self.address = address.rstrip("/")
        self._timeout = timeout

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call and validate one operation, converting transport failures."""
        model = RESPONSE_MODELS.get(name)
        if model is None:
            raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Unknown MCP tool {name!r}.")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.address}/tools/{name}", json=arguments)
        except httpx.RequestError as exc:
            raise VisionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                f"Vision engine is unavailable at {self.address}.",
                {"engine_address": self.address, "error": redact(exc)},
            ) from exc
        data = _json(response)
        if response.is_error:
            error = _error(data)
            try:
                code = ErrorCode(error.code)
            except ValueError:
                code = ErrorCode.INFERENCE_FAILED
            raise VisionError(code, error.message, error.details)
        try:
            validated = model.model_validate(data)
        except ValidationError as exc:
            raise VisionError(
                ErrorCode.INFERENCE_FAILED,
                "Engine returned a response that violates the shared API contract.",
                {"tool": name, "error": redact(exc)},
            ) from exc
        return validated.model_dump(mode="json")

    async def read_artifact(self, artifact_id: str) -> tuple[bytes, str]:
        """Fetch one validated artifact as bytes from the engine."""
        artifact_id = validate_artifact_id(artifact_id)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self.address}/artifacts/{artifact_id}")
        except httpx.RequestError as exc:
            raise VisionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                f"Vision engine is unavailable at {self.address}.",
                {"engine_address": self.address, "error": redact(exc)},
            ) from exc
        if response.is_error:
            error = _error(_json(response))
            try:
                code = ErrorCode(error.code)
            except ValueError:
                code = ErrorCode.INFERENCE_FAILED
            raise VisionError(code, error.message, error.details)
        media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        return response.content, media_type


def _json(response: httpx.Response) -> Any:
    """Decode JSON or produce a contract failure."""
    try:
        return response.json()
    except ValueError as exc:
        raise VisionError(
            ErrorCode.INFERENCE_FAILED,
            "Engine returned a non-JSON response.",
            {"status": response.status_code},
        ) from exc


def _error(data: Any) -> ErrorResponse:
    """Validate an engine error payload."""
    try:
        return ErrorResponse.model_validate(data)
    except ValidationError as exc:
        raise VisionError(
            ErrorCode.INFERENCE_FAILED,
            "Engine returned an invalid error response.",
            {"error": redact(exc)},
        ) from exc
