# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""FastAPI application exposing the local-only engine contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from vision_mcp import __version__
from vision_mcp.config import EngineConfig
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.security import redact

from .services import EngineServices


def create_app(config: EngineConfig) -> FastAPI:
    """Build an engine app around one validated configuration."""
    services = EngineServices(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services
        await services.start()
        try:
            yield
        finally:
            await services.stop()

    app = FastAPI(title="RF-DETR Vision Engine", version=__version__, lifespan=lifespan)

    @app.exception_handler(VisionError)
    async def vision_error(request: Request, exc: VisionError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        error = VisionError(
            ErrorCode.INVALID_ARGUMENT,
            "Request validation failed.",
            {"errors": exc.errors(include_url=False, include_input=False)},
        )
        return JSONResponse(status_code=400, content=error.to_payload())

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        error = VisionError(
            ErrorCode.INFERENCE_FAILED,
            "Engine request failed.",
            {"error": redact(exc)},
        )
        return JSONResponse(status_code=500, content=error.to_payload())

    @app.post("/tools/{name}")
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one typed M1-M6 operation."""
        result = await services.call(name, arguments)
        return result.model_dump(mode="json")

    @app.get("/streams/{stream_id}/live")
    async def live(stream_id: str) -> dict[str, Any]:
        """Cheap current state read with no database access."""
        return services.streams.snapshot(stream_id).model_dump(mode="json")

    @app.get("/streams/{stream_id}/frame.jpg")
    async def frame(stream_id: str, annotate: bool = Query(default=False)) -> Response:
        """Encode the latest already-decoded frame, optionally drawing overlays."""
        return Response(content=await services.frame_jpeg(stream_id, annotate), media_type="image/jpeg")

    @app.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: str) -> Response:
        """Return bytes for a generated, containment-checked artifact ID."""
        data, media_type = await services.artifacts.read(artifact_id)
        return Response(content=data, media_type=media_type)

    @app.get("/debug/preview")
    async def debug_preview(stream_id: str | None = None) -> Response:
        """Return one annotated debug frame when preview is enabled."""
        runtime = _preview_runtime(services, stream_id)
        return Response(content=await services.preview.snapshot(runtime), media_type="image/jpeg")

    @app.get("/debug/stream")
    async def debug_stream(stream_id: str | None = None) -> StreamingResponse:
        """Stream on-demand annotated JPEG frames to a browser."""
        runtime = _preview_runtime(services, stream_id)
        services.preview.require_enabled()
        return StreamingResponse(
            services.preview.mjpeg(runtime), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def _preview_runtime(services: EngineServices, stream_id: str | None) -> Any:
    """Resolve an explicit stream or the first configured one for debug routes."""
    if stream_id is not None:
        return services.streams.get(stream_id)
    runtime = next(iter(services.streams), None)
    if runtime is None:
        raise VisionError(ErrorCode.STREAM_NOT_FOUND, "No streams are configured.")
    return runtime
