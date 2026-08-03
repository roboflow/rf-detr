"""Low-level MCP registration and stdio transport."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import unquote, urlsplit

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.security import validate_artifact_id

from .client import EngineClient
from .definitions import resource_templates, resources, tool_definitions


def build_server(address: str) -> Server[Any]:
    """Build and register the entire MCP surface without contacting the engine."""
    client = EngineClient(address)

    async def list_tools(context: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_definitions())

    async def call_tool(context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            data = await client.call(params.name, params.arguments or {})
            return _tool_result(data)
        except VisionError as exc:
            return _tool_result(exc.to_payload(), is_error=True)

    async def list_resources(context: Any, params: Any) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=resources())

    async def list_templates(context: Any, params: Any) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resource_templates=resource_templates())

    async def read_resource(
        context: Any, params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        artifact_id = _artifact_id(uri)
        if artifact_id is not None:
            try:
                data, media_type = await client.read_artifact(artifact_id)
            except VisionError as exc:
                return _json_resource(uri, exc.to_payload())
            return types.ReadResourceResult(
                contents=[
                    types.BlobResourceContents(
                        uri=uri,
                        mime_type=media_type,
                        blob=base64.b64encode(data).decode("ascii"),
                    )
                ]
            )
        try:
            tool, arguments = _resource_call(uri)
            data = await client.call(tool, arguments)
        except VisionError as exc:
            data = exc.to_payload()
        return _json_resource(uri, data)

    return Server(
        "vision-mcp",
        version="0.1.0",
        description="Stateless RF-DETR Vision MCP facade",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_list_resource_templates=list_templates,
        on_read_resource=read_resource,
    )


async def run_server(address: str) -> None:
    """Run the MCP server over stdio; the engine is never spawned here."""
    server = build_server(address)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run(address: str) -> None:
    """Synchronous CLI bridge for AnyIO."""
    anyio.run(run_server, address)


def _tool_result(data: dict[str, Any], is_error: bool = False) -> types.CallToolResult:
    """Return both JSON text and structured content."""
    text = json.dumps(data, separators=(",", ":"))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=data,
        is_error=is_error,
    )


def _json_resource(uri: str, data: dict[str, Any]) -> types.ReadResourceResult:
    """Return one compact JSON text resource."""
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=uri, mime_type="application/json", text=json.dumps(data, separators=(",", ":"))
            )
        ]
    )


def _artifact_id(uri: str) -> str | None:
    """Return a validated artifact ID for an artifact URI, or None for another resource family."""
    parsed = urlsplit(uri)
    if parsed.netloc != "artifacts":
        return None
    path = [unquote(part) for part in parsed.path.split("/") if part]
    if len(path) != 1:
        candidate = path[-1] if path else ""
        validate_artifact_id(candidate)
        raise VisionError(ErrorCode.ARTIFACT_NOT_FOUND, "Unknown artifact id.")
    return validate_artifact_id(path[0])


def _resource_call(uri: str) -> tuple[str, dict[str, Any]]:
    """Map a registered vision URI to its engine operation."""
    parsed = urlsplit(uri)
    path = [unquote(part) for part in parsed.path.split("/") if part]
    if parsed.netloc == "system" and path == ["status"]:
        return "get_system_status", {}
    if parsed.netloc == "streams" and not path:
        return "list_active_streams", {}
    if parsed.netloc == "models" and not path:
        return "list_models", {}
    if parsed.netloc == "streams" and len(path) == 2:
        stream_id, leaf = path
        mapping = {
            "status": "get_stream_status",
            "counts": "get_current_counts",
            "events": "get_recent_detection_events",
        }
        if leaf in mapping:
            return mapping[leaf], {"stream_id": stream_id}
    if parsed.netloc == "models" and len(path) == 2 and path[1] == "status":
        return "get_model_status", {"model": path[0]}
    raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Unknown vision resource URI {uri!r}.")
