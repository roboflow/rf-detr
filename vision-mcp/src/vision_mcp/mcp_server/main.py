"""`vision-mcp` stdio entry point."""

from __future__ import annotations

import argparse
import os

from vision_mcp.logging_setup import configure_logging
from vision_mcp.mcp_server.server import run


def main() -> None:
    """Start the stateless MCP facade without probing or spawning the engine."""
    parser = argparse.ArgumentParser(description="RF-DETR Vision MCP stdio server")
    parser.add_argument(
        "--engine",
        default=os.environ.get("VISION_ENGINE_ADDRESS", "http://127.0.0.1:8765"),
        help="vision-engine HTTP address",
    )
    args = parser.parse_args()
    configure_logging(os.environ.get("VISION_MCP_LOG_LEVEL", "WARNING"))
    run(args.engine)


if __name__ == "__main__":
    main()
