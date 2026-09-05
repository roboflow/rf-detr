# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""`vision-engine` entry point."""

from __future__ import annotations

import os

# This must execute before app imports reach RF-DETR or Torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import uvicorn

from vision_mcp.config import load_config
from vision_mcp.engine.app import create_app
from vision_mcp.logging_setup import configure_logging, get_logger

logger = get_logger("vision-mcp.engine")


def main() -> None:
    """Load config and run a loopback-only Uvicorn server until SIGINT."""
    parser = argparse.ArgumentParser(description="RF-DETR local vision engine")
    parser.add_argument(
        "--config",
        default=os.environ.get("VISION_MCP_CONFIG", "config.yaml"),
        help="path to the engine YAML configuration",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    configure_logging(config.engine.log_level)
    logger.info(
        "engine starting",
        extra={
            "config": str(Path(args.config).resolve()),
            "bind": f"127.0.0.1:{config.engine.port}",
            "mps_fallback": True,
        },
    )
    uvicorn.run(
        create_app(config),
        host="127.0.0.1",
        port=config.engine.port,
        log_level=config.engine.log_level.lower(),
    )


if __name__ == "__main__":
    main()
