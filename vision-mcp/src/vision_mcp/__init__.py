# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Local-first RF-DETR vision engine and MCP server.

Two processes share this package: ``vision-engine`` owns cameras, models and storage; ``vision-mcp`` is a
stateless stdio MCP server that wraps the engine's HTTP API.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
