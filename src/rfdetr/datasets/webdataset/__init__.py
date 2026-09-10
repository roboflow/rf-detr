# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""WebDataset storage package.

Purpose: Separate shard index, packing and streaming-load responsibilities. Scope: internal implementation modules.
Usage: import index, pack or load by responsibility. Outputs: stable WebDataset index, shard and loader contracts.
Failure: module-specific validation errors surface unchanged. Used by: RF-DETR data and CLI paths.
"""
