# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Device resolution for Apple Silicon first, CUDA only when explicitly configured."""

from __future__ import annotations

import os

from vision_mcp.logging_setup import get_logger

logger = get_logger("vision-mcp.device")

MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


def enable_mps_fallback() -> None:
    """Allow unimplemented MPS ops to run on CPU.

    Must be called before torch is imported.
    """
    if os.environ.get(MPS_FALLBACK_ENV) != "1":
        os.environ[MPS_FALLBACK_ENV] = "1"
    logger.info("MPS CPU fallback active", extra={"env": MPS_FALLBACK_ENV})


def resolve_device(requested: str) -> str:
    """Turn a configured device string into a concrete torch device.

    ``auto`` prefers MPS and falls back to CPU; CUDA is used only when asked for by name and actually present.
    """
    import torch

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU", extra={"requested": requested})
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable; using CPU")
        return "cpu"
    return requested


def fallback_device(device: str) -> str | None:
    """The device to retry on after a load failure, or None when there is nothing left to try."""
    return None if device == "cpu" else "cpu"
