# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load and execute Core AI exports on a dedicated event loop."""

from __future__ import annotations

import asyncio
import inspect
import threading
import weakref
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.utilities.logger import get_logger

logger = get_logger()


async def _await_result(value: Any) -> Any:
    """Resolve an operation on runtime versions with async execution."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_coreai(
    session: Any, input_name: str | int, array: np.ndarray[Any, Any], metadata: ExportMetadata
) -> dict[str, Any]:
    """Run a Core AI function and copy its named results."""
    from coreai.runtime import NDArray

    result = await _await_result(session({input_name: NDArray(array)}))
    identifiers = list(metadata.outputs.values())
    if any(not isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("Core AI output mappings must use names.")
    names = [identifier for identifier in identifiers if isinstance(identifier, str)]
    outputs: dict[str, Any] = {}
    for name in names:
        value = result[name]
        if hasattr(value, "numpy"):
            value = value.numpy()
        outputs[name] = np.array(value, copy=True)
    return outputs


def _serve_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Keep one event loop alive for a loaded model."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Stop the loop when its bridge is no longer used."""
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if threading.current_thread() is not thread:
        thread.join(timeout=1)
    if not loop.is_running():
        loop.close()


class _CoreAILoopBridge:
    """Run a synchronous prediction API on one long-lived event loop."""

    def __init__(self) -> None:
        """Start the event loop for one Core AI session."""
        self.loop = asyncio.new_event_loop()
        self._call_lock = threading.Lock()
        self.thread = threading.Thread(target=_serve_loop, args=(self.loop,), daemon=True, name="rfdetr-coreai")
        self.thread.start()
        self._finalizer = weakref.finalize(self, _stop_loop, self.loop, self.thread)

    def run(self, operation: Any) -> Any:
        """Wait for one Core AI operation on the owning loop."""
        with self._call_lock:
            return asyncio.run_coroutine_threadsafe(operation, self.loop).result()


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str) -> Any:
    """Load Core AI with a validated compute policy."""
    from rfdetr.export._runtime.adapters import ExportRuntime, _input_array, _require_apple

    _require_apple("Core AI")
    try:
        from coreai.runtime import AIModel, SpecializationOptions
    except ImportError as exc:
        raise ImportError("Core AI inference requires coreai-core and macOS 27 or later.") from exc
    if device in {"gpu", "ane"}:
        raise ValueError("Core AI gpu and ane select a preferred compute unit, not a strict device. Use auto or cpu.")
    if device not in {"auto", "cpu"}:
        raise ValueError("Core AI device must be auto or cpu.")
    if metadata.task == "keypoints" and metadata.input_dtype == "float16" and device == "auto":
        # The Neural Engine aborts float16 keypoint inference; CPU execution preserves a usable artifact.
        logger.warning("Core AI float16 keypoint inference can abort on the Neural Engine; using CPU for auto.")
        device = "cpu"
    options = SpecializationOptions.cpu_only() if device == "cpu" else SpecializationOptions.default()

    async def load_coreai() -> Any:
        """Load the model and select its entry point."""
        loaded = await _await_result(AIModel.load(path, options))
        return await _await_result(loaded.load_function("main"))

    bridge = _CoreAILoopBridge()
    session = bridge.run(load_coreai())

    def execute(batch: torch.Tensor) -> dict[str, Any]:
        """Run one batch on the model's owning event loop."""
        return cast(
            dict[str, Any],
            bridge.run(_run_coreai(session, metadata.input_name, _input_array(batch, metadata), metadata)),
        )

    return ExportRuntime("coreai", metadata, session, device, metadata.input_name, execute, borrowed_outputs=False)
